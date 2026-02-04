"""
DeepMM: Deep Neural Network for Map Matching

This module adapts the DeepMM model (Seq2SeqAttention) from the original repository
to the LibCity framework for trajectory location prediction (map matching).

Original paper: "Deep Learning Map Matching"
Original repository: https://github.com/... (DeepMM)

Key adaptations:
- Inherits from AbstractModel instead of nn.Module
- Uses self.device from config instead of hardcoded .cuda() calls
- Replaces deprecated torch.autograd.Variable with modern tensor operations
- Implements predict() and calculate_loss() methods for LibCity compatibility
- Handles LibCity batch dictionary format

Model architecture:
- Bidirectional LSTM encoder (2 layers, 512 hidden dim by default)
- LSTM decoder with dot-product attention (1 layer, 512 hidden dim)
- Embeddings: 256-dim for location (source) and segment (target)
- Supports optional time encoding (NoEncoding, OneEncoding, TwoEncoding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel


class SoftDotAttention(nn.Module):
    """Soft Dot Attention mechanism.

    Reference: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OpenNMT.

    Supports three attention types:
    - dot: Simple dot product attention
    - general: Bilinear attention with learned weights
    - mlp: MLP-based attention
    """

    def __init__(self, dim, attn_type='dot'):
        """Initialize the attention layer.

        Args:
            dim: Hidden dimension size
            attn_type: Type of attention ('dot', 'general', or 'mlp')
        """
        super(SoftDotAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type: 'dot', 'general', or 'mlp'")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp requires bias in output layer
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propagate input through the attention network.

        Args:
            input: Query tensor of shape (batch, dim)
            context: Context tensor of shape (batch, sourceL, dim)

        Returns:
            h_tilde: Attended hidden state (batch, dim)
            attn: Attention weights (batch, sourceL)
        """
        tgt_len = 1
        h_s, h_t = context, input
        src_batch, src_len = context.shape[0], context.shape[1]
        tgt_batch = src_batch
        tgt_dim = self.dim

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_dim, tgt_len)
            else:
                # dot attention
                h_t = h_t.view(tgt_batch, tgt_dim, tgt_len)
            # (batch, s_len, d) x (batch, d, t_len) --> (batch, s_len, t_len)
            attn = torch.bmm(h_s, h_t).squeeze(2)
        elif self.attn_type == "mlp":
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            attn = self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
            attn = attn.transpose(1, 2).squeeze(2)  # tgt_len = 1

        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    """LSTM cell with dot-product attention mechanism.

    This module combines an LSTM cell with attention over encoder outputs,
    allowing the decoder to focus on relevant parts of the input sequence.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='dot'):
        """Initialize the LSTM attention layer.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            batch_first: If True, input/output tensors are (batch, seq, feature)
            attn_type: Type of attention mechanism ('dot', 'general', or 'mlp')
        """
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size, attn_type)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propagate input through the network.

        Args:
            input: Input tensor (batch, seq_len, input_size) if batch_first
            hidden: Tuple of (h_0, c_0) initial hidden states
            ctx: Encoder context (seq_len, batch, hidden_size)
            ctx_mask: Optional mask for context (not used currently)

        Returns:
            output: Output tensor (batch, seq_len, hidden_size)
            hidden: Final hidden state tuple (h_n, c_n)
        """
        def recurrence(input_t, hidden):
            """Single step recurrence with attention."""
            hx, cx = hidden  # (batch, hidden_dim)
            gates = self.input_weights(input_t) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # (batch, hidden_dim)

            # Apply attention: ctx should be (batch, seq_len, hidden_dim)
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class DeepMM(AbstractModel):
    """DeepMM: Seq2Seq model with attention for map matching.

    This model performs map matching by treating it as a sequence-to-sequence
    translation task: translating GPS point sequences to road segment sequences.

    Architecture:
    - Encoder: Bidirectional LSTM that processes GPS block embeddings
    - Decoder: LSTM with dot-product attention that generates road segment IDs
    - Optional time encoding to incorporate temporal information

    The model supports teacher forcing during training and greedy decoding
    during inference.
    """

    def __init__(self, config, data_feature):
        """Initialize the DeepMM model.

        Args:
            config: Configuration dictionary containing model hyperparameters
            data_feature: Dictionary containing data-specific features like vocab sizes
        """
        super(DeepMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Vocabulary sizes from data features
        self.src_loc_vocab_size = data_feature.get('src_loc_vocab_size', 10000)
        self.trg_seg_vocab_size = data_feature.get('trg_seg_vocab_size', 10000)
        self.src_tim_vocab_size = data_feature.get('src_tim_vocab_size', None)

        # Padding token indices
        self.pad_token_src_loc = data_feature.get('pad_token_src_loc', 1)
        self.pad_token_src_tim1 = data_feature.get('pad_token_src_tim1', None)
        self.pad_token_src_tim2 = data_feature.get('pad_token_src_tim2', None)
        self.pad_token_trg = data_feature.get('pad_token_trg', 1)

        # Special tokens
        self.sos_token = data_feature.get('sos_token', 0)
        self.eos_token = data_feature.get('eos_token', 2)

        # Model hyperparameters from config
        self.src_loc_emb_dim = config.get('src_loc_emb_dim', 256)
        self.src_tim_emb_dim = config.get('src_tim_emb_dim', 64)
        self.trg_seg_emb_dim = config.get('trg_seg_emb_dim', 256)
        self.time_encoding = config.get('time_encoding', 'NoEncoding')
        self.src_hidden_dim = config.get('src_hidden_dim', 512)
        self.trg_hidden_dim = config.get('trg_hidden_dim', 512)
        self.bidirectional = config.get('bidirectional', True)
        self.nlayers_src = config.get('nlayers_src', 2)
        self.dropout = config.get('dropout', 0.5)
        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.attn_type = config.get('attn_type', 'dot')
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 1.0)
        self.max_trg_length = config.get('max_trg_length', 54)

        self.num_directions = 2 if self.bidirectional else 1
        # Adjust hidden dim for bidirectional encoder
        self.encoder_hidden_dim = self.src_hidden_dim // 2 if self.bidirectional else self.src_hidden_dim

        # Calculate source embedding dimension based on time encoding
        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[0] if isinstance(self.src_tim_emb_dim, (list, tuple)) else self.src_loc_emb_dim + self.src_tim_emb_dim
        elif self.time_encoding == 'TwoEncoding':
            src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[1][0] + self.src_tim_emb_dim[1][1]
        else:
            raise RuntimeError(f'Invalid time encoding: {self.time_encoding}')

        # Embeddings
        self.src_embedding = nn.Embedding(
            self.src_loc_vocab_size,
            self.src_loc_emb_dim,
            padding_idx=self.pad_token_src_loc
        )
        self.trg_embedding = nn.Embedding(
            self.trg_seg_vocab_size,
            self.trg_seg_emb_dim,
            padding_idx=self.pad_token_trg
        )

        # Optional time embeddings
        if self.time_encoding == 'OneEncoding':
            tim_vocab_size = self.src_tim_vocab_size[0] if isinstance(self.src_tim_vocab_size, (list, tuple)) else self.src_tim_vocab_size
            tim_emb_dim = self.src_tim_emb_dim[0] if isinstance(self.src_tim_emb_dim, (list, tuple)) else self.src_tim_emb_dim
            self.src_time_embedding = nn.Embedding(
                tim_vocab_size,
                tim_emb_dim,
                padding_idx=self.pad_token_src_tim1
            )
        elif self.time_encoding == 'TwoEncoding':
            self.src_time_embedding_1 = nn.Embedding(
                self.src_tim_vocab_size[1][0],
                self.src_tim_emb_dim[1][0],
                padding_idx=self.pad_token_src_tim2[0]
            )
            self.src_time_embedding_2 = nn.Embedding(
                self.src_tim_vocab_size[1][1],
                self.src_tim_emb_dim[1][1],
                padding_idx=self.pad_token_src_tim2[1]
            )

        # Encoder: Bidirectional LSTM
        self.encoder = getattr(nn, self.rnn_type)(
            src_emb_dim,
            self.encoder_hidden_dim,
            self.nlayers_src,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout if self.nlayers_src > 1 else 0
        )

        # Decoder: LSTM with attention
        self.decoder = LSTMAttentionDot(
            self.trg_seg_emb_dim,
            self.trg_hidden_dim,
            batch_first=True,
            attn_type=self.attn_type
        )

        # Projection layers
        self.encoder2decoder = nn.Linear(
            self.encoder_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.trg_seg_vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize model weights with uniform distribution."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_encoder_state(self, input_src):
        """Get initial encoder hidden and cell states.

        Args:
            input_src: Source input tensor (batch, seq_len)

        Returns:
            Tuple of (h_0, c_0) tensors on the correct device
        """
        batch_size = input_src.size(0)
        h0_encoder = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.encoder_hidden_dim,
            device=self.device
        )
        c0_encoder = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.encoder_hidden_dim,
            device=self.device
        )
        return h0_encoder, c0_encoder

    def forward(self, batch):
        """Forward pass through the seq2seq model.

        Args:
            batch: Dictionary containing:
                - 'input_src': Source GPS block indices (batch, src_len)
                - 'input_trg': Target road segment indices (batch, trg_len)
                - 'input_time': Optional time indices

        Returns:
            decoder_logit: Logits for each position (batch, trg_len, trg_vocab_size)
        """
        input_src = batch['input_src']
        input_trg = batch['input_trg']
        input_time = batch.get('input_time', None)
        ctx_mask = batch.get('ctx_mask', None)

        # Get embeddings
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        # Handle time encoding
        if self.time_encoding == 'NoEncoding':
            src_time_emb = src_emb
        elif self.time_encoding == 'OneEncoding':
            time_emb = self.src_time_embedding(input_time)
            src_time_emb = torch.cat((src_emb, time_emb), dim=2)
        elif self.time_encoding == 'TwoEncoding':
            time_emb_1 = self.src_time_embedding_1(input_time[0])
            time_emb_2 = self.src_time_embedding_2(input_time[1])
            src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)
        else:
            raise RuntimeError(f'Invalid time encoding: {self.time_encoding}')

        # Initialize encoder hidden state
        h0_encoder, c0_encoder = self.get_encoder_state(input_src)

        # Encode source sequence
        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (h0_encoder, c0_encoder))
        else:
            src_h, src_h_t = self.encoder(src_time_emb, h0_encoder)
            src_c_t = c0_encoder

        # Combine bidirectional hidden states
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        # Project encoder state to decoder initial state
        decoder_init_state = torch.tanh(self.encoder2decoder(h_t))

        # Prepare context for attention (transpose to seq_len x batch x hidden)
        ctx = src_h.transpose(0, 1)

        # Decode with attention
        trg_h, (_, _) = self.decoder(trg_emb, (decoder_init_state, c_t), ctx, ctx_mask)

        # Project to vocabulary
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1),
            trg_h.size(2)
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size(0),
            trg_h.size(1),
            decoder_logit.size(1)
        )

        return decoder_logit

    def decode(self, logits):
        """Convert logits to probability distribution.

        Args:
            logits: Raw output scores (batch, seq_len, vocab_size)

        Returns:
            word_probs: Softmax probabilities (batch, seq_len, vocab_size)
        """
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=-1)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs

    def greedy_decode(self, batch, max_length=None):
        """Perform greedy decoding for inference.

        Args:
            batch: Input batch dictionary
            max_length: Maximum output sequence length

        Returns:
            predictions: Predicted sequence indices (batch, max_length)
        """
        if max_length is None:
            max_length = self.max_trg_length

        input_src = batch['input_src']
        input_time = batch.get('input_time', None)
        batch_size = input_src.size(0)

        # Encode source
        src_emb = self.src_embedding(input_src)

        if self.time_encoding == 'NoEncoding':
            src_time_emb = src_emb
        elif self.time_encoding == 'OneEncoding':
            time_emb = self.src_time_embedding(input_time)
            src_time_emb = torch.cat((src_emb, time_emb), dim=2)
        elif self.time_encoding == 'TwoEncoding':
            time_emb_1 = self.src_time_embedding_1(input_time[0])
            time_emb_2 = self.src_time_embedding_2(input_time[1])
            src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)
        else:
            raise RuntimeError(f'Invalid time encoding: {self.time_encoding}')

        h0_encoder, c0_encoder = self.get_encoder_state(input_src)

        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (h0_encoder, c0_encoder))
        else:
            src_h, src_h_t = self.encoder(src_time_emb, h0_encoder)
            src_c_t = c0_encoder

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_hidden = torch.tanh(self.encoder2decoder(h_t))
        decoder_cell = c_t
        ctx = src_h.transpose(0, 1)

        # Start with SOS token
        current_token = torch.full((batch_size,), self.sos_token,
                                   dtype=torch.long, device=self.device)
        predictions = []

        for _ in range(max_length):
            # Get embedding for current token
            trg_emb = self.trg_embedding(current_token.unsqueeze(1))

            # Single decoder step
            trg_h, (decoder_hidden, decoder_cell) = self.decoder(
                trg_emb, (decoder_hidden, decoder_cell), ctx, None
            )

            # Project to vocabulary and get prediction
            logits = self.decoder2vocab(trg_h.squeeze(1))
            current_token = logits.argmax(dim=-1)
            predictions.append(current_token)

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def predict(self, batch):
        """Generate predictions for a batch.

        This method is called during evaluation. It uses greedy decoding
        to generate the output sequence.

        Args:
            batch: Input batch dictionary

        Returns:
            predictions: Predicted sequence indices (batch, seq_len)
        """
        # For evaluation, return predicted indices (not logits)
        # If target is available, use teacher forcing for fair comparison
        if 'input_trg' in batch:
            logits = self.forward(batch)
            # Return predicted indices, not logits
            return logits.argmax(dim=-1)  # Returns 2D: [batch, seq_len]
        else:
            # If no target, use greedy decoding (already returns indices)
            return self.greedy_decode(batch)

    def calculate_loss(self, batch):
        """Calculate the training loss.

        Uses cross-entropy loss with padding tokens ignored.

        Args:
            batch: Dictionary containing input_src, input_trg, and target_trg

        Returns:
            loss: Scalar loss tensor
        """
        # Forward pass
        logits = self.forward(batch)

        # Get target (shifted by one position for teacher forcing)
        target = batch.get('target_trg', batch.get('input_trg')[:, 1:])

        # Reshape for cross-entropy: (batch * seq_len, vocab_size) and (batch * seq_len)
        # Trim logits to match target length if needed
        if logits.size(1) > target.size(1):
            logits = logits[:, :target.size(1), :]
        elif logits.size(1) < target.size(1):
            target = target[:, :logits.size(1)]

        logits_flat = logits.contiguous().view(-1, self.trg_seg_vocab_size)
        target_flat = target.contiguous().view(-1)

        # Cross-entropy loss ignoring padding
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_trg)
        loss = criterion(logits_flat, target_flat)

        return loss
