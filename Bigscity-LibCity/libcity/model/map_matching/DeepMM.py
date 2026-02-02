"""
DeepMM: Deep Learning-based Map Matching Model
Adapted from: repos/DeepMM/DeepMM/model.py (Seq2SeqAttention class, lines 825-1034)

Architecture:
- Encoder: Bidirectional LSTM
- Decoder: LSTM with soft attention
- Task: Map matching (GPS trajectory to road segment sequence)

Original Paper Reference:
- Deep learning based map matching

Key Adaptations for LibCity:
- Removed deprecated torch.autograd.Variable usage
- Made device-agnostic (no hardcoded .cuda() calls)
- Added LibCity config/data_feature interface
- Implemented predict() and calculate_loss() methods
- Fixed deprecated F.sigmoid/F.tanh to torch.sigmoid/torch.tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel


class SoftDotAttention(nn.Module):
    """Soft Dot Attention layer supporting dot, general, and mlp attention types.

    Reference: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OpenNMT.
    """

    def __init__(self, dim, attn_type='dot'):
        """Initialize attention layer.

        Args:
            dim: Hidden dimension size
            attn_type: Type of attention - 'dot', 'general', or 'mlp'
        """
        super(SoftDotAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type

        assert self.attn_type in ["dot", "general", "mlp"], \
            "Please select a valid attention type: dot, general, or mlp"

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, input, context):
        """Propagate input through the attention layer.

        Args:
            input: Decoder hidden state, shape (batch, dim)
            context: Encoder outputs, shape (batch, src_len, dim)

        Returns:
            h_tilde: Attention-weighted output, shape (batch, dim)
            attn: Attention weights, shape (batch, src_len)
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
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x src_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    """LSTM cell with attention mechanism.

    A custom LSTM implementation that incorporates attention at each timestep.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='dot'):
        """Initialize LSTM attention cell.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            batch_first: If True, input/output tensors are (batch, seq, feature)
            attn_type: Attention type - 'dot', 'general', or 'mlp'
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
        """Propagate input through the LSTM with attention.

        Args:
            input: Input sequence, shape (batch, seq_len, input_size) if batch_first
            hidden: Tuple of (h_0, c_0) initial states
            ctx: Encoder context, shape (seq_len, batch, hidden_size)
            ctx_mask: Optional mask for attention (not used currently)

        Returns:
            output: Output sequence with attention applied
            hidden: Final (h_n, c_n) states
        """
        def recurrence(input_t, hidden):
            """Single timestep recurrence with attention."""
            hx, cx = hidden  # (batch, hidden_dim)
            gates = self.input_weights(input_t) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            # Use torch functions instead of deprecated F.sigmoid/F.tanh
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # (batch, hidden_dim)

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
    """DeepMM: Deep Learning-based Map Matching Model.

    A sequence-to-sequence model with attention for map matching.
    Takes GPS trajectory (location IDs + optional time) as input and
    outputs road segment sequence.

    Architecture:
        - Encoder: Bidirectional LSTM
        - Decoder: LSTM with soft attention
        - Supports multiple attention types: dot, general, mlp
        - Optional time encoding: NoEncoding, OneEncoding, TwoEncoding

    Config Parameters:
        - src_loc_emb_dim: Source location embedding dimension (default: 256)
        - src_tim_emb_dim: Source time embedding dimension (default: 64)
        - trg_seg_emb_dim: Target segment embedding dimension (default: 256)
        - src_hidden_dim: Encoder hidden dimension (default: 512)
        - trg_hidden_dim: Decoder hidden dimension (default: 512)
        - bidirectional: Use bidirectional encoder (default: True)
        - nlayers_src: Number of encoder layers (default: 2)
        - dropout: Dropout rate (default: 0.1)
        - time_encoding: Time encoding type (default: 'NoEncoding')
        - rnn_type: RNN type for encoder (default: 'LSTM')
        - attn_type: Attention type (default: 'dot')

    Data Features:
        - src_loc_vocab_size: Source location vocabulary size
        - src_tim_vocab_size: Source time vocabulary size (for time encoding)
        - trg_seg_vocab_size: Target road segment vocabulary size
        - pad_token_src_loc: Padding token for source locations
        - pad_token_trg: Padding token for target segments
    """

    def __init__(self, config, data_feature):
        """Initialize DeepMM model.

        Args:
            config: Configuration dictionary with model hyperparameters
            data_feature: Data feature dictionary with vocabulary sizes
        """
        super(DeepMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Vocabulary sizes from data features
        self.src_loc_vocab_size = data_feature.get('src_loc_vocab_size')
        self.trg_seg_vocab_size = data_feature.get('trg_seg_vocab_size')

        # Padding tokens from data features
        self.pad_token_src_loc = data_feature.get('pad_token_src_loc', 0)
        self.pad_token_trg = data_feature.get('pad_token_trg', 0)

        # Embedding dimensions from config
        self.src_loc_emb_dim = config.get('src_loc_emb_dim', 256)
        self.trg_seg_emb_dim = config.get('trg_seg_emb_dim', 256)

        # Hidden dimensions
        self.src_hidden_dim = config.get('src_hidden_dim', 512)
        self.trg_hidden_dim = config.get('trg_hidden_dim', 512)

        # Model architecture settings
        self.bidirectional = config.get('bidirectional', True)
        self.nlayers_src = config.get('nlayers_src', 2)
        self.dropout = config.get('dropout', 0.1)
        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.attn_type = config.get('attn_type', 'dot')

        # Time encoding settings
        self.time_encoding = config.get('time_encoding', 'NoEncoding')
        self.src_tim_emb_dim = config.get('src_tim_emb_dim', 64)
        self.src_tim_vocab_size = data_feature.get('src_tim_vocab_size', None)
        self.pad_token_src_tim1 = data_feature.get('pad_token_src_tim1', 0)
        self.pad_token_src_tim2 = data_feature.get('pad_token_src_tim2', [0, 0])

        # Compute derived values
        self.num_directions = 2 if self.bidirectional else 1
        # For bidirectional, split hidden dim between forward and backward
        self.encoder_hidden_dim = self.src_hidden_dim // 2 if self.bidirectional else self.src_hidden_dim

        # Compute source embedding dimension based on time encoding
        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            if isinstance(self.src_tim_emb_dim, (list, tuple)):
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[0]
            else:
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim
        elif self.time_encoding == 'TwoEncoding':
            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                tim_dims = self.src_tim_emb_dim[1]
                src_emb_dim = self.src_loc_emb_dim + tim_dims[0] + tim_dims[1]
            else:
                src_emb_dim = self.src_loc_emb_dim + 2 * self.src_tim_emb_dim
        else:
            raise ValueError(f"Invalid time_encoding: {self.time_encoding}. "
                           f"Must be one of: NoEncoding, OneEncoding, TwoEncoding")

        # Source location embedding
        self.src_embedding = nn.Embedding(
            self.src_loc_vocab_size,
            self.src_loc_emb_dim,
            padding_idx=self.pad_token_src_loc
        )

        # Target segment embedding
        self.trg_embedding = nn.Embedding(
            self.trg_seg_vocab_size,
            self.trg_seg_emb_dim,
            padding_idx=self.pad_token_trg
        )

        # Time embeddings based on encoding type
        if self.time_encoding == 'OneEncoding':
            if isinstance(self.src_tim_vocab_size, (list, tuple)):
                tim_vocab = self.src_tim_vocab_size[0]
            else:
                tim_vocab = self.src_tim_vocab_size
            if isinstance(self.src_tim_emb_dim, (list, tuple)):
                tim_emb = self.src_tim_emb_dim[0]
            else:
                tim_emb = self.src_tim_emb_dim
            self.src_time_embedding = nn.Embedding(
                tim_vocab,
                tim_emb,
                padding_idx=self.pad_token_src_tim1
            )
        elif self.time_encoding == 'TwoEncoding':
            if isinstance(self.src_tim_vocab_size, (list, tuple)) and len(self.src_tim_vocab_size) > 1:
                tim_vocabs = self.src_tim_vocab_size[1]
            else:
                tim_vocabs = [self.src_tim_vocab_size, self.src_tim_vocab_size]
            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                tim_embs = self.src_tim_emb_dim[1]
            else:
                tim_embs = [self.src_tim_emb_dim, self.src_tim_emb_dim]
            self.src_time_embedding_1 = nn.Embedding(
                tim_vocabs[0],
                tim_embs[0],
                padding_idx=self.pad_token_src_tim2[0]
            )
            self.src_time_embedding_2 = nn.Embedding(
                tim_vocabs[1],
                tim_embs[1],
                padding_idx=self.pad_token_src_tim2[1]
            )

        # Encoder: BiLSTM or GRU
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

        # Bridge layer: encoder final state to decoder initial state
        self.encoder2decoder = nn.Linear(
            self.encoder_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        # Output projection: decoder hidden to vocabulary
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.trg_seg_vocab_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_encoder_state(self, batch_size):
        """Get initial encoder hidden states.

        Args:
            batch_size: Batch size

        Returns:
            h0: Initial hidden state, shape (num_layers * num_directions, batch, hidden)
            c0: Initial cell state, shape (num_layers * num_directions, batch, hidden)
        """
        h0 = torch.zeros(
            self.nlayers_src * self.num_directions,
            batch_size,
            self.encoder_hidden_dim,
            device=self.device
        )
        c0 = torch.zeros(
            self.nlayers_src * self.num_directions,
            batch_size,
            self.encoder_hidden_dim,
            device=self.device
        )
        return h0, c0

    def forward(self, batch):
        """Forward pass of the model.

        Args:
            batch: Dictionary containing:
                - 'input_src': Source GPS locations, shape (batch, src_len)
                - 'input_trg': Target road segments (teacher forcing), shape (batch, trg_len)
                - 'input_time': Optional time features for time encoding

        Returns:
            decoder_logit: Logits over target vocabulary, shape (batch, trg_len, vocab_size)
        """
        # Extract inputs from batch
        input_src = batch['input_src']  # (batch, src_len)
        input_trg = batch['input_trg']  # (batch, trg_len)
        input_time = batch.get('input_time', None)

        batch_size = input_src.size(0)

        # Embed source locations
        src_emb = self.src_embedding(input_src)  # (batch, src_len, emb_dim)

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
            raise ValueError(f"Invalid time_encoding: {self.time_encoding}")

        # Embed target segments
        trg_emb = self.trg_embedding(input_trg)  # (batch, trg_len, emb_dim)

        # Get initial encoder state
        h0, c0 = self.get_encoder_state(batch_size)

        # Encode source sequence
        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (h0, c0))
        else:
            src_h, src_h_t = self.encoder(src_time_emb, h0)
            src_c_t = c0

        # Combine bidirectional hidden states
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), dim=1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), dim=1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        # Transform encoder final state to decoder initial state
        decoder_init_state = torch.tanh(self.encoder2decoder(h_t))

        # Attention context: encoder outputs
        ctx = src_h.transpose(0, 1)  # (src_len, batch, hidden)

        # Decode with attention
        trg_h, _ = self.decoder(trg_emb, (decoder_init_state, c_t), ctx)

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
            logits: Raw model output, shape (batch, seq_len, vocab_size)

        Returns:
            word_probs: Probabilities over vocabulary, shape (batch, seq_len, vocab_size)
        """
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=-1)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs

    def predict(self, batch):
        """Generate predictions for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            predictions: Predicted token indices, shape (batch, seq_len)
        """
        logits = self.forward(batch)
        # Get most likely tokens
        predictions = torch.argmax(logits, dim=-1)
        return predictions

    def calculate_loss(self, batch):
        """Calculate training loss.

        Uses CrossEntropyLoss for sequence generation, ignoring padding tokens.

        Args:
            batch: Input batch dictionary containing:
                - 'input_src': Source sequence
                - 'input_trg': Target sequence for teacher forcing
                - 'target': Ground truth target sequence (shifted by 1)

        Returns:
            loss: Scalar loss tensor
        """
        logits = self.forward(batch)  # (batch, trg_len, vocab_size)

        # Target is the expected output (typically input_trg shifted by 1)
        target = batch['target']  # (batch, trg_len)

        # Reshape for CrossEntropyLoss: (batch * trg_len, vocab_size) and (batch * trg_len,)
        logits_flat = logits.view(-1, self.trg_seg_vocab_size)
        target_flat = target.view(-1)

        # CrossEntropyLoss with padding ignored
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_trg)
        loss = criterion(logits_flat, target_flat)

        return loss

    def greedy_decode(self, batch, max_len=100, sos_idx=1, eos_idx=2):
        """Greedy decoding for inference (without teacher forcing).

        Args:
            batch: Input batch dictionary with 'input_src'
            max_len: Maximum decoding length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index

        Returns:
            outputs: Generated sequences, shape (batch, decoded_len)
        """
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

        h0, c0 = self.get_encoder_state(batch_size)

        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (h0, c0))
        else:
            src_h, src_h_t = self.encoder(src_time_emb, h0)
            src_c_t = c0

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), dim=1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), dim=1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_hidden = torch.tanh(self.encoder2decoder(h_t))
        decoder_cell = c_t
        ctx = src_h.transpose(0, 1)

        # Start with SOS token
        current_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=self.device)
        outputs = [current_token]

        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len - 1):
            # Embed current token
            trg_emb = self.trg_embedding(current_token)  # (batch, 1, emb_dim)

            # Decode one step
            trg_h, (decoder_hidden, decoder_cell) = self.decoder(
                trg_emb, (decoder_hidden, decoder_cell), ctx
            )

            # Project to vocabulary
            logits = self.decoder2vocab(trg_h.squeeze(1))  # (batch, vocab_size)

            # Greedy selection
            current_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            outputs.append(current_token)

            # Check for EOS
            finished = finished | (current_token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        outputs = torch.cat(outputs, dim=1)  # (batch, decoded_len)
        return outputs
