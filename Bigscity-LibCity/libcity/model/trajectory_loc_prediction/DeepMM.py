"""
DeepMM: Deep Learning-based Map Matching Model for LibCity Framework

This module adapts the Seq2SeqAttention model from the DeepMM paper to the LibCity framework.
The original implementation is from: repos/DeepMM/DeepMM/model.py

Key Adaptations:
1. Removed deprecated torch.autograd.Variable wrappers
2. Replaced hardcoded .cuda() calls with device abstraction
3. Updated deprecated functions (F.sigmoid() -> torch.sigmoid())
4. Implemented LibCity interface (AbstractModel)
5. Added predict() and calculate_loss() methods

Original Model Classes Used:
- Seq2SeqAttention (lines 825-1034): Main encoder-decoder with attention
- LSTMAttentionDot (lines 391-443): LSTM decoder with dot-product attention
- SoftDotAttention (lines 303-388): Soft dot-product attention mechanism

Reference:
- DeepMM: Deep Learning based Map Matching with Heterogeneous Trajectory Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel


class SoftDotAttention(nn.Module):
    """Soft Dot Attention mechanism.

    Reference: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OpenNMT and the original DeepMM implementation.

    This attention mechanism supports three types:
    - 'dot': Simple dot product attention
    - 'general': Linear transformation before dot product
    - 'mlp': Multi-layer perceptron attention
    """

    def __init__(self, dim, attn_type='dot'):
        """Initialize the attention layer.

        Args:
            dim (int): Hidden dimension size
            attn_type (str): Attention type ('dot', 'general', or 'mlp')
        """
        super(SoftDotAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ["dot", "general", "mlp"], \
            "Please select a valid attention type: 'dot', 'general', or 'mlp'"

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp wants bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, input, context):
        """Compute attention-weighted context.

        Args:
            input: Query tensor of shape (batch, dim)
            context: Context tensor of shape (batch, source_len, dim)

        Returns:
            h_tilde: Attention-weighted output of shape (batch, dim)
            attn: Attention weights of shape (batch, source_len)
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
            else:  # dot
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
            attn = attn.transpose(1, 2).squeeze(2)  # tg_len = 1

        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    """LSTM cell with dot-product attention.

    This is a custom LSTM implementation where each step attends to
    the encoder outputs using the SoftDotAttention mechanism.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='dot'):
        """Initialize the LSTM attention layer.

        Args:
            input_size (int): Input embedding dimension
            hidden_size (int): Hidden state dimension
            batch_first (bool): Whether input is batch-first
            attn_type (str): Attention type ('dot', 'general', or 'mlp')
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
        """Process input through LSTM with attention.

        Args:
            input: Input tensor of shape (batch, seq_len, input_size) if batch_first
            hidden: Tuple of (h_0, c_0) initial states
            ctx: Encoder context of shape (seq_len, batch, hidden_size)
            ctx_mask: Optional context mask (not used in current implementation)

        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_size)
            hidden: Final (h_n, c_n) state tuple
        """
        def recurrence(input_t, hidden):
            """Process one time step."""
            hx, cx = hidden  # (batch, hidden_dim)
            gates = self.input_weights(input_t) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            # Updated: Use torch.sigmoid instead of deprecated F.sigmoid
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # (batch, hidden_dim)

            # Apply attention: ctx is (src_len, batch, hidden_dim)
            # attention_layer expects context as (batch, src_len, hidden_dim)
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)  # (seq_len, batch, input_size)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq_len, hidden_size)

        return output, hidden


class DeepMM(AbstractModel):
    """DeepMM: Deep Learning-based Map Matching Model.

    This model uses a sequence-to-sequence architecture with attention for
    map matching. It takes GPS trajectory points (discretized to grid cells)
    as input and outputs matched road segment sequences.

    The architecture consists of:
    1. Source location embedding layer
    2. Optional time embedding layers (OneEncoding or TwoEncoding)
    3. Bidirectional LSTM encoder
    4. LSTM decoder with dot-product attention
    5. Output projection layer

    Configuration Parameters:
        src_loc_emb_dim (int): Source location embedding dimension (default: 256)
        src_tim_emb_dim (int/list): Time embedding dimension(s) (default: 64)
        trg_seg_emb_dim (int): Target segment embedding dimension (default: 256)
        src_hidden_dim (int): Encoder hidden dimension (default: 512)
        trg_hidden_dim (int): Decoder hidden dimension (default: 512)
        bidirectional (bool): Use bidirectional encoder (default: True)
        nlayers_src (int): Number of encoder layers (default: 2)
        dropout (float): Dropout probability (default: 0.5)
        time_encoding (str): Time encoding type ('NoEncoding', 'OneEncoding', 'TwoEncoding')
        rnn_type (str): RNN type for encoder ('LSTM' or 'GRU') (default: 'LSTM')
        attn_type (str): Attention type ('dot', 'general', 'mlp') (default: 'dot')

    Data Features Required:
        src_loc_vocab_size (int): Source location vocabulary size
        trg_seg_vocab_size (int): Target segment vocabulary size
        pad_token_src_loc (int): Padding token ID for source
        pad_token_trg (int): Padding token ID for target
    """

    def __init__(self, config, data_feature):
        """Initialize the DeepMM model.

        Args:
            config (dict): Configuration dictionary with model hyperparameters
            data_feature (dict): Data features including vocabulary sizes
        """
        super(DeepMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # Vocabulary sizes from data features
        self.src_loc_vocab_size = data_feature.get('src_loc_vocab_size', 10000)
        self.trg_seg_vocab_size = data_feature.get('trg_seg_vocab_size', 5000)

        # Embedding dimensions
        self.src_loc_emb_dim = config.get('src_loc_emb_dim', 256)
        self.src_tim_emb_dim = config.get('src_tim_emb_dim', 64)
        self.trg_seg_emb_dim = config.get('trg_seg_emb_dim', 256)

        # Hidden dimensions
        self.src_hidden_dim = config.get('src_hidden_dim', 512)
        self.trg_hidden_dim = config.get('trg_hidden_dim', 512)

        # Architecture options
        self.bidirectional = config.get('bidirectional', True)
        self.nlayers_src = config.get('nlayers_src', 2)
        self.dropout = config.get('dropout', 0.5)
        self.time_encoding = config.get('time_encoding', 'NoEncoding')
        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.attn_type = config.get('attn_type', 'dot')

        # Padding tokens
        self.pad_token_src_loc = data_feature.get('pad_token_src_loc', 1)
        self.pad_token_trg = data_feature.get('pad_token_trg', 1)

        # Time encoding padding tokens (optional)
        self.pad_token_src_tim1 = config.get('pad_token_src_tim1', 1)
        self.pad_token_src_tim2 = config.get('pad_token_src_tim2', [1, 1])

        # Time vocabulary sizes (for time encoding)
        self.src_tim_vocab_size = data_feature.get('src_tim_vocab_size', [1000, [24, 60]])

        # Calculate number of directions
        self.num_directions = 2 if self.bidirectional else 1

        # Adjust encoder hidden dim for bidirectional
        self.encoder_hidden_dim = self.src_hidden_dim // 2 if self.bidirectional else self.src_hidden_dim

        # Build embedding layers
        self._build_embeddings()

        # Build encoder and decoder
        self._build_encoder_decoder()

        # Initialize weights
        self.init_weights()

    def _build_embeddings(self):
        """Build embedding layers for source locations, times, and target segments."""
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

        # Time embeddings (optional)
        if self.time_encoding == 'OneEncoding':
            tim_vocab_size = self.src_tim_vocab_size[0] if isinstance(
                self.src_tim_vocab_size, (list, tuple)) else self.src_tim_vocab_size
            tim_emb_dim = self.src_tim_emb_dim[0] if isinstance(
                self.src_tim_emb_dim, (list, tuple)) else self.src_tim_emb_dim
            self.src_time_embedding = nn.Embedding(
                tim_vocab_size,
                tim_emb_dim,
                padding_idx=self.pad_token_src_tim1
            )
        elif self.time_encoding == 'TwoEncoding':
            # Two separate time embeddings (e.g., hour and minute)
            if isinstance(self.src_tim_vocab_size, (list, tuple)) and len(self.src_tim_vocab_size) > 1:
                tim_vocab_sizes = self.src_tim_vocab_size[1]
            else:
                tim_vocab_sizes = [24, 60]

            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                tim_emb_dims = self.src_tim_emb_dim[1]
            else:
                tim_emb_dims = [32, 32]

            pad_tokens = self.pad_token_src_tim2 if isinstance(
                self.pad_token_src_tim2, (list, tuple)) else [1, 1]

            self.src_time_embedding_1 = nn.Embedding(
                tim_vocab_sizes[0],
                tim_emb_dims[0],
                padding_idx=pad_tokens[0]
            )
            self.src_time_embedding_2 = nn.Embedding(
                tim_vocab_sizes[1],
                tim_emb_dims[1],
                padding_idx=pad_tokens[1]
            )

    def _build_encoder_decoder(self):
        """Build encoder and decoder layers."""
        # Calculate source embedding dimension based on time encoding
        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            tim_emb_dim = self.src_tim_emb_dim[0] if isinstance(
                self.src_tim_emb_dim, (list, tuple)) else self.src_tim_emb_dim
            src_emb_dim = self.src_loc_emb_dim + tim_emb_dim
        elif self.time_encoding == 'TwoEncoding':
            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                tim_emb_dims = self.src_tim_emb_dim[1]
            else:
                tim_emb_dims = [32, 32]
            src_emb_dim = self.src_loc_emb_dim + tim_emb_dims[0] + tim_emb_dims[1]
        else:
            raise ValueError(f"Unknown time_encoding: {self.time_encoding}")

        # Encoder: Bidirectional LSTM or GRU
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

        # Encoder to decoder projection
        self.encoder2decoder = nn.Linear(
            self.encoder_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        # Decoder to vocabulary projection
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.trg_seg_vocab_size)

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
            batch_size (int): Batch size

        Returns:
            h0: Initial hidden state
            c0: Initial cell state
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
        """Forward pass through the model.

        Args:
            batch (dict): Batch dictionary containing:
                - 'input_src': Source location sequence (batch, src_seq_len)
                - 'input_trg': Target segment input for teacher forcing (batch, trg_seq_len)
                - 'input_time' (optional): Time sequence(s) for time encoding

        Returns:
            decoder_logit: Output logits of shape (batch, trg_seq_len, vocab_size)
        """
        input_src = batch['input_src']
        input_trg = batch['input_trg']
        input_time = batch.get('input_time', None)

        batch_size = input_src.size(0)

        # Source embeddings
        src_emb = self.src_embedding(input_src)

        # Add time embeddings if applicable
        if self.time_encoding == 'NoEncoding':
            src_time_emb = src_emb
        elif self.time_encoding == 'OneEncoding':
            if input_time is not None:
                time_emb = self.src_time_embedding(input_time)
                src_time_emb = torch.cat((src_emb, time_emb), dim=2)
            else:
                src_time_emb = src_emb
        elif self.time_encoding == 'TwoEncoding':
            if input_time is not None and isinstance(input_time, (list, tuple)):
                time_emb_1 = self.src_time_embedding_1(input_time[0])
                time_emb_2 = self.src_time_embedding_2(input_time[1])
                src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)
            else:
                src_time_emb = src_emb
        else:
            raise RuntimeError(f'Unknown time encoding: {self.time_encoding}')

        # Target embeddings
        trg_emb = self.trg_embedding(input_trg)

        # Get initial encoder state
        h0_encoder, c0_encoder = self.get_encoder_state(batch_size)

        # Encode source sequence
        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (h0_encoder, c0_encoder))
        else:  # GRU
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

        # Prepare context for attention: (src_seq_len, batch, hidden_dim)
        ctx = src_h.transpose(0, 1)

        # Decode with attention
        trg_h, (_, _) = self.decoder(trg_emb, (decoder_init_state, c_t), ctx)

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

    def predict(self, batch):
        """Generate predictions for evaluation.

        Args:
            batch (dict): Batch dictionary

        Returns:
            predictions: Predicted segment IDs of shape (batch, trg_seq_len)
        """
        logits = self.forward(batch)
        # Get most likely segment for each position
        predictions = torch.argmax(logits, dim=-1)
        return predictions

    def calculate_loss(self, batch):
        """Calculate cross-entropy loss for training.

        Args:
            batch (dict): Batch dictionary containing 'output_trg' or 'target'

        Returns:
            loss: Scalar loss tensor
        """
        logits = self.forward(batch)

        # Get target sequence
        target = batch.get('output_trg', batch.get('target'))

        # Reshape for cross-entropy: (batch * seq_len, vocab_size) vs (batch * seq_len)
        logits_flat = logits.view(-1, self.trg_seg_vocab_size)
        target_flat = target.view(-1)

        # Cross-entropy loss, ignoring padding tokens
        loss = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.pad_token_trg
        )

        return loss

    def decode(self, logits):
        """Return probability distribution over words.

        Args:
            logits: Output logits of shape (batch, seq_len, vocab_size)

        Returns:
            word_probs: Softmax probabilities of shape (batch, seq_len, vocab_size)
        """
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=-1)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs
