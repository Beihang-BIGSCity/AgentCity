"""
DeepMM: Deep Learning-based Map Matching Model

This model is adapted from the original DeepMM implementation for map matching tasks.
DeepMM uses a sequence-to-sequence architecture with attention mechanism to map
GPS trajectories to road segment sequences.

Original paper: "DeepMM: Deep Learning Based Map Matching with Data Augmentation"

Key Components:
1. SoftDotAttention - Attention mechanism (supports dot, general, mlp types)
2. LSTMAttentionDot - LSTM decoder with attention
3. Seq2SeqAttention - Main encoder-decoder model

Adaptations for LibCity:
- Inherits from AbstractModel following LibCity conventions
- Implements predict() and calculate_loss() methods
- Adapts batch input format to LibCity's Batch dictionary format
- Replaced deprecated Variable with direct tensor operations
- Replaced hard-coded .cuda() with config-based device handling
- Updated F.sigmoid/F.tanh to torch.sigmoid/torch.tanh (PyTorch 1.x+)

Required data_feature:
- src_loc_vocab_size: Number of source location tokens (GPS grid cells)
- trg_seg_vocab_size: Number of target segment tokens (road segments)
- pad_token_src_loc: Padding token for source location
- pad_token_trg: Padding token for target
- Optional time encoding vocabulary sizes

Required config parameters:
- src_loc_emb_dim: Source location embedding dimension (default: 256)
- src_tim_emb_dim: Source time embedding dimension (default: 64)
- trg_seg_emb_dim: Target segment embedding dimension (default: 256)
- src_hidden_dim: Encoder hidden dimension (default: 512)
- trg_hidden_dim: Decoder hidden dimension (default: 512)
- nlayers_src: Number of encoder layers (default: 2)
- dropout: Dropout rate (default: 0.5)
- bidirectional: Use bidirectional encoder (default: True)
- time_encoding: 'NoEncoding', 'OneEncoding', or 'TwoEncoding' (default: 'NoEncoding')
- rnn_type: 'LSTM' or 'GRU' (default: 'LSTM')
- attn_type: 'dot', 'general', or 'mlp' (default: 'dot')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.model.abstract_model import AbstractModel


class SoftDotAttention(nn.Module):
    """Soft Dot Attention mechanism.

    Supports three attention types:
    - 'dot': Simple dot product attention
    - 'general': Bilinear attention with learnable weight matrix
    - 'mlp': MLP-based attention

    Reference: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.

    Args:
        dim: Hidden dimension
        attn_type: Attention type ('dot', 'general', or 'mlp')
    """

    def __init__(self, dim, attn_type='dot'):
        """Initialize layer."""
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

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propagate input through the network.

        Args:
            input: Decoder hidden state (batch x dim)
            context: Encoder outputs (batch x sourceL x dim)

        Returns:
            h_tilde: Attention-weighted context (batch x dim)
            attn: Attention weights (batch x sourceL)
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
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class LSTMAttentionDot(nn.Module):
    """LSTM cell with attention mechanism.

    Implements a single-layer LSTM with soft dot attention over encoder outputs.
    Used as the decoder in the Seq2SeqAttention model.

    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden dimension
        batch_first: Whether batch is first dimension (default: True)
        attn_type: Attention type ('dot', 'general', or 'mlp')
    """

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='dot'):
        """Initialize params."""
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
            input: Input sequence (batch x seq_len x input_size) if batch_first
            hidden: Tuple of (h_0, c_0) initial hidden and cell states
            ctx: Encoder context (seq_len x batch x hidden_size)
            ctx_mask: Context mask (optional, not currently used)

        Returns:
            output: LSTM output sequence (batch x seq_len x hidden_size)
            hidden: Final (h_n, c_n) hidden and cell states
        """

        def recurrence(input_step, hidden_state):
            """Single LSTM step with attention."""
            hx, cx = hidden_state  # n_b x hidden_dim
            gates = self.input_weights(input_step) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim

            # Apply attention
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
    """
    DeepMM: Sequence-to-Sequence Map Matching Model with Attention.

    This model implements an encoder-decoder architecture for map matching,
    where GPS trajectory points are mapped to road segment sequences.
    The encoder is a bidirectional LSTM and the decoder is an LSTM with
    dot-product attention over encoder states.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocabulary sizes

    Required config parameters:
        - src_loc_emb_dim: Source location embedding dimension (default: 256)
        - src_tim_emb_dim: Source time embedding dimension (default: 64)
        - trg_seg_emb_dim: Target segment embedding dimension (default: 256)
        - src_hidden_dim: Encoder hidden dimension (default: 512)
        - trg_hidden_dim: Decoder hidden dimension (default: 512)
        - nlayers_src: Number of encoder layers (default: 2)
        - dropout: Dropout rate (default: 0.5)
        - bidirectional: Use bidirectional encoder (default: True)
        - time_encoding: 'NoEncoding', 'OneEncoding', or 'TwoEncoding' (default: 'NoEncoding')
        - rnn_type: 'LSTM' or 'GRU' (default: 'LSTM')
        - attn_type: 'dot', 'general', or 'mlp' (default: 'dot')

    Required data_feature:
        - src_loc_vocab_size: Number of source location tokens
        - trg_seg_vocab_size: Number of target segment tokens
        - pad_token_src_loc: Padding token for source location
        - pad_token_trg: Padding token for target
    """

    def __init__(self, config, data_feature):
        """Initialize DeepMM model."""
        super(DeepMM, self).__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', 'cpu')

        # Vocabulary sizes from data_feature
        self.src_loc_vocab_size = data_feature.get('src_loc_vocab_size', 10000)
        self.trg_seg_vocab_size = data_feature.get('trg_seg_vocab_size', 5000)

        # Time encoding vocabulary sizes (for OneEncoding and TwoEncoding)
        self.src_tim_vocab_size = data_feature.get('src_tim_vocab_size', [1440, [24, 60]])

        # Padding tokens
        self.pad_token_src_loc = data_feature.get('pad_token_src_loc', 0)
        self.pad_token_src_tim1 = data_feature.get('pad_token_src_tim1', 0)
        self.pad_token_src_tim2 = data_feature.get('pad_token_src_tim2', [0, 0])
        self.pad_token_trg = data_feature.get('pad_token_trg', 0)

        # Model hyperparameters from config
        self.src_loc_emb_dim = config.get('src_loc_emb_dim', 256)
        self.src_tim_emb_dim = config.get('src_tim_emb_dim', 64)
        self.trg_seg_emb_dim = config.get('trg_seg_emb_dim', 256)
        self.src_hidden_dim = config.get('src_hidden_dim', 512)
        self.trg_hidden_dim = config.get('trg_hidden_dim', 512)
        self.bidirectional = config.get('bidirectional', True)
        self.nlayers_src = config.get('nlayers_src', 2)
        self.dropout = config.get('dropout', 0.5)
        self.time_encoding = config.get('time_encoding', 'NoEncoding')
        self.rnn_type = config.get('rnn_type', 'LSTM')
        self.attn_type = config.get('attn_type', 'dot')
        self.batch_size = config.get('batch_size', 128)

        self.num_directions = 2 if self.bidirectional else 1
        # Adjust hidden dim for bidirectional encoder
        self._src_hidden_dim = self.src_hidden_dim // 2 if self.bidirectional else self.src_hidden_dim

        # Build model components
        self._build_model()

        self._logger.info(
            f"DeepMM initialized: src_vocab={self.src_loc_vocab_size}, "
            f"trg_vocab={self.trg_seg_vocab_size}, hidden={self.src_hidden_dim}, "
            f"attn_type={self.attn_type}, time_encoding={self.time_encoding}"
        )

    def _build_model(self):
        """Build all model components."""

        # Calculate source embedding dimension based on time encoding
        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            # src_tim_emb_dim can be a single value or list [dim]
            if isinstance(self.src_tim_emb_dim, (list, tuple)):
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[0]
            else:
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim
        elif self.time_encoding == 'TwoEncoding':
            # src_tim_emb_dim should be [single_dim, [hour_dim, minute_dim]]
            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[1][0] + self.src_tim_emb_dim[1][1]
            else:
                src_emb_dim = self.src_loc_emb_dim + 64 + 64  # Default time dims
        else:
            raise ValueError(f"Invalid time_encoding: {self.time_encoding}")

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

        # Time embeddings (if using time encoding)
        if self.time_encoding == 'OneEncoding':
            tim_emb_dim = self.src_tim_emb_dim[0] if isinstance(self.src_tim_emb_dim, (list, tuple)) else self.src_tim_emb_dim
            tim_vocab_size = self.src_tim_vocab_size[0] if isinstance(self.src_tim_vocab_size, (list, tuple)) else self.src_tim_vocab_size
            self.src_time_embedding = nn.Embedding(
                tim_vocab_size,
                tim_emb_dim,
                padding_idx=self.pad_token_src_tim1
            )
        elif self.time_encoding == 'TwoEncoding':
            # Two time embeddings for hour and minute
            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                tim_emb_dim_1 = self.src_tim_emb_dim[1][0]
                tim_emb_dim_2 = self.src_tim_emb_dim[1][1]
            else:
                tim_emb_dim_1 = tim_emb_dim_2 = 64

            if isinstance(self.src_tim_vocab_size, (list, tuple)) and len(self.src_tim_vocab_size) > 1:
                tim_vocab_size_1 = self.src_tim_vocab_size[1][0]
                tim_vocab_size_2 = self.src_tim_vocab_size[1][1]
            else:
                tim_vocab_size_1 = 24
                tim_vocab_size_2 = 60

            pad_tim2_0 = self.pad_token_src_tim2[0] if isinstance(self.pad_token_src_tim2, (list, tuple)) else 0
            pad_tim2_1 = self.pad_token_src_tim2[1] if isinstance(self.pad_token_src_tim2, (list, tuple)) else 0

            self.src_time_embedding_1 = nn.Embedding(
                tim_vocab_size_1,
                tim_emb_dim_1,
                padding_idx=pad_tim2_0
            )
            self.src_time_embedding_2 = nn.Embedding(
                tim_vocab_size_2,
                tim_emb_dim_2,
                padding_idx=pad_tim2_1
            )

        # Encoder RNN (LSTM or GRU)
        self.encoder = getattr(nn, self.rnn_type)(
            src_emb_dim,
            self._src_hidden_dim,
            self.nlayers_src,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout if self.nlayers_src > 1 else 0
        )

        # Decoder with attention
        self.decoder = LSTMAttentionDot(
            self.trg_seg_emb_dim,
            self.trg_hidden_dim,
            batch_first=True,
            attn_type=self.attn_type
        )

        # Encoder to decoder state projection
        self.encoder2decoder = nn.Linear(
            self._src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )

        # Output projection to vocabulary
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.trg_seg_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def _get_encoder_state(self, input):
        """Get initial encoder hidden and cell states.

        Args:
            input: Input tensor to determine batch size

        Returns:
            h0: Initial hidden state
            c0: Initial cell state
        """
        batch_size = input.size(0) if self.encoder.batch_first else input.size(1)

        h0 = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self._src_hidden_dim,
            device=self.device
        )
        c0 = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self._src_hidden_dim,
            device=self.device
        )

        return h0, c0

    def forward(self, batch):
        """Forward pass through the model.

        Args:
            batch: Dictionary containing:
                - 'input_src': Source sequence (batch x src_len)
                - 'input_trg': Target sequence (batch x trg_len)
                - 'input_time': Time sequence (optional, depends on time_encoding)

        Returns:
            decoder_logit: Output logits (batch x trg_len x vocab_size)
        """
        # Extract inputs from batch
        input_src = batch['input_src'].to(self.device)
        input_trg = batch['input_trg'].to(self.device)

        # Get time input if using time encoding
        input_time = None
        if self.time_encoding != 'NoEncoding':
            input_time = batch.get('input_time')
            if input_time is not None:
                if isinstance(input_time, (list, tuple)):
                    input_time = [t.to(self.device) for t in input_time]
                else:
                    input_time = input_time.to(self.device)

        # Source embedding
        src_emb = self.src_embedding(input_src)

        # Apply time encoding if specified
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

        # Target embedding
        trg_emb = self.trg_embedding(input_trg)

        # Initialize encoder state
        h0_encoder, c0_encoder = self._get_encoder_state(input_src)

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

        # Project encoder state to decoder state
        decoder_init_state = torch.tanh(self.encoder2decoder(h_t))

        # Prepare context for attention (src_len x batch x hidden)
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
            batch: Input batch dictionary

        Returns:
            predictions: Predicted token indices (batch x trg_len)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)
            # Get predictions by taking argmax
            predictions = logits.argmax(dim=-1)
        return predictions

    def calculate_loss(self, batch):
        """Calculate training loss.

        Args:
            batch: Dictionary containing:
                - 'input_src': Source sequence
                - 'input_trg': Target input sequence (shifted right)
                - 'target': Target output sequence for loss calculation
                - 'input_time': Time sequence (optional)

        Returns:
            loss: Cross-entropy loss
        """
        logits = self.forward(batch)

        # Get target for loss calculation
        if 'target' in batch:
            target = batch['target'].to(self.device)
        else:
            # Use input_trg shifted by 1 as target
            target = batch['input_trg'].to(self.device)

        # Reshape for cross-entropy loss
        logits_flat = logits.view(-1, self.trg_seg_vocab_size)
        target_flat = target.view(-1)

        # Create mask to ignore padding tokens
        mask = (target_flat != self.pad_token_trg)

        # Calculate masked cross-entropy loss
        loss = F.cross_entropy(
            logits_flat[mask],
            target_flat[mask],
            reduction='mean'
        )

        return loss

    def decode(self, logits):
        """Return probability distribution over words.

        Args:
            logits: Model output logits (batch x seq_len x vocab_size)

        Returns:
            word_probs: Softmax probabilities (batch x seq_len x vocab_size)
        """
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=-1)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs

    def greedy_decode(self, batch, max_length=None, sos_token=1, eos_token=2):
        """Greedy decoding for inference.

        Args:
            batch: Input batch with 'input_src' and optionally 'input_time'
            max_length: Maximum decoding length (default: 2x source length)
            sos_token: Start-of-sequence token ID
            eos_token: End-of-sequence token ID

        Returns:
            outputs: Decoded sequences (batch x decoded_len)
        """
        self.eval()
        with torch.no_grad():
            input_src = batch['input_src'].to(self.device)
            batch_size = input_src.size(0)
            src_len = input_src.size(1)

            if max_length is None:
                max_length = src_len * 2

            # Get time input if using time encoding
            input_time = None
            if self.time_encoding != 'NoEncoding':
                input_time = batch.get('input_time')
                if input_time is not None:
                    if isinstance(input_time, (list, tuple)):
                        input_time = [t.to(self.device) for t in input_time]
                    else:
                        input_time = input_time.to(self.device)

            # Source embedding
            src_emb = self.src_embedding(input_src)

            # Apply time encoding
            if self.time_encoding == 'NoEncoding':
                src_time_emb = src_emb
            elif self.time_encoding == 'OneEncoding':
                time_emb = self.src_time_embedding(input_time)
                src_time_emb = torch.cat((src_emb, time_emb), dim=2)
            elif self.time_encoding == 'TwoEncoding':
                time_emb_1 = self.src_time_embedding_1(input_time[0])
                time_emb_2 = self.src_time_embedding_2(input_time[1])
                src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)

            # Encode
            h0_encoder, c0_encoder = self._get_encoder_state(input_src)
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
            current_token = torch.full((batch_size,), sos_token, dtype=torch.long, device=self.device)
            outputs = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for _ in range(max_length):
                # Get embedding for current token
                trg_emb = self.trg_embedding(current_token.unsqueeze(1))

                # Decode one step
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    trg_emb, (decoder_hidden, decoder_cell), ctx
                )

                # Project to vocabulary and get next token
                logits = self.decoder2vocab(output.squeeze(1))
                current_token = logits.argmax(dim=-1)

                outputs.append(current_token)

                # Check for EOS
                finished = finished | (current_token == eos_token)
                if finished.all():
                    break

            outputs = torch.stack(outputs, dim=1)
            return outputs
