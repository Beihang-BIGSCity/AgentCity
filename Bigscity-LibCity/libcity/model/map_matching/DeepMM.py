"""
DeepMM: Deep Learning-based Map Matching with Seq2Seq Attention

Adapted from the original DeepMM repository:
    Original source: repos/DeepMM/DeepMM/model.py
    Main class: Seq2SeqAttention (lines 825-1035)

Architecture:
    - Bidirectional LSTM encoder for GPS trajectory sequences
    - LSTM decoder with soft dot attention for road segment prediction
    - Separate source (location/time) and target (road segment) embeddings
    - Configurable time encoding: NoEncoding, OneEncoding, TwoEncoding
    - Configurable attention type: dot, general, mlp
    - Configurable RNN type: LSTM, GRU

Task: Map Matching (GPS trajectory -> road segment sequence)
Base Class: AbstractModel (for neural map matching models)

Key adaptations from original code:
    1. Inherits from LibCity AbstractModel instead of nn.Module
    2. Config and data_feature passed via constructor
    3. Implements predict() and calculate_loss() per LibCity conventions
    4. Removed deprecated torch.autograd.Variable usage
    5. Replaced F.sigmoid/F.tanh with torch.sigmoid/torch.tanh
    6. Removed hardcoded .cuda() calls; uses device from config
    7. Removed hardcoded .cuda() on decoder2vocab Linear layer
    8. Batch data is extracted from LibCity batch dict format
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


# ==================== Attention Modules ====================

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.

    Supports three attention types: dot, general, mlp.
    """

    def __init__(self, dim, attn_type='dot'):
        """Initialize attention layer.

        Args:
            dim (int): Hidden dimension size.
            attn_type (str): Attention type, one of 'dot', 'general', 'mlp'.
        """
        super(SoftDotAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type: dot, general, or mlp.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp uses bias in the output linear layer
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, input, context):
        """Propagate input through the attention network.

        Args:
            input: (batch, dim) - current decoder hidden state
            context: (batch, sourceL, dim) - encoder outputs

        Returns:
            h_tilde: (batch, dim) - attended output
            attn: (batch, sourceL) - attention weights
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


# ==================== LSTM with Attention ====================

class LSTMAttentionDot(nn.Module):
    """A long short-term memory (LSTM) cell with dot attention.

    At each time step, the decoder LSTM produces a hidden state, which is
    then used to compute attention over encoder outputs. The attended context
    is combined with the hidden state to produce the final output.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='dot'):
        """Initialize params.

        Args:
            input_size (int): Input feature dimension.
            hidden_size (int): Hidden state dimension.
            batch_first (bool): If True, input/output tensors are (batch, seq, feature).
            attn_type (str): Attention type for SoftDotAttention.
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
            input: (batch, seq_len, input_size) if batch_first, else (seq_len, batch, input_size)
            hidden: tuple (h_0, c_0), each (batch, hidden_size)
            ctx: (sourceL, batch, hidden_size) - encoder context
            ctx_mask: optional mask for attention

        Returns:
            output: (batch, seq_len, hidden_size) if batch_first
            hidden: tuple (h_n, c_n)
        """
        def recurrence(input_step, hidden_state):
            """Recurrence helper for one time step."""
            hx, cx = hidden_state  # (batch, hidden_dim)
            gates = self.input_weights(input_step) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            # Adaptation: use torch.sigmoid/torch.tanh instead of deprecated F.sigmoid/F.tanh
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # (batch, hidden_dim)

            # Apply attention: ctx.transpose(0,1) -> (batch, sourceL, hidden_dim)
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


# ==================== Main DeepMM Model ====================

class DeepMM(AbstractModel):
    """DeepMM: Seq2Seq with Attention for Map Matching.

    This model translates GPS trajectory sequences to road segment sequences
    using a bidirectional LSTM encoder and an LSTM decoder with attention.

    Original paper uses the Seq2SeqAttention architecture from:
        repos/DeepMM/DeepMM/model.py (lines 825-1035)

    Task: Map Matching
    Base Class: AbstractModel (for neural map matching models)

    Required data_feature keys:
        - src_loc_vocab_size (int): Source location vocabulary size
        - trg_seg_vocab_size (int): Target road segment vocabulary size
        - pad_token_src_loc (int): Padding token for source location
        - pad_token_trg (int): Padding token for target segments

    Optional data_feature keys (for time encoding):
        - src_tim_vocab_size: Time vocabulary size(s)
        - pad_token_src_tim1: Padding token for time (OneEncoding)
        - pad_token_src_tim2: Padding tokens for time (TwoEncoding)

    Config parameters:
        - src_loc_emb_dim (int): Source location embedding dim (default: 256)
        - src_tim_emb_dim (int): Source time embedding dim (default: 64)
        - trg_seg_emb_dim (int): Target segment embedding dim (default: 256)
        - src_hidden_dim (int): Encoder hidden dim (default: 512)
        - trg_hidden_dim (int): Decoder hidden dim (default: 512)
        - bidirectional (bool): Use bidirectional encoder (default: True)
        - nlayers_src (int): Number of encoder layers (default: 2)
        - dropout (float): Dropout rate (default: 0.5)
        - time_encoding (str): Time encoding mode (default: 'NoEncoding')
        - rnn_type (str): RNN type for encoder (default: 'LSTM')
        - attn_type (str): Attention type (default: 'dot')
    """

    def __init__(self, config, data_feature):
        super(DeepMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # --- Extract vocabulary sizes and padding tokens from data_feature ---
        self.src_loc_vocab_size = data_feature.get('src_loc_vocab_size', 5000)
        self.trg_seg_vocab_size = data_feature.get('trg_seg_vocab_size', 5000)
        self.src_tim_vocab_size = data_feature.get('src_tim_vocab_size', None)
        self.pad_token_src_loc = data_feature.get('pad_token_src_loc', 0)
        self.pad_token_src_tim1 = data_feature.get('pad_token_src_tim1', 0)
        self.pad_token_src_tim2 = data_feature.get('pad_token_src_tim2', [0, 0])
        self.pad_token_trg = data_feature.get('pad_token_trg', 0)

        # --- Model hyperparameters from config ---
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

        # Derived configuration
        self.num_directions = 2 if self.bidirectional else 1
        # When bidirectional, each direction uses half the hidden dim
        self.src_hidden_dim_per_dir = self.src_hidden_dim // 2 if self.bidirectional else self.src_hidden_dim

        # --- Compute source embedding dimension (location + optional time) ---
        if self.time_encoding == 'NoEncoding':
            src_emb_dim = self.src_loc_emb_dim
        elif self.time_encoding == 'OneEncoding':
            # src_tim_emb_dim should be [dim] for OneEncoding
            if isinstance(self.src_tim_emb_dim, (list, tuple)):
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[0]
            else:
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim
        elif self.time_encoding == 'TwoEncoding':
            # src_tim_emb_dim should be [_, [dim1, dim2]] for TwoEncoding
            if isinstance(self.src_tim_emb_dim, (list, tuple)) and len(self.src_tim_emb_dim) > 1:
                src_emb_dim = self.src_loc_emb_dim + self.src_tim_emb_dim[1][0] + self.src_tim_emb_dim[1][1]
            else:
                raise RuntimeError('TwoEncoding requires src_tim_emb_dim = [_, [dim1, dim2]]')
        else:
            raise RuntimeError('Invalid time_encoding: {}. Use NoEncoding, OneEncoding, or TwoEncoding'.format(
                self.time_encoding))

        # --- Source location embedding ---
        self.src_embedding = nn.Embedding(
            self.src_loc_vocab_size,
            self.src_loc_emb_dim,
            padding_idx=self.pad_token_src_loc
        )

        # --- Target road segment embedding ---
        self.trg_embedding = nn.Embedding(
            self.trg_seg_vocab_size,
            self.trg_seg_emb_dim,
            padding_idx=self.pad_token_trg
        )

        # --- Optional time embeddings ---
        if self.time_encoding == 'OneEncoding':
            tim_vocab = self.src_tim_vocab_size[0] if isinstance(self.src_tim_vocab_size, (list, tuple)) \
                else self.src_tim_vocab_size
            tim_emb = self.src_tim_emb_dim[0] if isinstance(self.src_tim_emb_dim, (list, tuple)) \
                else self.src_tim_emb_dim
            self.src_time_embedding = nn.Embedding(
                tim_vocab,
                tim_emb,
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

        # --- Encoder: configurable RNN type (LSTM or GRU) ---
        self.encoder = getattr(nn, self.rnn_type)(
            src_emb_dim,
            self.src_hidden_dim_per_dir,
            self.nlayers_src,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        # --- Decoder: LSTM with dot attention ---
        self.decoder = LSTMAttentionDot(
            self.trg_seg_emb_dim,
            self.trg_hidden_dim,
            batch_first=True,
            attn_type=self.attn_type
        )

        # --- Bridge from encoder to decoder ---
        # Maps encoder final hidden state to decoder initial hidden state
        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim_per_dir * self.num_directions,
            self.trg_hidden_dim
        )

        # --- Output projection: decoder hidden state to vocabulary logits ---
        # Adaptation: removed original .cuda() call on this layer
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, self.trg_seg_vocab_size)

        self._init_weights()

        logger.info("DeepMM model initialized: src_vocab=%d, trg_vocab=%d, "
                     "hidden=%d, bidirectional=%s, rnn_type=%s, attn_type=%s, "
                     "time_encoding=%s",
                     self.src_loc_vocab_size, self.trg_seg_vocab_size,
                     self.src_hidden_dim, self.bidirectional,
                     self.rnn_type, self.attn_type, self.time_encoding)

    def _init_weights(self):
        """Initialize embedding and linear layer weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def _get_encoder_state(self, input_src):
        """Get initial encoder hidden and cell states.

        Adaptation: uses self.device instead of hardcoded .cuda()

        Args:
            input_src: (batch, seq_len) source input tensor

        Returns:
            h0: (num_layers * num_directions, batch, hidden_dim)
            c0: (num_layers * num_directions, batch, hidden_dim)
        """
        batch_size = input_src.size(0)
        h0 = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim_per_dir,
            device=self.device
        )
        c0 = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim_per_dir,
            device=self.device
        )
        return h0, c0

    def forward(self, batch):
        """Forward pass through the seq2seq attention model.

        Adaptation: accepts LibCity batch dict instead of separate tensor arguments.

        Expected batch keys:
            - 'input_src': (batch, src_seq_len) LongTensor of source location IDs
            - 'input_trg': (batch, trg_seq_len) LongTensor of target segment IDs
                           (teacher forcing: shifted ground truth during training)
            - 'input_time': (optional) time input for OneEncoding or TwoEncoding
            - 'output_trg': (batch, trg_seq_len) LongTensor of ground truth target

        Returns:
            decoder_logit: (batch, trg_seq_len, trg_seg_vocab_size) raw logits
        """
        # --- Extract inputs from batch ---
        input_src = batch['input_src']     # (batch, src_seq_len)
        input_trg = batch['input_trg']     # (batch, trg_seq_len)
        input_time = batch.get('input_time', None)

        # --- Source embedding ---
        src_emb = self.src_embedding(input_src)  # (batch, src_seq_len, src_loc_emb_dim)
        trg_emb = self.trg_embedding(input_trg)  # (batch, trg_seq_len, trg_seg_emb_dim)

        # --- Apply time encoding if configured ---
        if self.time_encoding == 'NoEncoding':
            src_time_emb = src_emb
        elif self.time_encoding == 'OneEncoding':
            time_emb = self.src_time_embedding(input_time)
            src_time_emb = torch.cat((src_emb, time_emb), dim=2)
        elif self.time_encoding == 'TwoEncoding':
            # input_time should be a list/tuple of two tensors
            time_emb_1 = self.src_time_embedding_1(input_time[0])
            time_emb_2 = self.src_time_embedding_2(input_time[1])
            src_time_emb = torch.cat((src_emb, time_emb_1, time_emb_2), dim=2)
        else:
            raise RuntimeError('Invalid time_encoding: {}'.format(self.time_encoding))

        # --- Encoder ---
        h0_encoder, c0_encoder = self._get_encoder_state(input_src)

        if self.rnn_type == "LSTM":
            src_h, (src_h_t, src_c_t) = self.encoder(src_time_emb, (h0_encoder, c0_encoder))
        else:
            # GRU does not have cell state
            src_h, src_h_t = self.encoder(src_time_emb, h0_encoder)
            src_c_t = c0_encoder

        # --- Combine bidirectional hidden states ---
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        # --- Bridge: project encoder state to decoder initial state ---
        decoder_init_state = torch.tanh(self.encoder2decoder(h_t))

        # --- Decoder with attention ---
        # Transpose encoder outputs for attention context: (src_seq_len, batch, hidden)
        ctx = src_h.transpose(0, 1)
        trg_h, (_, _) = self.decoder(trg_emb, (decoder_init_state, c_t), ctx)

        # --- Output projection ---
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
        """Predict road segment sequences for a batch of trajectories.

        During inference, returns the argmax predicted segment IDs.

        Args:
            batch: LibCity batch dict containing at minimum 'input_src' and 'input_trg'.

        Returns:
            predictions: (batch, trg_seq_len) LongTensor of predicted road segment IDs
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)  # (batch, trg_seq_len, vocab_size)
            predictions = logits.argmax(dim=-1)  # (batch, trg_seq_len)
        return predictions

    def calculate_loss(self, batch):
        """Compute cross-entropy loss with padding mask.

        The loss ignores positions where the target is the padding token,
        so variable-length sequences are handled correctly.

        Args:
            batch: LibCity batch dict containing 'input_src', 'input_trg', 'output_trg'.
                - 'output_trg': (batch, trg_seq_len) ground truth road segment IDs

        Returns:
            loss: scalar tensor
        """
        logits = self.forward(batch)  # (batch, trg_seq_len, vocab_size)

        # Get ground truth target
        target = batch.get('output_trg')
        if target is None:
            target = batch.get('target')
        if target is None:
            target = batch.get('tgt_roads')

        # Reshape for cross-entropy: (batch * seq_len, vocab_size) and (batch * seq_len,)
        logits_flat = logits.view(-1, self.trg_seg_vocab_size)
        target_flat = target.view(-1)

        # Cross-entropy loss ignoring padding token positions
        loss = F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.pad_token_trg
        )

        return loss

    def decode(self, logits):
        """Return probability distribution over words.

        Args:
            logits: (batch, seq_len, vocab_size) raw logits

        Returns:
            word_probs: (batch, seq_len, vocab_size) probability distributions
        """
        logits_reshape = logits.view(-1, self.trg_seg_vocab_size)
        word_probs = F.softmax(logits_reshape, dim=-1)
        word_probs = word_probs.view(
            logits.size(0), logits.size(1), logits.size(2)
        )
        return word_probs
