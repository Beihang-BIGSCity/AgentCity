"""
MulT-TTE: Multi-Task Learning for Travel Time Estimation

This model is adapted from the original MulT-TTE implementation.
Original paper: Multi-Task Learning for Travel Time Estimation with Masked Segment Prediction

Key Components:
1. BERT-based segment embedding learning (masked segment prediction)
2. Multiple embedding layers for highway type, week, date, time, and GPS
3. Custom LayerNormGRU for sequence encoding
4. Transformer decoder with multi-head attention
5. Multi-task learning: masked segment prediction + travel time estimation

Adapted for LibCity framework by inheriting from AbstractTrafficStateModel.
"""

import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

try:
    from transformers import BertConfig, BertForMaskedLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not installed. MulT_TTE requires transformers.")


# ============================================================================
# LayerNormGRU Components
# ============================================================================

class LayerNormGRUCell(nn.Module):
    """
    GRU Cell with Layer Normalization.
    Applies layer normalization to the gates and hidden state computations.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_h2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_cell_1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1(h_hat_first_half)
        h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))

        h_t = torch.mul(1 - z_t, h) + torch.mul(z_t, h_hat)

        h_t = h_t.view(h_t.size(0), -1)
        return h_t


class LayerNormGRU(nn.Module):
    """
    Multi-layer GRU with Layer Normalization.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, bias=True):
        super(LayerNormGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.hidden0 = nn.ModuleList([
            LayerNormGRUCell(
                input_size=(input_dim if layer == 0 else hidden_dim),
                hidden_size=hidden_dim,
                bias=bias
            )
            for layer in range(num_layers)
        ])

    def forward(self, input_tensor, seq_lens=None):
        seq_len, batch_size, _ = input_tensor.size()
        hx = input_tensor.new_zeros(self.num_layers, batch_size, self.hidden_dim, requires_grad=False)

        ht = []
        for i in range(seq_len):
            ht.append([None] * self.num_layers)

        seq_len_mask = input_tensor.new_ones(batch_size, seq_len, self.hidden_dim, requires_grad=False)
        if seq_lens is not None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1)

        device = input_tensor.device
        indices = (torch.LongTensor(seq_lens.cpu().numpy() if torch.is_tensor(seq_lens) else seq_lens) - 1).to(device)
        indices = indices.unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat([1, self.num_layers, 1, self.hidden_dim])

        h = hx

        for t, x in enumerate(input_tensor):
            for l, layer in enumerate(self.hidden0):
                ht_ = layer(x, h[l])
                ht[t][l] = ht_ * seq_len_mask[t]
                x = ht[t][l]
            ht[t] = torch.stack(ht[t])
            h = ht[t]

        y = torch.stack([h[-1] for h in ht])
        hy = torch.stack(list(torch.stack(ht).gather(dim=0, index=indices).squeeze(0)))

        return y, hy


# ============================================================================
# Transformer Decoder Components
# ============================================================================

class Norm(nn.Module):
    """Layer normalization with learnable parameters."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.attn_1 = nn.MultiheadAttention(embed_dim=d_model, dropout=dropout, num_heads=self.h)

    def forward(self, q, k, v, lens):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        S = q.shape[0]
        mask = torch.stack([
            torch.cat((torch.zeros(i), torch.ones(S - i)), 0) for i in lens
        ]).bool().to(k.device)
        attn_output, attn_output_weights = self.attn_1(q, k, v, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with multi-head attention and feed-forward network."""
    def __init__(self, d_model, heads=1, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, lens):
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, x2, x2, lens))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


def get_clones(module, N):
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    """Stack of decoder layers."""
    def __init__(self, d_model, N=3, heads=1, dropout=0.1):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, x, lens):
        for i in range(self.N):
            x = self.layers[i](x, lens)
        return self.norm(x)


# ============================================================================
# MulT_TTE Model (LibCity Adapted)
# ============================================================================

class MulT_TTE(AbstractTrafficStateModel):
    """
    MulT-TTE: Multi-Task Learning for Travel Time Estimation.

    This model uses BERT-based masked segment prediction as an auxiliary task
    to improve travel time estimation. It combines:
    - Segment embeddings learned via masked language modeling
    - Temporal embeddings (week, date, time)
    - Spatial embeddings (highway type, GPS coordinates)
    - LayerNormGRU for sequence encoding
    - Transformer decoder for final prediction

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Dictionary containing data-specific features
    """

    def __init__(self, config, data_feature):
        super(MulT_TTE, self).__init__(config, data_feature)

        if not HAS_TRANSFORMERS:
            raise ImportError("MulT_TTE requires the transformers library. "
                            "Install it with: pip install transformers")

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters from config
        self.input_dim = config.get('input_dim', 120)
        seq_input_dim = config.get('seq_input_dim', 120)
        seq_hidden_dim = config.get('seq_hidden_dim', 128)
        seq_layer = config.get('seq_layer', 2)
        bert_hidden_size = config.get('bert_hidden_size', 64)
        decoder_layer = config.get('decoder_layer', 3)
        decode_head = config.get('decode_head', 1)
        bert_hidden_layers = config.get('bert_hidden_layers', 4)
        bert_attention_heads = config.get('bert_attention_heads', 8)

        # Vocabulary size from data features or config
        self.vocab_size = data_feature.get('vocab_size', config.get('vocab_size', 27300))
        self.pad_token_id = data_feature.get('pad_token_id', self.vocab_size)

        # Multi-task learning weight
        self.beta = config.get('beta', 0.7)
        self.mask_rate = config.get('mask_rate', 0.4)

        # Loss function parameters
        self.loss_type = config.get('loss_type', 'smoothL1')
        self.loss_val = config.get('loss_val', 300.0)

        # Time normalization parameters from data features
        self.time_mean = data_feature.get('time_mean', 638.74)
        self.time_std = data_feature.get('time_std', 320.30)

        # Batch first flag (model uses seq_first internally)
        self.batch_first = False
        self.bidirectional = False

        # BERT for segment embedding learning
        self.bert_config = BertConfig(
            num_attention_heads=bert_attention_heads,
            hidden_size=bert_hidden_size,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size + 2,  # +2 for padding and mask tokens
            num_hidden_layers=bert_hidden_layers
        )
        self.seg_embedding_learning = BertForMaskedLM(self.bert_config)

        # Embedding layers
        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = nn.Embedding(367, 10)
        self.timeembed = nn.Embedding(1441, 20)
        self.gpsrep = nn.Linear(4, 16)

        # Time-aware network embedding dimension
        self.timene_dim = 3 + 10 + 20 + bert_hidden_size  # week + date + time + bert

        self.timene = nn.Sequential(
            nn.Linear(self.timene_dim, self.timene_dim),
            nn.LeakyReLU(),
            nn.Linear(self.timene_dim, self.timene_dim)
        )

        self.represent = nn.Sequential(
            nn.Linear(self.input_dim, seq_input_dim),
            nn.LeakyReLU(),
            nn.Linear(seq_input_dim, seq_input_dim)
        )

        # Sequence encoder
        self.sequence = LayerNormGRU(seq_input_dim, seq_hidden_dim, seq_layer)

        self.seq_hidden_dim = seq_hidden_dim * 2 if self.bidirectional else seq_hidden_dim
        self.decoder_embed_dim = seq_hidden_dim * 2 if self.bidirectional else seq_hidden_dim

        # Output layers
        self.input2hid = nn.Linear(seq_hidden_dim + 33, seq_hidden_dim)  # +33 for week(3)+date(10)+time(20)
        self.decoder = Decoder(d_model=self.decoder_embed_dim, N=decoder_layer, heads=decode_head)
        self.hid2out = nn.Linear(self.seq_hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for non-BERT layers."""
        for name, param in self.named_parameters():
            if 'seg_embedding_learning' not in name:
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)

    def pooling_sum(self, hiddens, lens):
        """Sum pooling over valid sequence positions."""
        lens = lens.to(hiddens.device)
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)
        batch_size = range(hiddens.shape[0])
        for i in batch_size:
            hiddens[i, 0] = torch.sum(hiddens[i, :lens[i]], dim=0)
        return hiddens[list(batch_size), 0]

    def seg_embedding(self, input_ids, attention_mask, labels):
        """
        BERT-based segment embedding with masked language modeling.

        Returns:
            loss: Masked LM loss
            hidden_states: Hidden states from the 4th layer
            logits: Prediction logits
        """
        bert_output = self.seg_embedding_learning(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return bert_output["loss"], bert_output["hidden_states"][4], bert_output["logits"]

    def _normalize(self, data):
        """Normalize data using time mean and std."""
        return (data - self.time_mean) / self.time_std

    def _inverse_normalize(self, data):
        """Inverse normalize data."""
        return data * self.time_std + self.time_mean

    def forward(self, batch):
        """
        Forward pass of MulT_TTE model.

        Args:
            batch: Dictionary containing:
                - 'links': Tensor of link features [batch, seq_len, features]
                - 'lens': Tensor of sequence lengths [batch]
                - 'linkindex': Tensor of masked link indices [batch, seq_len]
                - 'rawlinks': Tensor of original link indices [batch, seq_len]
                - 'encoder_attention_mask': Attention mask [batch, seq_len]
                - 'mask_label': Labels for masked positions [batch, seq_len]

        Returns:
            output: Predicted travel times [batch, 1]
            loss_1: Masked segment prediction loss
        """
        feature = batch['links']
        lens = batch['lens']

        # Highway embedding (feature[:, :, 0] is highway type)
        highwayrep = self.highwayembed(feature[:, :, 0].long())

        # Temporal embeddings
        weekrep = self.weekembed(feature[:, :, 3].long())
        daterep = self.dateembed(feature[:, :, 4].long())
        timerep = self.timeembed(feature[:, :, 5].long())

        # GPS representation
        gpsrep = self.gpsrep(feature[:, :, 6:10])

        # Concatenate temporal representations
        datetimerep = torch.cat([weekrep, daterep, timerep], dim=-1)

        # BERT masked segment prediction
        loss_1, hidden_states, prediction_scores = self.seg_embedding(
            batch['linkindex'],
            batch['encoder_attention_mask'],
            batch['mask_label']
        )

        # Time-aware network embedding
        raw_embeddings = self.seg_embedding_learning.bert.embeddings.word_embeddings(batch['rawlinks'])
        timene_input = torch.cat([raw_embeddings, datetimerep], dim=-1)
        timene = self.timene(timene_input) + timene_input  # Residual connection

        # Feature representation: length(1) + cumulative_length(1) + highway(5) + gps(16) + timene(97)
        representation = self.represent(torch.cat([
            feature[..., 1:3],  # Length and cumulative length
            highwayrep,
            gpsrep,
            timene
        ], dim=-1))

        # Transpose for sequence-first processing
        representation = representation if self.batch_first else representation.transpose(0, 1).contiguous()

        # Sequence encoding with LayerNormGRU
        hiddens, rnn_states = self.sequence(representation, seq_lens=lens.long())

        # Transformer decoder
        decoder_out = self.decoder(hiddens, lens)
        decoder_out = decoder_out if self.batch_first else decoder_out.transpose(0, 1).contiguous()

        # Pooling and final prediction
        pooled_decoder = self.pooling_sum(decoder_out, lens)
        pooled_hidden = torch.cat([
            pooled_decoder,
            weekrep[:, 0],
            daterep[:, 0],
            timerep[:, 0]
        ], dim=-1)

        hidden = F.leaky_relu(self.input2hid(pooled_hidden))
        output = self.hid2out(hidden)

        # Inverse normalize the output
        output = self._inverse_normalize(output)

        return output, loss_1

    def predict(self, batch):
        """
        Predict travel times for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted travel times [batch]
        """
        output, _ = self.forward(batch)
        return output.squeeze(-1)  # Change from [batch, 1] to [batch]

    def calculate_loss(self, batch):
        """
        Calculate the multi-task loss.

        The loss combines:
        1. Masked segment prediction loss (from BERT)
        2. Travel time estimation loss (SmoothL1 or other)

        Total loss = (1 - beta) * normalized_mlm_loss + beta * tte_loss

        Args:
            batch: Input batch dictionary with 'time' as ground truth

        Returns:
            Combined loss tensor
        """
        output, loss_1 = self.forward(batch)

        # Get ground truth travel time
        truth_data = batch['time']
        if truth_data.dim() == 1:
            truth_data = truth_data.unsqueeze(1)

        # Calculate TTE loss
        if self.loss_type == 'smoothL1':
            loss_func = nn.SmoothL1Loss(reduction='mean', beta=self.loss_val)
            loss_2 = loss_func(output.squeeze(), truth_data.squeeze())
        elif self.loss_type == 'mse':
            loss_2 = F.mse_loss(output.squeeze(), truth_data.squeeze())
        elif self.loss_type == 'mae':
            loss_2 = F.l1_loss(output.squeeze(), truth_data.squeeze())
        elif self.loss_type == 'mape':
            loss_2 = torch.mean(torch.abs(output.squeeze() - truth_data.squeeze()) / (truth_data.squeeze() + 0.1))
        else:
            loss_2 = F.smooth_l1_loss(output.squeeze(), truth_data.squeeze())

        # Combine losses with dynamic weighting
        # The original uses: (1 - beta) * loss_1 / (loss_1 / loss_2 + 1e-4).detach() + beta * loss_2
        # This normalizes the MLM loss relative to TTE loss
        if loss_1 is not None and loss_1.item() > 0:
            loss = (1 - self.beta) * loss_1 / (loss_1 / (loss_2 + 1e-8) + 1e-4).detach() + self.beta * loss_2
        else:
            loss = loss_2

        return loss
