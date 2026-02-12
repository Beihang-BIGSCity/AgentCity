"""
L2MM: Latent-to-Map-Matching with VAE-style Encoder-Decoder

Original Repository: repos/L2MM/mapmatching/
Main classes:
    - EncoderDecoder (repos/L2MM/mapmatching/model.py, lines 155-201)
    - Encoder (repos/L2MM/mapmatching/model.py, lines 9-32)
    - Decoder (repos/L2MM/mapmatching/model.py, lines 130-152)
    - LatentDistribution (repos/L2MM/mapmatching/model.py, lines 35-109)
    - StackingGRUCell (repos/L2MM/mapmatching/model.py, lines 203-225)
    - GlobalAttention (repos/L2MM/mapmatching/model.py, lines 112-127)
    - DenseLoss (repos/L2MM/mapmatching/util.py, lines 5-31)

Architecture:
    - Bidirectional GRU encoder for GPS grid cell sequences
    - LatentDistribution module (VAE-style with Gaussian mixture clustering)
      producing latent variable z from encoder hidden state
    - Stacked GRU cell decoder with GlobalAttention over encoder outputs
    - Output layer: Linear + LogSoftmax for road segment prediction
    - Three training modes: pretrain (no latent loss), train (full latent loss),
      test (deterministic latent encoding)

Task: Map Matching (GPS trajectory -> road segment sequence)
Base Class: AbstractModel (for neural map matching models)

Key adaptations from original code:
    1. Inherits from LibCity AbstractModel instead of nn.Module
    2. Config and data_feature passed via constructor
    3. Implements forward(), predict(), calculate_loss() per LibCity conventions
    4. Merged the external output layer (m1: Linear + LogSoftmax) into the model
    5. Removed deprecated torch.autograd.Variable usage
    6. Updated nn.Softmax() -> nn.Softmax(dim=-1) and nn.LogSoftmax() -> nn.LogSoftmax(dim=-1)
    7. Removed hardcoded .cuda() calls; uses device from config
    8. Batch data is extracted from LibCity batch dict format
    9. Integrated DenseLoss (masked cross-entropy) into calculate_loss
   10. LatentDistribution no longer loads from hardcoded file path;
       cluster centers are initialized randomly or from config
   11. Encoder no longer loads pretrained weights from hardcoded file path;
       pretrain loading is handled externally via config
   12. pack_padded_sequence uses enforce_sorted=False for robustness

Required data_feature keys:
    - input_vocab_size (int): Source grid cell vocabulary size (number of grid cells + padding)
    - output_vocab_size (int): Target road segment vocabulary size (number of roads + special tokens)

Config parameters:
    - embedding_size (int): Embedding dimension for grid cells and road segments (default: 256)
    - hidden_size (int): GRU hidden state dimension (default: 256)
    - num_layers (int): Number of encoder GRU layers (default: 2)
    - de_layer (int): Number of decoder GRU layers (default: 1)
    - dropout (float): Dropout rate (default: 0.1)
    - bidirectional (bool): Use bidirectional encoder (default: True)
    - cluster_size (int): Number of Gaussian mixture clusters in latent space (default: 10)
    - max_length (int): Maximum target sequence length for autoregressive decoding (default: 300)
    - criterion_name (str): Loss function type, 'CE' or 'NLL' (default: 'CE')
    - latent_weight (float): Weight for Gaussian latent loss term (default: 1.0)
    - cate_weight (float): Weight for category loss term (default: 0.1)
    - training_mode (str): 'pretrain' or 'train' (default: 'train')
    - BOS (int): Beginning-of-sequence token ID (default: 0)
    - EOS (int): End-of-sequence token ID (default: 2)
    - PAD (int): Padding token ID (default: 1)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from libcity.model.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


# ==================== Encoder Components ====================

class StackingGRUCell(nn.Module):
    """Stacked GRU cells for step-by-step decoding.

    Unlike nn.GRU which processes entire sequences, this module processes
    one time step at a time, maintaining separate hidden states per layer.

    Original: repos/L2MM/mapmatching/model.py lines 203-225
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """Process one time step through the stacked GRU layers.

        Args:
            input: (batch, input_size) - single time step input
            h0: (num_layers, batch, hidden_size) - hidden states per layer

        Returns:
            output: (batch, hidden_size) - output from top layer
            hn: (num_layers, batch, hidden_size) - updated hidden states
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn


class GlobalAttention(nn.Module):
    """Global attention mechanism for the decoder.

    Computes dot-product attention between decoder hidden state and all
    encoder outputs, then concatenates context with query and projects
    through a linear layer with tanh activation.

    Original: repos/L2MM/mapmatching/model.py lines 112-127
    Adaptation: replaced nn.Softmax() with nn.Softmax(dim=-1) to avoid
    deprecation warning.
    """

    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # Adaptation: specify dim explicitly to avoid deprecation warning
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """Compute attention-weighted context vector.

        Args:
            q: (batch, hidden_size) - decoder hidden state
            H: (batch, src_len, hidden_size) - encoder outputs

        Returns:
            (batch, hidden_size) - attended output
        """
        q1 = q.unsqueeze(2)                   # (batch, hidden_size, 1)
        a = torch.bmm(H, q1).squeeze(2)       # (batch, src_len)
        a = self.softmax(a)                    # (batch, src_len)
        a = a.unsqueeze(1)                     # (batch, 1, src_len)
        c = torch.bmm(a, H).squeeze(1)        # (batch, hidden_size)
        c = torch.cat([c, q], 1)              # (batch, 2 * hidden_size)
        return self.tanh(self.L2(c))           # (batch, hidden_size)


class Encoder(nn.Module):
    """Bidirectional GRU encoder with embedding.

    Encodes a sequence of grid cell IDs into a sequence of hidden states
    using a multi-layer bidirectional GRU.

    Original: repos/L2MM/mapmatching/model.py lines 9-32
    Adaptation: uses enforce_sorted=False in pack_padded_sequence for
    robustness when sequences are not pre-sorted by length.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 bidirectional, embedding):
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, input, lengths, h0=None):
        """Encode input sequence.

        Args:
            input: (seq_len, batch) or (batch, seq_len) LongTensor of grid cell IDs
            lengths: (batch,) or (1, batch) LongTensor of sequence lengths
            h0: optional initial hidden state

        Returns:
            hn: (num_layers * num_directions, batch, hidden_size) - final hidden states
            output: (seq_len, batch, hidden_size * num_directions) - all hidden states
        """
        embed = self.embedding(input)
        # Adaptation: convert lengths to a flat list for pack_padded_sequence
        lengths_list = lengths.data.view(-1).tolist()
        if lengths_list is not None:
            # Adaptation: enforce_sorted=False so input need not be sorted by length
            embed = pack_padded_sequence(embed, lengths_list, enforce_sorted=False)
        output, hn = self.rnn(embed, h0)
        if lengths_list is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output


class LatentDistribution(nn.Module):
    """Variational latent distribution with Gaussian mixture clustering.

    Projects encoder hidden state into a latent space using a VAE-style
    reparameterization trick. During training, computes Gaussian mixture
    cluster assignments and associated latent/category losses.

    Three modes:
        - 'pretrain': Sample z without latent loss computation
        - 'train': Sample z with full Gaussian mixture latent loss
        - 'test': Return deterministic mu_z (no sampling)

    Original: repos/L2MM/mapmatching/model.py lines 35-109
    Adaptation:
        - Removed hardcoded file loading for mu_c initialization
        - Removed hardcoded .cuda() calls; tensors follow model device
        - Added F.softmax(dim=-1) instead of deprecated F.softmax()
        - eps_z device is inferred from log_sigma_sq_z rather than hardcoded
    """

    def __init__(self, cluster_size, hidden_size):
        super(LatentDistribution, self).__init__()
        self.cluster_size = cluster_size
        self.hidden_size = hidden_size

        # Gaussian mixture cluster centers (learnable)
        mu_c = torch.rand(cluster_size, hidden_size)
        self.mu_c = nn.Parameter(mu_c, requires_grad=True)

        # Log variance of cluster centers (learnable, initialized to zero)
        log_sigma_sq_c = torch.zeros(cluster_size, hidden_size)
        self.log_sigma_sq_c = nn.Parameter(log_sigma_sq_c, requires_grad=True)

        # Projection from encoder hidden to latent mean
        self.cal_mu_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_mu_z.weight, std=0.02)
        nn.init.constant_(self.cal_mu_z.bias, 0.0)

        # Projection from encoder hidden to latent log-variance
        self.cal_log_sigma_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_log_sigma_z.weight, std=0.02)
        nn.init.constant_(self.cal_log_sigma_z.bias, 0.0)

    def batch_latent_loss(self, stack_log_sigma_sq_c, stack_mu_c,
                          stack_log_sigma_sq_z, stack_mu_z, att, log_sigma_sq_z):
        """Compute the KL-divergence-based latent loss and category loss.

        Args:
            stack_log_sigma_sq_c: (batch, cluster_size, hidden_size)
            stack_mu_c: (batch, cluster_size, hidden_size)
            stack_log_sigma_sq_z: (batch, cluster_size, hidden_size)
            stack_mu_z: (batch, cluster_size, hidden_size)
            att: (batch, cluster_size) - cluster assignment probabilities
            log_sigma_sq_z: (batch, hidden_size)

        Returns:
            batch_latent_loss: scalar tensor - Gaussian KL loss
            batch_cate_loss: scalar tensor - category uniformity loss
        """
        avg_ = torch.mean(
            stack_log_sigma_sq_c
            + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
            + torch.pow(stack_mu_z - stack_mu_c, 2) / torch.exp(stack_log_sigma_sq_c),
            dim=-1
        )

        sum_ = torch.sum(att * avg_, dim=-1).squeeze()
        mean_ = torch.mean(1 + log_sigma_sq_z, dim=-1).squeeze()

        batch_latent_loss = 0.5 * sum_ - 0.5 * mean_
        batch_latent_loss = torch.mean(batch_latent_loss).squeeze()

        cate_mean = torch.mean(att, dim=0).squeeze()
        batch_cate_loss = torch.mean(cate_mean * torch.log(cate_mean)).squeeze()
        batch_cate_loss = torch.mean(batch_cate_loss).squeeze()

        return batch_latent_loss, batch_cate_loss

    def forward(self, h, kind="train"):
        """Project hidden state into latent space.

        Args:
            h: (1, batch, hidden_size) - encoder final hidden state
            kind: str - 'pretrain', 'train', or 'test'

        Returns:
            If kind == 'test': mu_z (batch, hidden_size)
            If kind == 'pretrain': z (batch, hidden_size)
            If kind == 'train': (z, batch_latent_loss, batch_cate_loss)
        """
        h = h.squeeze(0)  # (batch, hidden_size)
        mu_z = self.cal_mu_z(h)

        if kind == "test":
            return mu_z

        log_sigma_sq_z = self.cal_log_sigma_z(h)
        # Adaptation: use device from tensor rather than hardcoded .cuda()
        eps_z = torch.rand(size=log_sigma_sq_z.shape, device=log_sigma_sq_z.device)

        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z

        if kind == "pretrain":
            return z
        else:
            # kind == "train": compute Gaussian mixture losses
            batch_size = z.shape[0]
            stack_mu_c = self.mu_c.unsqueeze(0).expand(batch_size, -1, -1)
            stack_log_sigma_sq_c = self.log_sigma_sq_c.unsqueeze(0).expand(batch_size, -1, -1)
            stack_mu_z = mu_z.unsqueeze(1).expand(-1, self.cluster_size, -1)
            stack_log_sigma_sq_z = log_sigma_sq_z.unsqueeze(1).expand(-1, self.cluster_size, -1)
            stack_z = z.unsqueeze(1).expand(-1, self.cluster_size, -1)

            att_logits = -torch.sum(
                torch.pow(stack_z - stack_mu_c, 2) / torch.exp(stack_log_sigma_sq_c),
                dim=-1
            )
            att_logits = att_logits.squeeze()
            # Adaptation: specify dim explicitly in F.softmax
            att = F.softmax(att_logits, dim=-1) + 1e-10

            batch_latent_loss, batch_cate_loss = self.batch_latent_loss(
                stack_log_sigma_sq_c, stack_mu_c,
                stack_log_sigma_sq_z, stack_mu_z,
                att, log_sigma_sq_z
            )
            return z, batch_latent_loss, batch_cate_loss


class Decoder(nn.Module):
    """GRU decoder with global attention over encoder outputs.

    Processes target tokens one at a time using StackingGRUCell and attends
    over encoder outputs at each step.

    Original: repos/L2MM/mapmatching/model.py lines 130-152
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers, dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """Decode target sequence step by step.

        Args:
            input: (seq_len, batch) LongTensor of target token IDs
            h: (num_layers, batch, hidden_size) initial decoder hidden state
            H: (src_len, batch, hidden_size) encoder outputs
            use_attention: whether to apply attention

        Returns:
            output: (seq_len, batch, hidden_size) decoder outputs
            h: (num_layers, batch, hidden_size) final hidden state
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        embed = self.embedding(input)
        output = []
        for e in embed.split(1):
            e = e.squeeze(0)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h


# ==================== Main L2MM Model ====================

class L2MM(AbstractModel):
    """L2MM: Latent-to-Map Matching with VAE-style Encoder-Decoder.

    This model matches GPS trajectory grid cell sequences to road segment
    sequences using a bidirectional GRU encoder, a VAE-style latent
    distribution module with Gaussian mixture clustering, and an attentive
    GRU decoder.

    The original codebase separates the model into:
        - m0 (EncoderDecoder): encoder + latent + decoder
        - m1 (Linear + LogSoftmax): output projection
    This adaptation merges them into a single nn.Module for clean
    integration with the LibCity training pipeline.

    Task: Map Matching
    Base Class: AbstractModel

    Required data_feature keys:
        - input_vocab_size (int): Grid cell vocabulary size (including padding at 0)
        - output_vocab_size (int): Road segment vocabulary size (including BOS, EOS, padding)

    Config parameters:
        - embedding_size (int): Embedding dimension (default: 256)
        - hidden_size (int): GRU hidden dimension (default: 256)
        - num_layers (int): Encoder GRU layers (default: 2)
        - de_layer (int): Decoder GRU layers (default: 1)
        - dropout (float): Dropout rate (default: 0.1)
        - bidirectional (bool): Bidirectional encoder (default: True)
        - cluster_size (int): Number of latent Gaussian clusters (default: 10)
        - max_length (int): Max decoding length for inference (default: 300)
        - criterion_name (str): 'CE' or 'NLL' (default: 'CE')
        - latent_weight (float): Weight scaling for Gaussian latent loss (default: 1.0)
        - cate_weight (float): Weight for category loss (default: 0.1)
        - training_mode (str): 'pretrain' or 'train' (default: 'train')
        - BOS (int): Beginning-of-sequence token (default: 1)
        - EOS (int): End-of-sequence token (default: 2)
        - PAD (int): Padding token (default: 0)
    """

    def __init__(self, config, data_feature):
        super(L2MM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', torch.device('cpu'))

        # --- Extract vocabulary sizes from data_feature ---
        self.input_vocab_size = data_feature.get('src_loc_vocab_size',
                            data_feature.get('input_vocab_size', 2245))
        self.output_vocab_size = data_feature.get('trg_seg_vocab_size',
                            data_feature.get('output_vocab_size', 6914))

        # --- Model hyperparameters from config ---
        self.embedding_size = config.get('embedding_size', 256)
        self.hidden_size = config.get('hidden_size', 256)
        self.num_layers = config.get('num_layers', 2)
        self.de_layer = config.get('de_layer', 1)
        self.dropout = config.get('dropout', 0.1)
        self.bidirectional = config.get('bidirectional', True)
        self.cluster_size = config.get('cluster_size', 10)
        self.max_length = config.get('max_length', 300)
        self.criterion_name = config.get('criterion_name', 'CE')
        self.latent_weight = config.get('latent_weight', 1.0)
        self.cate_weight = config.get('cate_weight', 0.1)
        self.training_mode = config.get('training_mode', 'train')

        # Special tokens
        self.BOS = data_feature.get('sos_token_trg', config.get('BOS', 0))
        self.EOS = data_feature.get('eos_token_trg', config.get('EOS', 2))
        self.PAD = data_feature.get('pad_token_trg', config.get('PAD', 1))

        # --- Build model components ---
        # Source embedding (grid cells)
        self.embedding_src = nn.Embedding(
            self.input_vocab_size, self.embedding_size, padding_idx=self.PAD
        )
        # Target embedding (road segments)
        self.embedding_trg = nn.Embedding(
            self.output_vocab_size, self.embedding_size, padding_idx=self.PAD
        )

        # Encoder: bidirectional GRU
        self.encoder = Encoder(
            self.embedding_size, self.hidden_size, self.num_layers,
            self.dropout, self.bidirectional, self.embedding_src
        )

        # Latent distribution: VAE-style with Gaussian mixture
        self.latent = LatentDistribution(self.cluster_size, self.hidden_size)

        # Decoder: stacked GRU with attention
        self.decoder = Decoder(
            self.embedding_size, self.hidden_size, self.de_layer,
            self.dropout, self.embedding_trg
        )

        # Output projection: merged from original m1 module
        # Adaptation: replaced nn.LogSoftmax() with nn.LogSoftmax(dim=-1)
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_vocab_size),
            nn.LogSoftmax(dim=-1)
        )

        # Loss function
        if self.criterion_name == "CE":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.PAD)
        else:
            self.loss_fn = nn.NLLLoss(ignore_index=self.PAD)

        logger.info(
            "L2MM model initialized: input_vocab=%d, output_vocab=%d, "
            "hidden=%d, num_layers=%d, de_layer=%d, bidirectional=%s, "
            "cluster_size=%d, training_mode=%s",
            self.input_vocab_size, self.output_vocab_size,
            self.hidden_size, self.num_layers, self.de_layer,
            self.bidirectional, self.cluster_size, self.training_mode
        )

    def encoder_hn2decoder_h0(self, h):
        """Convert bidirectional encoder hidden state to decoder initial state.

        For a bidirectional encoder with K layers, h has shape
        (2*K, batch, hidden_per_dir). This reshapes it to
        (K, batch, hidden_size) by concatenating forward/backward directions.

        Original: repos/L2MM/mapmatching/model.py lines 176-183

        Args:
            h: (num_layers * num_directions, batch, hidden_per_dir)

        Returns:
            (num_layers, batch, hidden_size)
        """
        if self.encoder.num_directions == 2:
            num_layers = h.size(0) // 2
            batch = h.size(1)
            hidden_size = h.size(2)
            return h.view(num_layers, 2, batch, hidden_size) \
                    .transpose(1, 2).contiguous() \
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def _unpack_batch(self, batch):
        """Extract and format tensors from the LibCity batch dict.

        Expected batch keys:
            - 'input_src': (batch, src_seq_len) LongTensor of grid cell IDs
            - 'src_lengths': (batch,) LongTensor of source sequence lengths
            - 'input_trg': (batch, trg_seq_len) LongTensor of target road IDs
                           (with BOS prepended, EOS appended)
            - 'trg_mask': (batch, trg_seq_len) ByteTensor mask for target
            - 'output_trg': (batch, trg_seq_len) ground truth target for loss

        All tensors are moved to self.device.

        Returns:
            src: (src_seq_len, batch) - transposed source
            lengths: (batch,) - source lengths
            trg: (trg_seq_len, batch) - transposed target (with BOS/EOS)
            trg_mask: (trg_seq_len, batch) - transposed mask, or None
            target: (trg_seq_len, batch) - transposed ground truth, or None
        """
        input_src = batch['input_src'].to(self.device)     # (batch, src_len)
        if 'src_lengths' in batch:
            src_lengths = batch['src_lengths'].to(self.device)
        else:
            # Compute lengths from padding (PAD token is 1 in DeepMMSeq2SeqDataset)
            pad_id = 1
            src_lengths = (input_src != pad_id).sum(dim=1)

        # Transpose to seq-first format: (seq_len, batch)
        src = input_src.transpose(0, 1).contiguous()
        lengths = src_lengths

        # Target sequences (may not be present during inference)
        trg = None
        trg_mask = None
        target = None

        if 'input_trg' in batch:
            input_trg = batch['input_trg'].to(self.device)  # (batch, trg_len)
            trg = input_trg.transpose(0, 1).contiguous()

        if 'trg_mask' in batch:
            mask = batch['trg_mask'].to(self.device)         # (batch, trg_len)
            trg_mask = mask.transpose(0, 1).contiguous()

        if 'output_trg' in batch:
            out_trg = batch['output_trg'].to(self.device)    # (batch, trg_len)
            target = out_trg.transpose(0, 1).contiguous()

        return src, lengths, trg, trg_mask, target

    def _forward_encdec(self, src, lengths, trg, kind="train"):
        """Core encoder-decoder forward pass.

        Mirrors the original EncoderDecoder.forward() logic with the
        three operating modes: pretrain, train, test.

        Original: repos/L2MM/mapmatching/model.py lines 185-200

        Args:
            src: (src_len, batch) source grid cell IDs
            lengths: (batch,) source sequence lengths
            trg: (trg_len, batch) target road IDs (with BOS/EOS)
            kind: 'pretrain', 'train', or 'test'

        Returns:
            If kind == 'train':
                output (seq_len, batch, hidden_size), batch_latent_loss, batch_cate_loss
            If kind == 'pretrain':
                output (seq_len, batch, hidden_size)
            If kind == 'test':
                z (batch, hidden_size) - latent encoding
        """
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)

        if kind == "train":
            z, batch_latent_loss, batch_cate_loss = self.latent(
                decoder_h0[-1].unsqueeze(0)
            )
            z = z.unsqueeze(0)
            output, de_hn = self.decoder(trg[:-1], z, H)
            return output, batch_latent_loss, batch_cate_loss

        elif kind == "pretrain":
            z = self.latent(decoder_h0[-1].unsqueeze(0), kind)
            z = z.unsqueeze(0)
            output, de_hn = self.decoder(trg[:-1], z, H)
            return output

        elif kind == "test":
            z = self.latent(decoder_h0[-1].unsqueeze(0), kind)
            return z

    def forward(self, batch):
        """Forward pass through the L2MM model.

        Adaptation: accepts LibCity batch dict and returns logits suitable
        for loss computation.

        During training, the full encoder-decoder-output pipeline is executed.
        The output projection (Linear + LogSoftmax) is applied to decoder outputs.

        Args:
            batch: LibCity batch dict with keys described in _unpack_batch

        Returns:
            output_logits: (batch, trg_seq_len - 1, output_vocab_size)
                Log-probability distribution over road segments for each
                target position (excluding the last target token, as the
                decoder receives trg[:-1] as input).
        """
        src, lengths, trg, trg_mask, target = self._unpack_batch(batch)

        kind = self.training_mode if self.training else "pretrain"

        if kind == "train":
            output, _, _ = self._forward_encdec(src, lengths, trg, kind="train")
        else:
            output = self._forward_encdec(src, lengths, trg, kind="pretrain")

        # output shape: (trg_len - 1, batch, hidden_size)
        # Apply output projection
        output_flat = output.view(-1, output.size(2))  # (seq_len * batch, hidden_size)
        logits_flat = self.output_projection(output_flat)  # (seq_len * batch, output_vocab)

        # Reshape back: (trg_len - 1, batch, output_vocab)
        logits = logits_flat.view(output.size(0), output.size(1), -1)

        # Transpose to batch-first: (batch, trg_len - 1, output_vocab)
        logits = logits.transpose(0, 1).contiguous()

        return logits

    def predict(self, batch):
        """Autoregressive prediction for inference.

        Performs greedy decoding: at each step, the argmax prediction is fed
        back as input to the next step. Decoding stops when EOS is produced
        or max_length is reached.

        Original: repos/L2MM/mapmatching/evaluate.py lines 9-29
        and repos/L2MM/mapmatching/train_with_bucket.py lines 69-91

        Adaptation: uses self.device instead of hardcoded Variable/cuda,
        processes entire batch at once.

        Args:
            batch: LibCity batch dict containing at minimum 'input_src' and 'src_lengths'

        Returns:
            predictions: (batch, max_length) LongTensor of predicted road segment IDs.
                         Positions after EOS are filled with PAD.
        """
        self.eval()
        with torch.no_grad():
            src, lengths, trg, trg_mask, target = self._unpack_batch(batch)

            # Encode
            encoder_hn, H = self.encoder(src, lengths)
            decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)

            # Get latent z (test mode: deterministic)
            z = self.latent(decoder_h0[-1].unsqueeze(0), 'test')
            h = z.unsqueeze(0)  # (1, batch, hidden_size)

            batch_size = src.size(1)
            # Start with BOS token
            input_token = torch.full(
                (1, batch_size), self.BOS, dtype=torch.long, device=self.device
            )

            predictions = torch.full(
                (batch_size, self.max_length), self.PAD,
                dtype=torch.long, device=self.device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for t in range(self.max_length):
                o, h = self.decoder(input_token, h, H)
                # o shape: (1, batch, hidden_size)
                o_flat = o.view(-1, o.size(2))          # (batch, hidden_size)
                logits = self.output_projection(o_flat)  # (batch, output_vocab)
                _, top_id = logits.topk(1, dim=-1)       # (batch, 1)
                top_id = top_id.squeeze(-1)              # (batch,)

                # Store predictions for non-finished sequences
                predictions[:, t] = torch.where(finished, self.PAD, top_id)

                # Mark sequences that produced EOS
                finished = finished | (top_id == self.EOS)

                if finished.all():
                    break

                # Next input: the predicted token
                input_token = top_id.unsqueeze(0)  # (1, batch)

        return predictions

    def calculate_loss(self, batch):
        """Compute training loss combining CE/NLL loss with latent losses.

        The total loss is:
            loss = CE_loss + latent_weight * gaussian_loss + cate_weight * cate_loss

        When cluster_size == 1, cate_loss is not used (equivalent to standard VAE).
        When training_mode == 'pretrain', only CE_loss is computed.

        Original loss logic from:
            repos/L2MM/mapmatching/train_with_bucket.py lines 242-249
            repos/L2MM/mapmatching/util.py DenseLoss class

        Adaptation: mask-based cross-entropy replaces the original DenseLoss
        which required separate target and mask tensors.

        Args:
            batch: LibCity batch dict with 'input_src', 'src_lengths',
                   'input_trg', 'output_trg', and optionally 'trg_mask'

        Returns:
            loss: scalar tensor
        """
        src, lengths, trg, trg_mask, target = self._unpack_batch(batch)

        if trg is None or target is None:
            logger.warning("No target provided in batch, returning zero loss.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        kind = self.training_mode

        if kind == "train":
            output, batch_gaussian_loss, batch_cate_loss = self._forward_encdec(
                src, lengths, trg, kind="train"
            )
        else:
            output = self._forward_encdec(src, lengths, trg, kind="pretrain")
            batch_gaussian_loss = torch.tensor(0.0, device=self.device)
            batch_cate_loss = torch.tensor(0.0, device=self.device)

        # output shape: (trg_len - 1, batch, hidden_size)
        # Apply output projection
        output_flat = output.view(-1, output.size(2))
        logits = self.output_projection(output_flat)

        # Target: trg[1:] (shifted by one, skip BOS)
        target_shifted = target[1:]  # (trg_len - 1, batch)

        if trg_mask is not None:
            mask_shifted = trg_mask[1:]  # (trg_len - 1, batch)
        else:
            # Build mask from non-PAD positions in target
            mask_shifted = (target_shifted != self.PAD).float()

        # Flatten for loss computation
        target_flat = target_shifted.contiguous().view(-1)  # (seq_len * batch)
        mask_flat = mask_shifted.contiguous().view(-1)      # (seq_len * batch)

        # Apply mask: only compute loss on valid (non-PAD) positions
        valid_mask = mask_flat.bool()

        if valid_mask.any():
            if self.criterion_name == "CE":
                # For CE loss, we need raw logits (before LogSoftmax)
                # Extract the linear layer output before LogSoftmax
                raw_logits = self.output_projection[0](output.view(-1, output.size(2)))
                ce_loss = F.cross_entropy(
                    raw_logits[valid_mask],
                    target_flat[valid_mask],
                    ignore_index=self.PAD
                )
            else:
                # NLL loss with log-probabilities
                ce_loss = F.nll_loss(
                    logits[valid_mask],
                    target_flat[valid_mask],
                    ignore_index=self.PAD
                )
        else:
            ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Combine losses
        total_loss = ce_loss

        if kind == "train":
            if self.cluster_size == 1:
                total_loss = total_loss + self.latent_weight * batch_gaussian_loss
            else:
                total_loss = (total_loss
                              + self.latent_weight / self.hidden_size * batch_gaussian_loss
                              + self.cate_weight * batch_cate_loss)

        return total_loss
