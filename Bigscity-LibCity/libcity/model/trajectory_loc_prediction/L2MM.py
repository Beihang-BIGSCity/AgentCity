"""
L2MM (Learning to Map Match) Model for LibCity Trajectory Location Prediction

Original repository: repos/L2MM/mapmatching/model.py
Adapted for LibCity framework by converting from GPS grid to road segment mapping
to POI next location prediction with unified vocabulary.

Key adaptations:
- Unified vocabulary: Both input and output use loc_size from data_feature
- Single-step prediction: Predicts next location only (not full sequence generation)
- LibCity batch format: Uses 'current_loc' and 'target' keys from BatchPAD
- Device management: Uses config['device'] for tensor placement

Core architecture preserved:
- Bidirectional GRU encoder
- VAE-GMM latent distribution for learning trajectory representations
- Global attention mechanism
- Stacking GRU decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from libcity.model.abstract_model import AbstractModel


class Encoder(nn.Module):
    """Bidirectional GRU Encoder for trajectory sequences."""

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
                          dropout=dropout if num_layers > 1 else 0)

    def forward(self, input, lengths, h0=None):
        """
        Args:
            input: (seq_len, batch_size) tensor of location indices
            lengths: (batch_size,) tensor of sequence lengths
            h0: initial hidden state

        Returns:
            hn: final hidden state (num_layers * num_directions, batch, hidden)
            output: all hidden states (seq_len, batch, hidden * num_directions)
        """
        embed = self.embedding(input)
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths, enforce_sorted=False)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0]
        return hn, output


class LatentDistribution(nn.Module):
    """
    VAE-GMM Latent Distribution module.

    Learns a Gaussian Mixture Model in the latent space for trajectory representation.
    Provides regularization through KL divergence loss components.
    """

    def __init__(self, cluster_size, hidden_size, device):
        super(LatentDistribution, self).__init__()
        self.cluster_size = cluster_size
        self.hidden_size = hidden_size
        self.device = device

        # Initialize cluster centers (mu_c) randomly
        mu_c = torch.rand(cluster_size, hidden_size)
        self.mu_c = nn.Parameter(mu_c, requires_grad=True)

        # Initialize cluster log variances to zeros
        log_sigma_sq_c = torch.zeros(cluster_size, hidden_size)
        self.log_sigma_sq_c = nn.Parameter(log_sigma_sq_c, requires_grad=True)

        # Linear layers to compute latent distribution parameters
        self.cal_mu_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_mu_z.weight, std=0.02)
        nn.init.constant_(self.cal_mu_z.bias, 0.0)

        self.cal_log_sigma_z = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.cal_log_sigma_z.weight, std=0.02)
        nn.init.constant_(self.cal_log_sigma_z.bias, 0.0)

    def batch_latent_loss(self, stack_log_sigma_sq_c, stack_mu_c,
                          stack_log_sigma_sq_z, stack_mu_z, att, log_sigma_sq_z):
        """
        Compute VAE-GMM latent losses.

        Returns:
            batch_latent_loss: KL divergence loss for the latent distribution
            batch_cate_loss: Categorical regularization loss
        """
        avg_ = torch.mean(stack_log_sigma_sq_c
                          + torch.exp(stack_log_sigma_sq_z) / torch.exp(stack_log_sigma_sq_c)
                          + torch.pow(stack_mu_z - stack_mu_c, 2) / torch.exp(stack_log_sigma_sq_c),
                          dim=-1)

        sum_ = torch.sum(att * avg_, dim=-1).squeeze()

        mean_ = torch.mean(1 + log_sigma_sq_z, dim=-1).squeeze()

        batch_latent_loss = 0.5 * sum_ - 0.5 * mean_
        batch_latent_loss = torch.mean(batch_latent_loss).squeeze()

        cate_mean = torch.mean(att, dim=0).squeeze()
        batch_cate_loss = torch.mean(cate_mean * torch.log(cate_mean + 1e-10)).squeeze()
        batch_cate_loss = torch.mean(batch_cate_loss).squeeze()

        return batch_latent_loss, batch_cate_loss

    def forward(self, h, mode="train"):
        """
        Args:
            h: encoder hidden state (1, batch, hidden_size)
            mode: "train" for full VAE-GMM, "pretrain" for simple VAE, "test" for deterministic

        Returns:
            In train mode: (z, batch_latent_loss, batch_cate_loss)
            In pretrain mode: z
            In test mode: mu_z (deterministic)
        """
        h = h.squeeze(0)  # (batch, hidden_size)
        mu_z = self.cal_mu_z(h)

        if mode == "test":
            return mu_z

        log_sigma_sq_z = self.cal_log_sigma_z(h)
        eps_z = torch.randn(size=log_sigma_sq_z.shape, device=self.device)

        z = mu_z + torch.sqrt(torch.exp(log_sigma_sq_z)) * eps_z

        if mode == "pretrain":
            return z
        else:
            # Full VAE-GMM training mode
            stack_mu_c = self.mu_c.unsqueeze(0).expand(z.shape[0], -1, -1)
            stack_log_sigma_sq_c = self.log_sigma_sq_c.unsqueeze(0).expand(z.shape[0], -1, -1)
            stack_mu_z = mu_z.unsqueeze(1).expand(-1, self.cluster_size, -1)
            stack_log_sigma_sq_z = log_sigma_sq_z.unsqueeze(1).expand(-1, self.cluster_size, -1)
            stack_z = z.unsqueeze(1).expand(-1, self.cluster_size, -1)

            # Compute attention weights (cluster assignment probabilities)
            att_logits = -torch.sum(
                torch.pow(stack_z - stack_mu_c, 2) / torch.exp(stack_log_sigma_sq_c),
                dim=-1
            )
            att = F.softmax(att_logits, dim=-1) + 1e-10

            batch_latent_loss, batch_cate_loss = self.batch_latent_loss(
                stack_log_sigma_sq_c, stack_mu_c,
                stack_log_sigma_sq_z, stack_mu_z,
                att, log_sigma_sq_z
            )
            return z, batch_latent_loss, batch_cate_loss


class GlobalAttention(nn.Module):
    """Global attention mechanism for decoder."""

    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Args:
            q: query tensor (batch, hidden_size)
            H: encoder outputs (batch, seq_len, hidden_size)

        Returns:
            context-enhanced representation (batch, hidden_size)
        """
        q1 = q.unsqueeze(2)  # (batch, hidden, 1)
        a = torch.bmm(H, q1).squeeze(2)  # (batch, seq_len)
        a = self.softmax(a)
        a = a.unsqueeze(1)  # (batch, 1, seq_len)
        c = torch.bmm(a, H).squeeze(1)  # (batch, hidden)
        c = torch.cat([c, q], 1)  # (batch, 2*hidden)
        return self.tanh(self.L2(c))


class StackingGRUCell(nn.Module):
    """Stacked GRU cells for decoder."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Args:
            input: (batch, input_size)
            h0: (num_layers, batch, hidden_size)

        Returns:
            output: (batch, hidden_size)
            hn: (num_layers, batch, hidden_size)
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


class Decoder(nn.Module):
    """Decoder with attention mechanism and stacking GRU."""

    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers, dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Args:
            input: (seq_len, batch) location indices
            h: initial hidden state (num_layers, batch, hidden_size)
            H: encoder outputs (seq_len, batch, hidden_size)
            use_attention: whether to use attention mechanism

        Returns:
            output: (seq_len, batch, hidden_size)
            h: final hidden state
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        embed = self.embedding(input)
        output = []
        for e in embed.split(1):
            e = e.squeeze(0)  # (batch, emb_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))  # H: (batch, seq, hidden)
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h

    def forward_step(self, input, h, H, use_attention=True):
        """
        Single step forward for prediction.

        Args:
            input: (batch,) location indices for single step
            h: hidden state (num_layers, batch, hidden_size)
            H: encoder outputs (seq_len, batch, hidden_size)

        Returns:
            output: (batch, hidden_size)
            h: updated hidden state
        """
        embed = self.embedding(input)  # (batch, emb_size)
        o, h = self.rnn(embed, h)
        if use_attention:
            o = self.attention(o, H.transpose(0, 1))
        o = self.dropout(o)
        return o, h


class L2MM(AbstractModel):
    """
    L2MM: Learning to Map Match model adapted for LibCity trajectory location prediction.

    This model uses a VAE-GMM encoder-decoder architecture with global attention
    for next location prediction in trajectory sequences.

    Original paper: "Learning to Map Match"
    Original repository: repos/L2MM

    Adaptations for LibCity:
    - Unified vocabulary for input and output (loc_size from data_feature)
    - Single-step next location prediction
    - LibCity batch format support (BatchPAD with 'current_loc', 'target')
    - Device management through config

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Dictionary containing data-specific features (loc_size, etc.)
    """

    def __init__(self, config, data_feature):
        super(L2MM, self).__init__(config, data_feature)

        # Data features
        self.loc_size = data_feature.get('loc_size')
        self.loc_pad = data_feature.get('loc_pad', 0)

        # Model hyperparameters from config
        self.hidden_size = config.get('hidden_size', 256)
        self.embedding_size = config.get('embedding_size', 256)
        self.num_layers = config.get('num_layers', 2)
        self.de_layer = config.get('de_layer', 1)
        self.dropout = config.get('dropout', 0.1)
        self.bidirectional = config.get('bidirectional', True)
        self.cluster_size = config.get('cluster_size', 10)
        self.device = config.get('device', torch.device('cpu'))

        # Training mode: 'pretrain' for simple VAE, 'train' for full VAE-GMM
        self.training_mode = config.get('training_mode', 'train')

        # Loss weights
        self.latent_weight = config.get('latent_weight', 1.0)
        self.cate_weight = config.get('cate_weight', 0.1)

        # Embeddings - unified vocabulary for both encoder and decoder
        self.embedding = nn.Embedding(self.loc_size, self.embedding_size, padding_idx=self.loc_pad)

        # Encoder (bidirectional GRU)
        self.encoder = Encoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            embedding=self.embedding
        )

        # Latent distribution (VAE-GMM)
        self.latent = LatentDistribution(
            cluster_size=self.cluster_size,
            hidden_size=self.hidden_size,
            device=self.device
        )

        # Decoder (stacking GRU with attention)
        self.decoder = Decoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.de_layer,
            dropout=self.dropout,
            embedding=self.embedding
        )

        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_size, self.loc_size)

        # Criterion for classification loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.loc_pad)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name and 'cal_' not in name:
                nn.init.constant_(param.data, 0)

    def encoder_hn2decoder_h0(self, h):
        """
        Transform encoder hidden state to decoder initial state.

        For bidirectional encoder, combines forward and backward hidden states.
        """
        if self.encoder.num_directions == 2:
            num_layers = h.size(0) // 2
            batch = h.size(1)
            hidden_size = h.size(2)
            h = h.view(num_layers, 2, batch, hidden_size)
            h = h.transpose(1, 2).contiguous()
            h = h.view(num_layers, batch, hidden_size * 2)
        return h

    def forward(self, batch):
        """
        Forward pass for training.

        Args:
            batch: LibCity batch dictionary with keys:
                - 'current_loc': (batch, seq_len) input location sequence
                - 'target': (batch,) target next location

        Returns:
            Dictionary containing:
                - 'logits': (batch, loc_size) prediction logits
                - 'latent_loss': VAE latent loss (if training mode)
                - 'cate_loss': Categorical loss (if training mode)
        """
        # Get input data
        loc = batch['current_loc']  # (batch, seq_len)
        batch_size = loc.shape[0]

        # Get sequence lengths
        if hasattr(batch, 'get_origin_len'):
            lengths = batch.get_origin_len('current_loc')
            lengths = torch.tensor(lengths, device=self.device)
        else:
            # Compute lengths from non-padding positions
            lengths = (loc != self.loc_pad).sum(dim=1)

        # Transpose to (seq_len, batch) for RNN
        loc = loc.transpose(0, 1)

        # Encode
        encoder_hn, H = self.encoder(loc, lengths)

        # Transform hidden state
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)

        # Get latent representation
        if self.training and self.training_mode == 'train':
            z, latent_loss, cate_loss = self.latent(decoder_h0[-1].unsqueeze(0), mode='train')
        elif self.training and self.training_mode == 'pretrain':
            z = self.latent(decoder_h0[-1].unsqueeze(0), mode='pretrain')
            latent_loss = torch.tensor(0.0, device=self.device)
            cate_loss = torch.tensor(0.0, device=self.device)
        else:
            z = self.latent(decoder_h0[-1].unsqueeze(0), mode='test')
            latent_loss = torch.tensor(0.0, device=self.device)
            cate_loss = torch.tensor(0.0, device=self.device)

        # For single-step prediction, use the last input location
        # Get the last valid position for each sequence
        last_idx = lengths - 1
        last_loc = loc.transpose(0, 1).gather(1, last_idx.unsqueeze(1).long()).squeeze(1)  # (batch,)

        # Initialize decoder hidden state with latent z
        # Expand z to match decoder layers
        decoder_h = z.unsqueeze(0).expand(self.de_layer, -1, -1).contiguous()

        # Single step decode
        output, _ = self.decoder.forward_step(last_loc, decoder_h, H, use_attention=True)

        # Project to vocabulary
        logits = self.output_layer(output)  # (batch, loc_size)

        return {
            'logits': logits,
            'latent_loss': latent_loss,
            'cate_loss': cate_loss
        }

    def predict(self, batch):
        """
        Predict next location.

        Args:
            batch: LibCity batch dictionary

        Returns:
            scores: (batch, loc_size) log probabilities
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)
            logits = output['logits']
            scores = F.log_softmax(logits, dim=-1)
        return scores

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Combines cross-entropy loss for next location prediction with
        VAE-GMM losses (latent KL divergence and categorical regularization).

        Args:
            batch: LibCity batch dictionary with 'current_loc' and 'target'

        Returns:
            total_loss: Combined loss tensor
        """
        output = self.forward(batch)
        logits = output['logits']
        target = batch['target']

        # Classification loss
        ce_loss = self.criterion(logits, target)

        # VAE-GMM losses
        latent_loss = output['latent_loss']
        cate_loss = output['cate_loss']

        # Combined loss
        total_loss = ce_loss + self.latent_weight * latent_loss + self.cate_weight * cate_loss

        return total_loss
