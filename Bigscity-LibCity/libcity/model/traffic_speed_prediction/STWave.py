"""
STWave Model Adaptation for LibCity Framework
==============================================

Original Paper: "When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks"
Original Repository: https://github.com/LMissher/STWave

This adaptation ports the STWave model to LibCity conventions for traffic speed prediction.
The model uses wavelet decomposition to disentangle traffic signals into low-frequency (trend)
and high-frequency (fluctuation) components, then processes them with a dual encoder architecture.

Key Adaptations:
- Inherits from AbstractTrafficStateModel
- Computes localadj, spawave, temwave from adj_mx instead of requiring pre-computed files
- Handles LibCity batch format: {'X': tensor, 'y': tensor}
- Implements required methods: forward, predict, calculate_loss

Author: Model Adaptation Agent
Date: 2026-01-30
"""

import math
from logging import getLogger

import pywt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def compute_laplacian(W):
    """
    Compute the normalized Laplacian of the weight matrix.
    L = I - D^{-1/2} W D^{-1/2}

    Args:
        W: Weight/adjacency matrix

    Returns:
        L: Normalized Laplacian matrix
    """
    d = np.array(W.sum(axis=0)).flatten()
    d = np.where(d > 0, 1 / np.sqrt(d), 0)
    D = sp.diags(d, 0)
    I = sp.identity(len(d), dtype=W.dtype)
    L = I - D @ W @ D
    return L


def get_largest_k_eigenvectors(L, k):
    """
    Compute the k largest eigenvalues and eigenvectors of the Laplacian.

    Args:
        L: Laplacian matrix
        k: Number of eigenvalues/eigenvectors to compute

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    try:
        lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    except Exception:
        # Fallback if eigsh fails (e.g., for very small graphs)
        L_dense = L.toarray() if sp.issparse(L) else L
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        idx = np.argsort(eigenvalues)[::-1][:k]
        lamb = eigenvalues[idx]
        U = eigenvectors[:, idx]
    return (lamb.astype(np.float32), U.astype(np.float32))


def compute_spawave(adj, dims):
    """
    Compute spatial graph wavelet eigenvalues and eigenvectors.

    Args:
        adj: Adjacency matrix
        dims: Number of dimensions (eigenvalues/eigenvectors to compute)

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    adj_with_self = adj + np.eye(adj.shape[0])
    L = compute_laplacian(sp.csr_matrix(adj_with_self))
    return get_largest_k_eigenvectors(L, dims)


def compute_localadj(adj):
    """
    Compute local adjacency (nearest neighbors based on shortest path distances).
    Uses Dijkstra's algorithm to find shortest paths, then selects log(N) nearest neighbors.

    Args:
        adj: Adjacency matrix

    Returns:
        localadj: Array of shape (N, log(N)) containing indices of nearest neighbors
    """
    num_nodes = adj.shape[0]
    sampled_nodes_number = max(1, int(math.log(num_nodes, 2)))

    # Create graph and compute shortest paths
    adj_with_self = adj + np.eye(num_nodes)
    graph = csr_matrix(adj_with_self)
    dist_matrix = dijkstra(csgraph=graph)

    # Set diagonal to large value to avoid self-selection
    dist_matrix[dist_matrix == 0] = dist_matrix.max() + 10
    np.fill_diagonal(dist_matrix, dist_matrix.max() + 10)

    # Handle inf values (disconnected nodes)
    dist_matrix = np.where(np.isinf(dist_matrix), dist_matrix.max() + 10, dist_matrix)

    # Select nearest neighbors
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, axis=-1)[:, :sampled_nodes_number]

    return localadj.astype(np.int64)


class Chomp1d(nn.Module):
    """
    Remove extra dimension added by padding in causal convolution.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemEmbedding(nn.Module):
    """
    Temporal embedding layer that creates embeddings from day-of-week and time-of-day features.
    """
    def __init__(self, D, vocab_size):
        super(TemEmbedding, self).__init__()
        self.ff_te = FeedForward([vocab_size + 7, D, D])

    def forward(self, TE, T=288):
        """
        Args:
            TE: Temporal encoding tensor of shape [B, T, 2] where
                TE[..., 0] is day of week and TE[..., 1] is time of day
            T: Number of time slots per day (default 288 for 5-min intervals)

        Returns:
            Temporal embedding of shape [B, T, 1, D]
        """
        # One-hot encode day of week
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)

        # One-hot encode time of day
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)

        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B, T, 7+T]
        TE = TE.unsqueeze(dim=2)  # [B, T, 1, 7+T]
        TE = self.ff_te(TE)  # [B, T, 1, D]

        return TE


class FeedForward(nn.Module):
    """
    Multi-layer feedforward network with optional residual connection and layer normalization.
    """
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x


class Sparse_Spatial_Attention(nn.Module):
    """
    Sparse Spatial Attention module that uses spectral graph information.
    Implements efficient attention by sampling nodes based on importance scores.
    """
    def __init__(self, heads, dims, samples, localadj):
        super(Sparse_Spatial_Attention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims
        self.s = samples
        self.la = localadj

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features)
        self.ff = FeedForward([features, features, features], True)
        self.proj = nn.Linear(self.la.shape[1], 1)

    def forward(self, x, spa_eigvalue, spa_eigvec, tem_eigvalue, tem_eigvec):
        """
        Args:
            x: Input tensor of shape [B, T, N, D]
            spa_eigvalue: Spatial eigenvalues [D]
            spa_eigvec: Spatial eigenvectors [N, D]
            tem_eigvalue: Temporal eigenvalues [D]
            tem_eigvec: Temporal eigenvectors [N, D]

        Returns:
            Output tensor of shape [B, T, N, D]
        """
        # Add spectral information to input
        x_ = x + torch.matmul(spa_eigvec, torch.diag_embed(spa_eigvalue)) + \
             torch.matmul(tem_eigvec, torch.diag_embed(tem_eigvalue))

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        # Multi-head split
        Q = torch.cat(torch.split(Q, self.d, -1), 0)
        K = torch.cat(torch.split(K, self.d, -1), 0)
        V = torch.cat(torch.split(V, self.d, -1), 0)

        B, T, N, D = K.shape

        # Calculate sampled Q_K based on local adjacency
        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), self.la, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Select top-k important nodes
        Sampled_Nodes = max(1, int(self.s * math.log(N, 2)))
        M = self.proj(Q_K_sample).squeeze(-1)
        M_top = M.topk(Sampled_Nodes, sorted=False)[1]

        # Compute attention with reduced queries
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(T)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        Q_K /= (self.d ** 0.5)
        attn = torch.softmax(Q_K, dim=-1)

        # Copy operation for output
        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        value = torch.matmul(attn, V).unsqueeze(-3).expand(B, T, N, M_top.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None], cp, :].squeeze(-2)

        # Multi-head concat
        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1)
        value = self.ofc(value)
        value = self.ln(value)

        return self.ff(value)


class TemporalAttention(nn.Module):
    """
    Temporal attention module with causal masking.
    """
    def __init__(self, heads, dims):
        super(TemporalAttention, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qfc = FeedForward([features, features])
        self.kfc = FeedForward([features, features])
        self.vfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, x, te, Mask=True):
        """
        Args:
            x: Input tensor of shape [B, T, N, F]
            te: Temporal embedding of shape [B, T, N, F]
            Mask: Whether to apply causal mask

        Returns:
            Output tensor of shape [B, T, N, F]
        """
        x = x + te

        query = self.qfc(x)
        key = self.kfc(x)
        value = self.vfc(x)

        # Multi-head split and permute
        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps).to(x.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attention).to(x.device)
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1)
        value = torch.matmul(attention, value)

        # Multi-head concat
        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)
        value = self.ofc(value)
        value = value + x
        value = self.ln(value)

        return self.ff(value)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions.
    """
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            conv = nn.Conv2d(features, features, (1, kernel_size),
                           dilation=(1, dilation_size), padding=(0, padding))
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            drop = nn.Dropout(dropout)
            layers += [nn.Sequential(conv, chomp, relu, drop)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh.transpose(1, 3)).transpose(1, 3)
        return xh


class Dual_Encoder(nn.Module):
    """
    Dual Encoder that processes low-frequency and high-frequency components separately.
    Uses temporal attention for low-frequency and TCN for high-frequency.
    """
    def __init__(self, heads, dims, samples, localadj, spawave, temwave):
        super(Dual_Encoder, self).__init__()
        self.temporal_conv = TemporalConvNet(heads * dims)
        self.temporal_att = TemporalAttention(heads, dims)

        self.spatial_att_l = Sparse_Spatial_Attention(heads, dims, samples, localadj)
        self.spatial_att_h = Sparse_Spatial_Attention(heads, dims, samples, localadj)

        # Spatial eigenvalues/eigenvectors
        spa_eigvalue = torch.from_numpy(spawave[0].astype(np.float32))
        self.spa_eigvalue = nn.Parameter(spa_eigvalue, requires_grad=True)
        self.register_buffer('spa_eigvec', torch.from_numpy(spawave[1].astype(np.float32)))

        # Temporal eigenvalues/eigenvectors
        tem_eigvalue = torch.from_numpy(temwave[0].astype(np.float32))
        self.tem_eigvalue = nn.Parameter(tem_eigvalue, requires_grad=True)
        self.register_buffer('tem_eigvec', torch.from_numpy(temwave[1].astype(np.float32)))

    def forward(self, xl, xh, te):
        """
        Args:
            xl: Low-frequency component [B, T, N, F]
            xh: High-frequency component [B, T, N, F]
            te: Temporal embedding [B, T, N, F]

        Returns:
            Tuple of (processed xl, processed xh)
        """
        xl = self.temporal_att(xl, te)
        xh = self.temporal_conv(xh)

        spa_statesl = self.spatial_att_l(xl, self.spa_eigvalue, self.spa_eigvec,
                                         self.tem_eigvalue, self.tem_eigvec)
        spa_statesh = self.spatial_att_h(xh, self.spa_eigvalue, self.spa_eigvec,
                                         self.tem_eigvalue, self.tem_eigvec)
        xl = spa_statesl + xl
        xh = spa_statesh + xh

        return xl, xh


class Adaptive_Fusion(nn.Module):
    """
    Adaptive Fusion module that combines low-frequency and high-frequency components
    using cross-attention mechanism.
    """
    def __init__(self, heads, dims):
        super(Adaptive_Fusion, self).__init__()
        features = heads * dims
        self.h = heads
        self.d = dims

        self.qlfc = FeedForward([features, features])
        self.khfc = FeedForward([features, features])
        self.vhfc = FeedForward([features, features])
        self.ofc = FeedForward([features, features])

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features, features, features], True)

    def forward(self, xl, xh, te, Mask=True):
        """
        Args:
            xl: Low-frequency component [B, T, N, F]
            xh: High-frequency component [B, T, N, F]
            te: Temporal embedding [B, T, N, F]
            Mask: Whether to apply causal mask

        Returns:
            Fused output tensor [B, T, N, F]
        """
        xl = xl + te
        xh = xh + te

        query = self.qlfc(xl)
        keyh = torch.relu(self.khfc(xh))
        valueh = torch.relu(self.vhfc(xh))

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        keyh = torch.cat(torch.split(keyh, self.d, -1), 0).permute(0, 2, 3, 1)
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)

        attentionh = torch.matmul(query, keyh)

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps).to(xl.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1)
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1) * torch.ones_like(attentionh).to(xl.device)
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)
        attentionh = F.softmax(attentionh, -1)

        value = torch.matmul(attentionh, valueh)
        value = torch.cat(torch.split(value, value.shape[0] // self.h, 0), -1).permute(0, 2, 1, 3)
        value = self.ofc(value)
        value = value + xl
        value = self.ln(value)

        return self.ff(value)


class STWave(AbstractTrafficStateModel):
    """
    STWave: Spatio-Temporal Wavelet Graph Neural Network for Traffic Forecasting.

    This model uses wavelet decomposition to disentangle traffic signals into
    low-frequency (trend) and high-frequency (fluctuation) components, then
    processes them with a dual encoder architecture combining temporal attention,
    temporal convolution, and sparse spatial attention.

    Args:
        config: LibCity configuration dictionary
        data_feature: Data feature dictionary containing:
            - adj_mx: Adjacency matrix
            - num_nodes: Number of nodes
            - feature_dim: Input feature dimension
            - output_dim: Output feature dimension
            - scaler: Data scaler for inverse transform
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)

        # Model hyperparameters
        self.heads = config.get('heads', 8)
        self.dims = config.get('dims', 16)
        self.layers = config.get('layers', 2)
        self.samples = config.get('samples', 1)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.wave = config.get('wave', 'coif1')
        self.level = config.get('level', 1)
        self.time_intervals = config.get('time_intervals', 300)  # 5 minutes default
        self.vocab_size = 24 * 60 * 60 // self.time_intervals  # Number of time slots per day

        features = self.heads * self.dims

        # Compute graph features from adj_mx
        self._logger.info('Computing graph features for STWave...')
        localadj = compute_localadj(self.adj_mx)
        spawave = compute_spawave(self.adj_mx, features)

        # For temporal adjacency, we use spatial adjacency as a fallback
        # In production, you might want to compute DTW-based temporal adjacency
        temwave = spawave  # Using spatial wave as temporal wave approximation

        self._logger.info(f'localadj shape: {localadj.shape}')
        self._logger.info(f'spawave eigenvalue shape: {spawave[0].shape}, eigenvector shape: {spawave[1].shape}')

        # Build local adjacency with self-loop
        I = torch.arange(localadj.shape[0]).unsqueeze(-1)
        localadj_tensor = torch.cat([I, torch.from_numpy(localadj)], -1)

        # Dual encoder layers
        self.dual_enc = nn.ModuleList([
            Dual_Encoder(self.heads, self.dims, self.samples, localadj_tensor, spawave, temwave)
            for _ in range(self.layers)
        ])

        # Adaptive fusion
        self.adp_f = Adaptive_Fusion(self.heads, self.dims)

        # Prediction layers
        self.pre_l = nn.Conv2d(self.input_window, self.output_window, (1, 1))
        self.pre_h = nn.Conv2d(self.input_window, self.output_window, (1, 1))
        self.pre = nn.Conv2d(self.input_window, self.output_window, (1, 1))

        # Embedding layers
        self.start_emb_l = FeedForward([self.output_dim, features, features])
        self.start_emb_h = FeedForward([self.output_dim, features, features])
        self.end_emb = FeedForward([features, features, self.output_dim])
        self.end_emb_l = FeedForward([features, features, self.output_dim])
        self.te_emb = TemEmbedding(features, self.vocab_size)

    def disentangle(self, x, w, j):
        """
        Disentangle input signal into low-frequency and high-frequency components
        using wavelet decomposition.

        Args:
            x: Input tensor of shape [B, T, N, F]
            w: Wavelet name (e.g., 'coif1')
            j: Decomposition level

        Returns:
            Tuple of (low-frequency component, high-frequency component)
        """
        x_np = x.cpu().numpy()
        x_np = x_np.transpose(0, 3, 2, 1)  # [B, F, N, T]

        coef = pywt.wavedec(x_np, w, level=j)

        # Low-frequency: keep only approximation coefficients
        coefl = [coef[0]] + [None] * (len(coef) - 1)

        # High-frequency: keep only detail coefficients
        coefh = [None] + coef[1:]

        xl = pywt.waverec(coefl, w).transpose(0, 3, 2, 1)
        xh = pywt.waverec(coefh, w).transpose(0, 3, 2, 1)

        xl = torch.from_numpy(xl).float().to(self.device)
        xh = torch.from_numpy(xh).float().to(self.device)

        return xl, xh

    def _forward(self, XL, XH, TE):
        """
        Internal forward pass with pre-processed inputs.

        Args:
            XL: Low-frequency component [B, T, N, F]
            XH: High-frequency component [B, T, N, F]
            TE: Temporal encoding [B, T, 2]

        Returns:
            Tuple of (main prediction, low-frequency prediction)
        """
        xl, xh = self.start_emb_l(XL), self.start_emb_h(XH)
        te = self.te_emb(TE, self.vocab_size)

        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:, :self.input_window, :, :])

        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        hat_y = self.adp_f(hat_y_l, hat_y_h, te[:, self.input_window:, :, :])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)

        return hat_y, hat_y_l

    def _extract_temporal_features(self, batch):
        """
        Extract temporal features from batch data.

        Args:
            batch: Dictionary with 'X' and 'y' tensors

        Returns:
            Tuple of (xl, xh, te) where:
                - xl: Low-frequency component
                - xh: High-frequency component
                - te: Temporal encoding
        """
        x = batch['X']  # [B, T, N, F]
        y = batch['y']

        # Split into traffic features and temporal features
        x_traffic = x[:, :, :, :self.output_dim]
        x_te = x[:, :, :, self.output_dim:]
        y_te = y[:, :, :, self.output_dim:]

        # Disentangle traffic signal
        xl, xh = self.disentangle(x_traffic, self.wave, self.level)

        # Create temporal encoding [B, T_in + T_out, 2]
        # Assuming temporal features are time_of_day and day_of_week
        if x_te.shape[-1] >= 2:
            # Use first two temporal features as day_of_week and time_of_day
            te_x = x_te[:, :, 0, :2]  # [B, T_in, 2]
            te_y = y_te[:, :, 0, :2]  # [B, T_out, 2]
            te = torch.cat([te_x, te_y], dim=1)  # [B, T_in + T_out, 2]
        elif x_te.shape[-1] == 1:
            # Only time_of_day available, create day_of_week from it
            te_x = x_te[:, :, 0, :]  # [B, T_in, 1]
            te_y = y_te[:, :, 0, :]  # [B, T_out, 1]
            te_combined = torch.cat([te_x, te_y], dim=1)
            # Scale time_of_day to vocab_size and create day info
            time_idx = (te_combined * self.vocab_size).long()
            day_idx = time_idx // self.vocab_size
            te = torch.cat([day_idx, time_idx], dim=-1)
        else:
            # No temporal features, create default
            batch_size = x.shape[0]
            total_len = self.input_window + self.output_window
            te = torch.zeros(batch_size, total_len, 2).to(x.device)
            # Create sequential time indices
            te[:, :, 1] = torch.arange(total_len).float().unsqueeze(0).expand(batch_size, -1)

        return xl, xh, te

    def forward(self, batch):
        """
        Forward pass for LibCity batch.

        Args:
            batch: Dictionary with 'X' tensor of shape [B, T, N, F]

        Returns:
            Prediction tensor of shape [B, T_out, N, output_dim]
        """
        xl, xh, te = self._extract_temporal_features(batch)
        hat_y, _ = self._forward(xl, xh, te)
        return hat_y

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch: Dictionary with 'X' tensor

        Returns:
            Prediction tensor of shape [B, T_out, N, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss.
        Uses masked MAE loss for both main prediction and low-frequency prediction.

        Args:
            batch: Dictionary with 'X' and 'y' tensors

        Returns:
            Loss tensor
        """
        y_true = batch['y']
        xl, xh, te = self._extract_temporal_features(batch)
        y_predicted, hat_y_l = self._forward(xl, xh, te)

        # Inverse transform predictions and targets
        y_true_inv = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted_inv = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        hat_y_l_inv = self._scaler.inverse_transform(hat_y_l[..., :self.output_dim])

        # Compute low-frequency component of target
        YL, _ = self.disentangle(y_true_inv, self.wave, self.level)

        # Combined loss: main prediction + low-frequency prediction
        main_loss = loss.masked_mae_torch(y_predicted_inv, y_true_inv, null_val=0.0)
        lf_loss = loss.masked_mae_torch(hat_y_l_inv, YL, null_val=0.0)

        return main_loss + lf_loss
