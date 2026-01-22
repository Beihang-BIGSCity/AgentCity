"""
DSTAGNN - Dynamic Spatial-Temporal Aware Graph Neural Network

Reference:
    Lan et al. "DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for
    Traffic Flow Forecasting" in ICML 2022.

Adapted to LibCity framework from original implementation:
    https://github.com/SYLan2019/DSTAGNN

Key Architecture Components:
    1. Temporal Attention Transformer (TAT): Multi-head self-attention for temporal patterns
    2. Spatial Attention (SAT): Captures dynamic spatial correlations
    3. Chebyshev Graph Convolution with Spatial Attention: Graph convolution on road network
    4. Gated Temporal Units (GTU): Multi-kernel temporal convolutions (kernels 3, 5, 7)
    5. Residual Connections: Skip connections for gradient flow
"""

from logging import getLogger
from scipy.sparse.linalg import eigs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def scaled_Laplacian(W):
    """
    Compute scaled Laplacian matrix ~L = 2L/lambda_max - I

    Args:
        W (np.ndarray): Adjacency matrix, shape (N, N)

    Returns:
        np.ndarray: Scaled Laplacian matrix, shape (N, N)
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    Compute Chebyshev polynomials from T_0 to T_{K-1}

    Args:
        L_tilde (np.ndarray): Scaled Laplacian matrix, shape (N, N)
        K (int): Maximum order of Chebyshev polynomials

    Returns:
        list[np.ndarray]: List of K Chebyshev polynomial matrices
    """
    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class SScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention for Spatial Attention (without value projection)"""

    def __init__(self, d_k):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        """
        Args:
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            attn_mask: [batch_size, n_heads, seq_len, seq_len] or None

        Returns:
            torch.Tensor: Attention scores [batch_size, n_heads, len_q, len_k]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        return scores


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention for Temporal Attention (with value projection)"""

    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        """
        Args:
            Q: [batch_size, num_of_d, n_heads, len_q, d_k]
            K: [batch_size, num_of_d, n_heads, len_k, d_k]
            V: [batch_size, num_of_d, n_heads, len_v, d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len] or None
            res_att: Residual attention from previous block

        Returns:
            tuple: (context, attention_scores)
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        return context, scores


class SMultiHeadAttention(nn.Module):
    """Multi-Head Attention for Spatial dimension"""

    def __init__(self, device, d_model, d_k, d_v, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.device = device

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        """
        Args:
            input_Q: [batch_size, len_q, d_model]
            input_K: [batch_size, len_k, d_model]
            attn_mask: [batch_size, seq_len, seq_len] or None

        Returns:
            torch.Tensor: Spatial attention scores [batch_size, n_heads, len_q, len_k]
        """
        batch_size = input_Q.size(0)

        # Project and reshape to multi-head format
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)

        return attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for Temporal dimension with residual connections"""

    def __init__(self, device, d_model, d_k, d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.device = device

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        """
        Args:
            input_Q: [batch_size, num_of_d, len_q, d_model]
            input_K: [batch_size, num_of_d, len_k, d_model]
            input_V: [batch_size, num_of_d, len_v, d_model]
            attn_mask: [batch_size, seq_len, seq_len] or None
            res_att: Residual attention from previous block

        Returns:
            tuple: (output, attention_scores)
        """
        residual = input_Q
        batch_size = input_Q.size(0)

        # Project and reshape: (B, num_of_d, len, d_model) -> (B, num_of_d, n_heads, len, d_k)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # Apply attention
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)

        # Concatenate heads
        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1, self.n_heads * self.d_v)
        output = self.fc(context)

        # Add residual and normalize
        output = self.layer_norm(output + residual)

        return output, res_attn


class cheb_conv_withSAt(nn.Module):
    """K-order Chebyshev Graph Convolution with Spatial Attention"""

    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)

        # Learnable parameters for each Chebyshev order
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)]
        )
        # Learnable masks for adaptive graph
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(self.DEVICE)) for _ in range(K)]
        )

    def forward(self, x, spatial_attention, adj_pa):
        """
        Chebyshev graph convolution with spatial attention

        Args:
            x: [batch_size, N, F_in, T]
            spatial_attention: [batch_size, K, N, N] - attention scores from SMultiHeadAttention
            adj_pa: [N, N] - pre-defined adjacency matrix

        Returns:
            torch.Tensor: [batch_size, N, F_out, T]
        """
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (B, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N, N)
                mask = self.mask[k]

                # Combine learned spatial attention with pre-defined adjacency
                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                # Apply spatial attention to Chebyshev polynomial
                T_k_with_at = T_k.mul(myspatial_attention)

                theta_k = self.Theta[k]

                # Graph convolution: (B, N, N) @ (B, N, F_in) -> (B, N, F_in)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)

                # Feature transformation: (B, N, F_in) @ (F_in, F_out) -> (B, N, F_out)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return self.relu(torch.cat(outputs, dim=-1))  # (B, N, F_out, T)


class Embedding(nn.Module):
    """Positional Embedding for Temporal or Spatial dimensions"""

    def __init__(self, nb_seq, d_Em, num_of_features, Etype, device):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.device = device
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        """
        Args:
            x: Input tensor
            batch_size: Batch size

        Returns:
            torch.Tensor: Embedded tensor with positional encoding
        """
        if self.Etype == 'T':
            # Temporal embedding
            pos = torch.arange(self.nb_seq, dtype=torch.long, device=self.device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features, self.nb_seq)
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            # Spatial embedding
            pos = torch.arange(self.nb_seq, dtype=torch.long, device=self.device)
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)

        return self.norm(embedding)


class GTU(nn.Module):
    """Gated Temporal Unit - Applies gating mechanism on temporal convolution"""

    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(
            in_channels,
            2 * in_channels,
            kernel_size=(1, kernel_size),
            stride=(1, time_strides)
        )

    def forward(self, x):
        """
        Args:
            x: [B, F, N, T]

        Returns:
            torch.Tensor: [B, F, N, T'] where T' depends on kernel_size and stride
        """
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, :self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class DSTAGNN_block(nn.Module):
    """
    DSTAGNN Block: Temporal Attention -> Spatial Attention -> Graph Conv -> Temporal Conv
    """

    def __init__(self, device, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, adj_TMD, num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(DSTAGNN_block, self).__init__()

        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

        # Convert adjacency matrix to tensor on the correct device
        self.adj_pa = torch.FloatTensor(adj_pa).to(device)

        # Projection layer to reduce temporal dimension
        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))

        # Embeddings
        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, num_of_d, 'T', device)
        self.EmbedS = Embedding(num_of_vertices, d_model, num_of_d, 'S', device)

        # Attention modules
        self.TAt = MultiHeadAttention(device, num_of_vertices, d_k, d_v, n_heads, num_of_d)
        self.SAt = SMultiHeadAttention(device, d_model, d_k, d_v, K)

        # Graph convolution with spatial attention
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)

        # Multi-scale temporal convolutions with different kernel sizes
        self.gtu3 = GTU(nb_time_filter, time_strides, 3)
        self.gtu5 = GTU(nb_time_filter, time_strides, 5)
        self.gtu7 = GTU(nb_time_filter, time_strides, 7)

        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.dropout = nn.Dropout(p=0.05)

        # Fusion layer for multi-scale temporal features
        self.fcmy = nn.Sequential(
            nn.Linear(3 * num_of_timesteps - 12, num_of_timesteps),
            nn.Dropout(0.05),
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, res_att):
        """
        Args:
            x: [batch_size, N, F_in, T]
            res_att: Residual attention from previous block

        Returns:
            tuple: (output, updated_res_att)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # === Temporal Attention Transformer (TAT) ===
        if num_of_features == 1:
            TEmx = self.EmbedT(x, batch_size)  # (B, F, T, N)
        else:
            TEmx = x.permute(0, 2, 3, 1)  # (B, F, T, N)

        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)  # (B, F, T, N)

        # Project to d_model dimension
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)  # (B, N, d_model)

        # === Spatial Attention (SAT) ===
        SEmx_TAt = self.EmbedS(x_TAt, batch_size)  # (B, N, d_model)
        SEmx_TAt = self.dropout(SEmx_TAt)
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)  # (B, K, N, N)

        # === Graph Convolution with Spatial Attention ===
        spatial_gcn = self.cheb_conv_SAt(x, STAt, self.adj_pa)  # (B, N, F, T)

        # === Multi-scale Temporal Convolution ===
        X = spatial_gcn.permute(0, 2, 1, 3)  # (B, F, N, T)
        x_gtu = []
        x_gtu.append(self.gtu3(X))  # (B, F, N, T-2)
        x_gtu.append(self.gtu5(X))  # (B, F, N, T-4)
        x_gtu.append(self.gtu7(X))  # (B, F, N, T-6)
        time_conv = torch.cat(x_gtu, dim=-1)  # (B, F, N, 3T-12)
        time_conv = self.fcmy(time_conv)  # (B, F, N, T)

        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)

        # === Residual Connection ===
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)

        # Layer normalization with residual
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual, re_At


class DSTAGNN(AbstractTrafficStateModel):
    """
    DSTAGNN - Dynamic Spatial-Temporal Aware Graph Neural Network

    This model implements a deep learning architecture for traffic flow forecasting
    that combines:
    - Temporal attention mechanisms to capture dynamic temporal patterns
    - Spatial attention to model evolving spatial correlations
    - Chebyshev graph convolutions for efficient graph operations
    - Multi-scale temporal convolutions with gating mechanisms
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self.device = config.get('device', torch.device('cpu'))

        # === Extract data features ===
        num_of_vertices = self.data_feature.get('num_nodes', 307)

        # Graph selection: 'AG' uses dynamic adaptive graph, 'G' uses static adjacency
        graph_use = config.get('graph_use', 'AG')
        adj_mx = self.data_feature.get('adj_mx')
        adj_TMD = self.data_feature.get('adj_TMD', adj_mx)  # Dynamic graph (STAG/STRG)
        adj_pa = self.data_feature.get('adj_pa', adj_mx)  # Pre-defined adjacency

        # Select which graph to use for Chebyshev polynomials
        adj_merge = adj_mx if graph_use == 'G' else adj_TMD

        # === Extract model hyperparameters ===
        num_of_d = config.get('in_channels', 1)  # Input feature dimension
        nb_block = config.get('nb_block', 4)  # Number of DSTAGNN blocks
        in_channels = config.get('in_channels', 1)
        K = config.get('K', 3)  # Order of Chebyshev polynomial
        nb_chev_filter = config.get('nb_chev_filter', 32)  # Graph conv output channels
        nb_time_filter = config.get('nb_time_filter', 32)  # Temporal conv output channels
        time_strides = 1  # Fixed to 1 for this implementation

        num_for_predict = config.get('output_window', 12)  # Prediction horizon
        len_input = config.get('input_window', 12)  # Input sequence length

        # Attention parameters
        d_model = config.get('d_model', 512)  # Model dimension
        d_k = config.get('d_k', 32)  # Key dimension
        d_v = config.get('d_k', 32)  # Value dimension (default same as d_k)
        n_heads = config.get('n_heads', 3)  # Number of attention heads

        # === Compute Chebyshev polynomials ===
        L_tilde = scaled_Laplacian(adj_merge)
        cheb_polynomials = [
            torch.from_numpy(i).type(torch.FloatTensor).to(self.device)
            for i in cheb_polynomial(L_tilde, K)
        ]

        # === Build DSTAGNN blocks ===
        # First block takes original input
        self.BlockList = nn.ModuleList([
            DSTAGNN_block(
                self.device, num_of_d, in_channels, K,
                nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                adj_pa, adj_TMD, num_of_vertices, len_input, d_model, d_k, d_v, n_heads
            )
        ])

        # Subsequent blocks take output from previous blocks
        self.BlockList.extend([
            DSTAGNN_block(
                self.device, num_of_d * nb_time_filter, nb_chev_filter, K,
                nb_chev_filter, nb_time_filter, 1, cheb_polynomials,
                adj_pa, adj_TMD, num_of_vertices, len_input // time_strides,
                d_model, d_k, d_v, n_heads
            )
            for _ in range(nb_block - 1)
        ])

        # === Output layers ===
        # Aggregate multi-block outputs
        self.final_conv = nn.Conv2d(
            int((len_input / time_strides) * nb_block),
            128,
            kernel_size=(1, nb_time_filter)
        )
        # Project to prediction horizon
        self.final_fc = nn.Linear(128, num_for_predict)

        self.to(self.device)
        self._init_parameters()

        self._logger.info(f'DSTAGNN model initialized with {nb_block} blocks, '
                         f'K={K}, d_model={d_model}, n_heads={n_heads}')

    def _init_parameters(self):
        """Initialize model parameters using Xavier uniform for weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        """
        Forward pass through DSTAGNN

        Args:
            x: [batch_size, N_nodes, F_in, T_in]

        Returns:
            torch.Tensor: [batch_size, N_nodes, T_out]
        """
        need_concat = []
        res_att = 0  # Initialize residual attention

        # Pass through all DSTAGNN blocks
        for block in self.BlockList:
            x, res_att = block(x, res_att)
            need_concat.append(x)

        # Concatenate all block outputs along time dimension
        final_x = torch.cat(need_concat, dim=-1)  # (B, N, F, T*nb_block)

        # Final convolution and projection
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)  # (B, N, 128)
        output = self.final_fc(output1)  # (B, N, T_out)

        return output

    def predict(self, batch):
        """
        Prediction method following LibCity convention

        Args:
            batch (dict): Dictionary with key 'X' of shape [B, T, N, F]

        Returns:
            torch.Tensor: Predictions of shape [B, T_out, N, 1]
        """
        x = batch['X']
        # Transform from LibCity format (B, T, N, F) to model format (B, N, F, T)
        x = x.permute(0, 2, 3, 1)

        # Forward pass
        output = self.forward(x)  # (B, N, T_out)

        # Transform back to LibCity format (B, T_out, N, 1)
        output = output.permute(0, 2, 1).unsqueeze(-1)

        return output

    def calculate_loss(self, batch):
        """
        Calculate loss using smooth L1 loss on inverse-transformed predictions

        Args:
            batch (dict): Dictionary with keys 'X' and 'y'

        Returns:
            torch.Tensor: Scalar loss value
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform to original scale for loss calculation
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
