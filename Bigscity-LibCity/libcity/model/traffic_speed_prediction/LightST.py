"""
LightST: A Lightweight Spatio-Temporal Model for Traffic Prediction

This is an adaptation of the STMLP (lightweight student model) from the LightST paper
for the LibCity framework. The model uses dilated inception convolutions and MLP layers
for efficient spatio-temporal traffic prediction.

Original source: repos/LightST/model/Teacher.py
Adapted classes: STMLP (student model)

Key Changes from Original:
1. Inherit from AbstractTrafficStateModel instead of nn.Module
2. Replace hardcoded .cuda() calls with self.device
3. Handle LibCity batch format (dict with 'X' and 'y' keys)
4. Implement calculate_loss and predict methods
5. Fix the final Linear layer to be a proper module instead of inline creation
6. Remove unused transformer encoder components to reduce memory footprint
"""

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class NConv(nn.Module):
    """Normalized Convolution for graph operations."""
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncwl,vw->ncvl', (x, adj))
        return x.contiguous()


class DyNconv(nn.Module):
    """Dynamic Normalized Convolution for graph operations."""
    def __init__(self):
        super(DyNconv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    """Linear layer implemented as 1x1 convolution."""
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class DilatedInception(nn.Module):
    """Dilated Inception module for temporal convolutions.

    Uses multiple kernel sizes (2, 3, 6, 7) with dilation to capture
    multi-scale temporal patterns.
    """
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class GraphConstructor(nn.Module):
    """Adaptive graph constructor for learning node relationships.

    Creates a learnable adjacency matrix using node embeddings.
    """
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fulla(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class LayerNorm(nn.Module):
    """Layer Normalization with optional element-wise affine parameters.

    This is a custom implementation that supports selective indexing
    of normalization parameters based on node indices.
    """
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, inputs, idx):
        if self.elementwise_affine:
            return F.layer_norm(inputs, tuple(inputs.shape[1:]),
                                self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(inputs, tuple(inputs.shape[1:]),
                                self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class LightST(AbstractTrafficStateModel):
    """LightST: Lightweight Spatio-Temporal MLP for Traffic Prediction.

    This is the student model (STMLP) from the LightST paper, adapted for LibCity.
    It uses dilated inception convolutions for temporal modeling and simple MLP
    layers instead of graph convolutions for spatial modeling, making it lightweight
    and efficient.

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Data feature dictionary containing dataset information

    Configuration Parameters:
        - input_window: Number of input time steps (default: 12)
        - output_window: Number of output time steps (default: 12)
        - output_dim: Output feature dimension (default: 1)
        - conv_channels: Convolution channels (default: 32)
        - residual_channels: Residual connection channels (default: 32)
        - skip_channels: Skip connection channels (default: 64)
        - end_channels: Final convolution channels (default: 128)
        - layers: Number of temporal layers (default: 3)
        - dropout: Dropout rate (default: 0.3)
        - subgraph_size: Size of subgraph for graph constructor (default: 20)
        - node_dim: Node embedding dimension (default: 40)
        - gcn_depth: GCN depth (default: 2, used for graph constructor)
        - dilation_exponential: Dilation exponential factor (default: 1)
        - propalpha: Propagation alpha (default: 0.05)
        - tanhalpha: Tanh alpha for graph constructor (default: 3)
        - layer_norm_affline: Whether to use affine in layer norm (default: True)
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Data features
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.num_batches = self.data_feature.get('num_batches', 1)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # Configuration parameters
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.output_dim = config.get('output_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.gcn_true = config.get('gcn_true', False)  # Set to False for STMLP
        self.buildA_true = config.get('buildA_true', True)
        self.gcn_depth = config.get('gcn_depth', 2)
        self.dropout = config.get('dropout', 0.3)
        self.subgraph_size = config.get('subgraph_size', 20)
        self.node_dim = config.get('node_dim', 40)
        self.dilation_exponential = config.get('dilation_exponential', 1)

        self.conv_channels = config.get('conv_channels', 32)
        self.residual_channels = config.get('residual_channels', 32)
        self.skip_channels = config.get('skip_channels', 64)
        self.end_channels = config.get('end_channels', 128)

        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)
        self.layer_norm_affline = config.get('layer_norm_affline', True)

        # Node indices for layer normalization
        self.idx = torch.arange(self.num_nodes).to(self.device)

        # Handle adjacency matrix
        if self.adj_mx is None:
            self.predefined_A = None
        else:
            self.predefined_A = torch.tensor(self.adj_mx) - torch.eye(self.num_nodes)
            self.predefined_A = self.predefined_A.to(self.device)
        self.static_feat = None

        # Initialize model layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norm = nn.ModuleList()

        # Student MLP layers (key difference from Teacher model)
        self.stu_mlp = nn.ModuleList()

        # Start convolution
        self.start_conv = nn.Conv2d(
            in_channels=self.feature_dim,
            out_channels=self.residual_channels,
            kernel_size=(1, 1)
        )

        # Graph constructor (kept for potential use)
        self.gc = GraphConstructor(
            self.num_nodes, self.subgraph_size, self.node_dim,
            self.device, alpha=self.tanhalpha, static_feat=self.static_feat
        )

        # Calculate receptive field
        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(
                self.output_dim + (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1)
                / (self.dilation_exponential - 1)
            )
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + self.output_dim

        # Build temporal layers
        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1)
                    / (self.dilation_exponential - 1)
                )
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1

            for j in range(1, self.layers + 1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (self.dilation_exponential ** j - 1)
                        / (self.dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                # Dilated inception convolutions for temporal modeling
                self.filter_convs.append(DilatedInception(
                    self.residual_channels,
                    self.conv_channels,
                    dilation_factor=new_dilation
                ))
                self.gate_convs.append(DilatedInception(
                    self.residual_channels,
                    self.conv_channels,
                    dilation_factor=new_dilation
                ))
                self.residual_convs.append(nn.Conv2d(
                    in_channels=self.conv_channels,
                    out_channels=self.residual_channels,
                    kernel_size=(1, 1)
                ))

                # Skip convolutions
                if self.input_window > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(
                        in_channels=self.conv_channels,
                        out_channels=self.skip_channels,
                        kernel_size=(1, self.input_window - rf_size_j + 1)
                    ))
                else:
                    self.skip_convs.append(nn.Conv2d(
                        in_channels=self.conv_channels,
                        out_channels=self.skip_channels,
                        kernel_size=(1, self.receptive_field - rf_size_j + 1)
                    ))

                # Layer normalization
                if self.input_window > self.receptive_field:
                    self.norm.append(LayerNorm(
                        (self.residual_channels, self.num_nodes, self.input_window - rf_size_j + 1),
                        elementwise_affine=self.layer_norm_affline
                    ))
                else:
                    self.norm.append(LayerNorm(
                        (self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),
                        elementwise_affine=self.layer_norm_affline
                    ))

                new_dilation *= self.dilation_exponential

        # Student MLP layers - hardcoded sizes based on temporal dimension shrinkage
        # These sizes depend on input_window=12, output_dim=1, layers=3, kernel_size=7
        # Layer 1: input has 13 time steps after convolution
        # Layer 2: input has 7 time steps after convolution
        # Layer 3: input has 1 time step after convolution
        self._init_student_mlp()

        # End convolutions
        self.end_conv_1 = nn.Conv2d(
            in_channels=self.skip_channels,
            out_channels=self.end_channels,
            kernel_size=(1, 1),
            bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.end_channels,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

        # Skip connections
        if self.input_window > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=self.feature_dim,
                out_channels=self.skip_channels,
                kernel_size=(1, self.input_window),
                bias=True
            )
            self.skipE = nn.Conv2d(
                in_channels=self.residual_channels,
                out_channels=self.skip_channels,
                kernel_size=(1, self.input_window - self.receptive_field + 1),
                bias=True
            )
        else:
            self.skip0 = nn.Conv2d(
                in_channels=self.feature_dim,
                out_channels=self.skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True
            )
            self.skipE = nn.Conv2d(
                in_channels=self.residual_channels,
                out_channels=self.skip_channels,
                kernel_size=(1, 1),
                bias=True
            )

        self._logger.info('LightST receptive_field: ' + str(self.receptive_field))

    def _init_student_mlp(self):
        """Initialize student MLP layers based on temporal dimension sizes.

        The temporal dimension shrinks through dilated convolutions:
        - After layer 1: 13 time steps (input_window + 1)
        - After layer 2: 7 time steps
        - After layer 3: 1 time step

        Note: These are hardcoded based on default input_window=12, layers=3
        For different configurations, these need to be recalculated.
        """
        # Calculate temporal dimensions after each layer
        kernel_size = 7
        new_dilation = 1
        rf_size_i = 0 * self.layers * (kernel_size - 1) + 1 if self.dilation_exponential <= 1 else 1

        time_dims = []
        for j in range(1, self.layers + 1):
            if self.dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (self.dilation_exponential ** j - 1)
                    / (self.dilation_exponential - 1)
                )
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            if self.input_window > self.receptive_field:
                time_dim = self.input_window - rf_size_j + 1
            else:
                time_dim = self.receptive_field - rf_size_j + 1
            time_dims.append(time_dim)
            new_dilation *= self.dilation_exponential

        # Create MLP layers for each temporal dimension
        for time_dim in time_dims:
            self.stu_mlp.append(nn.Sequential(
                nn.Linear(time_dim, time_dim),
                nn.Linear(time_dim, time_dim),
                nn.Linear(time_dim, time_dim)
            ))

    def forward(self, batch, idx=None):
        """Forward pass of the LightST model.

        Args:
            batch: Dictionary containing 'X' with shape (B, T, N, D)
                   where B=batch_size, T=input_window, N=num_nodes, D=feature_dim
            idx: Optional node indices for subgraph sampling

        Returns:
            Predictions with shape (B, T_out, N, D_out)
            where T_out=output_window, D_out=output_dim
        """
        inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)

        assert inputs.size(3) == self.input_window, \
            'input sequence length not equal to preset sequence length'

        # Pad if necessary
        if self.input_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field - self.input_window, 0, 0, 0))

        # Start convolution
        x = self.start_conv(inputs)
        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))

        # Temporal layers with gated convolutions and MLP
        for i in range(self.layers):
            residual = x

            # Gated temporal convolution
            filters = self.filter_convs[i](x)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)

            # Skip connection
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            # Student MLP layer (lightweight spatial modeling)
            x = self.stu_mlp[i](x)

            # Residual connection
            x = x + residual[:, :, :, -x.size(3):]

            # Layer normalization
            x = self.norm[i](x, self.idx)

        # Output layers
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # Output shape after end_conv_2: (B, output_window, N, 1)
        # This already matches LibCity format: (B, T, N, D)
        return x

    def predict(self, batch, idx=None):
        """Generate predictions for a batch.

        Args:
            batch: Input batch dictionary
            idx: Optional node indices

        Returns:
            Predictions with shape (B, T_out, N, D_out)
        """
        return self.forward(batch, idx)

    def calculate_loss(self, batch, idx=None):
        """Calculate the masked MAE loss.

        Args:
            batch: Dictionary containing 'X' and 'y'
            idx: Optional node indices for subgraph sampling

        Returns:
            Scalar loss tensor
        """
        y_true = batch['y']
        y_predicted = self.predict(batch, idx)

        # Inverse transform to original scale
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, 0)
