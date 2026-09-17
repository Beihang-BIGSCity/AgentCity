"""
AGCRN: Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting (NeurIPS 2020)

Paper:  "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
        https://arxiv.org/abs/2007.02842
Orig.:  https://github.com/LeiBAI/AGCRN

Original source files (all consolidated here to keep the port self-contained):
    model/AGCN.py       -> class AVWGCN    : adaptive vertex-weighted graph convolution
    model/AGCRNCell.py  -> class AGCRNCell : GRU-style recurrent cell built from two AVWGCN layers
    model/AGCRN.py      -> class AVWDCRNN  : stack of AGCRNCell (encoder)
                        -> class AGCRN     : encoder + Conv2d multi-step head

Adaptation notes:
    * The original repo is a standalone training script (model/Run.py + model/BasicTrainer.py);
      here the network is wrapped as a LibCity ``AbstractTrafficStateModel``.
    * LibCity batches are dicts: ``X`` is ``(B, input_window, N, feature_dim)`` and
      ``y`` is ``(B, output_window, N, output_dim)``. This is exactly the channels-last
      layout the original model expects, so **no transpose is performed**.
    * The original ``input_dim``/``horizon``/``num_nodes`` are taken from LibCity's
      ``data_feature``/``config`` (``input_window``/``output_window``).
    * No adjacency matrix is needed: the graph is learned from learnable node embeddings
      (``self.node_embeddings``), therefore ``adj_mx`` is intentionally unused.
    * Weight initialisation reproduces ``model/Run.py``:
      ``xavier_uniform_`` for tensors with ``dim > 1`` and ``uniform_`` otherwise.
    * The model is non-autoregressive: the full horizon is produced by a single
      ``Conv2d`` head applied to the last hidden state (matching the original ``forward``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class AVWGCN(nn.Module):
    """Adaptive Vertex-Weighted Graph Convolution (original ``model/AGCN.py``).

    Builds the graph from node embeddings and applies a Chebyshev-style
    polynomial of the (softmaxed, ReLU-ed) self-similarity adjacency.
    """

    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x: (B, N, C), node_embeddings: (N, D) -> supports: (N, N)
        # output: (B, N, C)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 2 (equivalent to the original "default cheb_k = 3" comment path)
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)                          # (cheb_k, N, N)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # (N, cheb_k, dim_in, dim_out)
        bias = torch.matmul(node_embeddings, self.bias_pool)               # (N, dim_out)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)                    # (B, cheb_k, N, dim_in)
        x_g = x_g.permute(0, 2, 1, 3)                                       # (B, N, cheb_k, dim_in)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias       # (B, N, dim_out)
        return x_gconv


class AGCRNCell(nn.Module):
    """GRU-style recurrent cell with adaptive graph convolutions (original ``model/AGCRNCell.py``)."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: (B, N, input_dim), state: (B, N, hidden_dim)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    """Encoder: ``num_layers`` stacked AGCRN cells with an explicit time loop (original ``AVWDCRNN``)."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # x: (B, T, N, D); init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: outputs of the last layer (B, T, N, hidden_dim)
        # output_hidden: last state of each layer (num_layers, B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(AbstractTrafficStateModel):
    """LibCity wrapper around the AGCRN network."""

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # ---------------- data shape / scaler ----------------
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.num_node = self.num_nodes  # alias kept for fidelity with the original naming
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        # original `input_dim` is the number of input channels == feature_dim
        self.input_dim = config.get('input_dim', self.feature_dim)
        self._scaler = self.data_feature.get('scaler')

        # ---------------- horizon / windows ----------------
        self.input_window = config.get('input_window', config.get('lag', 12))
        self.output_window = config.get('output_window', config.get('horizon', 12))
        self.horizon = self.output_window

        # ---------------- architecture hyper-parameters ----------------
        self.rnn_units = config.get('rnn_units', 64)
        self.hidden_dim = self.rnn_units
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        # original config key is `cheb_order`; also accept `cheb_k`
        self.cheb_k = config.get('cheb_order', config.get('cheb_k', 2))

        # ---------------- layers ----------------
        # learnable node embeddings define the adaptive graph (no adj_mx required)
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_nodes, self.input_dim, self.rnn_units,
                                self.cheb_k, self.embed_dim, self.num_layers)

        # non-autoregressive multi-step predictor
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim,
                                  kernel_size=(1, self.hidden_dim), bias=True)

        self._init_parameters()

    def _init_parameters(self):
        # Reproduces model/Run.py: xavier_uniform_ for dim > 1, uniform_ otherwise.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        # source: (B, T_in, N, input_dim)
        source = batch['X']

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)  # (B, T_in, N, hidden)
        output = output[:, -1:, :, :]                                       # (B, 1, N, hidden)

        # CNN based predictor
        output = self.end_conv(output)                                      # (B, T_out*out_dim, N, 1)
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                                 # (B, T_out, N, out_dim)
        return output

    def calculate_loss(self, batch):
        # Loss follows the LibCity convention used by other traffic state models:
        # compare in the original (inverse-transformed) value space with masked MAE.
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)

    def predict(self, batch):
        # Non-autoregressive: the whole horizon is produced in a single forward pass.
        return self.forward(batch)
