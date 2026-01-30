"""
ASeer: Asynchronous Spatio-temporal Graph Neural Network for Traffic Flow Prediction

This module implements the ASeer model adapted for the LibCity framework.
ASeer handles irregular time series with variable-length sequences using
asynchronous message passing with temporal encoding.

Original paper: ASeer - Asynchronous Spatio-temporal Graph Neural Networks

Key Features:
- Dual prediction: time periods AND traffic flow
- Variable-length sequence handling with masking
- Asynchronous spatio-temporal message passing
- Temporal encoding with individual and shared components

Adapted from:
- Original model: /repos/ASeer/model/net.py
- GNN components: /repos/ASeer/model/gnn.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse Graph Attention Layer with edge-based temporal encoding.

    Implements attention mechanism that considers:
    - Source node features
    - Destination node features (for encoding mode)
    - Edge features with temporal encoding

    Args:
        te_dim: Temporal encoding dimension
        in_feat: Input feature dimension
        nhid: Hidden dimension
        dropout: Dropout rate
        layer: Layer index
        is_pred: Whether this is a prediction (decoder) layer
    """

    def __init__(self, te_dim, in_feat, nhid, dropout, layer, is_pred):
        super(SpGraphAttentionLayer, self).__init__()
        self.is_pred = is_pred
        # Attention computation: combines source, destination (if not pred), and edge features
        # Edge features: 3 original (distance, delta_t, reachability) + te_dim
        # Node features: in_feat (2) for source and destination
        self.w_att = nn.Linear(4 + 3 + te_dim - 2 * is_pred, nhid, bias=True)
        self.va = nn.Parameter(torch.zeros(1, nhid))
        nn.init.normal_(self.va.data)

        # Output MLP: node features (2) + edge features (3 + te_dim)
        self.mlp_out = nn.Sequential(
            nn.Linear(2 + 3 + te_dim, nhid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, nhid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, nhid, bias=True),
            nn.ReLU(inplace=True)
        )

    def edge_attention(self, edges):
        """Edge UDF for computing attention scores."""
        if self.is_pred:
            xa = torch.cat([edges.src['h_x'], edges.data['h_edge']], dim=-1)
        else:
            xa = torch.cat([edges.src['h_x'], edges.dst['h_x'], edges.data['h_edge']], dim=-1)
        att_sim = torch.sum(self.va * torch.tanh(self.w_att(xa)), dim=-1)
        return {'att_sim': att_sim}

    def message_func(self, edges):
        """Message UDF for passing node and edge features."""
        return {'h_x': edges.src['h_x'], 'att_sim': edges.data['att_sim'], 'h_edge': edges.data['h_edge']}

    def reduce_func(self, nodes):
        """Reduce UDF for aggregating messages with attention."""
        alpha = F.softmax(nodes.mailbox['att_sim'], dim=1)
        alpha = alpha.unsqueeze(-1)
        nodes_msgs = torch.cat([nodes.mailbox['h_x'], nodes.mailbox['h_edge']], dim=-1)
        h_att = torch.sum(alpha * nodes_msgs, dim=1)
        return {'h_att': h_att}

    def edge_temporal_encoding(self, TE_Params, dt, dst_lane):
        """
        Compute temporal encoding for edges.

        Args:
            TE_Params: Tuple of (TE_w, sharedTE_w, TE_lam)
            dt: Delta time for edges, shape (N_edge, 1)
            dst_lane: Destination lane indices, shape (N_edge,)

        Returns:
            Temporal encoding tensor, shape (N_edge, te_dim)
        """
        N_edge, _ = dt.shape
        TE_w, sharedTE_w, TE_lam = TE_Params

        # Get individual TE parameters for destination nodes
        ret_TE_w = TE_w[dst_lane]  # (N_edge, D)
        ret_TE_lam = TE_lam[dst_lane]

        # Individual temporal encoding
        ind_sin = torch.sin(dt * ret_TE_w)
        ind_cos = torch.cos(dt * ret_TE_w)
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)

        # Shared temporal encoding
        shared_sin = torch.sin(sharedTE_w(dt))
        shared_cos = torch.cos(sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)

        # Combine with learnable lambda
        lam = torch.exp(-torch.square(ret_TE_lam))
        TE = (1 - lam) * te_ind + lam * te_shared

        return TE

    def forward(self, TE_Params, X_msg, g):
        """
        Forward pass through the attention layer.

        Args:
            TE_Params: Temporal encoding parameters
            X_msg: Node features, shape (num_nodes, in_features)
            g: DGL graph with edge features

        Returns:
            Output node features, shape (num_nodes, nhid)
        """
        N, in_features = X_msg.size()
        g.ndata['h_x'] = X_msg

        # Edge temporal encoding
        # Edge features: [distance, delta_t, reachability, dst_lane]
        delta_t = g.edata['feature'][..., 1:2]
        dst_lane = g.edata['feature'][..., 3].long()
        e_te = self.edge_temporal_encoding(TE_Params, delta_t, dst_lane)
        g.edata['h_edge'] = torch.cat([g.edata['feature'][..., :3], e_te], dim=-1)

        # Message passing
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_att = g.ndata.pop('h_att')

        h_conv = self.mlp_out(h_att)
        return h_conv


class AGDN(nn.Module):
    """
    Asynchronous Graph Diffusion Network.

    Multi-layer sparse graph attention network for asynchronous
    spatio-temporal message passing.

    Args:
        te_dim: Temporal encoding dimension
        in_feat: Input feature dimension
        nhid: Hidden dimension
        device: Computation device
        is_pred: Whether this is a prediction (decoder) module
        gathop: Number of attention layers
        dropout: Dropout rate
    """

    def __init__(self, te_dim, in_feat, nhid, device, is_pred=False, gathop=1, dropout=0):
        super(AGDN, self).__init__()
        self.nhid = nhid
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.gat_stacks = nn.ModuleList()

        for i in range(gathop):
            if i > 0:
                in_feat = nhid
            att_layer = SpGraphAttentionLayer(te_dim, in_feat, nhid, dropout=dropout, layer=i, is_pred=is_pred)
            self.gat_stacks.append(att_layer)

    def forward(self, TE_Params, X_msg, adj):
        """
        Forward pass through all attention layers.

        Args:
            TE_Params: Temporal encoding parameters
            X_msg: Input node features
            adj: DGL graph

        Returns:
            Output node features
        """
        for att_layer in self.gat_stacks:
            out = att_layer(TE_Params, X_msg, adj)
        return out


class SemiARDecoder(nn.Module):
    """
    Semi-Autoregressive Decoder with GRU updates.

    Performs semi-autoregressive prediction of both time periods and traffic flow.
    Uses GRU cells for hidden state updates between prediction iterations.

    Args:
        n_output: Number of output steps per iteration
        hid_dim: Hidden dimension
        in_feat: Input feature dimension
        device: Computation device
    """

    def __init__(self, n_output, hid_dim, in_feat, device):
        super(SemiARDecoder, self).__init__()
        self.n_output = n_output
        self.device = device
        self.update_GRU = nn.GRUCell(in_feat, hid_dim, bias=True)

        # Period prediction MLP
        self.MLP_period = nn.Sequential(
            nn.Linear(in_feat + 2 * hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, n_output, bias=True),
            nn.Sigmoid()
        )

        # Flow prediction MLP
        self.MLP_flow = nn.Sequential(
            nn.Linear(in_feat + 2 * hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, n_output, bias=True),
        )

    def decoder_individual_te(self, TE_Params, dt_onestep):
        """Compute temporal encoding for decoder."""
        dt = dt_onestep.unsqueeze(dim=-2)  # (N, 1, 1)
        N, L, _ = dt.shape
        TE_w, sharedTE_w, TE_lam = TE_Params

        ind_sin = torch.sin(dt * TE_w.repeat(1, L, 1))
        ind_cos = torch.cos(dt * TE_w.repeat(1, L, 1))
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)

        shared_sin = torch.sin(sharedTE_w(dt))
        shared_cos = torch.cos(sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)

        lam = torch.exp(-torch.square(TE_lam))
        TE = (1 - lam) * te_ind + lam * te_shared
        return TE.squeeze(dim=-2)

    def forward(self, TE_Params, h_t_int, dt_init, X, Y, is_test):
        """
        Forward pass through the decoder.

        Args:
            TE_Params: Temporal encoding parameters
            h_t_int: Integrated hidden state, shape (N, hid_dim)
            dt_init: Initial delta time, shape (N, 1)
            X: Input sequence
            Y: Target sequence, shape (N, Ly, F)
            is_test: Whether in test mode

        Returns:
            period_outputs: Predicted periods
            flow_outputs: Predicted flows (training)
            flow_test_outputs: Predicted flows (testing)
        """
        N, Ly, _ = Y.shape
        Y_period = Y[..., 0]
        Y_dt = Y[..., 2]
        n_iter = math.ceil(Ly / self.n_output)
        period_outputs = []
        flow_outputs = []
        flow_test_outputs = []

        dt_pred = dt_init.clone()
        elasped_time = dt_init.clone()
        ht_period = h_t_int.clone()

        # Period prediction loop
        for k in range(n_iter):
            ht_period = self.update_GRU(
                torch.cat([elasped_time, self.decoder_individual_te(TE_Params, elasped_time)], dim=-1),
                ht_period
            )
            input_p = torch.cat([dt_pred, self.decoder_individual_te(TE_Params, dt_pred), h_t_int, ht_period], dim=-1)
            periods = self.MLP_period(input_p)
            period_outputs.append(periods)

            dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True)
            elasped_time = torch.sum(periods, dim=-1, keepdim=True)

        # Flow prediction loop (training)
        elasped_time_truth = dt_init.clone()
        ht_flow = h_t_int.clone()
        for k in range(n_iter):
            dt_truth = Y_dt[:, k * self.n_output:k * self.n_output + 1]
            ht_flow = self.update_GRU(
                torch.cat([elasped_time_truth, self.decoder_individual_te(TE_Params, elasped_time_truth)], dim=-1),
                ht_flow
            )
            input_f = torch.cat([dt_truth, self.decoder_individual_te(TE_Params, dt_truth), h_t_int, ht_flow], dim=-1)
            flows = self.MLP_flow(input_f)
            flow_outputs.append(flows)

            elasped_time_truth = torch.sum(Y_period[:, k * self.n_output:(k + 1) * self.n_output], dim=-1, keepdim=True)

        # Test mode: autoregressive prediction
        if is_test:
            period_outputs = []
            dt_pred = dt_init.clone()
            elasped_time = dt_init.clone()
            ht = h_t_int.clone()
            k = 1
            while True:
                ht = self.update_GRU(
                    torch.cat([elasped_time, self.decoder_individual_te(TE_Params, elasped_time)], dim=-1),
                    ht
                )
                input_combined = torch.cat([dt_pred, self.decoder_individual_te(TE_Params, dt_pred), h_t_int, ht], dim=-1)
                periods = self.MLP_period(input_combined)
                periods = torch.clip(periods, 0.05, 1)
                period_outputs.append(periods)

                flows_test = self.MLP_flow(input_combined)
                flow_test_outputs.append(flows_test)

                dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True)
                elasped_time = torch.sum(periods, dim=-1, keepdim=True)

                if (k >= n_iter) and (300 * dt_pred.min().item() > 7200):
                    break
                k += 1

            flow_test_outputs = torch.cat(flow_test_outputs, dim=-1)

        period_outputs = torch.cat(period_outputs, dim=-1)
        flow_outputs = torch.cat(flow_outputs, dim=-1)

        return period_outputs, flow_outputs, flow_test_outputs


class SemiARDecoderMLP(nn.Module):
    """
    Semi-Autoregressive Decoder without GRU (MLP only).

    Simpler version of the decoder using only MLP layers.

    Args:
        n_output: Number of output steps per iteration
        hid_dim: Hidden dimension
        in_feat: Input feature dimension
        device: Computation device
    """

    def __init__(self, n_output, hid_dim, in_feat, device):
        super(SemiARDecoderMLP, self).__init__()
        self.n_output = n_output
        self.device = device

        self.MLP_period = nn.Sequential(
            nn.Linear(in_feat + hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, n_output, bias=True),
            nn.Sigmoid()
        )

        self.MLP_flow = nn.Sequential(
            nn.Linear(in_feat + hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, n_output, bias=True),
        )

    def decoder_individual_te(self, TE_Params, dt_onestep):
        """Compute temporal encoding for decoder."""
        dt = dt_onestep.unsqueeze(dim=-2)
        N, L, _ = dt.shape
        TE_w, sharedTE_w, TE_lam = TE_Params

        ind_sin = torch.sin(dt * TE_w.repeat(1, L, 1))
        ind_cos = torch.cos(dt * TE_w.repeat(1, L, 1))
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)

        shared_sin = torch.sin(sharedTE_w(dt))
        shared_cos = torch.cos(sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)

        lam = torch.exp(-torch.square(TE_lam))
        TE = (1 - lam) * te_ind + lam * te_shared
        return TE.squeeze(dim=-2)

    def forward(self, TE_Params, h_t_int, dt_init, X, Y, is_test):
        """Forward pass through the MLP decoder."""
        N, Ly, _ = Y.shape
        Y_period = Y[..., 0]
        Y_dt = Y[..., 2]
        n_iter = math.ceil(Ly / self.n_output)
        period_outputs = []
        flow_outputs = []
        flow_test_outputs = []

        dt_pred = dt_init.clone()
        for k in range(n_iter):
            input_p = torch.cat([dt_pred, self.decoder_individual_te(TE_Params, dt_pred), h_t_int], dim=-1)
            periods = self.MLP_period(input_p)
            period_outputs.append(periods)

            dt_truth = Y_dt[:, k * self.n_output:k * self.n_output + 1]
            input_f = torch.cat([dt_truth, self.decoder_individual_te(TE_Params, dt_truth), h_t_int], dim=-1)
            flows = self.MLP_flow(input_f)
            flow_outputs.append(flows)

            dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True)

        if is_test:
            period_outputs = []
            dt_pred = dt_init.clone()
            k = 1
            while True:
                input_combined = torch.cat([dt_pred, self.decoder_individual_te(TE_Params, dt_pred), h_t_int], dim=-1)
                periods = self.MLP_period(input_combined)
                periods = torch.clip(periods, 0.05, 1)
                period_outputs.append(periods)
                flows_test = self.MLP_flow(input_combined)
                flow_test_outputs.append(flows_test)

                dt_pred = dt_pred + torch.sum(periods, dim=-1, keepdim=True)

                if (k >= n_iter) and (300 * dt_pred.min().item() > 7200):
                    break
                k += 1

            flow_test_outputs = torch.cat(flow_test_outputs, dim=-1)

        period_outputs = torch.cat(period_outputs, dim=-1)
        flow_outputs = torch.cat(flow_outputs, dim=-1)

        return period_outputs, flow_outputs, flow_test_outputs


class ASeer(AbstractTrafficStateModel):
    """
    ASeer: Asynchronous Spatio-temporal Graph Neural Network.

    This model handles irregular time series with variable-length sequences
    using asynchronous message passing with temporal encoding.

    Key features:
    - Dual prediction architecture (time periods + traffic flow)
    - Temporal encoding with learnable individual and shared components
    - Time-aware Temporal Convolutional Network (TTCN) for sequence encoding
    - Asynchronous Graph Diffusion Network (AGDN) for spatial encoding

    Config parameters:
        hidden_dim (int): Hidden dimension, default 64
        time_emb_dim (int): Temporal embedding dimension, default 16
        n_output (int): Number of decoding steps per iteration, default 12
        beta (float): Weight for flow loss, default 1.0
        dropout (float): Dropout rate, default 0.0
        te_mode (str): Temporal encoding mode ('combine', 'share', 'ind'), default 'combine'
        decoder_type (str): Decoder type ('SEU' or 'MLP'), default 'SEU'

    Data features:
        num_nodes (int): Number of nodes in the graph
        adj_mx: Adjacency matrix or graph structure
        scaler: Data scaler for inverse transformation
    """

    def __init__(self, config, data_feature):
        # Extract data features first
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 3)  # [period, unit_flow, delta_t]

        super().__init__(config, data_feature)

        # Model hyperparameters from config
        self.hid_dim = config.get('hidden_dim', 64)
        self.te_dim = config.get('time_emb_dim', 16)
        self.n_output = config.get('n_output', 12)
        self.beta = config.get('beta', 1.0)  # Flow loss weight
        self.dropout_rate = config.get('dropout', 0.0)
        self.te_mode = config.get('te_mode', 'combine')  # 'combine', 'share', 'ind'
        self.decoder_type = config.get('decoder_type', 'SEU')  # 'SEU' or 'MLP'

        # LibCity specific
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.output_dim = data_feature.get('output_dim', 1)

        self._logger = getLogger()
        self._scaler = data_feature.get('scaler')

        # Check DGL availability
        if not HAS_DGL:
            self._logger.warning("DGL not available. ASeer requires DGL for graph operations.")

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Sequence temporal encoding (learnable per-node)
        self.sharedTE_w = nn.Linear(1, self.te_dim // 2, bias=False)
        self.TE_w = nn.Parameter(torch.zeros(self.num_nodes, 1, self.te_dim // 2))
        nn.init.normal_(self.TE_w.data)

        # Edge temporal encoding
        self.sharedTE_edge_w = nn.Linear(1, self.te_dim // 2, bias=False)
        self.TE_edge_w = nn.Parameter(torch.zeros(self.num_nodes, self.te_dim // 2))
        nn.init.normal_(self.TE_edge_w.data)

        # Temporal encoding combination parameters
        if self.te_mode == "combine":
            self.TE_lam = nn.Parameter(torch.zeros(self.num_nodes, 1, 1) + 1e-6)
            self.TE_edge_lam = nn.Parameter(torch.zeros(self.num_nodes, 1) + 1e-6)
        elif self.te_mode == "share":
            self.register_buffer('TE_lam', torch.zeros(self.num_nodes, 1, 1))
            self.register_buffer('TE_edge_lam', torch.zeros(self.num_nodes, 1))
        elif self.te_mode == "ind":
            self.register_buffer('TE_lam', torch.ones(self.num_nodes, 1, 1) * 1e9)
            self.register_buffer('TE_edge_lam', torch.ones(self.num_nodes, 1) * 1e9)
        else:
            self._logger.warning(f"Unknown te_mode: {self.te_mode}, using 'combine'")
            self.TE_lam = nn.Parameter(torch.zeros(self.num_nodes, 1, 1) + 1e-6)
            self.TE_edge_lam = nn.Parameter(torch.zeros(self.num_nodes, 1) + 1e-6)

        # TTCN (Time-aware Temporal Convolutional Network) components
        input_dim = 3 + self.te_dim + self.hid_dim  # [period, unit_flow, delta_t] + TE + hidden
        self.Filter_Generators = nn.Sequential(
            nn.Linear(input_dim, self.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, self.hid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.hid_dim, input_dim * self.hid_dim, bias=True)
        )
        self.T_bias = nn.Parameter(torch.zeros(1, self.hid_dim))
        nn.init.normal_(self.T_bias.data)

        # AGDN modules for encoder and rest (prediction)
        self.AGDN_EN = AGDN(self.te_dim, in_feat=2, nhid=self.hid_dim, device=self.device, is_pred=False)
        self.AGDN_REST = AGDN(self.te_dim, in_feat=2, nhid=self.hid_dim, device=self.device, is_pred=True)

        # Decoder
        decoder_in_feat = input_dim - 2 - self.hid_dim  # Remove delta_t duplicate and hidden
        if self.decoder_type == "SEU":
            self._logger.info("ASeer Decoder: SEU (Semi-autoregressive with GRU)")
            self.decoder = SemiARDecoder(self.n_output, self.hid_dim, decoder_in_feat, self.device)
        elif self.decoder_type == "MLP":
            self._logger.info("ASeer Decoder: MLP")
            self.decoder = SemiARDecoderMLP(self.n_output, self.hid_dim, decoder_in_feat, self.device)
        else:
            self._logger.warning(f"Unknown decoder_type: {self.decoder_type}, using SEU")
            self.decoder = SemiARDecoder(self.n_output, self.hid_dim, decoder_in_feat, self.device)

        self._logger.info(f"ASeer initialized with {self.num_nodes} nodes, "
                         f"hidden_dim={self.hid_dim}, te_dim={self.te_dim}, "
                         f"n_output={self.n_output}, beta={self.beta}")

    def individual_te(self, dt):
        """
        Compute individual temporal encoding for sequences.

        Args:
            dt: Delta time tensor, shape (N, L, 1)

        Returns:
            Temporal encoding tensor, shape (N, L, te_dim)
        """
        N, L, _ = dt.shape
        ind_sin = torch.sin(dt * self.TE_w.repeat(1, L, 1))
        ind_cos = torch.cos(dt * self.TE_w.repeat(1, L, 1))
        te_ind = torch.cat([ind_sin, ind_cos], dim=-1)

        shared_sin = torch.sin(self.sharedTE_w(dt))
        shared_cos = torch.cos(self.sharedTE_w(dt))
        te_shared = torch.cat([shared_sin, shared_cos], dim=-1)

        lam = torch.exp(-torch.square(self.TE_lam))
        TE = (1 - lam) * te_ind + lam * te_shared
        return TE

    def ttcn(self, X_int, mask_X):
        """
        Time-aware Temporal Convolutional Network.

        Aggregates sequence information with attention-like filtering.

        Args:
            X_int: Integrated input features, shape (N, Lx, F_in)
            mask_X: Sequence mask, shape (N, Lx, 1)

        Returns:
            Hidden representation, shape (N, hid_dim)
        """
        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int)  # (N, Lx, F_in*hid_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)

        # Normalize along sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*hid_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.hid_dim, -1)  # (N, Lx, hid_dim, F_in)

        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.hid_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1)  # (N, hid_dim)

        h_t = torch.relu(ttcn_out + self.T_bias)
        return h_t

    def _prepare_graphs(self, batch):
        """
        Prepare DGL graphs from batch data.

        This method handles conversion from LibCity batch format to DGL graphs.
        Expected batch keys:
            - 'adj_X': Encoder adjacency/graph
            - 'adj_Y': Decoder adjacency/graph

        If graphs are not provided, creates default based on adjacency matrix.

        Args:
            batch: Input batch dictionary

        Returns:
            Tuple of (adj_X, adj_Y) DGL graphs
        """
        # Try to get pre-computed graphs from batch
        # Note: Using try-except because LibCity's Batch class doesn't implement __contains__
        try:
            adj_X = batch['adj_X']
            adj_Y = batch['adj_Y']
            return adj_X, adj_Y
        except KeyError:
            pass  # Fallback to constructing from adjacency matrix

        # Fallback: try to construct from adjacency matrix in data_feature
        if hasattr(self, 'data_feature') and 'adj_mx' in self.data_feature:
            adj_mx = self.data_feature['adj_mx']
            # Convert adjacency matrix to DGL graph if needed
            if HAS_DGL and isinstance(adj_mx, np.ndarray):
                src, dst = np.where(adj_mx > 0)
                g = dgl.graph((src, dst), num_nodes=self.num_nodes)
                # Add default edge features
                num_edges = g.number_of_edges()
                g.edata['feature'] = torch.zeros(num_edges, 4, device=self.device)
                return g, g

        self._logger.warning("No graph structure provided in batch. Creating empty graphs.")
        if HAS_DGL:
            g = dgl.graph(([], []), num_nodes=self.num_nodes)
            g.edata['feature'] = torch.zeros(0, 4, device=self.device)
            return g, g
        else:
            raise RuntimeError("DGL is required for ASeer but not installed")

    def _prepare_mask(self, batch, X):
        """
        Prepare sequence mask from batch.

        Args:
            batch: Input batch dictionary
            X: Input tensor, shape (N, Lx, Fx)

        Returns:
            Mask tensor, shape (N, Lx, 1)
        """
        # Note: Using try-except because LibCity's Batch class doesn't implement __contains__
        try:
            mask_X = batch['mask_X']
            if mask_X.dim() == 3:
                mask_X = mask_X.unsqueeze(dim=-1)
            return mask_X
        except KeyError:
            pass  # Fallback to default mask

        # Default: all valid (mask of ones)
        N, Lx, _ = X.shape
        return torch.ones(N, Lx, 1, device=X.device)

    def forward(self, batch, is_test=False):
        """
        Forward pass through ASeer.

        Args:
            batch: Dictionary containing:
                - 'X': Input sequence, shape (batch, time_in, num_nodes, features)
                      or (1, N, Lx, Fx) in original format
                - 'y': Target sequence for training
                - 'mask_X': Optional sequence mask
                - 'adj_X', 'adj_Y': Optional DGL graphs
            is_test: Whether in test mode (enables autoregressive prediction)

        Returns:
            period_outputs: Predicted time periods
            flow_outputs: Predicted traffic flow
            flow_test_outputs: Test-time flow predictions (empty list if not test mode)
        """
        # Handle LibCity batch format: (batch, time, nodes, features)
        # Original format: (1, N, L, F) - batch dim contains nodes
        X = batch['X']
        Y = batch['y']

        # Determine data format and reshape if needed
        if X.dim() == 4:
            batch_size, time_in, num_nodes, features = X.shape
            if batch_size == 1 and num_nodes == self.num_nodes:
                # Original format: (1, N, Lx, Fx) - nodes are in time dimension
                X = X.squeeze(dim=0)  # (N, Lx, Fx)
            else:
                # LibCity format: (B, T, N, F) -> need to process per node
                # Transpose to (B, N, T, F) then reshape
                X = X.permute(0, 2, 1, 3)  # (B, N, T, F)
                X = X.reshape(-1, time_in, features)  # (B*N, T, F)

        if Y is not None and Y.dim() == 4:
            Y = Y.squeeze(dim=0) if Y.shape[0] == 1 else Y.permute(0, 2, 1, 3).reshape(-1, Y.shape[1], Y.shape[3])

        # Prepare mask
        mask_X = self._prepare_mask(batch, X)
        if mask_X.dim() == 2:
            mask_X = mask_X.unsqueeze(dim=-1)

        N, Lx, Fx = X.shape

        # Get graphs
        adj_X, adj_Y = self._prepare_graphs(batch)

        # Temporal encoding parameters
        TE_Params = (self.TE_w, self.sharedTE_w, self.TE_lam)
        TE_edge_Params = (self.TE_edge_w, self.sharedTE_edge_w, self.TE_edge_lam)

        # === Encoder ===
        # Message passing on input graph
        X_msg = X[..., :2].reshape(N * Lx, -1)  # (N*Lx, 2) - [period, unit_flow]
        hx_msg = self.AGDN_EN(TE_edge_Params, X_msg, adj_X)
        hx_msg = hx_msg.reshape(N, Lx, -1)  # (N, Lx, hid_dim)

        # TTCN - temporal aggregation
        te = self.individual_te(X[..., -1:])  # Temporal encoding from delta_t
        X_int = torch.cat([X, te, hx_msg], dim=-1)  # (N, Lx, F_in)
        h_t = self.ttcn(X_int, mask_X)  # (N, hid_dim)

        # === Decoder ===
        # Message passing for prediction
        Y_placeholder = torch.zeros(N, 1, 2, device=X.device)
        Y_msg = torch.cat([X_msg, Y_placeholder.reshape(N * 1, -1)], axis=0)  # (N*Lx+N, 2)
        hy_msg = self.AGDN_REST(TE_edge_Params, Y_msg, adj_Y)
        hy_msg = hy_msg[len(X_msg):].reshape(N, -1)  # (N, hid_dim)

        # Semi-autoregressive prediction
        h_t_int = h_t + hy_msg
        dt_start = Y[:, :1, 2] if Y is not None else torch.zeros(N, 1, device=X.device)

        period_outputs, flow_outputs, flow_test_outputs = self.decoder(
            TE_Params, h_t_int, dt_start, X, Y, is_test
        )

        return period_outputs, flow_outputs, flow_test_outputs

    def predict(self, batch):
        """
        Generate predictions for a batch.

        This method is called during evaluation/inference.

        Args:
            batch: Input batch dictionary

        Returns:
            flow_predictions: Traffic flow predictions in LibCity format
        """
        period_outputs, flow_outputs, flow_test_outputs = self.forward(batch, is_test=True)

        # For LibCity compatibility, return flow predictions
        # Reshape to (batch, time, nodes, features) if needed
        if len(flow_test_outputs) > 0 and isinstance(flow_test_outputs, torch.Tensor):
            predictions = flow_test_outputs
        else:
            predictions = flow_outputs

        # Reshape to LibCity format: (B, T, N, F)
        if predictions.dim() == 2:
            # (N, T) -> (1, T, N, 1)
            predictions = predictions.unsqueeze(0).unsqueeze(-1).permute(0, 2, 1, 3)

        return predictions

    def calculate_loss(self, batch):
        """
        Calculate the dual-task loss (period + flow).

        The total loss combines:
        1. Period prediction loss (BCE-like for normalized periods)
        2. Flow prediction loss (MAE) weighted by beta

        Args:
            batch: Input batch dictionary with 'X' and 'y'

        Returns:
            Combined loss tensor
        """
        Y = batch['y']

        # Forward pass
        period_outputs, flow_outputs, _ = self.forward(batch, is_test=False)

        # Reshape Y if needed
        if Y.dim() == 4:
            Y = Y.squeeze(dim=0) if Y.shape[0] == 1 else Y.permute(0, 2, 1, 3).reshape(-1, Y.shape[1], Y.shape[3])

        N, Ly, _ = Y.shape
        Y_period = Y[..., 0]  # Ground truth periods
        Y_flow = Y[..., 1]    # Ground truth flow (unit_flow)

        # Truncate predictions to match target length
        period_outputs = period_outputs[..., :Ly]
        flow_outputs = flow_outputs[..., :Ly]

        # Period loss (smooth L1 / Huber loss)
        period_loss = F.smooth_l1_loss(period_outputs, Y_period)

        # Flow loss (MAE)
        flow_loss = loss.masked_mae_torch(flow_outputs, Y_flow, null_val=0)

        # Combined loss
        total_loss = period_loss + self.beta * flow_loss

        return total_loss

    def get_period_predictions(self, batch):
        """
        Get period predictions separately.

        Useful for dual-task evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            period_outputs: Predicted time periods
        """
        period_outputs, _, _ = self.forward(batch, is_test=True)
        return period_outputs
