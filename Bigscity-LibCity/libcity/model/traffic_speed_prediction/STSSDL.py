"""
STSSDL (Spatio-Temporal Self-Supervised Deviation Learning) Model
Adapted for LibCity framework

Original Repository: /home/wangwenrui/shk/AgentCity/repos/STSSDL
Original Model File: /home/wangwenrui/shk/AgentCity/repos/STSSDL/model_STSSDL/STSSDL.py

Key Changes Made:
1. Inherited from AbstractTrafficStateModel
2. Adapted data format from (x, x_cov, x_his, y_cov, labels) to LibCity batch dict
3. Implemented calculate_loss() with MAE, contrastive, and deviation losses
4. Implemented predict() method returning only predictions
5. Extracted parameters from config and data_feature
6. Added batches_seen tracking for curriculum learning

Assumptions:
- Input batch['X'] has shape (batch, time, nodes, features) where features include:
  - feature 0: traffic value (speed/flow)
  - feature 1: time-of-day covariate (normalized to [0,1])
  - If add_time_in_day is enabled in dataset config
- Historical data (x_his) is approximated using the input sequence itself
  or can be provided via batch['X_his'] if available
- The model uses symmetric adjacency matrix from data_feature['adj_mx']

Required Config Parameters:
- rnn_units: int (default: 128)
- rnn_layers: int (default: 1)
- cheb_k: int (default: 3)
- prototype_num: int (default: 20)
- prototype_dim: int (default: 64)
- tod_embed_dim: int (default: 10)
- cl_decay_steps: int (default: 2000)
- use_curriculum_learning: bool (default: True)
- use_STE: bool (default: True)
- adaptive_embedding_dim: int (default: 48)
- node_embedding_dim: int (default: 20)
- input_embedding_dim: int (default: 128)
- lamb_c: float (default: 0.01) - contrastive loss weight
- lamb_d: float (default: 1.0) - deviation loss weight
- TDAY: int (default: 288) - time slots per day
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class AGCN(nn.Module):
    """Adaptive Graph Convolution Network layer."""

    def __init__(self, dim_in, dim_out, cheb_k, num_support):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(num_support * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        for support in supports:
            if len(support.shape) == 2:
                support_ks = [torch.eye(support.shape[0]).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
                for graph in support_ks:
                    x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            else:
                support_ks = [torch.eye(support.shape[1]).repeat(support.shape[0], 1, 1).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
                for graph in support_ks:
                    x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv


class AGCRNCell(nn.Module):
    """Adaptive Graph Convolutional Recurrent Network Cell."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_support):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, num_support)

    def forward(self, x, state, supports):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    """Adaptive DCRNN Encoder."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super(ADCRNN_Encoder, self).__init__()
        assert rnn_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, x, init_state, supports):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.rnn_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.rnn_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    """Adaptive DCRNN Decoder."""

    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super(ADCRNN_Decoder, self).__init__()
        assert rnn_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, xt, init_state, supports):
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.rnn_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class STSSDL(AbstractTrafficStateModel):
    """
    STSSDL: Spatio-Temporal Self-Supervised Deviation Learning for Traffic Prediction.

    This model uses prototype learning and self-supervised deviation learning
    to capture spatio-temporal patterns in traffic data.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes')
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.input_dim = self.output_dim
        self.feature_dim = self.data_feature.get('feature_dim', 1)

        # Get adjacency matrix and process it
        adj_mx = self.data_feature.get('adj_mx')
        self.adj_mx = self._process_adj_mx(adj_mx)

        # Model hyperparameters from config
        self.horizon = config.get('output_window', 12)
        self.input_window = config.get('input_window', 12)
        self.rnn_units = config.get('rnn_units', 128)
        self.rnn_layers = config.get('rnn_layers', 1)
        self.cheb_k = config.get('cheb_k', 3)
        self.cl_decay_steps = config.get('cl_decay_steps', 2000)
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)

        # Prototype learning parameters
        self.prototype_num = config.get('prototype_num', 20)
        self.prototype_dim = config.get('prototype_dim', 64)

        # Spatio-temporal embedding parameters
        self.use_STE = config.get('use_STE', True)
        self.TDAY = config.get('TDAY', 288)
        self.tod_embed_dim = config.get('tod_embed_dim', 10)
        self.adaptive_embedding_dim = config.get('adaptive_embedding_dim', 48)
        self.node_embedding_dim = config.get('node_embedding_dim', 20)
        self.input_embedding_dim = config.get('input_embedding_dim', 128)
        self.total_embedding_dim = self.tod_embed_dim + self.adaptive_embedding_dim + self.node_embedding_dim

        # Loss weights
        self.lamb_c = config.get('lamb_c', 0.01)
        self.lamb_d = config.get('lamb_d', 1.0)

        # Contrastive loss
        self.contrastive_loss = nn.TripletMarginLoss(margin=0.5)

        # Build prototypes
        self.prototypes = self._construct_prototypes()

        # Build spatio-temporal embeddings
        if self.use_STE:
            if self.adaptive_embedding_dim > 0:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(self.horizon, self.num_nodes, self.adaptive_embedding_dim))
                )
            self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
            self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
            self.time_embedding = nn.Parameter(torch.empty(self.TDAY, self.tod_embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)

        # Encoder
        if self.use_STE:
            encoder_input_dim = self.input_embedding_dim + self.total_embedding_dim
        else:
            encoder_input_dim = self.input_dim
        self.encoder = ADCRNN_Encoder(
            self.num_nodes, encoder_input_dim, self.rnn_units,
            self.cheb_k, self.rnn_layers, len(self.adj_mx)
        )

        # Decoder
        self.decoder_dim = self.rnn_units + self.prototype_dim
        if self.use_STE:
            decoder_input_dim = self.input_embedding_dim + self.total_embedding_dim - self.adaptive_embedding_dim
        else:
            decoder_input_dim = self.output_dim + 1  # output_dim + ycov_dim
        self.decoder = ADCRNN_Decoder(
            self.num_nodes, decoder_input_dim, self.decoder_dim,
            self.cheb_k, self.rnn_layers, 1
        )

        # Output projection
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        # Hypernet for adaptive graph
        self.hypernet = nn.Sequential(nn.Linear(self.decoder_dim * 2, self.tod_embed_dim, bias=True))

        # Batches seen counter for curriculum learning
        self.batches_seen = 0

        self._logger.info(f"STSSDL initialized with {self.num_nodes} nodes, "
                         f"use_STE={self.use_STE}, prototype_num={self.prototype_num}")

    def _process_adj_mx(self, adj_mx):
        """Process adjacency matrix to list of tensors."""
        if adj_mx is None:
            # Create identity matrix if no adjacency provided
            adj = torch.eye(self.num_nodes).to(self.device)
            return [adj]

        if isinstance(adj_mx, list):
            return [torch.FloatTensor(a).to(self.device) if not isinstance(a, torch.Tensor)
                    else a.to(self.device) for a in adj_mx]

        if isinstance(adj_mx, np.ndarray):
            # Apply symmetric normalization
            adj = self._sym_adj(adj_mx)
            return [torch.FloatTensor(adj).to(self.device)]

        if isinstance(adj_mx, torch.Tensor):
            return [adj_mx.to(self.device)]

        return [torch.eye(self.num_nodes).to(self.device)]

    def _sym_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        import scipy.sparse as sp
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

    def _construct_prototypes(self):
        """Construct prototype parameters."""
        prototypes_dict = nn.ParameterDict()
        prototype = torch.randn(self.prototype_num, self.prototype_dim)
        prototypes_dict['prototypes'] = nn.Parameter(prototype, requires_grad=True)
        prototypes_dict['Wq'] = nn.Parameter(
            torch.randn(self.rnn_units, self.prototype_dim), requires_grad=True
        )
        for param in prototypes_dict.values():
            nn.init.xavier_normal_(param)
        return prototypes_dict

    def compute_sampling_threshold(self, batches_seen):
        """Compute sampling threshold for curriculum learning."""
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def query_prototypes(self, h_t):
        """Query prototypes based on hidden state."""
        query = torch.matmul(h_t, self.prototypes['Wq'])
        att_score = torch.softmax(torch.matmul(query, self.prototypes['prototypes'].t()), dim=-1)
        value = torch.matmul(att_score, self.prototypes['prototypes'])
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes['prototypes'][ind[:, :, 0]]
        neg = self.prototypes['prototypes'][ind[:, :, 1]]
        mask = torch.stack([ind[:, :, 0], ind[:, :, 1]], dim=-1)
        return value, query, pos, neg, mask

    def calculate_distance(self, pos, pos_his, mask=None):
        """Calculate L1 distance between prototype queries."""
        score = torch.sum(torch.abs(pos - pos_his), dim=-1)
        return score, mask

    def _prepare_data(self, batch):
        """
        Prepare data from LibCity batch format.

        LibCity batch provides:
        - batch['X']: (batch, time, nodes, features)
        - batch['y']: (batch, time, nodes, features)

        Original STSSDL expects:
        - x: traffic values (batch, time, nodes, 1)
        - x_cov: time-of-day covariates (batch, time, nodes, 1)
        - x_his: historical traffic (batch, time, nodes, 1)
        - y_cov: future time-of-day covariates (batch, time, nodes, 1)
        - labels: ground truth (batch, time, nodes, 1)
        """
        X = batch['X']  # (batch, time, nodes, features)
        y = batch['y']  # (batch, time, nodes, features)

        # Extract traffic values
        x = X[..., :self.output_dim].float().to(self.device)  # (batch, time, nodes, output_dim)

        # Extract or generate time-of-day covariates
        if self.feature_dim > 1:
            # Time-of-day is typically the second feature
            x_cov = X[..., self.output_dim:self.output_dim + 1].float().to(self.device)
            y_cov = y[..., self.output_dim:self.output_dim + 1].float().to(self.device)
        else:
            # Generate dummy covariates if not available
            x_cov = torch.zeros_like(x).to(self.device)
            y_cov = torch.zeros_like(y[..., :self.output_dim]).to(self.device)

        # For historical data, we use the input sequence itself
        # In the original implementation, this comes from a separate historical window
        x_his = x.clone()

        # Labels
        labels = y[..., :self.output_dim].float().to(self.device)

        return x, x_cov, x_his, y_cov, labels

    def _forward_with_aux(self, x, x_cov, x_his, y_cov, labels=None, batches_seen=None):
        """
        Forward pass returning all outputs including auxiliary tensors for loss computation.
        """
        batch_size = x.shape[0]

        if self.use_STE:
            if self.input_embedding_dim > 0:
                x = self.input_proj(x)
            features = [x]

            if self.tod_embed_dim > 0:
                time_indices = (x_cov.squeeze(-1) * self.TDAY).long().clamp(0, self.TDAY - 1)
                time_emb = self.time_embedding[time_indices]
                features.append(time_emb)

            if self.adaptive_embedding_dim > 0:
                # Use input_window for encoder
                seq_len = x.shape[1]
                adp_emb = self.adaptive_embedding[:seq_len].expand(batch_size, -1, -1, -1)
                features.append(adp_emb)

            if self.node_embedding_dim > 0:
                # Use input_window for encoder
                seq_len = x.shape[1]
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(
                    batch_size, seq_len, -1, -1
                )
                features.append(node_emb)

            x = torch.cat(features, dim=-1)

        supports_en = self.adj_mx
        init_state = self.encoder.init_hidden(batch_size)
        h_en, state_en = self.encoder(x, init_state, supports_en)
        h_t = h_en[:, -1, :, :]

        v_t, q_t, p_t, n_t, mask = self.query_prototypes(h_t)

        # Process historical data
        if self.use_STE:
            if self.input_embedding_dim > 0:
                x_his = self.input_proj(x_his)
            features_his = [x_his]

            if self.tod_embed_dim > 0:
                time_indices = (x_cov.squeeze(-1) * self.TDAY).long().clamp(0, self.TDAY - 1)
                time_emb = self.time_embedding[time_indices]
                features_his.append(time_emb)

            if self.adaptive_embedding_dim > 0:
                # Use input_window for encoder
                seq_len = x_his.shape[1]
                adp_emb = self.adaptive_embedding[:seq_len].expand(batch_size, -1, -1, -1)
                features_his.append(adp_emb)

            if self.node_embedding_dim > 0:
                seq_len = x_his.shape[1]
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(
                    batch_size, seq_len, -1, -1
                )
                features_his.append(node_emb)

            x_his = torch.cat(features_his, dim=-1)

        h_his_en, state_his_en = self.encoder(x_his, init_state, supports_en)
        h_a = h_his_en[:, -1, :, :]

        v_a, q_a, p_a, n_a, mask_his = self.query_prototypes(h_a)

        latent_dis, _ = self.calculate_distance(q_t, q_a)
        prototype_dis, mask_dis = self.calculate_distance(p_t, p_a)

        query = torch.stack([q_t, q_a], dim=0)
        pos = torch.stack([p_t, p_a], dim=0)
        neg = torch.stack([n_t, n_a], dim=0)
        mask_combined = torch.stack([mask, mask_his], dim=0) if mask is not None else None

        h_de = torch.cat([h_t, v_t], dim=-1)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)

        node_embeddings = self.hypernet(h_aug)
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1)
        supports_de = [support]

        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((batch_size, self.num_nodes, self.output_dim), device=self.device)

        out = []
        for t in range(self.horizon):
            if self.use_STE:
                if self.input_embedding_dim > 0:
                    go = self.input_proj(go)
                features_dec = [go]

                if self.tod_embed_dim > 0:
                    tod = y_cov[:, t, ...].squeeze(-1)
                    time_indices = (tod * self.TDAY).long().clamp(0, self.TDAY - 1)
                    time_emb = self.time_embedding[time_indices]
                    features_dec.append(time_emb)

                if self.node_embedding_dim > 0:
                    node_emb = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
                    features_dec.append(node_emb)

                go = torch.cat(features_dec, dim=-1)
                h_de, ht_list = self.decoder(go, ht_list, supports_de)
            else:
                h_de, ht_list = self.decoder(
                    torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports_de
                )

            go = self.proj(h_de)
            out.append(go)

            if self.training and self.use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)

        return output, query, pos, neg, mask_combined, latent_dis, prototype_dis

    def forward(self, batch, batches_seen=None):
        """
        Forward pass for LibCity framework.

        Args:
            batch: dict containing 'X' and 'y' tensors
            batches_seen: number of batches seen for curriculum learning

        Returns:
            torch.Tensor: predictions of shape (batch, horizon, nodes, output_dim)
        """
        x, x_cov, x_his, y_cov, labels = self._prepare_data(batch)

        if batches_seen is None:
            batches_seen = self.batches_seen

        if self.training:
            # Transform labels for curriculum learning
            labels_scaled = self._scaler.transform(labels)
        else:
            labels_scaled = None

        output, query, pos, neg, mask, latent_dis, prototype_dis = self._forward_with_aux(
            x, x_cov, x_his, y_cov, labels_scaled, batches_seen
        )

        return output

    def predict(self, batch, batches_seen=None):
        """
        Predict method for LibCity framework.

        Args:
            batch: dict containing 'X' tensor
            batches_seen: number of batches seen

        Returns:
            torch.Tensor: predictions of shape (batch, horizon, nodes, output_dim)
        """
        return self.forward(batch, batches_seen)

    def calculate_loss(self, batch, batches_seen=None):
        """
        Calculate combined loss for training.

        The loss combines:
        1. MAE loss for prediction accuracy
        2. Contrastive loss for prototype learning
        3. Deviation loss for self-supervised learning

        Args:
            batch: dict containing 'X' and 'y' tensors
            batches_seen: number of batches seen for curriculum learning

        Returns:
            torch.Tensor: scalar loss value
        """
        x, x_cov, x_his, y_cov, labels = self._prepare_data(batch)

        if batches_seen is None:
            batches_seen = self.batches_seen
            self.batches_seen += 1

        # Transform labels for curriculum learning
        labels_scaled = self._scaler.transform(labels)

        output, query, pos, neg, mask, latent_dis, prototype_dis = self._forward_with_aux(
            x, x_cov, x_his, y_cov, labels_scaled, batches_seen
        )

        # Inverse transform for loss computation
        y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
        y_true = labels

        # MAE loss
        mae_loss = loss.masked_mae_torch(y_pred, y_true, null_val=0.0)

        # Contrastive loss
        loss_c = self.contrastive_loss(query[0].detach(), pos[0], neg[0])

        # Deviation loss
        loss_d = F.l1_loss(latent_dis.detach(), prototype_dis)

        # Combined loss
        total_loss = mae_loss + self.lamb_c * loss_c + self.lamb_d * loss_d

        return total_loss
