"""
MetaDG Model Adaptation for LibCity Framework

Original Paper: MetaDG: Meta-Learning for Dynamic Graph-based Traffic Flow Prediction
Original Repository: /home/wangwenrui/shk/AgentCity/repos/MetaDG

Key Changes Made:
1. Inherited from AbstractTrafficStateModel instead of nn.Module
2. Adapted forward signature from forward(x, y_cov, labels, batches_seen) to forward(batch)
3. Added predict() and calculate_loss() methods required by LibCity
4. Extracted time features (tod, dow) from LibCity's batch format
5. Added batches_seen tracking as class attribute for curriculum learning
6. Adapted input/output dimension handling for LibCity conventions

Required Config Parameters:
- node_emb_dim: Node embedding dimension (default: 16)
- tod_emb_dim: Time of day embedding dimension (default: 8)
- dow_emb_dim: Day of week embedding dimension (default: 8)
- continuous_time_emb_dim: Continuous time embedding dimension (default: 8)
- hidden_dim: Hidden state dimension (default: 64)
- layer_num: Number of encoder/decoder layers (default: 1)
- hop_num_k: Number of graph convolution hops (default: 1)
- use_curriculum_learning: Whether to use curriculum learning (default: True)
- cl_decay_steps: Curriculum learning decay steps (default: 6000)
- use_mask: Whether to use graph mask (default: True)
- refine_dynamic_node: Whether to refine dynamic node embeddings (default: True)
- refine_dynamic_graph: Whether to refine dynamic graphs (default: True)
- corr_enhance_mode: Correlation enhancement mode ['S', 'T', 'ST', 'TS', 'OFF'] (default: 'ST')

Assumptions/Limitations:
- Input X is expected to have shape (batch, time_in, num_nodes, features)
- The model expects add_time_in_day=True and add_day_in_week=True in config
- Time of day is expected as a normalized value [0, 1] in the feature dimension
- Day of week is expected as an integer [0, 6] in the feature dimension
"""

import numpy as np
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


def asym_adj(adj):
    """Asymmetric adjacency matrix normalization (row normalization)."""
    row_sum = adj.sum(-1)
    d_inv = 1 / row_sum
    if adj.dim() == 2:
        adj = torch.einsum('nm,n->nm', adj, d_inv)
    elif adj.dim() == 3:
        adj = torch.einsum('bnm,bn->bnm', adj, d_inv)
    return adj


class InstanceNorm(nn.Module):
    """Instance normalization with learnable parameters."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[-2, -1], keepdim=True)
        var = x.var(dim=[-2, -1], unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GCN(nn.Module):
    """Simple Graph Convolution layer."""
    def __init__(self):
        super(GCN, self).__init__()

    def forward(self, x, adj):
        if adj.dim() == 2:
            x = torch.einsum('nm,bmd->bnd', adj, x)
        elif adj.dim() == 3:
            x = torch.einsum('bnm,bmd->bnd', adj, x)
        else:
            raise Exception(f'Wrong Adj Matrix Dimension! Assert 2 or 3, got {adj.dim()}.')
        return x.contiguous()


class VariationalDropout(nn.Module):
    """Variational dropout that uses the same mask across time steps."""
    def __init__(self, p, device):
        super(VariationalDropout, self).__init__()
        self.dropout_rate = p
        self.device = device
        self.mask = None

    def forward(self, x, time_step):
        if self.training:
            if time_step == 0:
                self.mask = self.generate_mask(x.shape)
            x = self.mask * x
        return x

    def generate_mask(self, tensor_shape, device="cuda"):
        """Generate variational dropout mask."""
        if len(tensor_shape) == 3:
            batch_size, node_num, hidden_size = tensor_shape[0], tensor_shape[1], tensor_shape[2]
        else:
            batch_size, hidden_size = tensor_shape[0], tensor_shape[1]
            node_num = 1

        mask = torch.bernoulli(torch.ones([batch_size, node_num, hidden_size], device=self.device) * (1 - self.dropout_rate))
        mask = mask / (1 - self.dropout_rate)
        return mask


class MetaDGCN(nn.Module):
    """Meta-learning based Dynamic Graph Convolution Network."""
    def __init__(self, hop_num_k, hidden_dim, out_dim, graph_num, node_emb_dim):
        super(MetaDGCN, self).__init__()

        self.hop_num_k = hop_num_k
        self.hidden_dim = hidden_dim
        self.graph_num = graph_num
        self.node_emb_dim = node_emb_dim

        self.gconv = GCN()

        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.empty([node_emb_dim, (graph_num * hop_num_k + 1) * hidden_dim, out_dim])))
        self.bias_pool = nn.init.xavier_normal_(nn.Parameter(torch.empty([node_emb_dim, out_dim])))

    def forward(self, x, graph_list, node_emb):
        out = [x]
        for i in range(self.graph_num):
            x1 = x
            for k in range(self.hop_num_k):
                x2 = self.gconv(x1, graph_list[i])
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        weights = torch.einsum('bnd,dio->bnio', node_emb, self.weights_pool)
        bias = torch.einsum('bnd,do->bno', node_emb, self.bias_pool)
        h = torch.einsum('bni,bnio->bno', h, weights) + bias

        return h


class DynamicGraphQualification(nn.Module):
    """Dynamic Graph Qualification module for refining graph structure."""
    def __init__(self, node_emb_dim, delta=2., device=None):
        super(DynamicGraphQualification, self).__init__()

        self.delta = delta
        self.threshold_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(node_emb_dim, 1)))
        self.norm_phi = InstanceNorm()
        self.device = device

    def qualify(self, current_node_emb=None, prev_node_emb=None, static_node_emb=None):
        b, n, _ = current_node_emb.size()
        graph = torch.einsum('bnd,bmd->bnm', current_node_emb, prev_node_emb)
        static_graph_mask = torch.greater(torch.relu(torch.einsum('nd,md->nm', static_node_emb, static_node_emb)),
                                          torch.zeros([n, n]).to(self.device)).float()
        probs = asym_adj(torch.relu(torch.einsum('bnm,nm->bnm', graph, static_graph_mask)) + 1e-10)

        threshold = torch.sigmoid(torch.einsum('bnd,dg->bng', current_node_emb, self.threshold_pool))
        threshold_baseline = torch.diagonal(probs, dim1=-2, dim2=-1).unsqueeze(-1)
        threshold = threshold_baseline * threshold
        x = probs - threshold
        pos_mask = torch.greater_equal(x, torch.zeros([b, n, 1]).to(self.device)).float()
        neg_mask = 1 - pos_mask

        pos_mask = torch.sigmoid(x) * pos_mask
        tau_beta = torch.exp(self.norm_phi(pos_mask) * self.delta)
        phi = tau_beta * pos_mask + tau_beta * neg_mask

        return phi

    def forward(self, static_node_emb, current_node_emb, prev_node_emb):
        current_phi = self.qualify(current_node_emb=current_node_emb, prev_node_emb=prev_node_emb, static_node_emb=static_node_emb)
        return current_phi


class SpatialTemporalCorrelationEnhancement(nn.Module):
    """Spatial-Temporal Correlation Enhancement module."""
    def __init__(self, node_emb_dim, hidden_emb_dim, cross_attn_hidden_dim, corr_enhance_mode, device):
        super(SpatialTemporalCorrelationEnhancement, self).__init__()

        self.corr_enhance_mode = corr_enhance_mode

        if 'S' in self.corr_enhance_mode:
            self.linear_q = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_k = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_v = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_out = nn.Linear(cross_attn_hidden_dim, node_emb_dim)
            self.linear_out2 = nn.Linear(node_emb_dim, node_emb_dim)
            self.dropout1 = VariationalDropout(p=0.1, device=device)
            self.dropout2 = VariationalDropout(p=0.1, device=device)

        if 'T' in self.corr_enhance_mode:
            self.reset_proj = nn.Linear(hidden_emb_dim, node_emb_dim)

        self.scale = cross_attn_hidden_dim ** -0.5

    def spatial_correlation_enhancement(self, current_dynamic_node_emb, prev_dynamic_node_emb, time_step):
        q = self.linear_q(current_dynamic_node_emb)
        k = self.linear_k(prev_dynamic_node_emb)
        v = self.linear_v(prev_dynamic_node_emb)

        cross_attn = torch.einsum('bnh,bmh->bnm', q, k) * self.scale
        norm_cross_attn = torch.softmax(cross_attn, dim=-1)

        current_sce_node_emb = torch.einsum('bnm,bmh->bnh', norm_cross_attn, v)

        current_sce_node_emb = self.dropout1(current_dynamic_node_emb + self.linear_out(current_sce_node_emb), time_step)
        current_sce_node_emb = self.dropout2(current_dynamic_node_emb + self.linear_out2(current_sce_node_emb), time_step)

        return current_sce_node_emb

    def temporal_correlation_enhancement(self, current_dynamic_node_emb, prev_dynamic_node_emb, z):
        z_hat = torch.sigmoid(self.reset_proj(z))
        current_tce_node_emb = z_hat * prev_dynamic_node_emb + (1 - z_hat) * current_dynamic_node_emb
        return current_tce_node_emb

    def forward(self, current_dynamic_node_emb, prev_dynamic_node_emb, prev_enhanced_node_emb, z=None, time_step=None):
        if self.corr_enhance_mode == 'ST':
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, time_step)
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_enhanced_node_emb, prev_enhanced_node_emb, z)
        elif self.corr_enhance_mode == 'TS':
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, z)
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_enhanced_node_emb, prev_enhanced_node_emb, time_step)
        elif self.corr_enhance_mode == 'S':
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, time_step)
        elif self.corr_enhance_mode == 'T':
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, z)
            else:
                current_enhanced_node_emb = current_dynamic_node_emb
        else:
            current_enhanced_node_emb = current_dynamic_node_emb

        return current_enhanced_node_emb


class DynamicNodeGeneration(nn.Module):
    """Dynamic Node Embedding Generation module."""
    def __init__(self, node_emb_dim, time_emb_dim, hidden_emb_dim, graph_num):
        super(DynamicNodeGeneration, self).__init__()

        self.graph_num = graph_num
        gamma_src_dim = time_emb_dim * 2
        self.state_proj = nn.Linear(hidden_emb_dim, node_emb_dim)
        self.gamma_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(gamma_src_dim, graph_num, node_emb_dim)))

    def get_gamma(self, gamma_pool, emb=None):
        if emb.dim() == 2:
            gamma = torch.einsum('bd,dh->bh', emb, gamma_pool).unsqueeze(1)
        elif emb.dim() == 3:
            gamma = torch.einsum('bnd,dh->bnh', emb, gamma_pool)
        gamma = torch.sigmoid(gamma)
        return gamma

    def forward(self, static_node_emb, time_emb, state):
        state_emb = self.state_proj(state)
        current_dynamic_node_emb = []
        for i in range(self.graph_num):
            src_emb = torch.cat([time_emb[0], time_emb[1]], dim=-1)
            gamma = self.get_gamma(self.gamma_pool[:, i, :], src_emb)
            current_reserve_emb = gamma * static_node_emb + (1 - gamma) * state_emb
            current_dynamic_node_emb.append(current_reserve_emb)
        current_dynamic_node_emb = torch.stack(current_dynamic_node_emb, dim=1).squeeze(dim=1)

        return current_dynamic_node_emb


class MetaDGCRU(nn.Module):
    """Meta-learning based Dynamic Graph Convolutional Recurrent Unit."""
    def __init__(self,
                 hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim, node_num, graph_num,
                 use_mask, mask_pool, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST', device=None):
        super(MetaDGCRU, self).__init__()

        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.refine_dynamic_node = refine_dynamic_node
        self.refine_dynamic_graph = refine_dynamic_graph

        self.dynamic_node_generation = DynamicNodeGeneration(node_emb_dim=node_emb_dim,
                                                             time_emb_dim=time_emb_dim,
                                                             hidden_emb_dim=hidden_dim,
                                                             graph_num=graph_num)

        if self.refine_dynamic_node and self.refine_dynamic_graph:
            self.stce_p = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
            self.phi = DynamicGraphQualification(node_emb_dim=node_emb_dim, delta=2., device=device)
            self.stce_g = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
            self.stce_m = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
        elif self.refine_dynamic_node:
            self.stce_p = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
            self.stce_g = SpatialTemporalCorrelationEnhancement(node_emb_dim=node_emb_dim,
                                                                hidden_emb_dim=hidden_dim,
                                                                cross_attn_hidden_dim=64,
                                                                corr_enhance_mode=corr_enhance_mode,
                                                                device=device)
        elif self.refine_dynamic_graph:
            self.phi = DynamicGraphQualification(node_emb_dim=node_emb_dim, delta=2., device=device)

        self.use_mask = use_mask
        self.mask_pool = mask_pool
        if self.use_mask:
            p = 0.3
            self.dropout = nn.Dropout(p=p)
            if mask_pool is None:
                mask_emb_dim = node_emb_dim * 4
                self.mask_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(continuous_time_emb_dim * 2, 1, node_num, mask_emb_dim)))

        gcn_in_dim = in_dim + hidden_dim
        self.gate = MetaDGCN(hop_num_k, gcn_in_dim, out_dim * 2, graph_num, node_emb_dim)
        self.update = MetaDGCN(hop_num_k, gcn_in_dim, out_dim, graph_num, node_emb_dim)
        self.device = device

    def get_mask(self, time_emb, node_emb, mask_pool):
        mask_pool = torch.einsum('bns,dsh->bdnh', node_emb, mask_pool)
        mask_emb = torch.einsum('bd,bdnh->bnh', time_emb, mask_pool)
        graph_mask = torch.einsum('bnh,bmh->bnm', mask_emb, mask_emb)
        graph_mask = torch.relu(graph_mask)
        return graph_mask

    def forward(self,
                x, state, graph_list, static_node_emb, time_emb, continuous_time_emb,
                prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z, time_step):
        current_dynamic_node_emb = self.dynamic_node_generation(static_node_emb, time_emb, state)
        if self.refine_dynamic_node and self.refine_dynamic_graph:
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                B, _, _ = current_dynamic_node_emb.size()
                if time_step == 0:
                    prev_dynamic_node_emb = prev_dynamic_node_emb.repeat(B, 1, 1)
                    prev_stce_p = prev_stce_p.repeat(B, 1, 1)
                    prev_stce_g = prev_stce_g.repeat(B, 1, 1)
                    prev_stce_m = prev_stce_m.repeat(B, 1, 1)

                current_stce_p = self.stce_p(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_p, z=z, time_step=time_step)

                current_stce_g = self.stce_g(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_g, z=z, time_step=time_step)
                graph = torch.einsum('bnd,bmd->bnm', current_stce_g, current_stce_g)
                if self.use_mask:
                    graph_mask = self.get_mask(continuous_time_emb, current_stce_g, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph

                current_stce_m = self.stce_m(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_m, z=z, time_step=time_step)
                current_phi = self.phi(static_node_emb, current_stce_m, prev_stce_m)
                graph = torch.einsum('bnm,bnm->bnm', current_phi, graph)

                if self.use_mask:
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)

                current_graph_list.append(graph)
        elif self.refine_dynamic_node:
            current_stce_m = None
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                current_stce_p = self.stce_p(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_p, z=z, time_step=time_step)

                current_stce_g = self.stce_g(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_g, z=z, time_step=time_step)
                graph = torch.einsum('bnd,bmd->bnm', current_stce_g, current_stce_g)
                if self.use_mask:
                    graph_mask = self.get_mask(continuous_time_emb, current_stce_g, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)
        elif self.refine_dynamic_graph:
            current_stce_p, current_stce_g, current_stce_m = None, None, None
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                graph = torch.einsum('bnd,bmd->bnm', current_dynamic_node_emb, current_dynamic_node_emb)
                if self.use_mask:
                    graph_mask = self.get_mask(continuous_time_emb, current_dynamic_node_emb, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph
                current_phi = self.phi(static_node_emb, current_dynamic_node_emb, prev_dynamic_node_emb)
                graph = torch.einsum('bnm,bnm->bnm', current_phi, graph)
                if self.use_mask:
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)
        else:
            current_stce_p, current_stce_g, current_stce_m = None, None, None
            current_graph_list = graph_list

        input_and_state = torch.cat((x, state), dim=-1)
        r_z = torch.sigmoid(self.gate(input_and_state, current_graph_list, current_stce_p if current_stce_p is not None else current_dynamic_node_emb))
        r, z = torch.split(r_z, self.hidden_dim, dim=-1)
        temp_state = r * state
        temp = torch.cat((x, temp_state), dim=-1)
        c = torch.tanh(self.update(temp, current_graph_list, current_stce_p if current_stce_p is not None else current_dynamic_node_emb))
        h = z * state + (1 - z) * c
        return h, z, current_dynamic_node_emb, current_stce_p, current_stce_g, current_stce_m

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class Encoder(nn.Module):
    """Encoder module with stacked MetaDGCRU cells."""
    def __init__(self,
                 hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num, layer_num, use_mask, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST',
                 device=None):
        super(Encoder, self).__init__()

        self.cells = nn.ModuleList()
        for i in range(layer_num):
            self.cells.append(MetaDGCRU(hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                                        node_num, graph_num, use_mask, mask_pool,
                                        refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode,
                                        device=device))
        self.device = device

    def forward(self, x, graph_list, node_emb=None, time_emb=None, continuous_time_emb=None, use_mask=True):
        b, steps, n, _ = x.size()
        output_hidden = []
        for cell in self.cells:
            prev_dynamic_node_emb = prev_stce_p = prev_stce_g = prev_stce_m = node_emb
            z = None
            state = cell.init_hidden_state(b).to(self.device)
            inner_states = []
            for t in range(steps):
                if use_mask:
                    continuous_time_emb_t = continuous_time_emb[:, t, ...]
                else:
                    continuous_time_emb_t = None
                state, z, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m = cell(x[:, t, :, :], state, graph_list, node_emb,
                                                                                                [time_emb[:, 0, :], time_emb[:, t, :]],
                                                                                                continuous_time_emb_t,
                                                                                                prev_dynamic_node_emb,
                                                                                                prev_stce_p, prev_stce_g, prev_stce_m,
                                                                                                z=z, time_step=t)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)
        return current_input, output_hidden


class Decoder(nn.Module):
    """Decoder module with stacked MetaDGCRU cells."""
    def __init__(self,
                 hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num, layer_num, use_mask, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True, corr_enhance_mode='ST',
                 device=None):
        super(Decoder, self).__init__()

        self.cells = nn.ModuleList()
        for i in range(layer_num):
            self.cells.append(MetaDGCRU(hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                                        node_num, graph_num, use_mask, mask_pool,
                                        refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph,
                                        corr_enhance_mode=corr_enhance_mode,
                                        device=device))
        self.device = device

    def forward(self,
                x_t, init_state, graph_list, node_emb=None, time_emb=None, continuous_time_emb=None,
                prev_dynamic_node_emb=None, prev_stce_p=None, prev_stce_g=None, prev_stce_m=None, z=None, time_step=None):
        current_input = x_t
        output_hidden = []
        for i, cell in enumerate(self.cells):
            state, z, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m = cell(current_input, init_state[i], graph_list, node_emb,
                                                                                          [time_emb[1], time_emb[0]],
                                                                                          continuous_time_emb,
                                                                                          prev_dynamic_node_emb,
                                                                                          prev_stce_p, prev_stce_g, prev_stce_m,
                                                                                          z=z, time_step=time_step)
            output_hidden.append(state)
            current_input = state

        return current_input, output_hidden, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z


class MetaDG(AbstractTrafficStateModel):
    """
    MetaDG: Meta-Learning for Dynamic Graph-based Traffic Flow Prediction

    Adapted for LibCity framework from original implementation.
    """
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # Get data features from LibCity
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        # Get config parameters
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)

        # MetaDG specific parameters
        self.node_emb_dim = config.get('node_emb_dim', 16)
        self.tod_emb_dim = config.get('tod_emb_dim', 8)
        self.dow_emb_dim = config.get('dow_emb_dim', 8)
        self.continuous_time_emb_dim = config.get('continuous_time_emb_dim', 8)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.layer_num = config.get('layer_num', 1)
        self.hop_num_k = config.get('hop_num_k', 1)

        # Curriculum learning parameters
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.cl_decay_steps = config.get('cl_decay_steps', 6000)

        # Dynamic graph parameters
        self.use_mask = config.get('use_mask', True)
        self.refine_dynamic_node = config.get('refine_dynamic_node', True)
        self.refine_dynamic_graph = config.get('refine_dynamic_graph', True)
        self.corr_enhance_mode = config.get('corr_enhance_mode', 'ST')

        # Check if time features are available
        self.add_time_in_day = config.get('add_time_in_day', True)
        self.add_day_in_week = config.get('add_day_in_week', True)

        # Calculate input dimension: flow features + optional time features handled separately
        # Original model expects [flow, tod, dow] in input
        # LibCity adds tod and dow as additional feature dimensions
        self.in_dim = self.output_dim  # Only flow features as input to GRU

        # Batches seen counter for curriculum learning
        self.batches_seen = 0

        self._logger = getLogger()
        self._logger.info(f'MetaDG Config: node_emb_dim={self.node_emb_dim}, tod_emb_dim={self.tod_emb_dim}, '
                         f'dow_emb_dim={self.dow_emb_dim}, continuous_time_emb_dim={self.continuous_time_emb_dim}')

        # Initialize embeddings
        self.node_emb = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.num_nodes, self.node_emb_dim)))
        self.tod_embedding = nn.Embedding(288, self.tod_emb_dim)  # 5-minute intervals: 288 per day
        self.dow_embedding = nn.Embedding(7, self.dow_emb_dim)

        # Time mask parameters
        if self.use_mask:
            time_emb_dim = self.continuous_time_emb_dim
            self.time_frequency_emb = nn.init.xavier_normal_(nn.Parameter(torch.empty(1, 1, time_emb_dim)))
            self.scale = (1 / time_emb_dim) ** 0.5
            mask_emb_dim = self.node_emb_dim * 4
            self.mask_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(time_emb_dim * 2, 1, self.node_emb_dim, mask_emb_dim)))
        else:
            self.mask_pool = None

        # Build encoder and decoder
        time_emb_dim_total = self.tod_emb_dim + self.dow_emb_dim

        self.encoder = Encoder(
            hop_num_k=self.hop_num_k,
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            node_emb_dim=self.node_emb_dim,
            time_emb_dim=time_emb_dim_total,
            continuous_time_emb_dim=self.continuous_time_emb_dim,
            node_num=self.num_nodes,
            graph_num=1,
            layer_num=self.layer_num,
            use_mask=self.use_mask,
            mask_pool=self.mask_pool,
            refine_dynamic_node=self.refine_dynamic_node,
            refine_dynamic_graph=self.refine_dynamic_graph,
            corr_enhance_mode=self.corr_enhance_mode,
            device=self.device
        )

        self.decoder = Decoder(
            hop_num_k=self.hop_num_k,
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            node_emb_dim=self.node_emb_dim,
            time_emb_dim=time_emb_dim_total,
            continuous_time_emb_dim=self.continuous_time_emb_dim,
            node_num=self.num_nodes,
            graph_num=1,
            layer_num=self.layer_num,
            use_mask=self.use_mask,
            mask_pool=self.mask_pool,
            refine_dynamic_node=self.refine_dynamic_node,
            refine_dynamic_graph=self.refine_dynamic_graph,
            corr_enhance_mode=self.corr_enhance_mode,
            device=self.device
        )

        self.fc_final = nn.Linear(self.hidden_dim, self.output_dim)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _extract_time_features(self, x):
        """
        Extract time of day and day of week features from input tensor.

        LibCity adds time features as additional dimensions in X:
        - If add_time_in_day=True: time_in_day is added (normalized [0, 1])
        - If add_day_in_week=True: day_in_week is added (integer [0, 6])

        Args:
            x: Input tensor of shape (batch, time, nodes, features)
               where features = [traffic_data..., time_in_day (optional), day_in_week (optional)]

        Returns:
            tuple: (flow_data, tod, dow)
                - flow_data: (batch, time, nodes, output_dim)
                - tod: (batch, time) normalized time of day [0, 1]
                - dow: (batch, time) day of week [0, 6]
        """
        b, t, n, f = x.shape

        # Determine feature indices based on config
        # Assuming feature order: [traffic_features, time_in_day, day_in_week]
        flow_dim = self.output_dim

        if self.add_time_in_day and self.add_day_in_week:
            # Features: [flow, tod, dow]
            flow_data = x[..., :flow_dim]
            tod = x[:, :, 0, flow_dim]  # time_in_day from first node
            dow = x[:, :, 0, flow_dim + 1]  # day_in_week from first node
        elif self.add_time_in_day:
            # Features: [flow, tod]
            flow_data = x[..., :flow_dim]
            tod = x[:, :, 0, flow_dim]
            dow = torch.zeros(b, t, device=x.device)
        elif self.add_day_in_week:
            # Features: [flow, dow]
            flow_data = x[..., :flow_dim]
            tod = torch.zeros(b, t, device=x.device)
            dow = x[:, :, 0, flow_dim]
        else:
            # No time features
            flow_data = x[..., :flow_dim]
            tod = torch.zeros(b, t, device=x.device)
            dow = torch.zeros(b, t, device=x.device)

        return flow_data, tod, dow

    def _prepare_decoder_covariates(self, batch):
        """
        Prepare decoder covariates (future time features) from batch.

        For decoder, we need future time features. In LibCity, y contains the target values
        but may also have time features appended.

        Args:
            batch: LibCity batch dictionary

        Returns:
            tuple: (y_cov_tod, y_cov_dow) for decoder time embeddings
        """
        y = batch['y']
        b, t, n, f = y.shape

        # Extract time features from y similar to x
        flow_dim = self.output_dim

        if self.add_time_in_day and self.add_day_in_week:
            tod = y[:, :, 0, flow_dim]
            dow = y[:, :, 0, flow_dim + 1]
        elif self.add_time_in_day:
            tod = y[:, :, 0, flow_dim]
            dow = torch.zeros(b, t, device=y.device)
        elif self.add_day_in_week:
            tod = torch.zeros(b, t, device=y.device)
            dow = y[:, :, 0, flow_dim]
        else:
            # If no time features, generate based on input
            tod = torch.zeros(b, t, device=y.device)
            dow = torch.zeros(b, t, device=y.device)

        return tod, dow

    def forward(self, batch):
        """
        Forward pass adapted for LibCity's batch format.

        Args:
            batch: Dictionary containing 'X' and 'y' tensors
                X shape: (batch, input_window, num_nodes, feature_dim)
                y shape: (batch, output_window, num_nodes, output_dim)

        Returns:
            torch.Tensor: Predictions of shape (batch, output_window, num_nodes, output_dim)
        """
        x = batch['X']
        y = batch['y']
        b, in_step, n, _ = x.size()

        # Extract flow data and time features from input
        flow_data, tod, dow = self._extract_time_features(x)

        # Prepare continuous time embedding for mask
        if self.use_mask:
            # Continuous time based on tod and dow
            continuous_time = 2 * np.pi * (tod + dow / 7.0)
            continuous_time = self.time_frequency_emb * continuous_time.unsqueeze(-1)
            time_cos = torch.cos(continuous_time)
            time_sin = torch.sin(continuous_time)
            continuous_time_emb = torch.stack([time_cos, time_sin], dim=-1).reshape(b, in_step, -1)
            continuous_time_emb = continuous_time_emb * self.scale
        else:
            continuous_time_emb = None

        # Prepare time embeddings
        # Convert normalized tod [0, 1] to index [0, 287]
        stepwise_tod_emb = self.tod_embedding((tod * 288).long().clamp(0, 287))
        stepwise_dow_emb = self.dow_embedding(dow.long().clamp(0, 6))
        stepwise_time_embedding = torch.cat([stepwise_tod_emb, stepwise_dow_emb], dim=-1)

        # Generate initial graph from node embeddings
        node_emb = self.node_emb
        graph = torch.softmax(torch.relu(node_emb @ node_emb.T), dim=-1)

        # Encode
        h, _ = self.encoder(flow_data, [graph], node_emb, stepwise_time_embedding,
                           continuous_time_emb=continuous_time_emb, use_mask=self.use_mask)
        h_last = h[:, -1, :, :]

        # Prepare decoder state
        state = [h_last] * self.layer_num
        go = torch.zeros((b, self.num_nodes, self.output_dim), device=self.device)

        # Get decoder time features
        y_tod, y_dow = self._prepare_decoder_covariates(batch)

        # Prepare decoder continuous time embedding
        if self.use_mask:
            continuous_time = 2 * np.pi * (y_tod + y_dow / 7.0)
            continuous_time = self.time_frequency_emb * continuous_time.unsqueeze(-1)
            time_cos = torch.cos(continuous_time)
            time_sin = torch.sin(continuous_time)
            continuous_time_emb_dec = torch.stack([time_cos, time_sin], dim=-1).reshape(b, self.output_window, -1)
            continuous_time_emb_dec = continuous_time_emb_dec * self.scale
        else:
            continuous_time_emb_dec = None

        # Prepare decoder time embeddings
        stepwise_tod_emb_dec = self.tod_embedding((y_tod * 288).long().clamp(0, 287))
        stepwise_dow_emb_dec = self.dow_embedding(y_dow.long().clamp(0, 6))
        stepwise_time_embedding_dec = torch.cat([stepwise_tod_emb_dec, stepwise_dow_emb_dec], dim=-1)

        # Decode with autoregressive generation
        out = []
        prev_dynamic_node_emb = prev_stce_p = prev_stce_g = prev_stce_m = node_emb
        z = None

        # Get labels for curriculum learning
        labels = y[..., :self.output_dim] if self.training and self.use_curriculum_learning else None

        for t in range(self.output_window):
            current_input = go
            time_emb = [stepwise_time_embedding_dec[:, t, :], stepwise_time_embedding_dec[:, -1, :]]

            if self.use_mask:
                continuous_time_emb_t = continuous_time_emb_dec[:, t, ...]
            else:
                continuous_time_emb_t = None

            h_de, state, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z = self.decoder(
                current_input, state, [graph], node_emb=node_emb,
                time_emb=time_emb,
                continuous_time_emb=continuous_time_emb_t,
                prev_dynamic_node_emb=prev_dynamic_node_emb,
                prev_stce_p=prev_stce_p,
                prev_stce_g=prev_stce_g,
                prev_stce_m=prev_stce_m,
                z=z,
                time_step=t
            )

            go = self.fc_final(h_de)
            out.append(go)

            # Curriculum learning: use ground truth with decreasing probability
            if self.training and self.use_curriculum_learning and labels is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(self.batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)
        return output

    def compute_sampling_threshold(self, batches_seen):
        """Compute sampling threshold for curriculum learning."""
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def predict(self, batch):
        """
        Make predictions for a batch.

        Args:
            batch: LibCity batch dictionary

        Returns:
            torch.Tensor: Predictions of shape (batch, output_window, num_nodes, output_dim)
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training.

        Args:
            batch: LibCity batch dictionary

        Returns:
            torch.Tensor: Loss value
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform for loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Update batches seen for curriculum learning
        if self.training:
            self.batches_seen += 1

        return loss.masked_mae_torch(y_predicted, y_true, 0)
