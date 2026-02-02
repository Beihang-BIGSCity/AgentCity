# coding: utf-8
"""
TrajSDE: Stochastic Differential Equation-based Trajectory Prediction Model

Adapted for LibCity framework from:
    Original repository: repos/TrajSDE
    Main model class: PredictionModelSDENet

This model uses SDE (Stochastic Differential Equations) for trajectory prediction:
- Encoder: LocalEncoderSDE with AA (Actor-Actor) interactions
- Aggregator: GlobalInteractor with graph-based multi-head attention
- Decoder: SDEDecoder for multi-modal trajectory generation

Dependencies:
- torchsde 0.2.5
- PyTorch Geometric 2.2.0

Migration Notes:
- Removed PyTorch Lightning dependencies
- Simplified for single-dataset usage (removed cross-domain components)
- Adapted data format from TemporalData to LibCity batch format
- Made lane-related components (AL encoder) optional for POI check-in data
- Core SDE components preserved intact
- Fixed batch immutability issue by using instance variables instead of modifying batch
"""

from __future__ import print_function
from __future__ import division

from typing import Optional, Tuple, List
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data, Batch as PyGBatch
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.typing import Adj, OptTensor, Size
    from torch_geometric.utils import softmax, subgraph
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("PyTorch Geometric not found. TrajSDE model requires torch_geometric.")

# TorchSDE imports
try:
    import torchsde
    from torchsde._core import base_sde, misc
    from torchsde._core.base_sde import BaseSDE
    from torchsde._core.methods.euler import Euler
    from torchsde._core.base_solver import interp, adaptive_stepping
    from torchsde._brownian import BaseBrownian, BrownianInterval
    from torchsde.settings import LEVY_AREA_APPROXIMATIONS, METHODS, NOISE_TYPES, SDE_TYPES
    HAS_TORCHSDE = True
except ImportError:
    HAS_TORCHSDE = False
    warnings.warn("torchsde not found. TrajSDE model requires torchsde >= 0.2.5.")


# ============================================================================
# Utility Functions and Classes
# ============================================================================

def init_weights(m: nn.Module) -> None:
    """Initialize network weights."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)


def init_network_weights(net, std=0.1):
    """Initialize network weights with normal distribution."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


class DistanceDropEdge:
    """Drop edges based on distance threshold."""

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self, edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


class TemporalData(Data if HAS_TORCH_GEOMETRIC else object):
    """Custom PyTorch Geometric Data class for temporal trajectory data."""

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 rotate_mat: Optional[torch.Tensor] = None,
                 lane_positions: Optional[torch.Tensor] = None,
                 lane_paddings: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 agent_index: Optional[torch.Tensor] = None,
                 batch: Optional[torch.Tensor] = None,
                 source: Optional[torch.Tensor] = None,
                 **kwargs) -> None:
        if not HAS_TORCH_GEOMETRIC:
            # Simple dict-like storage when PyG is not available
            self._data = {}
            if x is not None:
                self._data['x'] = x
            if positions is not None:
                self._data['positions'] = positions
            if edge_index is not None:
                self._data['edge_index'] = edge_index
            if y is not None:
                self._data['y'] = y
            if num_nodes is not None:
                self._data['num_nodes'] = num_nodes
            if padding_mask is not None:
                self._data['padding_mask'] = padding_mask
            if bos_mask is not None:
                self._data['bos_mask'] = bos_mask
            if rotate_angles is not None:
                self._data['rotate_angles'] = rotate_angles
            if rotate_mat is not None:
                self._data['rotate_mat'] = rotate_mat
            if lane_positions is not None:
                self._data['lane_positions'] = lane_positions
            if lane_paddings is not None:
                self._data['lane_paddings'] = lane_paddings
            if lane_actor_index is not None:
                self._data['lane_actor_index'] = lane_actor_index
            if lane_actor_vectors is not None:
                self._data['lane_actor_vectors'] = lane_actor_vectors
            if agent_index is not None:
                self._data['agent_index'] = agent_index
            if batch is not None:
                self._data['batch'] = batch
            if source is not None:
                self._data['source'] = source
            for k, v in kwargs.items():
                self._data[k] = v
            return

        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(
            x=x, positions=positions, edge_index=edge_index, y=y,
            num_nodes=num_nodes, padding_mask=padding_mask, bos_mask=bos_mask,
            rotate_angles=rotate_angles, rotate_mat=rotate_mat,
            lane_positions=lane_positions,
            lane_paddings=lane_paddings, lane_actor_index=lane_actor_index,
            lane_actor_vectors=lane_actor_vectors, agent_index=agent_index,
            batch=batch, source=source, **kwargs
        )

    def __getitem__(self, key):
        if HAS_TORCH_GEOMETRIC:
            return super().__getitem__(key)
        return self._data.get(key)

    def __setitem__(self, key, value):
        if HAS_TORCH_GEOMETRIC:
            super().__setitem__(key, value)
        else:
            self._data[key] = value

    def __inc__(self, key, value, *args, **kwargs):
        if not HAS_TORCH_GEOMETRIC:
            return 0
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


# ============================================================================
# Embedding Modules
# ============================================================================

class SingleInputEmbedding(nn.Module):
    """Single input embedding layer."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    """Multiple input embedding layer with aggregation."""

    def __init__(self, in_channels: List[int], out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)


# ============================================================================
# GRU Unit for SDE Integration
# ============================================================================

class GRU_Unit(nn.Module):
    """GRU unit for integrating SDE outputs with observations."""

    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim))
        init_network_weights(self.new_state_net)

    def forward(self, h_cur, input_tensor, mask):
        y_concat = torch.cat([h_cur, input_tensor], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        new_state = self.new_state_net(combined)

        h_next = (1 - update_gate) * new_state + update_gate * h_cur
        h_next = mask.unsqueeze(-1) * h_next + ~mask.unsqueeze(-1) * h_cur

        return h_next


# ============================================================================
# SDE Components
# ============================================================================

class FFunc(nn.Module):
    """Posterior drift function for SDE."""

    def __init__(self, embed_dim, num_layers=2):
        super(FFunc, self).__init__()
        net_list = [nn.Linear(embed_dim + 2, embed_dim)]
        for _ in range(num_layers):
            net_list.append(nn.Tanh())
            net_list.append(nn.Linear(embed_dim, embed_dim))
        self.net = nn.Sequential(*nn.ModuleList(net_list))

    def forward(self, t, y):
        _t = torch.ones(y.size(0), 1, device=y.device, dtype=y.dtype) * float(t)
        inp = torch.cat((y, torch.sin(_t), torch.cos(_t)), dim=-1)
        return self.net(inp)


class HFunc(nn.Module):
    """Prior drift function for SDE (Ornstein-Uhlenbeck process)."""

    def __init__(self, theta=1.0, mu=0.0):
        super(HFunc, self).__init__()
        self.theta = nn.Parameter(torch.tensor([[theta]]), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)

    def forward(self, t, y):
        return self.theta * (self.mu - y)


class GFunc(nn.Module):
    """Diffusion function for SDE."""

    def __init__(self, embed_dim, sigma=0.5, num_layers=2):
        super(GFunc, self).__init__()
        net_list = [nn.Linear(embed_dim + 2, embed_dim)]
        for _ in range(num_layers - 1):
            net_list.append(nn.Tanh())
            net_list.append(nn.Linear(embed_dim, embed_dim))
        net_list.append(nn.Tanh())
        net_list.append(nn.Linear(embed_dim, 1))
        self.net = nn.Sequential(*nn.ModuleList(net_list))

    def forward(self, t, y):
        _t = torch.ones(y.size(0), 1, device=y.device, dtype=y.dtype) * float(t)
        out = self.net(torch.cat((y, torch.sin(_t), torch.cos(_t)), dim=-1))
        return torch.sigmoid(out)


class LSDEFunc(torchsde.SDEIto if HAS_TORCHSDE else nn.Module):
    """Latent SDE function combining drift and diffusion."""

    def __init__(self, f, g, h, embed_dim, order=1):
        if HAS_TORCHSDE:
            super().__init__(noise_type="diagonal")
        else:
            super().__init__()
        self.order = order
        self.intloss = None
        self.sensitivity = None
        self.f_func = f
        self.g_func = g
        self.h_func = h
        self.fnfe = 0
        self.gnfe = 0
        self.hnfe = 0
        self.embed_dim = embed_dim

    def forward(self, s, x):
        pass

    def h(self, s, x):
        """Prior drift."""
        self.hnfe += 1
        return self.h_func(t=s, y=x)

    def f(self, s, x):
        """Posterior drift."""
        self.fnfe += 1
        return self.f_func(t=s, y=x)

    def g(self, s, x):
        """Diffusion."""
        self.gnfe += 1
        out = self.g_func(t=s, y=x).repeat(1, self.embed_dim)
        return out


# ============================================================================
# Custom SDE Solver
# ============================================================================

if HAS_TORCHSDE:
    from torchsde._core import base_solver

    class BaseSDESolverPrivate(base_solver.BaseSDESolver):
        """Custom SDE solver for TrajSDE integration."""

        def integrate(self, y0, ts, extra0):
            step_size = self.dt
            prev_t = curr_t = ts[0]
            prev_y = curr_y = y0
            curr_extra = extra0

            ys = [y0]
            prev_error_ratio = None

            for out_t in ts[1:]:
                while curr_t < out_t:
                    next_t = min(curr_t + step_size, ts[-1])
                    prev_t, prev_y = curr_t, curr_y
                    curr_y, curr_extra = self.step(curr_t, next_t, curr_y, curr_extra)
                    curr_t = next_t
                ys.append(interp.linear_interp(t0=prev_t, y0=prev_y, t1=curr_t, y1=curr_y, t=out_t))

            return torch.stack(ys, dim=0), curr_extra

    class EulerSolver(BaseSDESolverPrivate):
        """Euler-Maruyama solver for SDE."""

        weak_order = 1.0
        sde_type = SDE_TYPES.ito
        noise_types = NOISE_TYPES.all()
        levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

        def __init__(self, sde, **kwargs):
            self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
            super(EulerSolver, self).__init__(sde=sde, **kwargs)

        def step(self, t0, t1, y0, extra0):
            del extra0
            dt = t1 - t0
            I_k = self.bm(t0, t1)
            f, g_prod = self.sde.f_and_g_prod(t0, y0, I_k)
            y1 = y0 + f * dt + g_prod
            return y1, ()


class ForwardSDE(BaseSDE if HAS_TORCHSDE else nn.Module):
    """Forward SDE wrapper for integration."""

    def __init__(self, sde):
        if HAS_TORCHSDE:
            super(ForwardSDE, self).__init__(sde_type=sde.sde_type, noise_type=sde.noise_type)
        else:
            super().__init__()
        self._base_sde = sde
        self.f = sde.f
        self.g = sde.g
        self.f_and_g = self.f_and_g_default
        self.f_and_g_prod = self.f_and_g_prod_default
        self.prod = self.prod_diagonal

    def f_and_g_default(self, t, y):
        return self.f(t, y), self.g(t, y)

    def prod_diagonal(self, g, v):
        return g * v

    def f_and_g_prod_default(self, t, y, v):
        f, g = self.f_and_g(t, y)
        return f, self.prod(g, v)


def sdeint_simple(sde, y0, ts, dt=1e-3, rtol=1e-5, atol=1e-4, method='euler'):
    """Simplified SDE integration function."""
    if not HAS_TORCHSDE:
        raise RuntimeError("torchsde is required for SDE integration")

    forward_sde = ForwardSDE(sde)

    batch_size = y0.size(0)
    noise_size = y0.size(1)

    bm = BrownianInterval(
        t0=ts[0], t1=ts[-1],
        size=(batch_size, noise_size),
        dtype=y0.dtype, device=y0.device,
        levy_area_approximation=LEVY_AREA_APPROXIMATIONS.none
    )

    solver = EulerSolver(
        sde=forward_sde,
        bm=bm,
        dt=dt,
        adaptive=False,
        rtol=rtol,
        atol=atol,
        dt_min=1e-5,
        options={}
    )

    extra0 = solver.init_extra_solver_state(ts[0], y0)
    ys, _ = solver.integrate(y0, ts, extra0)

    return ys


# ============================================================================
# Actor-Actor Encoder (Simplified for LibCity)
# ============================================================================

class AAEncoderSimple(nn.Module):
    """Simplified Actor-Actor interaction encoder for LibCity batch format.

    This version does not rely on PyTorch Geometric MessagePassing and works
    with the simpler batch format provided by LibCity.
    """

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(AAEncoderSimple, self).__init__()

        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, bos_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (num_actors, historical_steps, node_dim)
            bos_mask: Beginning-of-sequence mask (num_actors, historical_steps)

        Returns:
            Encoded features of shape (historical_steps, num_actors, embed_dim)
        """
        # x: (num_actors, historical_steps, node_dim)
        num_actors, seq_len, _ = x.shape

        # Embed each timestep
        # Reshape to (num_actors * historical_steps, node_dim)
        x_flat = x.reshape(-1, x.size(-1))
        center_embed = self.center_embed(x_flat)  # (num_actors * historical_steps, embed_dim)
        center_embed = center_embed.view(num_actors, seq_len, -1)  # (num_actors, historical_steps, embed_dim)

        # Apply bos_mask: replace with bos_token where mask is True
        # bos_mask: (num_actors, historical_steps)
        bos_token_expanded = self.bos_token.unsqueeze(0).expand(num_actors, -1, -1)  # (num_actors, historical_steps, embed_dim)
        center_embed = torch.where(
            bos_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            bos_token_expanded,
            center_embed
        )

        # Self-attention across time steps for each actor
        center_embed = center_embed.transpose(0, 1)  # (historical_steps, num_actors, embed_dim)

        # Simple self-attention without edge information
        x_norm = self.norm1(center_embed)

        # Compute Q, K, V
        Q = self.lin_q(x_norm)  # (historical_steps, num_actors, embed_dim)
        K = self.lin_k(x_norm)
        V = self.lin_v(x_norm)

        # Reshape for multi-head attention
        Q = Q.view(seq_len, num_actors, self.num_heads, self.embed_dim // self.num_heads)
        K = K.view(seq_len, num_actors, self.num_heads, self.embed_dim // self.num_heads)
        V = V.view(seq_len, num_actors, self.num_heads, self.embed_dim // self.num_heads)

        # Attention across time steps
        # (seq_len, num_actors, num_heads, head_dim) -> (num_actors, num_heads, seq_len, head_dim)
        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)

        scale = (self.embed_dim // self.num_heads) ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (num_actors, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)  # (num_actors, num_heads, seq_len, head_dim)
        out = out.permute(2, 0, 1, 3).contiguous()  # (seq_len, num_actors, num_heads, head_dim)
        out = out.view(seq_len, num_actors, self.embed_dim)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        center_embed = center_embed + out
        center_embed = center_embed + self.mlp(self.norm2(center_embed))

        return center_embed  # (historical_steps, num_actors, embed_dim)


# ============================================================================
# Simplified Local Encoder for LibCity
# ============================================================================

class LocalEncoderSimple(nn.Module):
    """Simplified SDE-based local encoder for LibCity POI data.

    This version:
    - Works with LibCity's POI check-in data format
    - Does not require lane information (AL encoder is removed)
    - Uses simplified self-attention instead of graph message passing
    """

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 ref_time: int = 20,
                 max_past_t: float = 2.0,
                 sde_layers: int = 2,
                 minimum_step: float = 0.1,
                 rtol: float = 1e-3,
                 atol: float = 1e-3,
                 method: str = 'euler',
                 run_backwards: bool = True) -> None:
        super(LocalEncoderSimple, self).__init__()

        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.ref_time = ref_time
        self.max_past_t = max_past_t
        self.minimum_step = minimum_step
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.run_backwards = run_backwards

        self.aa_encoder = AAEncoderSimple(
            historical_steps=historical_steps,
            node_dim=node_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.gru_unit = GRU_Unit(embed_dim, embed_dim, n_units=embed_dim)

        # SDE components
        sigma, theta, mu = 0.5, 1.0, 0.0
        post_drift = FFunc(embed_dim, num_layers=sde_layers)
        prior_drift = HFunc(theta=theta, mu=mu)
        diffusion = GFunc(embed_dim, num_layers=sde_layers, sigma=sigma)
        self.lsde_func = LSDEFunc(f=post_drift, g=diffusion, h=prior_drift, embed_dim=embed_dim)
        if HAS_TORCHSDE:
            self.lsde_func.noise_type = 'diagonal'
            self.lsde_func.sde_type = 'ito'

        self.hidden = nn.Parameter(torch.Tensor(embed_dim))
        nn.init.normal_(self.hidden, mean=0., std=.02)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor,
                bos_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the local encoder.

        Args:
            x: Input features (num_actors, historical_steps, node_dim)
            padding_mask: Padding mask (num_actors, historical_steps)
            bos_mask: Beginning-of-sequence mask (num_actors, historical_steps)

        Returns:
            Encoded representations (num_actors, embed_dim)
        """
        num_actors = x.shape[0]
        prev_hidden = self.hidden.unsqueeze(0).repeat(num_actors, 1)

        actors_mask = ~padding_mask[:, :self.ref_time + 1]

        # AA encoding
        aa_out = self.aa_encoder(x, bos_mask)  # (historical_steps, num_actors, embed_dim)

        # SDE integration over time
        past_time_steps = torch.linspace(-self.max_past_t, 0, self.historical_steps)
        past_time_steps = -1 * past_time_steps
        time_points_iter = range(0, past_time_steps.size(-1))

        if self.run_backwards:
            prev_t, t_i = past_time_steps[-1] - 0.01, past_time_steps[-1]
            time_points_iter = reversed(time_points_iter)
        else:
            prev_t, t_i = past_time_steps[0] - 0.01, past_time_steps[0]

        latent_ys = []

        for idx, t in enumerate(time_points_iter):
            time_points = torch.tensor([prev_t, t_i], device=prev_hidden.device)

            # SDE integration step
            if HAS_TORCHSDE:
                pred_y = sdeint_simple(
                    self.lsde_func, prev_hidden, time_points,
                    dt=self.minimum_step, rtol=self.rtol, atol=self.atol, method=self.method
                )
                # pred_y shape: (time_steps, batch_size, embed_dim)
                # Take the last time step directly
                yi_ode = pred_y[-1]  # shape: (batch_size, embed_dim)
            else:
                # Fallback without SDE
                yi_ode = prev_hidden

            xi = aa_out[t]
            maski = actors_mask[:, t] if t < actors_mask.size(1) else torch.ones(num_actors, dtype=torch.bool, device=x.device)

            yi = self.gru_unit(input_tensor=xi, h_cur=yi_ode, mask=maski).squeeze(0)

            prev_hidden = yi
            if idx + 1 < past_time_steps.size(-1):
                if self.run_backwards:
                    prev_t, t_i = past_time_steps[t], past_time_steps[t - 1]
                else:
                    prev_t, t_i = past_time_steps[t], past_time_steps[t + 1]

            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys)  # (historical_steps, num_actors, embed_dim)

        # Get final embedding at end-of-sequence
        eos_idcs = self.ref_time - torch.argmax(bos_mask.float(), dim=1)
        eos_idcs = eos_idcs.clamp(0, latent_ys.size(0) - 1)
        out = latent_ys[eos_idcs, torch.arange(latent_ys.size(1)), :]

        return out


# ============================================================================
# Simplified Global Interactor for LibCity
# ============================================================================

class GlobalInteractorSimple(nn.Module):
    """Simplified global interactor for LibCity.

    Uses standard multi-head attention instead of graph-based message passing.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 num_modes: int = 10,
                 dropout: float = 0.1) -> None:
        super(GlobalInteractorSimple, self).__init__()

        self.num_modes = num_modes
        self.embed_dim = embed_dim

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(self, local_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_embed: (num_actors, embed_dim)

        Returns:
            Multi-modal embeddings: (num_modes, num_actors, embed_dim)
        """
        # Add batch dimension for attention
        x = local_embed.unsqueeze(0)  # (1, num_actors, embed_dim)

        for attn, norm, mlp in zip(self.attention_layers, self.norms, self.mlps):
            x_norm = norm(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + mlp(norm(x))

        x = self.norm_final(x).squeeze(0)  # (num_actors, embed_dim)
        x = self.multihead_proj(x)  # (num_actors, num_modes * embed_dim)
        x = x.view(-1, self.num_modes, self.embed_dim)  # (num_actors, num_modes, embed_dim)
        x = x.transpose(0, 1)  # (num_modes, num_actors, embed_dim)
        return x


# ============================================================================
# Simplified SDE Decoder for LibCity
# ============================================================================

class SDEDecoderSimple(nn.Module):
    """Simplified SDE-based decoder for POI prediction.

    Outputs probabilities over POI locations instead of continuous coordinates.
    """

    def __init__(self,
                 global_channels: int,
                 local_channels: int,
                 num_modes: int = 10,
                 future_steps: int = 1,
                 loc_size: int = 100,
                 max_fut_t: float = 6.0,
                 min_stepsize: float = 0.1,
                 rtol: float = 1e-3,
                 atol: float = 1e-3,
                 method: str = 'euler') -> None:
        super(SDEDecoderSimple, self).__init__()

        self.num_modes = num_modes
        self.future_steps = future_steps
        self.loc_size = loc_size
        self.min_stepsize = min_stepsize
        self.rtol = rtol
        self.atol = atol
        self.method = method

        self.input_size = global_channels
        self.hidden_size = local_channels

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))

        # SDE components for decoder
        sigma, theta, mu = 0.5, 1.0, 0.0
        post_drift = FFunc(self.hidden_size)
        prior_drift = HFunc(theta=theta, mu=mu)
        diffusion = GFunc(self.hidden_size, sigma=sigma)
        self.lsde_func = LSDEFunc(f=post_drift, g=diffusion, h=prior_drift, embed_dim=self.hidden_size)
        if HAS_TORCHSDE:
            self.lsde_func.noise_type = 'diagonal'
            self.lsde_func.sde_type = 'ito'

        # Output projection to location probabilities
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, loc_size))

        # Mode probability predictor
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))

        self.hidden = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.normal_(self.hidden, mean=0., std=.02)

        self.ts_pred = torch.linspace(0, max_fut_t, future_steps + 1)

        self.apply(init_weights)

    def forward(self, local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> dict:
        """Forward pass through the decoder.

        Args:
            local_embed: (num_actors, hidden_size)
            global_embed: (num_modes, num_actors, hidden_size)

        Returns:
            dict with 'loc' (location logits) and 'pi' (mode probabilities)
        """
        # Combine local and global embeddings
        loc_emb = self.aggr_embed(
            torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))

        num_actors = loc_emb.shape[1]
        hidden_0 = loc_emb.view(self.num_modes * num_actors, self.hidden_size)

        # SDE integration for trajectory prediction
        ts_pred = self.ts_pred.to(hidden_0.device)

        if HAS_TORCHSDE and self.future_steps > 1:
            from torchsde import sdeint
            sol_y = sdeint(
                self.lsde_func, hidden_0, ts_pred,
                dt=self.min_stepsize, dt_min=self.min_stepsize,
                rtol=self.rtol, atol=self.atol, method=self.method
            )[-1]  # Take only the last timestep
        else:
            # Use aggregated embedding directly for single-step prediction
            sol_y = hidden_0

        # Compute mode probabilities
        pi = self.pi(
            torch.cat((local_embed.expand(self.num_modes, *local_embed.shape), global_embed), dim=-1)
        ).squeeze(-1).t()  # (num_actors, num_modes)

        # Decode location probabilities
        loc_logits = self.decoder(sol_y).view(self.num_modes, num_actors, self.loc_size)

        # Aggregate across modes using mode probabilities
        # Use the mode with highest probability for prediction
        out = {'loc_logits': loc_logits, 'pi': pi}

        return out


# ============================================================================
# Main TrajSDE Model for LibCity
# ============================================================================

class TrajSDE(AbstractModel):
    """
    TrajSDE: Stochastic Differential Equation-based Trajectory Prediction

    Adapted for LibCity's POI check-in trajectory prediction task.

    This model uses SDEs for both encoding and decoding, with multi-head
    attention for global interactions.

    Args:
        config (dict): Configuration dictionary containing model parameters
        data_feature (dict): Data feature dictionary from LibCity dataset

    Configuration Parameters:
        - embed_dim (int): Embedding dimension (default: 64)
        - num_modes (int): Number of prediction modes (default: 6)
        - num_heads (int): Number of attention heads (default: 8)
        - dropout (float): Dropout rate (default: 0.1)
        - historical_steps (int): Number of historical time steps (default: 10)
        - hidden_size (int): Hidden layer size (default: 128)
        - num_global_layers (int): Number of global interaction layers (default: 3)
        - sde_layers (int): Number of SDE layers (default: 2)
        - rtol (float): Relative tolerance for SDE solver (default: 0.001)
        - atol (float): Absolute tolerance for SDE solver (default: 0.001)
        - step_size (float): Step size for SDE solver (default: 0.1)
    """

    def __init__(self, config, data_feature):
        super(TrajSDE, self).__init__(config, data_feature)

        # Check dependencies
        if not HAS_TORCHSDE:
            warnings.warn("torchsde not found. Model will use fallback methods.")

        self.device = config.get('device', 'cpu')

        # Data features from LibCity
        self.loc_size = data_feature.get('loc_size', 1000)
        self.tim_size = data_feature.get('tim_size', 48)
        self.uid_size = data_feature.get('uid_size', 100)
        self.loc_pad = data_feature.get('loc_pad', 0)

        # Model configuration
        self.embed_dim = config.get('embed_dim', 64)
        self.num_modes = config.get('num_modes', 6)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.historical_steps = config.get('historical_steps', 10)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_global_layers = config.get('num_global_layers', 3)
        self.sde_layers = config.get('sde_layers', 2)
        self.rtol = config.get('rtol', 0.001)
        self.atol = config.get('atol', 0.001)
        self.step_size = config.get('step_size', 0.1)

        # Reference time (typically last observation step)
        self.ref_time = self.historical_steps - 1

        # Embedding layers for LibCity data
        self.loc_embedding = nn.Embedding(self.loc_size, self.embed_dim, padding_idx=self.loc_pad)
        self.tim_embedding = nn.Embedding(self.tim_size, self.embed_dim // 2)

        # Input projection (location + time embeddings)
        self.input_proj = nn.Linear(self.embed_dim + self.embed_dim // 2, self.embed_dim)

        # Build model components
        self._build_encoder()
        self._build_aggregator()
        self._build_decoder()

        # Loss function
        self.loss_eps = 1e-6

    def _build_encoder(self):
        """Build the simplified SDE-based local encoder."""
        self.encoder = LocalEncoderSimple(
            historical_steps=self.historical_steps,
            node_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            ref_time=self.ref_time,
            max_past_t=2.0,
            sde_layers=self.sde_layers,
            minimum_step=self.step_size,
            rtol=self.rtol,
            atol=self.atol,
            method='euler',
            run_backwards=True
        )

    def _build_aggregator(self):
        """Build the simplified global interaction aggregator."""
        self.aggregator = GlobalInteractorSimple(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_global_layers,
            num_modes=self.num_modes,
            dropout=self.dropout
        )

    def _build_decoder(self):
        """Build the simplified SDE-based decoder."""
        self.decoder = SDEDecoderSimple(
            global_channels=self.embed_dim,
            local_channels=self.embed_dim,
            num_modes=self.num_modes,
            future_steps=1,  # Single-step prediction for POI
            loc_size=self.loc_size,
            max_fut_t=6.0,
            min_stepsize=self.step_size,
            rtol=self.rtol,
            atol=self.atol,
            method='euler'
        )

    def _prepare_data(self, batch):
        """
        Convert LibCity batch format to internal format.

        LibCity batch typically contains:
            - current_loc: Location sequence (batch_size, seq_len)
            - current_tim: Time sequence (batch_size, seq_len)
            - target: Target location (batch_size,)

        Returns:
            Tuple of (x, padding_mask, bos_mask, target)
        """
        # Extract data from batch - handle both dict and Batch object
        if hasattr(batch, '__getitem__'):
            current_loc = batch['current_loc']
            current_tim = batch.get('current_tim', None) if hasattr(batch, 'get') else batch['current_tim']
            target = batch.get('target', None) if hasattr(batch, 'get') else batch['target']
        else:
            current_loc = batch.current_loc
            current_tim = getattr(batch, 'current_tim', None)
            target = getattr(batch, 'target', None)

        # Handle time if not available
        if current_tim is None:
            current_tim = torch.zeros_like(current_loc)

        batch_size, seq_len = current_loc.shape

        # Adjust sequence length to match historical_steps
        if seq_len < self.historical_steps:
            # Pad with zeros
            pad_len = self.historical_steps - seq_len
            current_loc = F.pad(current_loc, (pad_len, 0), value=self.loc_pad)
            current_tim = F.pad(current_tim, (pad_len, 0), value=0)
            seq_len = self.historical_steps
        elif seq_len > self.historical_steps:
            # Truncate (keep last historical_steps)
            current_loc = current_loc[:, -self.historical_steps:]
            current_tim = current_tim[:, -self.historical_steps:]
            seq_len = self.historical_steps

        # Create embeddings
        loc_emb = self.loc_embedding(current_loc)  # (batch_size, seq_len, embed_dim)
        tim_emb = self.tim_embedding(current_tim.clamp(0, self.tim_size - 1))  # (batch_size, seq_len, embed_dim/2)

        # Concatenate and project
        x = self.input_proj(torch.cat([loc_emb, tim_emb], dim=-1))  # (batch_size, seq_len, embed_dim)

        # Create padding mask (True where padded)
        padding_mask = (current_loc == self.loc_pad)

        # Create BOS mask (True at beginning-of-sequence positions)
        # For LibCity, we mark padded positions as BOS
        bos_mask = padding_mask.clone()

        return x, padding_mask, bos_mask, target

    def forward(self, batch) -> dict:
        """
        Forward pass through the model.

        Args:
            batch: Input batch (LibCity Batch object or dict)

        Returns:
            dict: Output dictionary containing:
                - loc_logits: Location logits [num_modes, batch_size, loc_size]
                - pi: Mode probabilities [batch_size, num_modes]
        """
        # Prepare data (do not modify batch, create local variables instead)
        x, padding_mask, bos_mask, target = self._prepare_data(batch)

        # Encode
        local_embed = self.encoder(x, padding_mask, bos_mask)

        # Aggregate
        global_embed = self.aggregator(local_embed)

        # Decode
        out = self.decoder(local_embed, global_embed)

        return out

    def predict(self, batch) -> torch.Tensor:
        """
        Predict next location for the given batch.

        Args:
            batch: Input batch

        Returns:
            torch.Tensor: Predicted location probabilities
                Shape: [batch_size, loc_size]
        """
        output = self.forward(batch)

        # Get location logits and mode probabilities
        loc_logits = output['loc_logits']  # (num_modes, batch_size, loc_size)
        pi = output['pi']  # (batch_size, num_modes)

        # Weight by mode probabilities and aggregate
        pi_softmax = F.softmax(pi, dim=-1)  # (batch_size, num_modes)

        # Weighted average of location logits across modes
        loc_logits = loc_logits.permute(1, 0, 2)  # (batch_size, num_modes, loc_size)
        weighted_logits = (loc_logits * pi_softmax.unsqueeze(-1)).sum(dim=1)  # (batch_size, loc_size)

        return F.log_softmax(weighted_logits, dim=-1)

    def calculate_loss(self, batch) -> torch.Tensor:
        """
        Calculate training loss (Cross-entropy loss for POI prediction).

        Args:
            batch: Input batch

        Returns:
            torch.Tensor: Scalar loss value
        """
        output = self.forward(batch)

        # Get target from batch
        if hasattr(batch, '__getitem__'):
            target = batch['target']
        else:
            target = batch.target

        # Get location logits and mode probabilities
        loc_logits = output['loc_logits']  # (num_modes, batch_size, loc_size)
        pi = output['pi']  # (batch_size, num_modes)

        batch_size = target.shape[0]

        # Compute loss for each mode
        mode_losses = []
        for m in range(self.num_modes):
            mode_logits = loc_logits[m]  # (batch_size, loc_size)
            mode_loss = F.cross_entropy(mode_logits, target, reduction='none')  # (batch_size,)
            mode_losses.append(mode_loss)

        mode_losses = torch.stack(mode_losses, dim=1)  # (batch_size, num_modes)

        # Weight by mode probabilities
        pi_softmax = F.softmax(pi, dim=-1)  # (batch_size, num_modes)
        weighted_loss = (mode_losses * pi_softmax).sum(dim=1)  # (batch_size,)

        # Add mode diversity regularization
        pi_entropy = -(pi_softmax * torch.log(pi_softmax + self.loss_eps)).sum(dim=1)
        diversity_loss = -pi_entropy.mean() * 0.1  # Encourage mode diversity

        loss = weighted_loss.mean() + diversity_loss

        return loss

    def save(self, path):
        """Save model to path."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from path."""
        self.load_state_dict(torch.load(path, map_location=self.device))
