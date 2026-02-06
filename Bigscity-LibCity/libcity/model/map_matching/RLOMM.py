# coding=utf-8
"""
RLOMM: Reinforcement Learning for Online Map Matching

This model is adapted from the original RLOMM implementation for map matching tasks.
RLOMM uses a Double DQN architecture with contrastive learning for map matching
GPS trajectories to road segment sequences.

Original paper: "RLOMM: An Efficient Reinforcement Learning-Based Approach for Online Map Matching"

Key Components:
1. RoadGIN - Graph Isomorphism Network for road graph encoding
2. TraceGCN - Directed GCN for GPS trace graph encoding
3. QNetwork - DQN network with attention mechanism for action selection
4. MMAgent - Main RL agent that manages training and inference
5. Memory - Experience replay buffer for RL training

Adaptations for LibCity:
- Inherits from AbstractModel following LibCity conventions
- Implements predict() and calculate_loss() methods
- Adapts batch input format to LibCity's Batch dictionary format
- Integrates road graph and trace graph handling
- Provides both single-step and episode-based training modes

Required data_feature:
- num_roads: Number of road segments
- num_grids: Number of grid cells for trace encoding
- road_adj: Road graph adjacency (SparseTensor)
- road_x: Road node features
- trace_in_edge_index: Trace graph incoming edges
- trace_out_edge_index: Trace graph outgoing edges
- trace_weight: Trace graph edge weights
- map_matrix: Grid to road mapping matrix
- singleton_grid_mask: Mask for singleton grids
- singleton_grid_location: Location features for singleton grids
- connectivity_distances: Precomputed road connectivity distances

Required config parameters:
- road_emb_dim: Road embedding dimension (default: 128)
- traces_emb_dim: Trace embedding dimension (default: 128)
- num_layers: Number of RNN layers (default: 3)
- gin_depth: Depth of GIN layers (default: 3)
- gcn_depth: Depth of GCN layers (default: 3)
- gamma: RL discount factor (default: 0.99)
- target_update_interval: Target network update interval (default: 10)
- match_interval: Number of points to match at once (default: 4)
- memory_capacity: Experience replay capacity (default: 100)
- optimize_batch_size: Batch size for optimization (default: 8)
- correct_reward: Reward for correct match (default: 1.0)
- mask_reward: Reward for masked positions (default: 0.0)
- continuous_success_reward: Bonus for continuous successes (default: 0.5)
- connectivity_reward: Reward for connected roads (default: 0.5)
- detour_penalty: Penalty for detour (default: 0.3)
- lambda_ctr: Weight for contrastive loss (default: 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import logging

from libcity.model.abstract_model import AbstractModel

# Optional imports for graph neural networks
try:
    from torch_geometric.nn import GINConv, MLP, GCNConv
    from torch_sparse import SparseTensor
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    logging.warning("PyTorch Geometric not found. RLOMM may have limited functionality.")

# Set up logger
_logger = logging.getLogger(__name__)


# ============================================================================
# Memory: Experience Replay Buffer
# ============================================================================

State = namedtuple('State', (
    'traces_encoding',
    'matched_road_segments_encoding',
    'trace',
    'matched_road_segments_id',
    'candidates',
    'positive_samples',
    'negative_samples'
))

Transition = namedtuple('Transition', (
    'state',
    'action',
    'next_state',
    'reward'
))


class Memory:
    """Experience replay buffer for reinforcement learning."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self,
             last_traces_encoding, last_matched_road_segments_encoding,
             trace, matched_road_segments_id, candidates,
             last_positive_samples, last_negative_samples,
             traces_encoding, matched_road_segments_encoding,
             next_trace, next_matched_road_segments_id, next_candidates,
             next_positive_samples, next_negative_samples,
             action, reward):
        """Store a transition in the replay buffer."""
        state_namedtuple = State(last_traces_encoding, last_matched_road_segments_encoding,
                                 trace, matched_road_segments_id, candidates,
                                 last_positive_samples, last_negative_samples)

        next_state_namedtuple = State(traces_encoding, matched_road_segments_encoding,
                                      next_trace, next_matched_road_segments_id, next_candidates,
                                      next_positive_samples, next_negative_samples)

        self.memory.append(Transition(state_namedtuple, action, next_state_namedtuple, reward))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """Clear the memory buffer."""
        self.memory.clear()


# ============================================================================
# RoadGIN: Graph Isomorphism Network for Road Graph
# ============================================================================

class RoadGIN(nn.Module):
    """
    Road Graph Encoder using Graph Isomorphism Network.

    Encodes road network structure using GIN layers with max pooling
    across layers for feature aggregation.
    """

    def __init__(self, emb_dim, depth=3, mlp_layers=2):
        super().__init__()
        self.depth = depth
        self.emb_dim = emb_dim

        if HAS_PYGEOMETRIC:
            self.gins = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            for _ in range(self.depth):
                mlp = MLP(in_channels=emb_dim,
                          hidden_channels=2 * emb_dim,
                          out_channels=emb_dim,
                          num_layers=mlp_layers)
                self.gins.append(GINConv(nn=mlp, train_eps=True))
                self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        else:
            # Fallback to simple MLP if PyG not available
            self.layers = nn.ModuleList()
            for _ in range(self.depth):
                self.layers.append(nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim)
                ))

    def forward(self, x, adj_t):
        """
        Forward pass through RoadGIN.

        Args:
            x: Node features [num_nodes, emb_dim]
            adj_t: Adjacency matrix (SparseTensor or edge_index)

        Returns:
            x: Encoded road features with padding [num_nodes + 1, emb_dim]
        """
        layer_outputs = []

        if HAS_PYGEOMETRIC:
            for i in range(self.depth):
                x = self.gins[i](x, adj_t.to(x.device))
                x = F.relu(self.batch_norms[i](x))
                layer_outputs.append(x)
        else:
            # Fallback: simple MLP layers
            for i in range(self.depth):
                x = self.layers[i](x)
                layer_outputs.append(x)

        # Max pooling across layers
        x = torch.stack(layer_outputs, dim=0)
        x = torch.max(x, dim=0)[0]

        # Add padding token (zero vector)
        zero_tensor = torch.zeros(1, x.size(1), device=x.device)
        x = torch.cat((x, zero_tensor), dim=0)
        return x


# ============================================================================
# TraceGCN: Directed GCN for Trace Graph
# ============================================================================

class GCNLayer(nn.Module):
    """Single GCN layer with linear combination."""

    def __init__(self, in_feats, out_feats, bias=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)

        if HAS_PYGEOMETRIC:
            self.gcnconv = GCNConv(in_channels=in_feats,
                                   out_channels=out_feats,
                                   add_self_loops=False,
                                   bias=bias)
        else:
            # Fallback linear layer
            self.gcn_linear = nn.Linear(in_feats, out_feats, bias)

    def forward(self, x, edge_index, edge_weight=None):
        hl = self.linear(x)
        if HAS_PYGEOMETRIC:
            hr = self.gcnconv(x, edge_index, edge_weight)
        else:
            hr = self.gcn_linear(x)
        return hl + hr


class DiGCN(nn.Module):
    """Directed Graph Convolutional Network."""

    def __init__(self, embed_dim, depth=3):
        super(DiGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.depth = depth
        for _ in range(self.depth):
            self.convs.append(GCNLayer(embed_dim, embed_dim))
            self.bns.append(nn.BatchNorm1d(embed_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for idx in range(self.depth):
            x = self.convs[idx](x, edge_index, edge_weight)
            x = self.bns[idx](x)
            if idx != self.depth - 1:
                x = F.relu(x)
        return x


class TraceGCN(nn.Module):
    """
    Trace Graph Encoder using Bidirectional GCN.

    Encodes GPS trace graph with separate GCNs for incoming and outgoing edges.
    """

    def __init__(self, emb_dim, depth=3):
        super(TraceGCN, self).__init__()
        self.emb_dim = emb_dim
        self.gcn1 = DiGCN(self.emb_dim, depth)
        self.gcn2 = DiGCN(self.emb_dim, depth)

    def forward(self, feats, in_edge_index, out_edge_index, edge_weight=None):
        """
        Forward pass through TraceGCN.

        Args:
            feats: Node features [num_grids, emb_dim]
            in_edge_index: Incoming edge indices
            out_edge_index: Outgoing edge indices
            edge_weight: Edge weights

        Returns:
            Concatenated in/out embeddings [num_grids, 2*emb_dim]
        """
        emb_ind = self.gcn1(feats, in_edge_index, edge_weight)
        emb_oud = self.gcn2(feats, out_edge_index, edge_weight)
        ans = torch.cat([emb_ind, emb_oud], 1)
        return ans


# ============================================================================
# Attention Module
# ============================================================================

class Attention(nn.Module):
    """Attention mechanism for combining trace and segment features."""

    def __init__(self, combined_dim, candidate_dim, d_model):
        super(Attention, self).__init__()
        self.proj = nn.Linear(combined_dim, d_model)
        self.proj_candidates = nn.Linear(candidate_dim, d_model)

    def forward(self, trace_encoded, segments_encoded, candidates):
        """
        Compute attention scores for candidate selection.

        Args:
            trace_encoded: Encoded traces [batch, seq_len, trace_dim]
            segments_encoded: Encoded segments [batch, seq_len, seg_dim]
            candidates: Candidate embeddings [batch, seq_len, num_cands, cand_dim]

        Returns:
            scores: Attention scores [batch, seq_len, num_candidates]
        """
        trace_segments_combined = torch.cat((trace_encoded, segments_encoded), dim=2)
        x_proj = torch.tanh(self.proj(trace_segments_combined))
        candidates_proj = torch.tanh(self.proj_candidates(candidates))
        x_proj = x_proj.unsqueeze(2)
        scores = torch.matmul(x_proj, candidates_proj.transpose(2, 3)).squeeze(2)
        return scores


# ============================================================================
# QNetwork: Deep Q-Network
# ============================================================================

class QNetwork(nn.Module):
    """
    Q-Network for RLOMM.

    Combines road graph encoding, trace graph encoding, and RNN-based
    sequence modeling for Q-value estimation.
    """

    def __init__(self, road_emb_dim=128, traces_emb_dim=128, num_layers=3,
                 gin_depth=3, gcn_depth=3):
        super(QNetwork, self).__init__()
        self.emb_dim = traces_emb_dim
        self.road_emb_dim = road_emb_dim
        self.num_layers = num_layers

        # Feature projection layers
        self.fc = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.road_feat_fc = nn.Linear(28, road_emb_dim)  # 3*8 + 4 features
        self.trace_feat_fc = nn.Linear(4, traces_emb_dim)

        # Graph encoders
        self.road_gin = RoadGIN(road_emb_dim, depth=gin_depth)
        self.trace_gcn = TraceGCN(traces_emb_dim, depth=gcn_depth)

        # RNN for traces
        self.rnn_traces = nn.RNN(
            input_size=traces_emb_dim + 1,
            hidden_size=traces_emb_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # RNN for road segments
        self.rnn_segments = nn.RNN(
            input_size=road_emb_dim,
            hidden_size=road_emb_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Projection layers for attention
        self.trace_weight = nn.Linear(traces_emb_dim, traces_emb_dim // 2)
        self.segment_weight = nn.Linear(road_emb_dim, road_emb_dim // 2)

        # Attention mechanism
        self.attention = Attention(traces_emb_dim, road_emb_dim, 128)

    def forward(self, traces_encoding, matched_road_segments_encoding,
                traces, matched_road_segments_id, candidates,
                road_graph, trace_graph):
        """
        Forward pass through Q-Network.

        Args:
            traces_encoding: Previous trace hidden state
            matched_road_segments_encoding: Previous segment hidden state
            traces: Input traces [batch, seq_len, 2] (grid_id + time_delta)
            matched_road_segments_id: Matched road segment IDs [batch, seq_len, 1]
            candidates: Candidate road IDs [batch, seq_len, num_cands]
            road_graph: Road graph data object
            trace_graph: Trace graph data object

        Returns:
            traces_hidden: New trace hidden state
            segments_hidden: New segment hidden state
            action_values: Q-values for each candidate
            road_emb: Road embeddings
            full_grid_emb: Grid embeddings
        """
        device = traces.device

        # Encode road graph
        road_emb = self.road_feat_fc(road_graph.road_x)
        road_emb = self.road_gin(road_emb, road_graph.road_adj)

        # Get segment embeddings
        segments_emb = road_emb[matched_road_segments_id.squeeze(-1).long()]
        candidates_emb = road_emb[candidates.squeeze(-1).long()]

        # Build grid embeddings from road embeddings
        pure_grid_feat = torch.mm(trace_graph.map_matrix, road_emb[:-1, :])

        # Handle singleton grids
        if hasattr(trace_graph, 'singleton_grid_mask'):
            pure_grid_feat[trace_graph.singleton_grid_mask] = self.trace_feat_fc(
                trace_graph.singleton_grid_location
            )

        # Encode trace graph
        full_grid_emb = torch.zeros(trace_graph.num_grids + 1, 2 * self.emb_dim, device=device)
        full_grid_emb[1:, :] = self.trace_gcn(
            pure_grid_feat,
            trace_graph.trace_in_edge_index,
            trace_graph.trace_out_edge_index,
            trace_graph.trace_weight
        )

        # Extract trace embeddings
        trace_id = traces[:, :, 0].long()
        timestamp = traces[:, :, 1].unsqueeze(-1)

        full_grid_emb = self.fc(full_grid_emb)
        traces_emb = full_grid_emb[trace_id]
        traces_emb = torch.cat((traces_emb, timestamp), dim=-1)

        # RNN encoding
        traces_output, traces_hidden = self.rnn_traces(traces_emb, traces_encoding)
        segments_output, segments_hidden = self.rnn_segments(segments_emb, matched_road_segments_encoding)

        # Project and compute attention
        traces_encoded = self.trace_weight(traces_output)
        segments_encoded = self.segment_weight(segments_output)
        action_values = self.attention(traces_encoded, segments_encoded, candidates_emb)

        return traces_hidden, segments_hidden, action_values, road_emb, full_grid_emb


# ============================================================================
# MMAgent: Map Matching Agent
# ============================================================================

class MMAgent(nn.Module):
    """
    Map Matching Agent using Double DQN.

    Manages the main and target networks, experience replay,
    and provides methods for action selection and training.
    """

    def __init__(self, correct_reward=1.0, mask_reward=0.0,
                 continuous_success_reward=0.5, connectivity_reward=0.5,
                 detour_penalty=0.3, memory_capacity=100,
                 road_emb_dim=128, traces_emb_dim=128, num_layers=3,
                 gin_depth=3, gcn_depth=3):
        super(MMAgent, self).__init__()

        # Reward parameters
        self.correct_reward = correct_reward
        self.mask_reward = mask_reward
        self.continuous_success_reward = continuous_success_reward
        self.connectivity_reward = connectivity_reward
        self.detour_penalty = detour_penalty

        # Networks
        self.main_net = QNetwork(road_emb_dim, traces_emb_dim, num_layers,
                                 gin_depth, gcn_depth)
        self.target_net = QNetwork(road_emb_dim, traces_emb_dim, num_layers,
                                   gin_depth, gcn_depth)
        self.target_net.eval()

        # Experience replay
        self.memory = Memory(memory_capacity)

        # State tracking
        self.continuous_successes = None
        self.short_term_history = None
        self.short_term_history_size = 4

    def select_action(self, last_traces_encoding, last_matched_road_segments_encoding,
                      traces, matched_road_segments_id, candidates, road_graph, trace_graph):
        """
        Select actions using the main network (greedy policy).

        Returns:
            traces_encoding: Updated trace encoding
            matched_road_segments_encoding: Updated segment encoding
            action: Selected action indices
        """
        with torch.no_grad():
            traces_encoding, matched_road_segments_encoding, action_values, _, _ = self.main_net(
                last_traces_encoding,
                last_matched_road_segments_encoding,
                traces,
                matched_road_segments_id,
                candidates,
                road_graph,
                trace_graph
            )
            return traces_encoding, matched_road_segments_encoding, torch.argmax(action_values, dim=-1)

    def reset_continuous_successes(self, batch_size):
        """Reset continuous success counters."""
        self.continuous_successes = torch.zeros(batch_size, dtype=torch.int32)

    def init_short_history(self, batch_size):
        """Initialize short-term history for detour detection."""
        self.short_term_history = [[-1, -1] for _ in range(batch_size)]

    def update_short_term_history(self, matched_road_segments_id):
        """Update short-term history with new matches."""
        for i, road_id in enumerate(matched_road_segments_id):
            self.short_term_history[i].extend(road_id.tolist())
            self.short_term_history[i] = self.short_term_history[i][-self.short_term_history_size:]

    def step(self, last_matched_road, road_graph, action, candidates_id,
             tgt_roads, trace_lens, point_idx):
        """
        Compute rewards for a batch of actions.

        Args:
            last_matched_road: Last matched road segment IDs
            road_graph: Road graph data object
            action: Selected actions
            candidates_id: Candidate road IDs
            tgt_roads: Ground truth road indices
            trace_lens: Trace lengths for masking
            point_idx: Current point index

        Returns:
            rewards: Reward tensor [batch, seq_len]
        """
        seq_len = candidates_id.size(1)
        rewards = [[] for _ in range(seq_len)]
        continuous_success_threshold = 3

        self.init_short_history(action.size(0))

        for i in range(action.size(0)):
            last_road_id = last_matched_road[i].item()

            for j in range(seq_len):
                if trace_lens[i] <= point_idx + 1 - (seq_len - 1) + j:
                    reward = self.mask_reward
                    self.continuous_successes[i] = 0
                else:
                    selected_candidate_id = candidates_id[i, j, action[i, j]]

                    self.short_term_history[i][1] = self.short_term_history[i][0]
                    self.short_term_history[i][0] = selected_candidate_id.item()

                    if action[i, j] == tgt_roads[i, j].item():
                        reward = self.correct_reward
                        self.continuous_successes[i] += 1
                        if self.continuous_successes[i] >= continuous_success_threshold:
                            reward += self.continuous_success_reward
                    else:
                        reward = -self.correct_reward

                        # Check connectivity
                        if hasattr(road_graph, 'connectivity_distances'):
                            connectivity = road_graph.connectivity_distances.get(
                                (last_road_id, selected_candidate_id.item()), -1
                            )
                            if connectivity == -1:
                                reward += -self.connectivity_reward
                            elif connectivity > 2:
                                reward += -self.connectivity_reward / 2

                        self.continuous_successes[i] = 0

                        # Check for detour
                        if (selected_candidate_id.item() != self.short_term_history[i][0] and
                                selected_candidate_id.item() == self.short_term_history[i][1]):
                            reward += -self.detour_penalty

                    last_road_id = selected_candidate_id.item()

                rewards[j].append(reward)

        return torch.tensor(rewards, device=action.device).transpose(0, 1).float()

    def update_target_net(self):
        """Update target network with main network weights."""
        self.target_net.load_state_dict(self.main_net.state_dict())


# ============================================================================
# Helper Functions
# ============================================================================

def contrastive_loss(features, positive_samples, negative_samples, temperature=0.5, eps=1e-8):
    """
    Compute contrastive loss for aligning trace and road embeddings.

    Args:
        features: Trace features [batch * seq_len, feat_dim]
        positive_samples: Positive road embeddings [batch * seq_len, feat_dim]
        negative_samples: Negative road embeddings [batch * seq_len, num_neg, feat_dim]
        temperature: Temperature for softmax

    Returns:
        loss: Contrastive loss value
    """
    batch_size, match_interval, feature_dim = features.shape

    features = features.view(batch_size * match_interval, 1, feature_dim)
    positive_samples = positive_samples.view(batch_size * match_interval, feature_dim)
    negative_samples = negative_samples.view(batch_size * match_interval, -1, feature_dim)

    positive_similarity = torch.sum(features * positive_samples.unsqueeze(1), dim=-1) / temperature
    negative_similarity = torch.bmm(negative_samples, features.transpose(1, 2)) / temperature

    max_similarity = torch.max(positive_similarity, torch.max(negative_similarity, dim=1).values)
    positive_similarity = positive_similarity - max_similarity
    negative_similarity = negative_similarity - max_similarity.unsqueeze(1)

    positive_similarity_exp = torch.exp(positive_similarity)
    negative_similarity_exp = torch.exp(negative_similarity).sum(dim=1)

    loss = -torch.log((positive_similarity_exp + eps) / (positive_similarity_exp + negative_similarity_exp + eps))

    return loss.mean()


def get_positive_negative_samples(roads_slice, candidates_slice):
    """
    Extract positive and negative samples from candidates.

    Args:
        roads_slice: Ground truth road indices [batch, seq_len]
        candidates_slice: Candidate road IDs [batch, seq_len, num_cands]

    Returns:
        positive_samples: Positive sample IDs [batch, seq_len]
        negative_samples: Negative sample IDs [batch, seq_len, num_cands-1]
    """
    batch_size, seq_len, _ = candidates_slice.shape
    positive_samples = torch.gather(candidates_slice, 2, roads_slice.unsqueeze(-1)).squeeze(-1)
    all_indices = torch.arange(candidates_slice.shape[2], device=candidates_slice.device).expand(batch_size, seq_len, -1)
    mask = all_indices != roads_slice.unsqueeze(-1)
    negative_samples = torch.masked_select(candidates_slice, mask).view(batch_size, seq_len, -1)
    return positive_samples, negative_samples


# ============================================================================
# RoadGraph and TraceGraph Data Classes
# ============================================================================

class RoadGraphData:
    """Container for road graph data."""

    def __init__(self, road_x, road_adj, connectivity_distances=None, device='cpu'):
        self.road_x = road_x.to(device)
        self.road_adj = road_adj.to(device) if hasattr(road_adj, 'to') else road_adj
        self.connectivity_distances = connectivity_distances or {}
        self.device = device


class TraceGraphData:
    """Container for trace graph data."""

    def __init__(self, num_grids, trace_in_edge_index, trace_out_edge_index,
                 trace_weight, map_matrix, singleton_grid_mask=None,
                 singleton_grid_location=None, device='cpu'):
        self.num_grids = num_grids
        self.trace_in_edge_index = trace_in_edge_index.to(device)
        self.trace_out_edge_index = trace_out_edge_index.to(device)
        self.trace_weight = trace_weight.to(device)
        self.map_matrix = map_matrix.to(device)
        self.singleton_grid_mask = singleton_grid_mask.to(device) if singleton_grid_mask is not None else None
        self.singleton_grid_location = singleton_grid_location.to(device) if singleton_grid_location is not None else None
        self.device = device


# ============================================================================
# RLOMM: Main Model Class for LibCity
# ============================================================================

class RLOMM(AbstractModel):
    """
    RLOMM: Reinforcement Learning for Online Map Matching

    This model uses Double DQN with contrastive learning for map matching
    GPS trajectories to road segment sequences.

    Architecture:
    1. RoadGIN: Encodes road network topology
    2. TraceGCN: Encodes GPS trace graph structure
    3. QNetwork: Combines encodings with RNN for sequence modeling
    4. MMAgent: RL agent with experience replay

    Args:
        config: LibCity configuration dictionary
        data_feature: Data feature dictionary containing graph data
    """

    def __init__(self, config, data_feature):
        super(RLOMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Model hyperparameters
        self.road_emb_dim = config.get('road_emb_dim', 128)
        self.traces_emb_dim = config.get('traces_emb_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.gin_depth = config.get('gin_depth', 3)
        self.gcn_depth = config.get('gcn_depth', 3)

        # RL hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.target_update_interval = config.get('target_update_interval', 10)
        self.match_interval = config.get('match_interval', 4)
        self.memory_capacity = config.get('memory_capacity', 100)
        self.optimize_batch_size = config.get('optimize_batch_size', 8)

        # Reward parameters
        self.correct_reward = config.get('correct_reward', 1.0)
        self.mask_reward = config.get('mask_reward', 0.0)
        self.continuous_success_reward = config.get('continuous_success_reward', 0.5)
        self.connectivity_reward = config.get('connectivity_reward', 0.5)
        self.detour_penalty = config.get('detour_penalty', 0.3)
        self.lambda_ctr = config.get('lambda_ctr', 0.1)

        # Data features
        self.num_roads = data_feature.get('num_roads', 8533)
        self.num_grids = data_feature.get('num_grids', 10551)

        # Candidate generation parameters
        self.candidate_size = config.get('candidate_size', 10)

        # Store data feature for graph construction
        self.data_feature = data_feature

        # Store map_matrix for candidate generation (used when candidates_id not provided)
        map_matrix = data_feature.get('map_matrix', None)
        if map_matrix is not None:
            self.map_matrix = map_matrix
        else:
            # Fallback: identity-like mapping (each grid maps to itself as road)
            self.map_matrix = None

        # Build agent
        self._build_model()

        # Initialize graphs (will be set by executor or from data_feature)
        self._init_graphs(data_feature)

        # Training state
        self.steps_done = 0

        _logger.info(f"RLOMM: Initialized with road_emb_dim={self.road_emb_dim}, "
                     f"traces_emb_dim={self.traces_emb_dim}, "
                     f"num_roads={self.num_roads}, num_grids={self.num_grids}")

    def _build_model(self):
        """Build the MMAgent model."""
        self.agent = MMAgent(
            correct_reward=self.correct_reward,
            mask_reward=self.mask_reward,
            continuous_success_reward=self.continuous_success_reward,
            connectivity_reward=self.connectivity_reward,
            detour_penalty=self.detour_penalty,
            memory_capacity=self.memory_capacity,
            road_emb_dim=self.road_emb_dim,
            traces_emb_dim=self.traces_emb_dim,
            num_layers=self.num_layers,
            gin_depth=self.gin_depth,
            gcn_depth=self.gcn_depth
        )

    def _init_graphs(self, data_feature):
        """Initialize road and trace graphs from data_feature."""
        self.road_graph = None
        self.trace_graph = None

        # Try to construct graphs from data_feature
        if 'road_x' in data_feature and 'road_adj' in data_feature:
            self.road_graph = RoadGraphData(
                road_x=data_feature['road_x'],
                road_adj=data_feature['road_adj'],
                connectivity_distances=data_feature.get('connectivity_distances'),
                device=self.device
            )

        if 'trace_in_edge_index' in data_feature and 'trace_out_edge_index' in data_feature:
            self.trace_graph = TraceGraphData(
                num_grids=data_feature.get('num_grids', self.num_grids),
                trace_in_edge_index=data_feature['trace_in_edge_index'],
                trace_out_edge_index=data_feature['trace_out_edge_index'],
                trace_weight=data_feature['trace_weight'],
                map_matrix=data_feature['map_matrix'],
                singleton_grid_mask=data_feature.get('singleton_grid_mask'),
                singleton_grid_location=data_feature.get('singleton_grid_location'),
                device=self.device
            )

    def set_graphs(self, road_graph, trace_graph):
        """
        Set road and trace graphs externally.

        This allows the executor to provide pre-loaded graph data.
        """
        self.road_graph = road_graph
        self.trace_graph = trace_graph

    def _prepare_batch(self, batch):
        """
        Convert LibCity batch format to RLOMM expected format.

        Expected batch keys:
        - traces: [batch, seq_len, 2] (grid_id + time_delta)
        - tgt_roads: [batch, seq_len] (ground truth indices into candidates)
        - candidates_id: [batch, seq_len, num_candidates]
        - trace_lens: [batch] sequence lengths

        Alternative keys for LibCity compatibility:
        - X, y, input_gps, output_trg, etc.

        DeepMapMatchingDataset batch format:
        - grid_traces: [batch, seq_len] (grid cell IDs, 1-indexed, 0=padding)
        - tgt_roads: [batch, seq_len] (ground truth road IDs, not indices)
        - traces_lens: list of actual trace lengths
        - road_lens: list of actual road lengths
        - traces_gps: [batch, seq_len, 2] (GPS coordinates)
        """
        # Get traces - add 'grid_traces' as valid key
        traces = batch.get('traces', batch.get('X', batch.get('input_traces', batch.get('grid_traces'))))
        if traces is None:
            raise KeyError("Batch must contain 'traces', 'X', 'input_traces', or 'grid_traces'")
        traces = traces.to(self.device)

        # Handle 1D grid traces (convert to 2D format with zero time delta)
        # DeepMapMatchingDataset provides grid_traces as [batch, seq_len]
        if traces.dim() == 2:
            # [batch, seq_len] -> [batch, seq_len, 2] with (grid_id, time_delta=0)
            traces = torch.stack([traces, torch.zeros_like(traces)], dim=-1).float()

        # Get target roads (ground truth - could be indices into candidates or road IDs)
        tgt_roads = batch.get('tgt_roads', batch.get('y', batch.get('target', batch.get('output_trg'))))
        if tgt_roads is not None:
            tgt_roads = tgt_roads.to(self.device)

        # Get trace lengths - also accept 'traces_lens' as alternative key
        # NOTE: Must be extracted BEFORE sequence alignment block
        trace_lens = batch.get('trace_lens', batch.get('lengths', batch.get('src_lens', batch.get('traces_lens'))))
        if trace_lens is None:
            trace_lens = [traces.size(1)] * traces.size(0)

        # Handle sequence length mismatch between traces and roads
        # NOTE: Must happen BEFORE candidate generation to ensure tgt_roads is expanded
        # DeepMapMatchingDataset provides:
        # - grid_traces: [batch, trace_len] (e.g., 208 GPS points)
        # - tgt_roads: [batch, road_len] (e.g., 29 road segments)
        # - sample_Idx: [batch, trace_len] - maps each GPS point to its corresponding road index
        # RLOMM expects aligned sequences where tgt_roads has the same length as traces
        sample_idx = batch.get('sample_Idx', batch.get('sample_idx'))
        if sample_idx is not None and tgt_roads is not None:
            # Check if expansion is needed (tgt_roads shorter than traces)
            if tgt_roads.size(1) < traces.size(1):
                sample_idx = sample_idx.to(self.device)
                batch_size, trace_len = traces.size(0), traces.size(1)

                # Create expanded tgt_roads tensor [batch, trace_len]
                expanded_tgt_roads = torch.full((batch_size, trace_len), -1, dtype=torch.long, device=self.device)

                for b in range(batch_size):
                    trace_len_b = trace_lens[b] if isinstance(trace_lens, list) else trace_lens[b].item()
                    for t in range(int(trace_len_b)):
                        idx = sample_idx[b, t].item()
                        if 0 <= idx < tgt_roads.size(1):
                            expanded_tgt_roads[b, t] = tgt_roads[b, idx]

                tgt_roads = expanded_tgt_roads
                _logger.debug(f"Expanded tgt_roads from {batch.get('tgt_roads', batch.get('y')).size(1)} to {trace_len} using sample_Idx")

        # Get candidates - generate from map_matrix if not provided
        # NOTE: Must happen AFTER tgt_roads expansion to receive properly aligned tensor
        candidates_id = batch.get('candidates_id', batch.get('candidates', batch.get('cands')))
        if candidates_id is None:
            # Generate candidates using grid-to-road mapping
            candidates_id, tgt_roads = self._generate_candidates(traces, tgt_roads)
        else:
            candidates_id = candidates_id.to(self.device)

        return traces, tgt_roads, candidates_id, trace_lens

    def _generate_candidates(self, traces, tgt_roads):
        """
        Generate candidate roads for each position based on grid-to-road mapping.

        This method is used when the batch does not provide pre-computed candidates
        (e.g., when using DeepMapMatchingDataset).

        Args:
            traces: [batch, seq_len, 2] tensor where traces[:,:,0] contains grid IDs
            tgt_roads: [batch, seq_len] tensor containing ground truth road IDs
                       (Note: these are road IDs, not indices into candidates)

        Returns:
            candidates_id: [batch, seq_len, candidate_size] tensor of candidate road IDs
            tgt_indices: [batch, seq_len] tensor of indices into candidates for ground truth
        """
        batch_size, seq_len, _ = traces.shape
        device = traces.device

        # Extract grid IDs from traces (first element of last dimension)
        grid_ids = traces[:, :, 0].long()

        # Initialize candidates and target indices
        candidates_id = torch.zeros(batch_size, seq_len, self.candidate_size, dtype=torch.long, device=device)
        tgt_indices = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Get map_matrix for grid-to-road mapping
        if self.map_matrix is None:
            # No map_matrix available - use target roads as only candidates
            _logger.warning("No map_matrix available for candidate generation. Using target roads only.")
            for b in range(batch_size):
                for s in range(seq_len):
                    if tgt_roads is not None and tgt_roads[b, s] >= 0:
                        candidates_id[b, s, 0] = tgt_roads[b, s]
                        tgt_indices[b, s] = 0
                        # Fill remaining with random roads
                        random_roads = torch.randint(0, self.num_roads, (self.candidate_size - 1,), device=device)
                        candidates_id[b, s, 1:] = random_roads
            return candidates_id, tgt_indices

        # Use map_matrix to find candidate roads for each grid
        # map_matrix shape: [num_grids, num_roads]
        map_matrix = self.map_matrix
        if not isinstance(map_matrix, torch.Tensor):
            map_matrix = torch.tensor(map_matrix, device=device)
        else:
            map_matrix = map_matrix.to(device)

        for b in range(batch_size):
            for s in range(seq_len):
                grid_id = grid_ids[b, s].item()

                # Get candidate roads from map_matrix
                # Grid IDs from DeepMapMatchingDataset are 1-indexed (0 = padding)
                # Adjust for 0-indexed map_matrix if needed
                adjusted_grid_id = max(0, grid_id - 1) if grid_id > 0 else 0

                if adjusted_grid_id < map_matrix.size(0):
                    # Get roads that map to this grid (non-zero entries in map_matrix row)
                    grid_road_mapping = map_matrix[adjusted_grid_id]
                    candidate_roads = torch.nonzero(grid_road_mapping, as_tuple=True)[0]
                else:
                    candidate_roads = torch.tensor([], dtype=torch.long, device=device)

                # Prepare candidate list
                candidates_list = []

                # Always include target road first if available (ensures training signal)
                target_road = None
                if tgt_roads is not None and tgt_roads[b, s] >= 0:
                    target_road = tgt_roads[b, s].item()
                    candidates_list.append(target_road)

                # Add roads from grid mapping
                for road in candidate_roads.tolist():
                    if road not in candidates_list:
                        candidates_list.append(road)
                    if len(candidates_list) >= self.candidate_size:
                        break

                # Fill remaining slots with random roads if needed
                while len(candidates_list) < self.candidate_size:
                    random_road = torch.randint(0, self.num_roads, (1,), device=device).item()
                    if random_road not in candidates_list:
                        candidates_list.append(random_road)

                # Truncate to candidate_size
                candidates_list = candidates_list[:self.candidate_size]

                # Store candidates
                candidates_id[b, s, :] = torch.tensor(candidates_list, dtype=torch.long, device=device)

                # Find target index in candidates
                if target_road is not None and target_road in candidates_list:
                    tgt_indices[b, s] = candidates_list.index(target_road)
                else:
                    tgt_indices[b, s] = 0  # Default to first candidate

        return candidates_id, tgt_indices

    def forward(self, batch):
        """
        Forward pass for inference.

        Args:
            batch: Input batch dictionary

        Returns:
            matched_road_ids: Matched road segment IDs [batch, seq_len]
        """
        if self.road_graph is None or self.trace_graph is None:
            raise RuntimeError("Road graph and trace graph must be set before forward pass. "
                              "Use set_graphs() or provide graph data in data_feature.")

        traces, tgt_roads, candidates_id, trace_lens = self._prepare_batch(batch)

        batch_size = traces.size(0)
        seq_len = traces.size(1)

        # Initialize matched segments
        matched_road_segments_id = torch.full(
            (batch_size, self.match_interval, 1), -1,
            dtype=torch.long, device=self.device
        )

        # Initialize hidden states
        last_traces_encoding = None
        last_matched_road_segments_encoding = None

        # Collect all matched IDs
        all_matched_ids = []

        # Process in intervals
        for point_idx in range(self.match_interval - 1, seq_len - self.match_interval, self.match_interval):
            sub_traces = traces[:, point_idx + 1 - self.match_interval:point_idx + 1, :]
            sub_candidates = candidates_id[:, point_idx + 1 - self.match_interval:point_idx + 1, :]

            # Select actions
            traces_encoding, matched_road_segments_encoding, action = self.agent.select_action(
                last_traces_encoding,
                last_matched_road_segments_encoding,
                sub_traces,
                matched_road_segments_id,
                sub_candidates,
                self.road_graph,
                self.trace_graph
            )

            # Get matched road IDs
            cur_matched_road_segments_id = candidates_id[:, point_idx + 1 - self.match_interval, :].gather(
                -1, action[:, 0].unsqueeze(1)
            ).unsqueeze(-1)

            for i in range(1, self.match_interval):
                cur_matched_road_segments_id = torch.cat((
                    cur_matched_road_segments_id,
                    candidates_id[:, point_idx + 1 - self.match_interval + i, :].gather(
                        -1, action[:, i].unsqueeze(1)
                    ).unsqueeze(-1)
                ), dim=1)

            all_matched_ids.append(cur_matched_road_segments_id)

            # Update state
            matched_road_segments_id = cur_matched_road_segments_id
            last_traces_encoding = traces_encoding
            last_matched_road_segments_encoding = matched_road_segments_encoding

        # Concatenate all matched IDs
        if all_matched_ids:
            matched_road_ids = torch.cat(all_matched_ids, dim=1).squeeze(-1)
        else:
            matched_road_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=self.device)

        return matched_road_ids

    def predict(self, batch):
        """
        Generate predictions for evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            predictions: Predicted road segment IDs [batch, seq_len]
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)
        return predictions

    def calculate_loss(self, batch):
        """
        Calculate training loss using Double DQN + Contrastive Loss.

        This method performs one step of RL training:
        1. Process batch through the network
        2. Compute Q-values and rewards
        3. Calculate TD error (RL loss)
        4. Calculate contrastive loss
        5. Return combined loss

        Args:
            batch: Dictionary containing trajectory data and targets

        Returns:
            loss: Combined RL + contrastive loss
        """
        if self.road_graph is None or self.trace_graph is None:
            raise RuntimeError("Road graph and trace graph must be set before training. "
                              "Use set_graphs() or provide graph data in data_feature.")

        traces, tgt_roads, candidates_id, trace_lens = self._prepare_batch(batch)

        if tgt_roads is None:
            _logger.warning("No target roads in batch, returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_size = traces.size(0)
        seq_len = traces.size(1)

        # Initialize
        self.agent.reset_continuous_successes(batch_size)
        matched_road_segments_id = torch.full(
            (batch_size, self.match_interval, 1), -1,
            dtype=torch.long, device=self.device
        )

        last_traces_encoding = None
        last_matched_road_segments_encoding = None

        total_rl_loss = 0.0
        total_ctr_loss = 0.0
        num_steps = 0

        # Process in intervals and collect experiences
        for point_idx in range(self.match_interval - 1, seq_len - self.match_interval, self.match_interval):
            sub_traces = traces[:, point_idx + 1 - self.match_interval:point_idx + 1, :]
            sub_candidates = candidates_id[:, point_idx + 1 - self.match_interval:point_idx + 1, :]
            sub_tgt_roads = tgt_roads[:, point_idx + 1 - self.match_interval:point_idx + 1]

            # Forward through main network
            traces_encoding, matched_road_segments_encoding, action_values, road_emb, full_grid_emb = \
                self.agent.main_net(
                    last_traces_encoding,
                    last_matched_road_segments_encoding,
                    sub_traces,
                    matched_road_segments_id,
                    sub_candidates,
                    self.road_graph,
                    self.trace_graph
                )

            # Select actions (greedy for training)
            action = torch.argmax(action_values, dim=-1)

            # Compute rewards
            reward = self.agent.step(
                matched_road_segments_id[:, -1, :],
                self.road_graph,
                action,
                sub_candidates,
                sub_tgt_roads,
                trace_lens,
                point_idx
            )

            # Get current matched segments
            cur_matched_road_segments_id = sub_candidates[:, 0, :].gather(
                -1, action[:, 0].unsqueeze(1)
            ).unsqueeze(-1)
            for i in range(1, self.match_interval):
                cur_matched_road_segments_id = torch.cat((
                    cur_matched_road_segments_id,
                    sub_candidates[:, i, :].gather(-1, action[:, i].unsqueeze(1)).unsqueeze(-1)
                ), dim=1)

            # Prepare next state data
            next_traces = traces[:, point_idx + 1:point_idx + 1 + self.match_interval, :]
            next_candidates = candidates_id[:, point_idx + 1:point_idx + 1 + self.match_interval, :]
            next_matched_road_segments_id = cur_matched_road_segments_id

            # Get positive and negative samples for contrastive loss
            positive_samples, negative_samples = get_positive_negative_samples(
                sub_tgt_roads, sub_candidates
            )

            # Store experience
            self.agent.memory.push(
                last_traces_encoding, last_matched_road_segments_encoding,
                sub_traces, matched_road_segments_id, sub_candidates,
                positive_samples, negative_samples,
                traces_encoding, matched_road_segments_encoding,
                next_traces, next_matched_road_segments_id, next_candidates,
                None, None,
                action, reward
            )

            # Update state
            matched_road_segments_id = next_matched_road_segments_id
            last_traces_encoding = traces_encoding
            last_matched_road_segments_encoding = matched_road_segments_encoding

            # Compute Q-values for current actions
            state_action_values = action_values[:, 0, :].gather(-1, action[:, 0].unsqueeze(1))
            for i in range(1, self.match_interval):
                state_action_values = torch.cat((
                    state_action_values,
                    action_values[:, i, :].gather(-1, action[:, i].unsqueeze(1))
                ), dim=1)

            # Compute target Q-values using Double DQN
            if next_traces.size(1) == self.match_interval:
                with torch.no_grad():
                    # Get actions from main network
                    _, _, q_values_next_main, _, _ = self.agent.main_net(
                        traces_encoding, matched_road_segments_encoding,
                        next_traces, next_matched_road_segments_id, next_candidates,
                        self.road_graph, self.trace_graph
                    )
                    max_next_action = q_values_next_main.max(-1)[1]

                    # Get Q-values from target network
                    _, _, q_values_next_target, _, _ = self.agent.target_net(
                        traces_encoding, matched_road_segments_encoding,
                        next_traces, next_matched_road_segments_id, next_candidates,
                        self.road_graph, self.trace_graph
                    )

                    next_state_values = q_values_next_target[:, 0, :].gather(
                        -1, max_next_action[:, 0].unsqueeze(1)
                    )
                    for i in range(1, self.match_interval):
                        next_state_values = torch.cat((
                            next_state_values,
                            q_values_next_target[:, i, :].gather(-1, max_next_action[:, i].unsqueeze(1))
                        ), dim=1)

                    expected_state_action_values = (next_state_values * self.gamma) + reward
            else:
                expected_state_action_values = reward

            # Compute mask for valid positions
            mask = (reward != self.mask_reward).float()

            # RL loss (Smooth L1)
            rl_loss = F.smooth_l1_loss(
                state_action_values * mask,
                expected_state_action_values * mask
            )

            # Contrastive loss
            features = full_grid_emb[sub_traces[:, :, 0].long()]
            positive_features = road_emb[positive_samples.long()]
            negative_features = road_emb[negative_samples.long()]

            ctr_loss = contrastive_loss(
                features * mask.unsqueeze(-1),
                positive_features * mask.unsqueeze(-1),
                negative_features * mask.unsqueeze(-1).unsqueeze(-1)
            )

            total_rl_loss += rl_loss
            total_ctr_loss += ctr_loss
            num_steps += 1

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_interval == 0:
            self.agent.update_target_net()

        # Combine losses
        if num_steps > 0:
            avg_rl_loss = total_rl_loss / num_steps
            avg_ctr_loss = total_ctr_loss / num_steps
            loss = avg_rl_loss + self.lambda_ctr * avg_ctr_loss
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def get_memory_size(self):
        """Return current experience replay buffer size."""
        return len(self.agent.memory)

    def clear_memory(self):
        """Clear the experience replay buffer."""
        self.agent.memory.clear()
