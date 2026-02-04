"""
GraphMM: Graph-based Map Matching Model

This model is adapted from the original GraphMM implementation for map matching tasks.
GraphMM uses graph neural networks to encode both road network topology and trajectory
point relationships, then employs a sequence-to-sequence model with optional CRF layer
for map matching.

Original paper: "GraphMM: Graph-based Vehicular Map Matching by Leveraging
Trajectory and Road Correlations"

Key Components:
1. RoadGIN - Graph Isomorphism Network for road network encoding
2. TraceGCN - Directed Graph Convolutional Network for trajectory grid encoding
3. Seq2Seq - Sequence-to-sequence model with attention for decoding
4. CRF - Conditional Random Field for structured prediction (optional)

Adaptations for LibCity:
- Inherits from AbstractModel following LibCity conventions
- Implements predict() and calculate_loss() methods
- Adapts batch input format to LibCity's Batch dictionary format
- GraphData is constructed from data_feature provided by dataset
- Added configuration parameters accessible via config dictionary

Required data_feature:
- num_roads: Number of road segments in the network
- num_grids: Number of grid cells for trajectory discretization
- road_x: Road segment features tensor
- road_adj: Road network adjacency (SparseTensor)
- trace_in_edge_index: Trace graph incoming edge indices
- trace_out_edge_index: Trace graph outgoing edge indices
- trace_weight: Trace graph edge weights
- map_matrix: Grid to road mapping matrix
- singleton_grid_mask: Mask for singleton grids
- singleton_grid_location: GPS coordinates for singleton grids
- A_matrix: Road network adjacency matrix for CRF

Required config parameters:
- emb_dim: Embedding dimension (default: 256)
- dropout: Dropout probability (default: 0.5)
- use_crf: Whether to use CRF layer (default: True)
- use_attention: Whether to use attention in seq2seq (default: True)
- bidirectional: Whether to use bidirectional GRU (default: True)
- teacher_forcing_ratio: Teacher forcing ratio (default: 0.5)
- topn: Top-n candidates for CRF decoding (default: 5)
- neg_nums: Negative sampling number for CRF training (default: 800)
- layer: Number of hops for adjacency polynomial (default: 4)
- gamma: Penalty for unreachable transitions (default: 10000)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from libcity.model.abstract_model import AbstractModel

try:
    from torch_geometric.nn import GINConv, MLP, GCNConv
    from torch_sparse import SparseTensor
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class RoadGIN(nn.Module):
    """Graph Isomorphism Network for road network encoding.

    Uses multiple GIN layers with batch normalization to learn road segment
    embeddings that capture the road network topology.

    Args:
        emb_dim: Embedding dimension
        depth: Number of GIN layers (default: 3)
        mlp_layers: Number of MLP layers in each GIN (default: 2)
    """

    def __init__(self, emb_dim, depth=3, mlp_layers=2):
        super().__init__()
        self.depth = depth
        self.gins = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.depth):
            mlp = MLP(
                in_channels=emb_dim,
                hidden_channels=2 * emb_dim,
                out_channels=emb_dim,
                num_layers=mlp_layers
            )
            self.gins.append(GINConv(nn=mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, adj_t):
        """Forward pass through GIN layers.

        Args:
            x: Node features (num_roads, emb_dim)
            adj_t: Adjacency SparseTensor

        Returns:
            Road embeddings (num_roads, emb_dim)
        """
        layer_outputs = []
        for i in range(self.depth):
            x = self.gins[i](x, adj_t.to(x.device))
            x = F.relu(self.batch_norms[i](x))
            layer_outputs.append(x)
        # Max pooling across layers
        x = torch.stack(layer_outputs, dim=0)
        x = torch.max(x, dim=0)[0]
        return x


class GCNLayer(nn.Module):
    """Single GCN layer with linear transformation.

    Combines a linear transformation with graph convolution for
    directed graph processing.

    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension
        bias: Whether to use bias (default: False)
    """

    def __init__(self, in_feats, out_feats, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.gcnconv = GCNConv(
            in_channels=in_feats,
            out_channels=out_feats,
            add_self_loops=False,
            bias=bias
        )

    def forward(self, x, edge_index, edge_weight):
        """Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights

        Returns:
            Transformed features
        """
        hl = self.linear(x)
        hr = self.gcnconv(x, edge_index, edge_weight)
        return hl + hr


class DiGCN(nn.Module):
    """Directed Graph Convolutional Network.

    Stack of GCN layers for processing directed graphs.

    Args:
        embed_dim: Embedding dimension
        depth: Number of GCN layers (default: 2)
    """

    def __init__(self, embed_dim, depth=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.depth = depth

        for _ in range(self.depth):
            self.convs.append(GCNLayer(embed_dim, embed_dim))
            self.bns.append(nn.BatchNorm1d(embed_dim))

    def forward(self, x, edge_index, edge_weight):
        """Forward pass through DiGCN layers.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Edge weights

        Returns:
            Transformed node features
        """
        for idx in range(self.depth):
            x = self.convs[idx](x, edge_index, edge_weight)
            x = self.bns[idx](x)
            if idx != self.depth - 1:
                x = F.relu(x)
        return x


class TraceGCN(nn.Module):
    """Trace Graph Convolutional Network.

    Processes trajectory grids using two directed GCN branches for
    incoming and outgoing edges, then concatenates the results.

    Args:
        emb_dim: Embedding dimension
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.gcn1 = DiGCN(self.emb_dim)
        self.gcn2 = DiGCN(self.emb_dim)

    def forward(self, feats, in_edge_index, out_edge_index, edge_weight):
        """Forward pass.

        Args:
            feats: Grid features (num_grids, emb_dim)
            in_edge_index: Incoming edge indices
            out_edge_index: Outgoing edge indices
            edge_weight: Edge weights

        Returns:
            Grid embeddings (num_grids, 2*emb_dim)
        """
        emb_ind = self.gcn1(feats, in_edge_index, edge_weight)
        emb_oud = self.gcn2(feats, out_edge_index, edge_weight)
        ans = torch.cat([emb_ind, emb_oud], 1)
        return ans


class Attention(nn.Module):
    """Bahdanau-style attention for sequence-to-sequence model.

    Args:
        enc_hid_dim: Encoder hidden dimension
        dec_hid_dim: Decoder hidden dimension
    """

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        """Calculate attention weights.

        Args:
            hidden: Decoder hidden state (1, batch_size, hidden_dim)
            encoder_outputs: Encoder outputs (batch_size, src_len, hidden_dim)
            attn_mask: Attention mask (batch_size, src_len)

        Returns:
            Attention weights (batch_size, src_len)
        """
        src_len = encoder_outputs.shape[1]
        # (batch_size, src_len, hidden_dim)
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(attn_mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with attention.

    Encoder-decoder architecture using GRU cells with optional attention
    mechanism for trajectory decoding.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        atten_flag: Whether to use attention (default: True)
        bi: Whether to use bidirectional encoder (default: True)
        drop_prob: Dropout probability (default: 0.5)
    """

    def __init__(self, input_size, hidden_size, atten_flag=True, bi=True, drop_prob=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.atten_flag = atten_flag
        self.drop_prob = drop_prob
        self.bi = bi
        self.D = 2 if self.bi else 1

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.bi
        )

        dec_input_dim = hidden_size * self.D
        if self.atten_flag:
            self.attn = Attention(
                enc_hid_dim=hidden_size * self.D,
                dec_hid_dim=hidden_size
            )
            dec_input_dim += hidden_size

        self.decoder = nn.GRU(
            input_size=dec_input_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

    def encode(self, src, src_len):
        """Encode input sequence.

        Args:
            src: Input sequence (batch_size, seq_len, input_size)
            src_len: Sequence lengths (list)

        Returns:
            outputs: Encoder outputs (batch_size, seq_len, hidden_size * D)
            hiddens: Final hidden state (1, batch_size, hidden_size)
        """
        self.encoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            src, src_len, batch_first=True, enforce_sorted=False
        )
        packed_outputs, hiddens = self.encoder(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        if self.bi:
            hiddens = torch.sum(hiddens, dim=0, keepdims=True)

        return outputs, hiddens

    def decode(self, src, hidden, encoder_outputs, attn_mask):
        """Decode one step.

        Args:
            src: Decoder input (batch_size, 1, emb_dim)
            hidden: Previous hidden state (1, batch_size, hidden_size)
            encoder_outputs: Encoder outputs (batch_size, src_len, hidden_size * D)
            attn_mask: Attention mask (batch_size, src_len)

        Returns:
            outputs: Decoder output (batch_size, 1, hidden_size)
            hiddens: Updated hidden state (1, batch_size, hidden_size)
        """
        self.decoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)

        if self.atten_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            a = a.unsqueeze(1)  # (batch_size, 1, src_len)
            weighted = torch.bmm(a, encoder_outputs)  # (batch_size, 1, hidden_size * D)
            src = torch.cat((weighted, src), dim=2)

        outputs, hiddens = self.decoder(src, hidden)
        return outputs, hiddens


class CRF(nn.Module):
    """Conditional Random Field for structured prediction.

    Implements CRF with learnable transition matrix based on road embeddings
    and network topology. Uses negative sampling for efficient training.

    Args:
        num_tags: Number of road segments (output classes)
        emb_dim: Embedding dimension
        topn: Top-n candidates for Viterbi decoding
        neg_nums: Number of negative samples
        device: Computation device
        batch_first: Whether batch is first dimension (default: True)
    """

    def __init__(self, num_tags, emb_dim, topn, neg_nums, device='cpu', batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.device = device
        self.topn = topn
        self.neg_nums = neg_nums
        self.W = nn.Linear(emb_dim, emb_dim, bias=False)

    def get_transitions(self, full_road_emb, A_list):
        """Compute transition matrix.

        Args:
            full_road_emb: Road embeddings (num_roads, emb_dim)
            A_list: Adjacency polynomial matrix (num_roads, num_roads)

        Returns:
            Transition scores (num_roads, num_roads)
        """
        r = self.W(full_road_emb) @ full_road_emb.T
        energy = A_list * F.relu(r)
        return energy

    def forward(self, emissions, tags, full_road_emb, A_list, mask):
        """Compute negative log likelihood.

        Args:
            emissions: Emission scores (batch_size, seq_len, num_tags)
            tags: Target sequences (batch_size, seq_len)
            full_road_emb: Road embeddings
            A_list: Adjacency polynomial
            mask: Sequence mask (batch_size, seq_len)

        Returns:
            Negative log likelihood (scalar)
        """
        batch_size = mask.size(0)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        transitions = self.get_transitions(full_road_emb, A_list)
        numerator = self._compute_score(emissions, tags, transitions, mask)

        # Sample negative tags
        seq_ends = mask.long().sum(dim=0) - 1
        neg_tag_sets = set()
        for i in range(batch_size):
            neg_tag_sets |= set(tags[:seq_ends[i] + 1, i].detach().cpu().numpy().tolist())

        assert len(neg_tag_sets) < self.neg_nums
        remain_nums = self.neg_nums - len(neg_tag_sets)

        if remain_nums > 0:
            _, indices = torch.topk(emissions, dim=-1, k=3)
            tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
            cand_set = [i for i in tag_sets if i not in neg_tag_sets]
            cand_num = len(cand_set)
            neg_tag_sets |= set(np.random.choice(
                cand_set, min(remain_nums, cand_num), replace=False
            ).tolist())

        neg_tag_sets = sorted(list(neg_tag_sets))
        trans = transitions[neg_tag_sets, :]
        trans = trans[:, neg_tag_sets]

        denominator = self._compute_normalizer(emissions, trans, neg_tag_sets, mask)
        llh = numerator - denominator

        return llh.sum() / mask.float().sum()

    def decode(self, emissions, full_road_emb, A_list, mask):
        """Viterbi decoding for inference.

        Args:
            emissions: Emission scores (batch_size, seq_len, num_tags)
            full_road_emb: Road embeddings
            A_list: Adjacency polynomial
            mask: Sequence mask (batch_size, seq_len)

        Returns:
            Best tag sequences (list of lists)
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        transitions = self.get_transitions(full_road_emb, A_list)
        return self._viterbi_decode(emissions, transitions, mask)

    def _compute_score(self, emissions, tags, transitions, mask):
        """Compute sequence score.

        Args:
            emissions: (seq_len, batch_size, num_tags)
            tags: (seq_len, batch_size)
            transitions: (num_tags, num_tags)
            mask: (seq_len, batch_size)

        Returns:
            Sequence scores (batch_size,)
        """
        seq_length, batch_size = tags.shape
        mask = mask.float()

        score = torch.zeros(batch_size).to(self.device)
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        return score

    def _compute_normalizer(self, emissions, trans, neg_tag_sets, mask):
        """Compute log partition function.

        Args:
            emissions: (seq_len, batch_size, num_tags)
            trans: (k, k) subsampled transition matrix
            neg_tag_sets: List of sampled tag indices
            mask: (seq_len, batch_size)

        Returns:
            Log partition values (batch_size,)
        """
        seq_length = emissions.size(0)
        score = emissions[0, :, neg_tag_sets]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i, :, neg_tag_sets].unsqueeze(1)
            next_score = broadcast_score + trans + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, transitions, mask):
        """Viterbi decoding algorithm.

        Args:
            emissions: (seq_len, batch_size, num_tags)
            transitions: (num_tags, num_tags)
            mask: (seq_len, batch_size)

        Returns:
            Best tag sequences (list of lists)
        """
        seq_length, batch_size = mask.shape

        # Get top-k candidates
        _, indices = torch.topk(emissions, dim=-1, k=self.topn)
        tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
        tag_sets = sorted(tag_sets)
        tag_map = {i: tag for i, tag in enumerate(tag_sets)}

        # Subsampled transition matrix
        trans = transitions[tag_sets, :]
        trans = trans[:, tag_sets]

        score = emissions[0, :, tag_sets]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i, :, tag_sets].unsqueeze(1)
            next_score = broadcast_score + trans + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Backtrack
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags = [tag_map[t] for t in best_tags]
            tags_len = len(best_tags)
            best_tags_list.append(best_tags + [-1] * (seq_length - tags_len))

        return best_tags_list


class GraphData:
    """Container for graph-related data.

    Holds all graph structures and features needed by GraphMM model.
    Can be initialized from data_feature dictionary.

    Args:
        data_feature: Dictionary containing graph data
        device: Computation device
        layer: Number of hops for adjacency polynomial
        gamma: Penalty for unreachable transitions
    """

    def __init__(self, data_feature, device, layer=4, gamma=10000):
        self.device = device

        # Get dimensions
        self.num_roads = data_feature.get('num_roads', 1000)
        self.num_grids = data_feature.get('num_grids', 5000)

        # Road network data
        self.road_x = data_feature.get('road_x')
        if self.road_x is not None:
            self.road_x = self.road_x.to(device)

        self.road_adj = data_feature.get('road_adj')
        if self.road_adj is not None:
            self.road_adj = self.road_adj.to(device)

        # Trace graph data
        self.trace_weight = data_feature.get('trace_weight')
        if self.trace_weight is not None:
            self.trace_weight = self.trace_weight.float().to(device)

        self.trace_in_edge_index = data_feature.get('trace_in_edge_index')
        if self.trace_in_edge_index is not None:
            self.trace_in_edge_index = self.trace_in_edge_index.to(device)

        self.trace_out_edge_index = data_feature.get('trace_out_edge_index')
        if self.trace_out_edge_index is not None:
            self.trace_out_edge_index = self.trace_out_edge_index.to(device)

        # Grid-road mapping
        self.map_matrix = data_feature.get('map_matrix')
        if self.map_matrix is not None:
            self.map_matrix = self.map_matrix.to(device)

        self.singleton_grid_mask = data_feature.get('singleton_grid_mask')
        if self.singleton_grid_mask is not None:
            self.singleton_grid_mask = self.singleton_grid_mask.to(device)

        self.singleton_grid_location = data_feature.get('singleton_grid_location')
        if self.singleton_grid_location is not None:
            self.singleton_grid_location = self.singleton_grid_location.to(device)

        # Adjacency polynomial
        A = data_feature.get('A_matrix')
        if A is not None:
            self.A_list = self._get_adj_poly(A, layer, gamma)
        else:
            # Fall back to pre-computed A_list from dataset
            self.A_list = data_feature.get('A_list')
            if self.A_list is not None:
                self.A_list = self.A_list.to(device)

    def _get_adj_poly(self, A, layer, gamma):
        """Compute adjacency polynomial A^k with penalty.

        Args:
            A: Adjacency matrix
            layer: Number of hops
            gamma: Penalty for unreachable pairs

        Returns:
            Adjacency polynomial with penalties
        """
        A_ = A.to(self.device)
        ans = A_.clone()
        for _ in range(layer - 1):
            ans = ans @ A_
        ans[ans != 0] = 1.
        ans[ans == 0] = -gamma
        return ans


class GraphMM(AbstractModel):
    """
    GraphMM: Graph-based Map Matching Model for LibCity.

    This model uses graph neural networks to encode road network and trajectory
    relationships, then applies a sequence-to-sequence decoder with optional CRF
    for structured prediction of road segment sequences.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including graph structures

    Required config parameters:
        - emb_dim: Embedding dimension (default: 256)
        - dropout: Dropout probability (default: 0.5)
        - use_crf: Whether to use CRF layer (default: True)
        - use_attention: Whether to use attention in seq2seq (default: True)
        - bidirectional: Whether to use bidirectional GRU (default: True)
        - teacher_forcing_ratio: Teacher forcing ratio (default: 0.5)
        - topn: Top-n candidates for CRF decoding (default: 5)
        - neg_nums: Negative sampling number for CRF training (default: 800)
        - layer: Number of hops for adjacency polynomial (default: 4)
        - gamma: Penalty for unreachable transitions (default: 10000)

    Required data_feature:
        - num_roads: Number of road segments
        - num_grids: Number of grid cells
        - road_x: Road features tensor (num_roads, 28)
        - road_adj: Road network adjacency (SparseTensor)
        - trace_in_edge_index, trace_out_edge_index, trace_weight: Trace graph
        - map_matrix: Grid to road mapping matrix
        - singleton_grid_mask: Mask for singleton grids
        - singleton_grid_location: GPS for singleton grids (num_singletons, 4)
        - A_matrix: Adjacency matrix for CRF
    """

    def __init__(self, config, data_feature):
        super(GraphMM, self).__init__(config, data_feature)

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "GraphMM requires torch_geometric and torch_sparse. "
                "Please install them with: pip install torch_geometric torch_sparse"
            )

        self._logger = getLogger()
        self.device = config.get('device', 'cpu')

        # Model hyperparameters
        self.emb_dim = config.get('emb_dim', 256)
        self.dropout = config.get('dropout', 0.5)
        self.use_crf = config.get('use_crf', True)
        self.use_attention = config.get('use_attention', True)
        self.bidirectional = config.get('bidirectional', True)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.topn = config.get('topn', 5)
        self.neg_nums = config.get('neg_nums', 800)
        self.layer = config.get('layer', 4)
        self.gamma = config.get('gamma', 10000)

        # Data dimensions
        self.num_roads = data_feature.get('num_roads', 1000)
        self.num_grids = data_feature.get('num_grids', 5000)

        # Build graph data container
        self.gdata = GraphData(
            data_feature=data_feature,
            device=self.device,
            layer=self.layer,
            gamma=self.gamma
        )

        # Build model components
        self._build_model()

        self._logger.info(
            f"GraphMM initialized: emb_dim={self.emb_dim}, "
            f"num_roads={self.num_roads}, num_grids={self.num_grids}, "
            f"use_crf={self.use_crf}"
        )

    def _build_model(self):
        """Build all model components."""

        # Road network encoder
        self.road_gin = RoadGIN(self.emb_dim)

        # Trace graph encoder
        self.trace_gcn = TraceGCN(self.emb_dim)

        # Sequence-to-sequence decoder
        self.seq2seq = Seq2Seq(
            input_size=2 * self.emb_dim,
            hidden_size=self.emb_dim,
            atten_flag=self.use_attention,
            bi=self.bidirectional,
            drop_prob=self.dropout
        )

        # Feature projection layers
        # Road features: 28 dimensions (3*8 + 4) in original
        road_feat_dim = 28
        if self.gdata.road_x is not None:
            road_feat_dim = self.gdata.road_x.size(-1)
        self.road_feat_fc = nn.Linear(road_feat_dim, self.emb_dim)

        # Trace features: 4 dimensions (GPS coordinates) for singleton grids
        trace_feat_dim = 4
        if self.gdata.singleton_grid_location is not None:
            trace_feat_dim = self.gdata.singleton_grid_location.size(-1)
        self.trace_feat_fc = nn.Linear(trace_feat_dim, self.emb_dim)

        # Input projection: 2*emb_dim (grid emb) + 2 (GPS) + 1 (sample idx) = 2*emb_dim + 3
        self.fc_input = nn.Linear(2 * self.emb_dim + 3, 2 * self.emb_dim)

        # CRF layer (optional)
        if self.use_crf:
            self.crf = CRF(
                num_tags=self.num_roads,
                emb_dim=self.emb_dim,
                topn=self.topn,
                neg_nums=self.neg_nums,
                device=self.device
            )

    def get_emb(self):
        """Get road and grid embeddings.

        Returns:
            full_road_emb: Road embeddings (num_roads, emb_dim)
            full_grid_emb: Grid embeddings (num_grids+1, 2*emb_dim)
        """
        # Road embedding
        road_x = self.road_feat_fc(self.gdata.road_x)
        full_road_emb = self.road_gin(road_x, self.gdata.road_adj)

        # Grid embedding from road mapping
        pure_grid_feat = torch.mm(self.gdata.map_matrix, full_road_emb)

        # Handle singleton grids (grids not mapped to roads)
        if self.gdata.singleton_grid_mask is not None:
            pure_grid_feat[self.gdata.singleton_grid_mask] = self.trace_feat_fc(
                self.gdata.singleton_grid_location
            )

        # Full grid embedding with trace GCN
        full_grid_emb = torch.zeros(
            self.gdata.num_grids + 1, 2 * self.emb_dim
        ).to(self.device)

        full_grid_emb[1:, :] = self.trace_gcn(
            pure_grid_feat,
            self.gdata.trace_in_edge_index,
            self.gdata.trace_out_edge_index,
            self.gdata.trace_weight
        )

        return full_road_emb, full_grid_emb

    def get_probs(self, grid_traces, tgt_roads, traces_gps, sample_idx,
                  trace_lens, road_lens, tf_ratio, full_road_emb, full_grid_emb):
        """Decode to get emission probabilities.

        Args:
            grid_traces: Grid cell IDs (batch_size, seq_len)
            tgt_roads: Target road IDs (batch_size, tgt_len) or None
            traces_gps: GPS coordinates (batch_size, seq_len, 2)
            sample_idx: Sampling indices (batch_size, seq_len)
            trace_lens: Actual trace lengths (list)
            road_lens: Actual road sequence lengths (list)
            tf_ratio: Teacher forcing ratio
            full_road_emb: Road embeddings
            full_grid_emb: Grid embeddings

        Returns:
            Emission probabilities (batch_size, max_road_len, num_roads)
        """
        if tgt_roads is not None:
            B, max_RL = tgt_roads.shape
        else:
            B = grid_traces.shape[0]
            max_RL = int(max(road_lens))

        # Get input embeddings
        rnn_input = full_grid_emb[grid_traces]

        # Concatenate with GPS and sample index
        rnn_input = torch.cat([
            rnn_input,
            traces_gps,
            sample_idx.unsqueeze(-1).float()
        ], dim=-1)
        rnn_input = self.fc_input(rnn_input)

        # Encode trajectory
        encoder_outputs, hiddens = self.seq2seq.encode(rnn_input, trace_lens)

        # Initialize decoder
        probs = torch.zeros(B, max_RL, self.num_roads).to(self.device)
        inputs = torch.zeros(B, 1, self.seq2seq.hidden_size).to(self.device)

        # Attention mask
        attn_mask = None
        if self.use_attention:
            attn_mask = torch.zeros(B, int(max(trace_lens))).to(self.device)
            for i in range(len(trace_lens)):
                attn_mask[i, :trace_lens[i]] = 1.

        # First decode step
        inputs, hiddens = self.seq2seq.decode(inputs, hiddens, encoder_outputs, attn_mask)
        probs[:, 0, :] = inputs.squeeze(1) @ full_road_emb.detach().T

        # Teacher forcing decision
        teacher_force = random.random() < tf_ratio
        if teacher_force and tgt_roads is not None:
            lst_road_id = tgt_roads[:, 0]
        else:
            lst_road_id = probs[:, 0, :].argmax(1)

        # Decode remaining steps
        for t in range(1, max_RL):
            if teacher_force and tgt_roads is not None:
                inputs = full_road_emb[lst_road_id].view(B, 1, -1)
            inputs, hiddens = self.seq2seq.decode(inputs, hiddens, encoder_outputs, attn_mask)
            probs[:, t, :] = inputs.squeeze(1) @ full_road_emb.detach().T

            teacher_force = random.random() < tf_ratio
            if teacher_force and tgt_roads is not None:
                lst_road_id = tgt_roads[:, t]
            else:
                lst_road_id = probs[:, t, :].argmax(1)

        return probs

    def forward(self, batch, tf_ratio=None):
        """Forward pass for training.

        Args:
            batch: LibCity Batch object containing:
                - 'grid_traces': Grid cell IDs (batch_size, trace_len)
                - 'tgt_roads': Target road sequence (batch_size, road_len)
                - 'traces_gps': GPS coordinates (batch_size, trace_len, 2)
                - 'sample_idx': Sampling indices (batch_size, trace_len)
                - 'trace_lens': Actual trace lengths
                - 'road_lens': Actual road sequence lengths
            tf_ratio: Teacher forcing ratio (default: self.teacher_forcing_ratio)

        Returns:
            loss: Training loss
        """
        if tf_ratio is None:
            tf_ratio = self.teacher_forcing_ratio if self.training else 0.0

        # Extract data from batch
        grid_traces = batch['grid_traces'].to(self.device)
        tgt_roads = batch['tgt_roads'].to(self.device)
        traces_gps = batch['traces_gps'].float().to(self.device)
        sample_idx = batch['sample_Idx'].to(self.device)

        # Get sequence lengths
        if 'traces_lens' in batch:
            trace_lens = batch['traces_lens']
            if isinstance(trace_lens, torch.Tensor):
                trace_lens = trace_lens.tolist()
        else:
            trace_lens = [grid_traces.size(1)] * grid_traces.size(0)

        if 'road_lens' in batch:
            road_lens = batch['road_lens']
            if isinstance(road_lens, torch.Tensor):
                road_lens = road_lens.tolist()
        else:
            road_lens = [tgt_roads.size(1)] * tgt_roads.size(0)

        # Get embeddings
        full_road_emb, full_grid_emb = self.get_emb()

        # Get emission probabilities
        emissions = self.get_probs(
            grid_traces=grid_traces,
            tgt_roads=tgt_roads,
            traces_gps=traces_gps,
            sample_idx=sample_idx,
            trace_lens=trace_lens,
            road_lens=road_lens,
            tf_ratio=tf_ratio,
            full_road_emb=full_road_emb,
            full_grid_emb=full_grid_emb
        )

        # Calculate loss
        if self.use_crf:
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens))).to(self.device)
            for i in range(len(road_lens)):
                tgt_mask[i, :road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool()
            loss = -self.crf(emissions, tgt_roads, full_road_emb.detach(),
                            self.gdata.A_list, tgt_mask)
        else:
            mask = (tgt_roads.view(-1) != -1)
            loss = F.cross_entropy(
                emissions.view(-1, self.num_roads)[mask],
                tgt_roads.view(-1)[mask]
            )

        return loss

    def infer(self, batch, tf_ratio=0.0):
        """Inference to get predicted road sequences.

        Args:
            batch: LibCity Batch object
            tf_ratio: Teacher forcing ratio (default: 0.0 for inference)

        Returns:
            predictions: Predicted road sequences (batch_size, max_road_len)
                        or probabilities if not using CRF
        """
        # Extract data from batch
        grid_traces = batch['grid_traces'].to(self.device)
        traces_gps = batch['traces_gps'].float().to(self.device)
        sample_idx = batch['sample_Idx'].to(self.device)

        # Get sequence lengths
        if 'traces_lens' in batch:
            trace_lens = batch['traces_lens']
            if isinstance(trace_lens, torch.Tensor):
                trace_lens = trace_lens.tolist()
        else:
            trace_lens = [grid_traces.size(1)] * grid_traces.size(0)

        if 'road_lens' in batch:
            road_lens = batch['road_lens']
            if isinstance(road_lens, torch.Tensor):
                road_lens = road_lens.tolist()
        else:
            # Estimate road lens if not provided
            road_lens = trace_lens

        # Get embeddings
        full_road_emb, full_grid_emb = self.get_emb()

        # Get emission probabilities
        emissions = self.get_probs(
            grid_traces=grid_traces,
            tgt_roads=None,
            traces_gps=traces_gps,
            sample_idx=sample_idx,
            trace_lens=trace_lens,
            road_lens=road_lens,
            tf_ratio=tf_ratio,
            full_road_emb=full_road_emb,
            full_grid_emb=full_grid_emb
        )

        # Decode
        if self.use_crf:
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens))).to(self.device)
            for i in range(len(road_lens)):
                tgt_mask[i, :road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool()
            preds = self.crf.decode(emissions, full_road_emb, self.gdata.A_list, tgt_mask)
            return preds
        else:
            preds = F.softmax(emissions, dim=-1)
            return preds

    def predict(self, batch):
        """Prediction method for LibCity evaluation.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted road sequences or probabilities
        """
        self.eval()
        with torch.no_grad():
            return self.infer(batch)

    def calculate_loss(self, batch):
        """Calculate training loss for LibCity.

        Args:
            batch: LibCity Batch object

        Returns:
            loss: Training loss tensor
        """
        return self.forward(batch)
