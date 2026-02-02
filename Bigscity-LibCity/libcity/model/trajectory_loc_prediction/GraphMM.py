"""
GraphMM: Graph-based Map Matching Model

This model is adapted from the original GraphMM (GMM) implementation for the LibCity framework.

Original Paper:
"GraphMM: Graph-based Vehicular Map Matching by Leveraging Trajectory and Road Correlations"

Key Components:
1. RoadGIN: Graph Isomorphism Network encoder for road network graph
2. TraceGCN: Directed GCN encoder for trajectory trace graph
3. Seq2Seq: Sequence-to-sequence decoder with attention mechanism
4. CRF: Optional Conditional Random Field for structured prediction

Task:
Map matching - matching GPS trajectory points to road segments in a road network.

Adaptations for LibCity:
- Replaced standalone module classes with integrated class structure
- Adapted __init__() to use LibCity's config and data_feature pattern
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Extracted hyperparameters to config parameters
- Handled device management through LibCity's config
- Preserved dual-graph architecture (road graph + trace graph)
- Maintained PyG compatibility (GINConv, GCNConv layers)

Original files:
- repos/GraphMM/model/gmm.py (GMM main model)
- repos/GraphMM/model/road_gin.py (RoadGIN encoder)
- repos/GraphMM/model/trace_gcn.py (TraceGCN encoder)
- repos/GraphMM/model/seq2seq.py (Seq2Seq decoder with attention)
- repos/GraphMM/model/crf.py (CRF layer)
- repos/GraphMM/graph_data.py (GraphData container)

Dependencies:
- torch-geometric (for GINConv, GCNConv, MLP layers)
- torch-sparse (for SparseTensor)
- Standard PyTorch modules
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINConv, GCNConv, MLP
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

try:
    from torch_sparse import SparseTensor
    HAS_TORCH_SPARSE = True
except ImportError:
    HAS_TORCH_SPARSE = False

from libcity.model.abstract_model import AbstractModel


# ========================== RoadGIN Encoder ==========================

class RoadGIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder for road network graph.

    Uses multiple GIN layers to encode road segment features, with max-pooling
    over layer outputs to capture multi-hop neighborhood information.

    Args:
        emb_dim: Embedding dimension for road features
        depth: Number of GIN layers (default: 3)
        mlp_layers: Number of layers in MLP for each GIN layer (default: 2)
    """

    def __init__(self, emb_dim, depth=3, mlp_layers=2):
        super().__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for RoadGIN. "
                "Please install it: pip install torch-geometric"
            )

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
        """
        Forward pass through GIN layers.

        Args:
            x: Node features of shape (num_roads, emb_dim)
            adj_t: Sparse adjacency matrix (SparseTensor or edge_index)

        Returns:
            Encoded road embeddings of shape (num_roads, emb_dim)
        """
        layer_outputs = []
        for i in range(self.depth):
            x = self.gins[i](x, adj_t.to(x.device))
            x = F.relu(self.batch_norms[i](x))
            layer_outputs.append(x)

        # Max pooling over layer outputs
        x = torch.stack(layer_outputs, dim=0)
        x = torch.max(x, dim=0)[0]
        return x


# ========================== TraceGCN Encoder ==========================

class GCNLayer(nn.Module):
    """
    Single GCN layer combining linear transformation and graph convolution.

    Args:
        in_feats: Input feature dimension
        out_feats: Output feature dimension
        bias: Whether to use bias (default: False)
    """

    def __init__(self, in_feats, out_feats, bias=False):
        super(GCNLayer, self).__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for GCNLayer. "
                "Please install it: pip install torch-geometric"
            )

        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.gcnconv = GCNConv(
            in_channels=in_feats,
            out_channels=out_feats,
            add_self_loops=False,
            bias=bias
        )

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass combining linear and GCN transformations.

        Args:
            x: Node features of shape (num_nodes, in_feats)
            edge_index: Edge indices of shape (2, num_edges)
            edge_weight: Edge weights of shape (num_edges,)

        Returns:
            Transformed features of shape (num_nodes, out_feats)
        """
        hl = self.linear(x)
        hr = self.gcnconv(x, edge_index, edge_weight)
        return hl + hr


class DiGCN(nn.Module):
    """
    Directed Graph Convolutional Network with multiple layers.

    Args:
        embed_dim: Embedding dimension
        depth: Number of GCN layers (default: 2)
    """

    def __init__(self, embed_dim, depth=2):
        super(DiGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.depth = depth

        for _ in range(self.depth):
            self.convs.append(GCNLayer(embed_dim, embed_dim))
            self.bns.append(nn.BatchNorm1d(embed_dim))

    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass through DiGCN layers.

        Args:
            x: Node features of shape (num_nodes, embed_dim)
            edge_index: Edge indices of shape (2, num_edges)
            edge_weight: Edge weights of shape (num_edges,)

        Returns:
            Encoded features of shape (num_nodes, embed_dim)
        """
        for idx in range(self.depth):
            x = self.convs[idx](x, edge_index, edge_weight)
            x = self.bns[idx](x)
            if idx != self.depth - 1:
                x = F.relu(x)
        return x


class TraceGCN(nn.Module):
    """
    Trace Graph Convolutional Network for encoding trajectory trace graphs.

    Uses two separate DiGCN encoders for incoming and outgoing edges,
    then concatenates the results for bidirectional representation.

    Args:
        emb_dim: Embedding dimension
    """

    def __init__(self, emb_dim):
        super(TraceGCN, self).__init__()
        self.emb_dim = emb_dim
        self.gcn1 = DiGCN(self.emb_dim)
        self.gcn2 = DiGCN(self.emb_dim)

    def forward(self, feats, in_edge_index, out_edge_index, edge_weight):
        """
        Forward pass through TraceGCN.

        Args:
            feats: Node features of shape (num_grids, emb_dim)
            in_edge_index: Incoming edge indices of shape (2, num_edges)
            out_edge_index: Outgoing edge indices of shape (2, num_edges)
            edge_weight: Edge weights of shape (num_edges,)

        Returns:
            Encoded features of shape (num_grids, 2*emb_dim)
        """
        emb_ind = self.gcn1(feats, in_edge_index, edge_weight)
        emb_oud = self.gcn2(feats, out_edge_index, edge_weight)
        ans = torch.cat([emb_ind, emb_oud], 1)
        return ans


# ========================== Attention Mechanism ==========================

class Attention(nn.Module):
    """
    Attention mechanism for Seq2Seq decoder.

    Computes attention weights between decoder hidden state and encoder outputs.

    Args:
        enc_hid_dim: Encoder hidden dimension
        dec_hid_dim: Decoder hidden dimension
    """

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        """
        Compute attention weights.

        Args:
            hidden: Decoder hidden state of shape (1, batch_size, hidden_dim)
            encoder_outputs: Encoder outputs of shape (batch_size, src_len, hidden_dim)
            attn_mask: Attention mask of shape (batch_size, src_len)

        Returns:
            Attention weights of shape (batch_size, src_len)
        """
        src_len = encoder_outputs.shape[1]
        # Repeat hidden for each source position
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)

        # Compute attention energy
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )
        attention = self.v(energy).squeeze(2)

        # Apply mask to force attention only over non-padding elements
        attention = attention.masked_fill(attn_mask == 0, -1e10)

        return F.softmax(attention, dim=1)


# ========================== Seq2Seq Decoder ==========================

class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model with optional attention for decoding.

    Uses bidirectional GRU encoder and unidirectional GRU decoder.

    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        atten_flag: Whether to use attention (default: True)
        bi: Whether encoder is bidirectional (default: True)
        drop_prob: Dropout probability (default: 0.5)
    """

    def __init__(self, input_size, hidden_size, atten_flag=True, bi=True, drop_prob=0.5):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.atten_flag = atten_flag
        self.drop_prob = drop_prob
        self.bi = bi
        self.D = 2 if self.bi else 1

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.bi
        )

        # Decoder input dimension depends on attention
        dec_input_dim = hidden_size * self.D
        if self.atten_flag:
            self.attn = Attention(
                enc_hid_dim=hidden_size * self.D,
                dec_hid_dim=hidden_size
            )
            dec_input_dim += hidden_size

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=dec_input_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

    def encode(self, src, src_len):
        """
        Encode source sequence.

        Args:
            src: Source sequence of shape (batch_size, seq_len, input_size)
            src_len: List of actual sequence lengths

        Returns:
            outputs: Encoder outputs of shape (batch_size, seq_len, hidden_size * D)
            hiddens: Final hidden state of shape (1, batch_size, hidden_size)
        """
        self.encoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)

        # Pack sequence for efficient computation
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            src, src_len, batch_first=True, enforce_sorted=False
        )
        packed_outputs, hiddens = self.encoder(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # Sum bidirectional hidden states
        if self.bi:
            hiddens = torch.sum(hiddens, dim=0, keepdims=True)

        return outputs, hiddens

    def decode(self, src, hidden, encoder_outputs, attn_mask):
        """
        Decode one step.

        Args:
            src: Input embedding of shape (batch_size, 1, emb_dim)
            hidden: Previous hidden state of shape (1, batch_size, hidden_size)
            encoder_outputs: Encoder outputs of shape (batch_size, src_len, hidden_size * D)
            attn_mask: Attention mask of shape (batch_size, src_len)

        Returns:
            outputs: Decoder output of shape (batch_size, 1, hidden_size)
            hiddens: Updated hidden state of shape (1, batch_size, hidden_size)
        """
        self.decoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)

        if self.atten_flag:
            # Compute attention
            a = self.attn(hidden, encoder_outputs, attn_mask)
            a = a.unsqueeze(1)
            # Compute weighted context
            weighted = torch.bmm(a, encoder_outputs)
            src = torch.cat((weighted, src), dim=2)

        outputs, hiddens = self.decoder(src, hidden)
        return outputs, hiddens


# ========================== CRF Layer ==========================

class CRF(nn.Module):
    """
    Conditional Random Field for structured prediction.

    Uses road embeddings and adjacency matrix to compute transition scores.
    Supports negative sampling for efficient training.

    Args:
        num_tags: Number of road segment tags
        emb_dim: Embedding dimension
        topn: Top-N candidates for Viterbi decoding
        neg_nums: Number of negative samples for training
        device: Computation device
        batch_first: Whether input is batch-first (default: True)
    """

    def __init__(self, num_tags, emb_dim, topn, neg_nums, device='cpu', batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.device = device
        self.topn = topn
        self.neg_nums = neg_nums

        # Transition weight matrix
        self.W = nn.Linear(emb_dim, emb_dim, bias=False)

    def get_transitions(self, full_road_emb, A_list):
        """
        Compute transition scores between road segments.

        Args:
            full_road_emb: Road embeddings of shape (num_roads, emb_dim)
            A_list: Adjacency matrix with penalties for unreachable roads

        Returns:
            Transition scores of shape (num_roads, num_roads)
        """
        r = self.W(full_road_emb) @ full_road_emb.T
        energy = A_list * F.relu(r)
        return energy

    def forward(self, emissions, tags, full_road_emb, A_list, mask):
        """
        Compute the conditional log likelihood.

        Args:
            emissions: Emission scores of shape (batch_size, seq_length, num_tags)
            tags: Ground truth tags of shape (batch_size, seq_length)
            full_road_emb: Road embeddings of shape (num_roads, emb_dim)
            A_list: Adjacency penalty matrix
            mask: Sequence mask of shape (batch_size, seq_length)

        Returns:
            Negative log likelihood (to be maximized)
        """
        batch_size = mask.size(0)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Get transition matrix
        transitions = self.get_transitions(full_road_emb, A_list)

        # Compute numerator (score of correct sequence)
        numerator = self._compute_score(emissions, tags, transitions, mask)

        # Sample negative tags for efficient normalizer computation
        seq_ends = mask.long().sum(dim=0) - 1
        neg_tag_sets = set()

        for i in range(batch_size):
            neg_tag_sets |= set(tags[:seq_ends[i] + 1, i].detach().cpu().numpy().tolist())

        assert len(neg_tag_sets) < self.neg_nums
        remain_nums = self.neg_nums - len(neg_tag_sets)

        # Sample additional tags from top-k emissions
        if remain_nums > 0:
            _, indices = torch.topk(emissions, dim=-1, k=3)
            tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
            cand_set = [i for i in tag_sets if i not in neg_tag_sets]
            cand_num = len(cand_set)
            neg_tag_sets |= set(
                np.random.choice(cand_set, min(remain_nums, cand_num), replace=False).tolist()
            )

        neg_tag_sets = sorted(list(neg_tag_sets))
        trans = transitions[neg_tag_sets, :]
        trans = trans[:, neg_tag_sets]

        # Compute denominator (log partition function)
        denominator = self._compute_normalizer(emissions, trans, neg_tag_sets, mask)

        # Log likelihood
        llh = numerator - denominator

        return llh.sum() / mask.float().sum()

    def decode(self, emissions, full_road_emb, A_list, mask):
        """
        Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores of shape (batch_size, seq_length, num_tags)
            full_road_emb: Road embeddings of shape (num_roads, emb_dim)
            A_list: Adjacency penalty matrix
            mask: Sequence mask of shape (batch_size, seq_length)

        Returns:
            List of best tag sequences for each batch element
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        transitions = self.get_transitions(full_road_emb, A_list)
        return self._viterbi_decode(emissions, transitions, mask)

    def _compute_score(self, emissions, tags, transitions, mask):
        """
        Compute the score of the correct tag sequence.

        Args:
            emissions: (seq_length, batch_size, num_tags)
            tags: (seq_length, batch_size)
            transitions: (num_tags, num_tags)
            mask: (seq_length, batch_size)

        Returns:
            Score tensor of shape (batch_size,)
        """
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start with first emission score
        score = torch.zeros(batch_size).to(self.device)
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Add transition score (masked)
            score += transitions[tags[i - 1], tags[i]] * mask[i]
            # Add emission score (masked)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        return score

    def _compute_normalizer(self, emissions, trans, neg_tag_sets, mask):
        """
        Compute the log partition function using forward algorithm.

        Args:
            emissions: (seq_length, batch_size, num_tags)
            trans: Subsampled transition matrix (k, k)
            neg_tag_sets: List of sampled tag indices
            mask: (seq_length, batch_size)

        Returns:
            Log partition function of shape (batch_size,)
        """
        seq_length = emissions.size(0)

        # Initialize with first emission
        score = emissions[0, :, neg_tag_sets]

        for i in range(1, seq_length):
            # Broadcast for efficient computation
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i, :, neg_tag_sets].unsqueeze(1)

            # Compute next scores
            next_score = broadcast_score + trans + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)

            # Apply mask
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, transitions, mask):
        """
        Viterbi decoding to find best tag sequence.

        Args:
            emissions: (seq_length, batch_size, num_tags)
            transitions: (num_tags, num_tags)
            mask: (seq_length, batch_size)

        Returns:
            List of best tag sequences
        """
        seq_length, batch_size = mask.shape

        # Select top-k candidates for efficient decoding
        _, indices = torch.topk(emissions, dim=-1, k=self.topn)
        tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
        tag_sets = sorted(tag_sets)
        tag_map = {i: tag for i, tag in enumerate(tag_sets)}

        # Extract subsampled transition matrix
        trans = transitions[tag_sets, :]
        trans = trans[:, tag_sets]

        # Initialize with first emission
        score = emissions[0, :, tag_sets]
        history = []

        # Viterbi forward pass
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i, :, tag_sets].unsqueeze(1)

            next_score = broadcast_score + trans + broadcast_emission
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Backtrack to find best sequences
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # Trace back through history
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags = [tag_map[t] for t in best_tags]
            tags_len = len(best_tags)
            best_tags_list.append(best_tags + [-1] * (seq_length - tags_len))

        return best_tags_list


# ========================== GraphData Container ==========================

class GraphData:
    """
    Container for graph-related data used by GraphMM.

    Stores road graph, trace graph, and various mappings needed for
    the dual-graph map matching model.

    This class is instantiated from data_feature in the model's __init__.
    """

    def __init__(self, device='cpu'):
        self.device = device

        # Road graph data
        self.num_roads = 0
        self.road_x = None
        self.road_adj = None
        self.A_list = None

        # Trace graph data
        self.num_grids = 0
        self.trace_weight = None
        self.trace_in_edge_index = None
        self.trace_out_edge_index = None

        # Grid-road mapping
        self.map_matrix = None
        self.singleton_grid_mask = None
        self.singleton_grid_location = None

    @classmethod
    def from_data_feature(cls, data_feature, layer=4, gamma=10000, device='cpu'):
        """
        Create GraphData from LibCity data_feature dictionary.

        Args:
            data_feature: Dictionary containing graph data
            layer: Number of hops for adjacency polynomial
            gamma: Penalty for unreachable roads
            device: Computation device

        Returns:
            Initialized GraphData instance
        """
        gdata = cls(device=device)

        # Load road graph data
        gdata.num_roads = data_feature.get('num_roads', data_feature.get('loc_size', 1000))

        road_x = data_feature.get('road_x', data_feature.get('road_features', None))
        if road_x is not None:
            if not isinstance(road_x, torch.Tensor):
                road_x = torch.FloatTensor(np.asarray(road_x))
            gdata.road_x = road_x.to(device)
        else:
            # Default road features (identity matrix or random)
            gdata.road_x = torch.eye(gdata.num_roads, 28).to(device)

        # Road adjacency
        road_adj = data_feature.get('road_adj', data_feature.get('road_edge_index', None))
        if road_adj is not None:
            # Check if road_adj is in edge index format (2, num_edges)
            is_edge_index_format = False
            if isinstance(road_adj, tuple):
                is_edge_index_format = True
            elif isinstance(road_adj, torch.Tensor) and road_adj.dim() == 2 and road_adj.shape[0] == 2:
                is_edge_index_format = True
            elif isinstance(road_adj, np.ndarray) and road_adj.ndim == 2 and road_adj.shape[0] == 2:
                is_edge_index_format = True

            if is_edge_index_format:
                # Edge index format
                if isinstance(road_adj, tuple):
                    row, col = road_adj
                else:
                    row, col = road_adj[0], road_adj[1]
                # Ensure row and col are torch tensors
                if not isinstance(row, torch.Tensor):
                    row = torch.LongTensor(np.asarray(row))
                if not isinstance(col, torch.Tensor):
                    col = torch.LongTensor(np.asarray(col))
                if HAS_TORCH_SPARSE:
                    gdata.road_adj = SparseTensor(
                        row=row, col=col,
                        sparse_sizes=(gdata.num_roads, gdata.num_roads)
                    ).to(device)
                else:
                    gdata.road_adj = torch.stack([row, col]).to(device)
            else:
                # Dense adjacency matrix format
                if not isinstance(road_adj, torch.Tensor):
                    road_adj = torch.FloatTensor(np.asarray(road_adj))
                gdata.road_adj = road_adj.to(device)
        else:
            # Default: self-loops
            indices = torch.arange(gdata.num_roads)
            if HAS_TORCH_SPARSE:
                gdata.road_adj = SparseTensor(
                    row=indices, col=indices,
                    sparse_sizes=(gdata.num_roads, gdata.num_roads)
                ).to(device)
            else:
                gdata.road_adj = torch.stack([indices, indices]).to(device)

        # Load trace graph data
        gdata.num_grids = data_feature.get('num_grids', gdata.num_roads)

        trace_weight = data_feature.get('trace_weight', None)
        if trace_weight is not None:
            if not isinstance(trace_weight, torch.Tensor):
                trace_weight = torch.FloatTensor(np.asarray(trace_weight))
            gdata.trace_weight = trace_weight.to(device)

        trace_in_edge_index = data_feature.get('trace_in_edge_index', None)
        if trace_in_edge_index is not None:
            if not isinstance(trace_in_edge_index, torch.Tensor):
                trace_in_edge_index = torch.LongTensor(np.asarray(trace_in_edge_index))
            gdata.trace_in_edge_index = trace_in_edge_index.to(device)

        trace_out_edge_index = data_feature.get('trace_out_edge_index', None)
        if trace_out_edge_index is not None:
            if not isinstance(trace_out_edge_index, torch.Tensor):
                trace_out_edge_index = torch.LongTensor(np.asarray(trace_out_edge_index))
            gdata.trace_out_edge_index = trace_out_edge_index.to(device)

        # Grid-road mapping
        map_matrix = data_feature.get('map_matrix', data_feature.get('grid_road_map', None))
        if map_matrix is not None:
            if not isinstance(map_matrix, torch.Tensor):
                map_matrix = torch.FloatTensor(np.asarray(map_matrix))
            gdata.map_matrix = map_matrix.to(device)

        singleton_grid_mask = data_feature.get('singleton_grid_mask', None)
        if singleton_grid_mask is not None:
            if not isinstance(singleton_grid_mask, torch.Tensor):
                singleton_grid_mask = torch.BoolTensor(np.asarray(singleton_grid_mask))
            gdata.singleton_grid_mask = singleton_grid_mask.to(device)

        singleton_grid_location = data_feature.get('singleton_grid_location', None)
        if singleton_grid_location is not None:
            if not isinstance(singleton_grid_location, torch.Tensor):
                singleton_grid_location = torch.FloatTensor(np.asarray(singleton_grid_location))
            gdata.singleton_grid_location = singleton_grid_location.to(device)

        # Compute A^k adjacency polynomial
        A = data_feature.get('A', data_feature.get('adjacency_matrix', None))
        if A is not None:
            if not isinstance(A, torch.Tensor):
                A = torch.FloatTensor(np.asarray(A))
            gdata.A_list = gdata._get_adj_poly(A, layer, gamma, device)
        else:
            # Default: identity with penalty
            gdata.A_list = torch.eye(gdata.num_roads).to(device)

        return gdata

    def _get_adj_poly(self, A, layer, gamma, device):
        """
        Compute adjacency polynomial A^k with penalty for unreachable roads.

        Args:
            A: Base adjacency matrix
            layer: Power k for A^k
            gamma: Penalty value for zero entries
            device: Computation device

        Returns:
            Processed adjacency polynomial with penalties
        """
        A_ = A.to(device)
        ans = A_.clone()
        for _ in range(layer - 1):
            ans = ans @ A_
        ans[ans != 0] = 1.
        ans[ans == 0] = -gamma
        return ans


# ========================== Main GraphMM Model ==========================

class GraphMM(AbstractModel):
    """
    GraphMM: Graph-based Map Matching Model for LibCity.

    This model performs map matching by leveraging dual-graph encoders
    (road network graph + trajectory trace graph) with a Seq2Seq decoder
    and optional CRF for structured prediction.

    Key Features:
    1. RoadGIN: Encodes road network structure using Graph Isomorphism Network
    2. TraceGCN: Encodes trajectory trace graph using bidirectional GCN
    3. Seq2Seq: Decodes road segment sequence with attention mechanism
    4. CRF: Optional structured prediction for sequence decoding

    Args:
        config: Configuration dictionary containing model hyperparameters
        data_feature: Data features including graph data, vocab sizes, etc.

    Config Parameters:
        - emb_dim: Embedding dimension (default: 256)
        - layer: K-hop neighbors for adjacency polynomial (default: 4)
        - tf_ratio: Teacher forcing ratio during training (default: 0.5)
        - drop_prob: Dropout probability (default: 0.5)
        - gamma: Penalty for unreachable roads in CRF (default: 10000)
        - topn: Top-N candidates for CRF Viterbi decoding (default: 5)
        - neg_nums: Number of negative samples for CRF training (default: 800)
        - use_crf: Whether to use CRF layer (default: True)
        - bi: Whether to use bidirectional GRU (default: True)
        - atten_flag: Whether to use attention in Seq2Seq (default: True)
        - road_feat_dim: Road feature dimension (default: 28)
        - gin_depth: Depth of RoadGIN (default: 3)
        - gin_mlp_layers: MLP layers in RoadGIN (default: 2)
        - digcn_depth: Depth of DiGCN in TraceGCN (default: 2)

    Data Feature Requirements:
        - num_roads / loc_size: Number of road segments
        - num_grids: Number of grid cells (for trace graph)
        - road_x / road_features: Road segment features
        - road_adj / road_edge_index: Road graph adjacency
        - trace_in_edge_index: Trace graph incoming edges
        - trace_out_edge_index: Trace graph outgoing edges
        - trace_weight: Trace graph edge weights
        - map_matrix: Grid-to-road mapping matrix
        - A / adjacency_matrix: Base adjacency matrix for A^k computation
    """

    def __init__(self, config, data_feature):
        super(GraphMM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Model hyperparameters from config
        self.emb_dim = config.get('emb_dim', 256)
        self.layer = config.get('layer', 4)
        self.tf_ratio = config.get('tf_ratio', 0.5)
        self.drop_prob = config.get('drop_prob', 0.5)
        self.gamma = config.get('gamma', 10000)
        self.topn = config.get('topn', 5)
        self.neg_nums = config.get('neg_nums', 800)
        self.use_crf = config.get('use_crf', True)
        self.bi = config.get('bi', True)
        self.atten_flag = config.get('atten_flag', True)

        # Feature dimensions
        self.road_feat_dim = config.get('road_feat_dim', 28)
        self.trace_feat_dim = config.get('trace_feat_dim', 4)
        self.gps_feat_dim = config.get('gps_feat_dim', 2)

        # Architecture parameters
        self.gin_depth = config.get('gin_depth', 3)
        self.gin_mlp_layers = config.get('gin_mlp_layers', 2)
        self.digcn_depth = config.get('digcn_depth', 2)

        # Data dimensions from data_feature
        self.num_roads = data_feature.get('num_roads', data_feature.get('loc_size', 1000))
        self.target_size = self.num_roads

        # Initialize graph data container
        self.gdata = GraphData.from_data_feature(
            data_feature,
            layer=self.layer,
            gamma=self.gamma,
            device=self.device
        )

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""

        # Feature projections
        self.road_feat_fc = nn.Linear(self.road_feat_dim, self.emb_dim)
        self.trace_feat_fc = nn.Linear(self.trace_feat_dim, self.emb_dim)

        # Encoders
        self.road_gin = RoadGIN(
            self.emb_dim,
            depth=self.gin_depth,
            mlp_layers=self.gin_mlp_layers
        )
        self.trace_gcn = TraceGCN(self.emb_dim)

        # Seq2Seq decoder
        self.seq2seq = Seq2Seq(
            input_size=2 * self.emb_dim,
            hidden_size=self.emb_dim,
            atten_flag=self.atten_flag,
            bi=self.bi,
            drop_prob=self.drop_prob
        )

        # Input fusion layer (grid embedding + GPS + sample index)
        self.fc_input = nn.Linear(2 * self.emb_dim + 3, 2 * self.emb_dim)

        # Optional CRF layer
        if self.use_crf:
            self.crf = CRF(
                num_tags=self.target_size,
                emb_dim=self.emb_dim,
                topn=self.topn,
                neg_nums=self.neg_nums,
                device=self.device
            )

    def get_emb(self, gdata):
        """
        Get road embeddings and grid embeddings through graph encoders.

        Args:
            gdata: GraphData container with graph information

        Returns:
            full_road_emb: Road embeddings of shape (num_roads, emb_dim)
            full_grid_emb: Grid embeddings of shape (num_grids+1, 2*emb_dim)
        """
        # Encode road network
        road_x = self.road_feat_fc(gdata.road_x)
        full_road_emb = self.road_gin(road_x, gdata.road_adj)

        # Compute grid features from road embeddings
        if gdata.map_matrix is not None:
            pure_grid_feat = torch.mm(gdata.map_matrix, full_road_emb)
        else:
            pure_grid_feat = full_road_emb[:gdata.num_grids]

        # Handle singleton grids (grids not mapped to any road)
        if gdata.singleton_grid_mask is not None and gdata.singleton_grid_location is not None:
            pure_grid_feat[gdata.singleton_grid_mask] = self.trace_feat_fc(
                gdata.singleton_grid_location
            )

        # Encode trace graph
        full_grid_emb = torch.zeros(gdata.num_grids + 1, 2 * self.emb_dim).to(self.device)

        if (gdata.trace_in_edge_index is not None and
            gdata.trace_out_edge_index is not None and
            gdata.trace_weight is not None):
            full_grid_emb[1:, :] = self.trace_gcn(
                pure_grid_feat,
                gdata.trace_in_edge_index,
                gdata.trace_out_edge_index,
                gdata.trace_weight
            )
        else:
            # Fallback: use road embeddings directly
            full_grid_emb[1:gdata.num_grids+1, :self.emb_dim] = pure_grid_feat
            full_grid_emb[1:gdata.num_grids+1, self.emb_dim:] = pure_grid_feat

        return full_road_emb, full_grid_emb

    def get_probs(self, grid_traces, tgt_roads, traces_gps, sample_Idx,
                  trace_lens, road_lens, tf_ratio, full_road_emb,
                  full_grid_emb, gdata):
        """
        Decode trajectory to get emission probabilities for each road segment.

        Args:
            grid_traces: Grid cell IDs of trajectory points (batch_size, seq_len)
            tgt_roads: Ground truth road segments (batch_size, seq_len1) or None for inference
            traces_gps: GPS coordinates (batch_size, seq_len, 2)
            sample_Idx: Sample indices (batch_size, seq_len)
            trace_lens: List of actual trajectory lengths
            road_lens: List of actual road sequence lengths
            tf_ratio: Teacher forcing ratio
            full_road_emb: Road embeddings (num_roads, emb_dim)
            full_grid_emb: Grid embeddings (num_grids+1, 2*emb_dim)
            gdata: GraphData container

        Returns:
            Emission probabilities of shape (batch_size, max_road_len, num_roads)
        """
        if tgt_roads is not None:
            B, max_RL = tgt_roads.shape
        else:
            B = grid_traces.shape[0]
            max_RL = int(max(road_lens))

        # Get grid embeddings for trajectory points
        rnn_input = full_grid_emb[grid_traces]

        # Concatenate with GPS and sample index
        rnn_input = torch.cat([rnn_input, traces_gps, sample_Idx.unsqueeze(-1)], dim=-1)
        rnn_input = self.fc_input(rnn_input)

        # Encode trajectory sequence
        encoder_outputs, hiddens = self.seq2seq.encode(rnn_input, trace_lens)

        # Initialize decoder
        probs = torch.zeros(B, max_RL, gdata.num_roads).to(self.device)
        inputs = torch.zeros(B, 1, self.seq2seq.hidden_size).to(self.device)

        # Create attention mask
        attn_mask = None
        if self.atten_flag:
            attn_mask = torch.zeros(B, int(max(trace_lens)))
            for i in range(len(trace_lens)):
                attn_mask[i][:trace_lens[i]] = 1.
            attn_mask = attn_mask.to(self.device)

        # Decode first step
        inputs, hiddens = self.seq2seq.decode(inputs, hiddens, encoder_outputs, attn_mask)
        probs[:, 0, :] = inputs.squeeze(1) @ full_road_emb.detach().T

        # Determine teacher forcing for first step
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

            # Determine teacher forcing for next step
            teacher_force = random.random() < tf_ratio
            if teacher_force and tgt_roads is not None:
                lst_road_id = tgt_roads[:, t]
            else:
                lst_road_id = probs[:, t, :].argmax(1)

        return probs

    def forward(self, batch):
        """
        Forward pass for training.

        Args:
            batch: LibCity Batch object containing:
                - 'grid_traces': Grid cell IDs (batch_size, seq_len)
                - 'tgt_roads' / 'target': Ground truth roads (batch_size, road_len)
                - 'traces_gps' / 'gps': GPS coordinates (batch_size, seq_len, 2)
                - 'sample_Idx': Sample indices (batch_size, seq_len)
                - 'traces_lens': List of trajectory lengths
                - 'road_lens': List of road sequence lengths

        Returns:
            Loss tensor
        """
        # Extract data from batch
        grid_traces = batch.get('grid_traces', batch.get('X', None))
        tgt_roads = batch.get('tgt_roads', batch.get('target', batch.get('y', None)))
        traces_gps = batch.get('traces_gps', batch.get('gps', None))
        sample_Idx = batch.get('sample_Idx', batch.get('sample_idx', None))
        traces_lens = batch.get('traces_lens', batch.get('trace_lens', None))
        road_lens = batch.get('road_lens', None)

        # Get teacher forcing ratio (may be overridden in batch)
        tf_ratio = batch.get('tf_ratio', self.tf_ratio)

        # Convert to tensors if needed
        if not isinstance(grid_traces, torch.Tensor):
            grid_traces = torch.LongTensor(grid_traces)
        grid_traces = grid_traces.to(self.device)

        if not isinstance(tgt_roads, torch.Tensor):
            tgt_roads = torch.LongTensor(tgt_roads)
        tgt_roads = tgt_roads.to(self.device)

        if not isinstance(traces_gps, torch.Tensor):
            traces_gps = torch.FloatTensor(traces_gps)
        traces_gps = traces_gps.to(self.device)

        if sample_Idx is not None:
            if not isinstance(sample_Idx, torch.Tensor):
                sample_Idx = torch.LongTensor(sample_Idx)
            sample_Idx = sample_Idx.to(self.device)
        else:
            # Create default sample indices (all zeros or sequential)
            sample_Idx = torch.zeros_like(grid_traces).to(self.device)

        # Convert length lists
        if isinstance(traces_lens, torch.Tensor):
            traces_lens = traces_lens.tolist()
        if isinstance(road_lens, torch.Tensor):
            road_lens = road_lens.tolist()

        # Infer lengths if not provided
        if traces_lens is None:
            traces_lens = [grid_traces.shape[1]] * grid_traces.shape[0]
        if road_lens is None:
            road_lens = [tgt_roads.shape[1]] * tgt_roads.shape[0]

        # Get embeddings
        full_road_emb, full_grid_emb = self.get_emb(self.gdata)

        # Get emission probabilities
        emissions = self.get_probs(
            grid_traces=grid_traces,
            tgt_roads=tgt_roads,
            traces_gps=traces_gps,
            trace_lens=traces_lens,
            road_lens=road_lens,
            tf_ratio=tf_ratio,
            full_grid_emb=full_grid_emb,
            gdata=self.gdata,
            sample_Idx=sample_Idx,
            full_road_emb=full_road_emb
        )

        # Compute loss
        if self.use_crf:
            # Create target mask
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
            for i in range(len(road_lens)):
                tgt_mask[i][:road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool().to(self.device)

            # CRF loss (negative log likelihood)
            loss = -self.crf(
                emissions, tgt_roads, full_road_emb.detach(),
                self.gdata.A_list, tgt_mask
            )
        else:
            # Cross-entropy loss with mask
            mask = (tgt_roads.view(-1) != -1)
            loss = F.cross_entropy(
                emissions.view(-1, self.target_size)[mask],
                tgt_roads.view(-1)[mask]
            )

        return loss

    def predict(self, batch):
        """
        Prediction method for inference.

        Args:
            batch: LibCity Batch object containing trajectory data

        Returns:
            Predicted road segment sequences (list of lists or tensor)
        """
        # Extract data from batch
        grid_traces = batch.get('grid_traces', batch.get('X', None))
        traces_gps = batch.get('traces_gps', batch.get('gps', None))
        sample_Idx = batch.get('sample_Idx', batch.get('sample_idx', None))
        traces_lens = batch.get('traces_lens', batch.get('trace_lens', None))
        road_lens = batch.get('road_lens', None)

        # Convert to tensors if needed
        if not isinstance(grid_traces, torch.Tensor):
            grid_traces = torch.LongTensor(grid_traces)
        grid_traces = grid_traces.to(self.device)

        if not isinstance(traces_gps, torch.Tensor):
            traces_gps = torch.FloatTensor(traces_gps)
        traces_gps = traces_gps.to(self.device)

        if sample_Idx is not None:
            if not isinstance(sample_Idx, torch.Tensor):
                sample_Idx = torch.LongTensor(sample_Idx)
            sample_Idx = sample_Idx.to(self.device)
        else:
            sample_Idx = torch.zeros_like(grid_traces).to(self.device)

        # Convert length lists
        if isinstance(traces_lens, torch.Tensor):
            traces_lens = traces_lens.tolist()
        if isinstance(road_lens, torch.Tensor):
            road_lens = road_lens.tolist()

        if traces_lens is None:
            traces_lens = [grid_traces.shape[1]] * grid_traces.shape[0]
        if road_lens is None:
            road_lens = traces_lens  # Use trace lens as estimate

        # Get embeddings
        full_road_emb, full_grid_emb = self.get_emb(self.gdata)

        # Get emission probabilities (no teacher forcing)
        emissions = self.get_probs(
            grid_traces=grid_traces,
            tgt_roads=None,
            traces_gps=traces_gps,
            trace_lens=traces_lens,
            road_lens=road_lens,
            tf_ratio=0.,  # No teacher forcing during inference
            full_grid_emb=full_grid_emb,
            sample_Idx=sample_Idx,
            gdata=self.gdata,
            full_road_emb=full_road_emb
        )

        # Decode predictions
        if self.use_crf:
            # Create target mask
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
            for i in range(len(road_lens)):
                tgt_mask[i][:road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool().to(self.device)

            # Viterbi decoding
            preds = self.crf.decode(
                emissions, full_road_emb, self.gdata.A_list, tgt_mask
            )
        else:
            # Argmax decoding
            preds = F.softmax(emissions, dim=-1)

        return preds

    def calculate_loss(self, batch):
        """
        Calculate training loss for LibCity.

        Args:
            batch: LibCity Batch object containing trajectory and target data

        Returns:
            Loss tensor
        """
        return self.forward(batch)

    def infer(self, grid_traces, traces_gps, traces_lens, road_lens,
              sample_Idx, gdata, tf_ratio=0.):
        """
        Direct inference method (for compatibility with original interface).

        Args:
            grid_traces: Grid cell IDs (batch_size, seq_len)
            traces_gps: GPS coordinates (batch_size, seq_len, 2)
            traces_lens: List of trajectory lengths
            road_lens: List of road sequence lengths
            sample_Idx: Sample indices (batch_size, seq_len)
            gdata: GraphData container (can be None to use self.gdata)
            tf_ratio: Teacher forcing ratio (default: 0)

        Returns:
            Predicted road segment sequences
        """
        if gdata is None:
            gdata = self.gdata

        # Get embeddings
        full_road_emb, full_grid_emb = self.get_emb(gdata)

        # Get emission probabilities
        emissions = self.get_probs(
            grid_traces=grid_traces,
            tgt_roads=None,
            traces_gps=traces_gps,
            trace_lens=traces_lens,
            road_lens=road_lens,
            tf_ratio=tf_ratio,
            full_grid_emb=full_grid_emb,
            sample_Idx=sample_Idx,
            gdata=gdata,
            full_road_emb=full_road_emb
        )

        if self.use_crf:
            tgt_mask = torch.zeros(emissions.shape[0], int(max(road_lens)))
            for i in range(len(road_lens)):
                tgt_mask[i][:road_lens[i]] = 1.
            tgt_mask = tgt_mask.bool().to(self.device)
            preds = self.crf.decode(emissions, full_road_emb, gdata.A_list, tgt_mask)
        else:
            preds = F.softmax(emissions, dim=-1)

        return preds

    def update_gdata(self, data_feature):
        """
        Update graph data from new data features.

        Useful when switching datasets or updating graph information.

        Args:
            data_feature: New data feature dictionary
        """
        self.gdata = GraphData.from_data_feature(
            data_feature,
            layer=self.layer,
            gamma=self.gamma,
            device=self.device
        )
        self.num_roads = self.gdata.num_roads
        self.target_size = self.num_roads
