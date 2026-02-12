"""
GraphMM: Graph-based Map Matching Model

Original Repository: repos/GraphMM
Paper: GraphMM - Graph-based Vehicular Map Matching

This model performs map matching of GPS trajectories to road segments using:
1. RoadGIN: Graph Isomorphism Network for road network embedding
2. TraceGCN: Directed GCN for GPS trace grid graph embedding
3. Seq2Seq: Sequence-to-sequence decoder with attention for road prediction
4. CRF: Conditional Random Field for structured prediction (optional)

Architecture Overview:
    GPS Trace -> Grid Embedding (TraceGCN) -> Seq2Seq Encoder
    Road Network -> Road Embedding (RoadGIN) -> Transition Matrix (CRF)
    Seq2Seq Decoder -> Emission Scores -> CRF Decode -> Road Sequence

Task: Map Matching
Base Class: AbstractModel (for neural map matching models)

Adaptations for LibCity:
- Inherits from AbstractModel
- Implements forward(), predict(), calculate_loss() methods
- Accepts batch dict from DeepMapMatchingDataset:
    batch = {
        'grid_traces':  [batch, trace_len]        grid cell IDs (1-indexed, 0=pad)
        'tgt_roads':    [batch, road_len]          target road IDs (-1=pad)
        'traces_gps':   [batch, trace_len, 2]      GPS coordinates (lat, lon)
        'sample_Idx':   [batch, trace_len]          alignment indices
        'traces_lens':  list[int]                   actual trace lengths
        'road_lens':    list[int]                   actual road lengths
    }
- Graph data (road_adj, trace edges, etc.) loaded from data_feature at init

Required data_feature keys:
    num_roads, num_grids, road_adj, road_x,
    trace_in_edge_index, trace_out_edge_index, trace_weight,
    map_matrix, A_list,
    singleton_grid_mask, singleton_grid_location

Required config parameters:
    emb_dim (default 128), topn (default 30), neg_nums (default 100),
    atten_flag (default True), drop_prob (default 0.5), bi (default True),
    use_crf (default True), tf_ratio (default 0.5),
    road_feature_dim (default 28), trace_feature_dim (default 4),
    gin_depth (default 3), gin_mlp_layers (default 2), digcn_depth (default 2)
"""

import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel

# Optional imports for graph neural networks
try:
    from torch_geometric.nn import GINConv, MLP, GCNConv
    from torch_sparse import SparseTensor
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    logging.warning(
        "PyTorch Geometric not found. GraphMM will use fallback MLP layers "
        "instead of GINConv/GCNConv. Install torch-geometric for full functionality."
    )

_logger = logging.getLogger(__name__)


# =============================================================================
# Component 1: RoadGIN -- Graph Isomorphism Network for road graph encoding
# Original: repos/GraphMM/model/road_gin.py
# =============================================================================

class RoadGIN(nn.Module):
    """
    Road Graph Encoder using Graph Isomorphism Network.

    Encodes road network structure via multiple GIN layers with batch
    normalization and max-pooling across layers for feature aggregation.

    Args:
        emb_dim: Embedding dimension for road features
        depth: Number of GIN layers (default 3)
        mlp_layers: Number of MLP layers inside each GINConv (default 2)
    """

    def __init__(self, emb_dim, depth=3, mlp_layers=2):
        super().__init__()
        self.depth = depth
        self.emb_dim = emb_dim

        if HAS_PYGEOMETRIC:
            self.gins = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            for _ in range(self.depth):
                mlp = MLP(
                    in_channels=emb_dim,
                    hidden_channels=2 * emb_dim,
                    out_channels=emb_dim,
                    num_layers=mlp_layers,
                )
                self.gins.append(GINConv(nn=mlp, train_eps=True))
                self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        else:
            # Fallback: simple MLP layers when PyG is not available
            self.layers = nn.ModuleList()
            for _ in range(self.depth):
                self.layers.append(nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                ))

    def forward(self, x, adj_t):
        """
        Args:
            x: Road node features [num_roads, emb_dim]
            adj_t: Sparse adjacency (SparseTensor or edge_index)

        Returns:
            x: Road embeddings [num_roads, emb_dim] (max-pooled across layers)
        """
        layer_outputs = []
        if HAS_PYGEOMETRIC:
            for i in range(self.depth):
                x = self.gins[i](x, adj_t.to(x.device))
                x = F.relu(self.batch_norms[i](x))
                layer_outputs.append(x)
        else:
            for i in range(self.depth):
                x = self.layers[i](x)
                layer_outputs.append(x)

        x = torch.stack(layer_outputs, dim=0)
        x = torch.max(x, dim=0)[0]
        return x


# =============================================================================
# Component 2: TraceGCN -- Directed GCN for GPS trace graph encoding
# Original: repos/GraphMM/model/trace_gcn.py
# =============================================================================

class GCNLayer(nn.Module):
    """Single GCN layer: linear + graph convolution (additive combination)."""

    def __init__(self, in_feats, out_feats, bias=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias)
        if HAS_PYGEOMETRIC:
            self.gcnconv = GCNConv(
                in_channels=in_feats,
                out_channels=out_feats,
                add_self_loops=False,
                bias=bias,
            )
        else:
            self.gcn_linear = nn.Linear(in_feats, out_feats, bias)

    def forward(self, x, edge_index, edge_weight=None):
        hl = self.linear(x)
        if HAS_PYGEOMETRIC:
            hr = self.gcnconv(x, edge_index, edge_weight)
        else:
            hr = self.gcn_linear(x)
        return hl + hr


class DiGCN(nn.Module):
    """Directed Graph Convolutional Network (multi-layer)."""

    def __init__(self, embed_dim, depth=2):
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
    Trace Graph Encoder using bi-directional GCN.

    Separately encodes incoming and outgoing edge flows, then
    concatenates embeddings -> [num_grids, 2 * emb_dim].
    """

    def __init__(self, emb_dim, depth=2):
        super(TraceGCN, self).__init__()
        self.emb_dim = emb_dim
        self.gcn1 = DiGCN(self.emb_dim, depth)
        self.gcn2 = DiGCN(self.emb_dim, depth)

    def forward(self, feats, in_edge_index, out_edge_index, edge_weight=None):
        """
        Args:
            feats: Grid node features [num_grids, emb_dim]
            in_edge_index: Incoming edges [2, E]
            out_edge_index: Outgoing edges [2, E]
            edge_weight: Edge weights [E]

        Returns:
            Concatenated embeddings [num_grids, 2 * emb_dim]
        """
        emb_ind = self.gcn1(feats, in_edge_index, edge_weight)
        emb_oud = self.gcn2(feats, out_edge_index, edge_weight)
        return torch.cat([emb_ind, emb_oud], dim=1)


# =============================================================================
# Component 3: Seq2Seq -- Encoder-Decoder with Attention
# Original: repos/GraphMM/model/seq2seq.py
# =============================================================================

class Seq2SeqAttention(nn.Module):
    """Bahdanau-style attention for the Seq2Seq decoder."""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        """
        Args:
            hidden: Decoder hidden [1, batch, hidden_dim]
            encoder_outputs: Encoder outputs [batch, src_len, enc_dim]
            attn_mask: [batch, src_len] (1 for valid, 0 for pad)

        Returns:
            Attention weights [batch, src_len]
        """
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(src_len, 1, 1).permute(1, 0, 2)
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    """
    GRU-based Seq2Seq model with optional bidirectional encoder and
    Bahdanau attention.

    The encoder processes the grid-embedded trajectory and the decoder
    auto-regressively produces road segment logits at each time step.
    """

    def __init__(self, input_size, hidden_size, atten_flag=True,
                 bi=True, drop_prob=0.5):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.atten_flag = atten_flag
        self.drop_prob = drop_prob
        self.bi = bi
        self.D = 2 if self.bi else 1

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.bi,
        )

        dec_input_dim = hidden_size * self.D
        if self.atten_flag:
            self.attn = Seq2SeqAttention(
                enc_hid_dim=hidden_size * self.D,
                dec_hid_dim=hidden_size,
            )
            dec_input_dim += hidden_size

        self.decoder = nn.GRU(
            input_size=dec_input_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def encode(self, src, src_len):
        """
        Encode the trajectory sequence.

        Args:
            src: [batch, seq_len, input_size]
            src_len: list of actual lengths

        Returns:
            outputs: [batch, seq_len, hidden_size * D]
            hiddens: [1, batch, hidden_size]
        """
        self.encoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)
        packed = nn.utils.rnn.pack_padded_sequence(
            src, src_len, batch_first=True, enforce_sorted=False,
        )
        packed_outputs, hiddens = self.encoder(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True,
        )
        if self.bi:
            hiddens = torch.sum(hiddens, dim=0, keepdims=True)
        return outputs, hiddens

    def decode(self, src, hidden, encoder_outputs, attn_mask):
        """
        One decoding step.

        Args:
            src: [batch, 1, emb_dim]
            hidden: [1, batch, hidden_size]
            encoder_outputs: [batch, src_len, D * hidden_size]
            attn_mask: [batch, src_len]

        Returns:
            outputs: [batch, 1, hidden_size]
            hiddens: [1, batch, hidden_size]
        """
        self.decoder.flatten_parameters()
        src = F.dropout(src, self.drop_prob, training=self.training)
        if self.atten_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            a = a.unsqueeze(1)
            weighted = torch.bmm(a, encoder_outputs)
            src = torch.cat((weighted, src), dim=2)
        outputs, hiddens = self.decoder(src, hidden)
        return outputs, hiddens


# =============================================================================
# Component 4: CRF -- Conditional Random Field for structured prediction
# Original: repos/GraphMM/model/crf.py
# =============================================================================

class CRF(nn.Module):
    """
    Conditional Random Field for map matching.

    Learns transition potentials between road segments based on road
    embeddings and the adjacency polynomial A^k.  Uses negative sampling
    for efficient partition function computation.
    """

    def __init__(self, num_tags, emb_dim, topn, neg_nums,
                 device='cpu', batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.device = device
        self.topn = topn
        self.neg_nums = neg_nums
        self.W = nn.Linear(emb_dim, emb_dim, bias=False)

    def get_transitions(self, full_road_emb, A_list):
        """Compute pairwise transition scores."""
        r = self.W(full_road_emb) @ full_road_emb.T
        energy = A_list * F.relu(r)
        return energy

    def forward(self, emissions, tags, full_road_emb, A_list, mask):
        """
        Compute the conditional log-likelihood.

        Args:
            emissions: [batch, seq_len, num_tags]
            tags: [batch, seq_len]
            full_road_emb: [num_tags, emb_dim]
            A_list: [num_tags, num_tags] adjacency polynomial
            mask: [batch, seq_len] boolean

        Returns:
            Negative log-likelihood (scalar, to be minimized)
        """
        batch_size = mask.size(0)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        transitions = self.get_transitions(full_road_emb, A_list)
        numerator = self._compute_score(emissions, tags, transitions, mask)

        # Negative sampling for partition function
        seq_ends = mask.long().sum(dim=0) - 1
        neg_tag_sets = set()
        for i in range(batch_size):
            neg_tag_sets |= set(
                tags[:seq_ends[i] + 1, i].detach().cpu().numpy().tolist()
            )
        # If neg_nums is larger than current set, sample more from top-k
        remain_nums = self.neg_nums - len(neg_tag_sets)
        if remain_nums > 0:
            _, indices = torch.topk(emissions, dim=-1, k=min(3, emissions.size(-1)))
            tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
            cand_set = [i for i in tag_sets if i not in neg_tag_sets]
            cand_num = len(cand_set)
            if cand_num > 0:
                neg_tag_sets |= set(
                    np.random.choice(
                        cand_set, min(remain_nums, cand_num), replace=False
                    ).tolist()
                )
        neg_tag_sets = sorted(list(neg_tag_sets))
        trans = transitions[neg_tag_sets, :][:, neg_tag_sets]

        denominator = self._compute_normalizer(emissions, trans, neg_tag_sets, mask)
        llh = numerator - denominator
        return llh.sum() / mask.float().sum()

    def decode(self, emissions, full_road_emb, A_list, mask):
        """
        Viterbi decoding: find the most likely tag sequence.

        Returns:
            List of lists -- best tag sequence per sample.
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        transitions = self.get_transitions(full_road_emb, A_list)
        return self._viterbi_decode(emissions, transitions, mask)

    # ----- internal helpers -----

    def _compute_score(self, emissions, tags, transitions, mask):
        seq_length, batch_size = tags.shape
        mask = mask.float()
        score = torch.zeros(batch_size).to(self.device)
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        return score

    def _compute_normalizer(self, emissions, trans, neg_tag_sets, mask):
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
        seq_length, batch_size = mask.shape

        _, indices = torch.topk(emissions, dim=-1, k=min(self.topn, emissions.size(-1)))
        tag_sets = indices.flatten().unique().detach().cpu().numpy().tolist()
        tag_sets = sorted(tag_sets)
        tag_map = {i: tag for i, tag in enumerate(tag_sets)}

        trans = transitions[tag_sets, :][:, tag_sets]
        score = emissions[0, :, tag_sets]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i, :, tag_sets].unsqueeze(1)
            next_score = broadcast_score + trans + broadcast_emission
            next_score, idx = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(idx)

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        for b in range(batch_size):
            _, best_last_tag = score[b].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(history[:seq_ends[b]]):
                best_last_tag = hist[b][best_tags[-1]]
                best_tags.append(best_last_tag.item())
            best_tags.reverse()
            best_tags = [tag_map[t] for t in best_tags]
            tags_len = len(best_tags)
            best_tags_list.append(best_tags + [-1] * (seq_length - tags_len))
        return best_tags_list


# =============================================================================
# GraphData container -- holds pre-computed graph tensors on device
# =============================================================================

class GraphDataContainer:
    """
    Lightweight container that mirrors the original GraphData class.

    All tensors are moved to the target device at construction time so that
    the model can reference them directly during forward passes.
    """

    def __init__(self, data_feature, device):
        self.device = device
        self.num_roads = data_feature.get('num_roads', 0)
        self.num_grids = data_feature.get('num_grids', 0)

        # Road graph tensors
        self.road_x = self._to_device(data_feature.get('road_x'))
        self.road_adj = self._to_device(data_feature.get('road_adj'))
        self.A_list = self._to_device(data_feature.get('A_list'))

        # Trace graph tensors
        self.trace_weight = self._to_device(data_feature.get('trace_weight'))
        self.trace_in_edge_index = self._to_device(
            data_feature.get('trace_in_edge_index')
        )
        self.trace_out_edge_index = self._to_device(
            data_feature.get('trace_out_edge_index')
        )

        # Grid-road mapping
        self.map_matrix = self._to_device(data_feature.get('map_matrix'))
        self.singleton_grid_mask = self._to_device(
            data_feature.get('singleton_grid_mask')
        )
        self.singleton_grid_location = self._to_device(
            data_feature.get('singleton_grid_location')
        )

    def _to_device(self, tensor):
        if tensor is None:
            return None
        if hasattr(tensor, 'to'):
            return tensor.to(self.device)
        return tensor


# =============================================================================
# GraphMM -- Main LibCity model class
# =============================================================================

class GraphMM(AbstractModel):
    """
    GraphMM: Graph-based Map Matching Model adapted for LibCity.

    Wraps the original GMM architecture (RoadGIN + TraceGCN + Seq2Seq + CRF)
    and exposes the standard LibCity interface:
        - forward(batch)        -> emissions tensor
        - predict(batch)        -> predicted road IDs [batch, road_len]
        - calculate_loss(batch) -> scalar loss

    Args:
        config: LibCity config dict
        data_feature: dict returned by DeepMapMatchingDataset.get_data_feature()
    """

    def __init__(self, config, data_feature):
        super(GraphMM, self).__init__(config, data_feature)

        # -- Device --
        self.device = config.get('device', torch.device('cpu'))

        # -- Hyper-parameters (with defaults matching GraphMM.json) --
        self.emb_dim = config.get('emb_dim', 128)
        self.topn = config.get('topn', 30)
        self.neg_nums = config.get('neg_nums', 100)
        self.atten_flag = config.get('atten_flag', True)
        self.drop_prob = config.get('drop_prob', 0.5)
        self.bi = config.get('bi', True)
        self.use_crf = config.get('use_crf', True)
        self.tf_ratio = config.get('tf_ratio', 0.5)
        self.road_feature_dim = config.get('road_feature_dim', 28)
        self.trace_feature_dim = config.get('trace_feature_dim', 4)
        self.gin_depth = config.get('gin_depth', 3)
        self.gin_mlp_layers = config.get('gin_mlp_layers', 2)
        self.digcn_depth = config.get('digcn_depth', 2)

        # -- Data features --
        self.num_roads = data_feature.get('num_roads', 0)
        self.num_grids = data_feature.get('num_grids', 0)
        self.target_size = self.num_roads  # Number of road classes

        if self.num_roads == 0:
            raise ValueError(
                "data_feature must contain 'num_roads' (> 0). "
                "Ensure DeepMapMatchingDataset is configured correctly."
            )

        # -- Build graph data container (moves tensors to device) --
        self.gdata = GraphDataContainer(data_feature, self.device)

        # -- Sub-modules --
        # 1. Road embedding: project raw features -> emb_dim, then GIN
        self.road_gin = RoadGIN(
            self.emb_dim, depth=self.gin_depth, mlp_layers=self.gin_mlp_layers,
        )
        # 2. Trace graph embedding
        self.trace_gcn = TraceGCN(self.emb_dim, depth=self.digcn_depth)

        # 3. Feature projection layers
        # road_x has self.road_feature_dim features (default 28 = 3*8+4)
        self.road_feat_fc = nn.Linear(self.road_feature_dim, self.emb_dim)
        # singleton grid location has self.trace_feature_dim features (default 4)
        self.trace_feat_fc = nn.Linear(self.trace_feature_dim, self.emb_dim)
        # Concatenation of grid embedding (2*emb_dim) + gps (2) + sample_idx (1) -> 2*emb_dim
        self.fc_input = nn.Linear(2 * self.emb_dim + 3, 2 * self.emb_dim)

        # 4. Seq2Seq (input = 2 * emb_dim after fc_input)
        self.seq2seq = Seq2Seq(
            input_size=2 * self.emb_dim,
            hidden_size=self.emb_dim,
            atten_flag=self.atten_flag,
            bi=self.bi,
            drop_prob=self.drop_prob,
        )

        # 5. CRF (optional)
        if self.use_crf:
            self.crf = CRF(
                num_tags=self.target_size,
                emb_dim=self.emb_dim,
                topn=self.topn,
                neg_nums=self.neg_nums,
                device=self.device,
            )

        _logger.info(
            "GraphMM initialized: emb_dim=%d, num_roads=%d, num_grids=%d, "
            "use_crf=%s, atten_flag=%s, bi=%s",
            self.emb_dim, self.num_roads, self.num_grids,
            self.use_crf, self.atten_flag, self.bi,
        )

    # -----------------------------------------------------------------
    # Graph embedding helpers (mirrors original GMM.get_emb)
    # -----------------------------------------------------------------

    def _get_embeddings(self):
        """
        Compute road and grid embeddings from the stored graph data.

        Returns:
            full_road_emb: [num_roads, emb_dim]
            full_grid_emb: [num_grids + 1, 2 * emb_dim]
                           Index 0 is a zero-vector padding token.
        """
        gdata = self.gdata

        # Road embedding
        road_x = self.road_feat_fc(gdata.road_x)
        full_road_emb = self.road_gin(road_x, gdata.road_adj)

        # Grid embedding from road embedding via mapping matrix
        pure_grid_feat = torch.mm(gdata.map_matrix, full_road_emb)

        # Override singleton grids (grids with no road mapping)
        if (gdata.singleton_grid_mask is not None and
                gdata.singleton_grid_location is not None and
                gdata.singleton_grid_mask.numel() > 0):
            pure_grid_feat[gdata.singleton_grid_mask] = self.trace_feat_fc(
                gdata.singleton_grid_location
            )

        # Build full grid embedding with padding at index 0
        full_grid_emb = torch.zeros(
            gdata.num_grids + 1, 2 * self.emb_dim, device=self.device
        )
        full_grid_emb[1:, :] = self.trace_gcn(
            pure_grid_feat,
            gdata.trace_in_edge_index,
            gdata.trace_out_edge_index,
            gdata.trace_weight,
        )

        return full_road_emb, full_grid_emb

    # -----------------------------------------------------------------
    # Emission probability computation (mirrors original GMM.get_probs)
    # -----------------------------------------------------------------

    def _get_emissions(self, grid_traces, tgt_roads, traces_gps, sample_Idx,
                       traces_lens, road_lens, tf_ratio,
                       full_road_emb, full_grid_emb):
        """
        Run Seq2Seq decoder to produce emission logits.

        Args:
            grid_traces: [B, max_trace_len]
            tgt_roads: [B, max_road_len] or None (inference)
            traces_gps: [B, max_trace_len, 2]
            sample_Idx: [B, max_trace_len]
            traces_lens: list[int]
            road_lens: list[int]
            tf_ratio: teacher forcing ratio
            full_road_emb: [num_roads, emb_dim]
            full_grid_emb: [num_grids + 1, 2 * emb_dim]

        Returns:
            probs: [B, max_road_len, num_roads]
        """
        if tgt_roads is not None:
            B, max_RL = tgt_roads.shape
        else:
            B = grid_traces.shape[0]
            max_RL = int(max(road_lens))

        # Look up grid embeddings and concatenate GPS + sample index
        rnn_input = full_grid_emb[grid_traces]                      # [B, T, 2*D]
        rnn_input = torch.cat(
            [rnn_input, traces_gps, sample_Idx.unsqueeze(-1).float()],
            dim=-1,
        )                                                           # [B, T, 2*D+3]
        rnn_input = self.fc_input(rnn_input)                        # [B, T, 2*D]

        # Encode
        encoder_outputs, hiddens = self.seq2seq.encode(rnn_input, traces_lens)

        # Decode step-by-step
        probs = torch.zeros(B, max_RL, self.num_roads, device=self.device)
        inputs = torch.zeros(B, 1, self.seq2seq.hidden_size, device=self.device)

        # Build attention mask
        attn_mask = None
        if self.atten_flag:
            attn_mask = torch.zeros(B, int(max(traces_lens)), device=self.device)
            for i in range(len(traces_lens)):
                attn_mask[i][:traces_lens[i]] = 1.0

        # First decoding step
        inputs, hiddens = self.seq2seq.decode(
            inputs, hiddens, encoder_outputs, attn_mask,
        )
        probs[:, 0, :] = inputs.squeeze(1) @ full_road_emb.detach().T

        # Teacher forcing decision
        teacher_force = random.random() < tf_ratio
        if teacher_force and tgt_roads is not None:
            lst_road_id = tgt_roads[:, 0]
        else:
            lst_road_id = probs[:, 0, :].argmax(1)

        # Remaining steps
        for t in range(1, max_RL):
            if teacher_force and tgt_roads is not None:
                inputs = full_road_emb[lst_road_id].view(B, 1, -1)
            inputs, hiddens = self.seq2seq.decode(
                inputs, hiddens, encoder_outputs, attn_mask,
            )
            probs[:, t, :] = inputs.squeeze(1) @ full_road_emb.detach().T

            teacher_force = random.random() < tf_ratio
            if teacher_force and tgt_roads is not None:
                lst_road_id = tgt_roads[:, t]
            else:
                lst_road_id = probs[:, t, :].argmax(1)

        return probs

    # -----------------------------------------------------------------
    # Batch unpacking
    # -----------------------------------------------------------------

    def _unpack_batch(self, batch):
        """
        Convert LibCity batch dict to the tensors expected by the model.

        Args:
            batch: dict from DeepMapMatchingDataset collate function

        Returns:
            grid_traces, tgt_roads, traces_gps, sample_Idx,
            traces_lens, road_lens  (all on self.device)
        """
        grid_traces = batch['grid_traces'].to(self.device)
        tgt_roads = batch.get('tgt_roads')
        if tgt_roads is not None:
            tgt_roads = tgt_roads.to(self.device)
        traces_gps = batch['traces_gps'].to(self.device)
        sample_Idx = batch['sample_Idx'].to(self.device)
        traces_lens = batch['traces_lens']
        road_lens = batch['road_lens']

        # Ensure traces_lens is a plain list of ints
        if isinstance(traces_lens, torch.Tensor):
            traces_lens = traces_lens.tolist()
        if isinstance(road_lens, torch.Tensor):
            road_lens = road_lens.tolist()

        return grid_traces, tgt_roads, traces_gps, sample_Idx, traces_lens, road_lens

    # -----------------------------------------------------------------
    # LibCity interface methods
    # -----------------------------------------------------------------

    def forward(self, batch):
        """
        Forward pass: compute emission logits.

        This is used internally by calculate_loss and predict.

        Args:
            batch: dict from DeepMapMatchingDataset

        Returns:
            emissions: [batch, max_road_len, num_roads]
        """
        (grid_traces, tgt_roads, traces_gps, sample_Idx,
         traces_lens, road_lens) = self._unpack_batch(batch)

        full_road_emb, full_grid_emb = self._get_embeddings()

        emissions = self._get_emissions(
            grid_traces=grid_traces,
            tgt_roads=tgt_roads,
            traces_gps=traces_gps,
            sample_Idx=sample_Idx,
            traces_lens=traces_lens,
            road_lens=road_lens,
            tf_ratio=self.tf_ratio if self.training else 0.0,
            full_road_emb=full_road_emb,
            full_grid_emb=full_grid_emb,
        )
        return emissions

    def calculate_loss(self, batch):
        """
        Compute training loss.

        With CRF:  negative log-likelihood of the CRF.
        Without CRF:  cross-entropy on emission logits.

        Args:
            batch: dict from DeepMapMatchingDataset

        Returns:
            loss: scalar tensor
        """
        (grid_traces, tgt_roads, traces_gps, sample_Idx,
         traces_lens, road_lens) = self._unpack_batch(batch)

        if tgt_roads is None:
            _logger.warning("No target roads in batch, returning zero loss.")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        full_road_emb, full_grid_emb = self._get_embeddings()

        emissions = self._get_emissions(
            grid_traces=grid_traces,
            tgt_roads=tgt_roads,
            traces_gps=traces_gps,
            sample_Idx=sample_Idx,
            traces_lens=traces_lens,
            road_lens=road_lens,
            tf_ratio=self.tf_ratio,
            full_road_emb=full_road_emb,
            full_grid_emb=full_grid_emb,
        )

        if self.use_crf:
            # Build CRF target mask
            tgt_mask = torch.zeros(
                emissions.shape[0], int(max(road_lens)), device=self.device,
            )
            for i in range(len(road_lens)):
                tgt_mask[i][:road_lens[i]] = 1.0
            tgt_mask = tgt_mask.bool()

            # CRF returns negative log-likelihood (negated inside CRF.forward)
            loss = -self.crf(
                emissions, tgt_roads,
                full_road_emb.detach(), self.gdata.A_list, tgt_mask,
            )
        else:
            # Standard cross-entropy
            mask = (tgt_roads.view(-1) != -1)
            loss = F.cross_entropy(
                emissions.view(-1, self.target_size)[mask],
                tgt_roads.view(-1)[mask],
            )

        return loss

    def predict(self, batch):
        """
        Generate predictions for evaluation.

        With CRF:    Viterbi decoding -> list of road ID sequences.
        Without CRF: argmax on softmax emissions.

        Args:
            batch: dict from DeepMapMatchingDataset

        Returns:
            preds: [batch, max_road_len] tensor of predicted road IDs
                   (or list of lists when using CRF)
        """
        self.eval()
        with torch.no_grad():
            (grid_traces, tgt_roads, traces_gps, sample_Idx,
             traces_lens, road_lens) = self._unpack_batch(batch)

            full_road_emb, full_grid_emb = self._get_embeddings()

            emissions = self._get_emissions(
                grid_traces=grid_traces,
                tgt_roads=None,
                traces_gps=traces_gps,
                sample_Idx=sample_Idx,
                traces_lens=traces_lens,
                road_lens=road_lens,
                tf_ratio=0.0,
                full_road_emb=full_road_emb,
                full_grid_emb=full_grid_emb,
            )

            if self.use_crf:
                tgt_mask = torch.zeros(
                    emissions.shape[0], int(max(road_lens)), device=self.device,
                )
                for i in range(len(road_lens)):
                    tgt_mask[i][:road_lens[i]] = 1.0
                tgt_mask = tgt_mask.bool()

                preds = self.crf.decode(
                    emissions, full_road_emb, self.gdata.A_list, tgt_mask,
                )
                # Convert list of lists to tensor for consistent interface
                preds = torch.tensor(preds, dtype=torch.long, device=self.device)
            else:
                preds = F.softmax(emissions, dim=-1).argmax(dim=-1)

        return preds
