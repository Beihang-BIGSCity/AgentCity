# coding: utf-8
"""
JGRM: Joint GPS and Route Modeling for Trajectory Representation Learning

This module adapts the JGRM model from the original repository for LibCity framework.

Original repository: repos/JGRM/
Original files used:
- JGRM.py: Main JGRMModel class with dual-branch architecture
- cl_loss.py: GPS-Route matching loss function
- dcl.py: Decoupled Contrastive Loss

Architecture Overview:
JGRM is a trajectory representation learning model with a dual-branch architecture:
1. Route Encoding Branch: GAT (Graph Attention Network) + Transformer
   - Uses road network graph structure
   - Encodes route segment sequences with temporal features
2. GPS Encoding Branch: GRU + GRU
   - Intra-segment GRU for GPS points within each road segment
   - Inter-segment GRU for modeling across road segments
3. Joint Encoding: Shared Transformer
   - Fuses GPS and route representations
   - Enables cross-modal learning

Training Objectives (Three Loss Functions):
1. Route MLM Loss: Masked Language Model loss for route segments
2. GPS MLM Loss: Masked Language Model loss for GPS-derived segments
3. GPS-Route Matching Loss: Contrastive matching between modalities

Key Adaptations for LibCity:
1. Inherits from AbstractModel instead of BaseModel
2. Implements predict() and calculate_loss() following LibCity conventions
3. Adapts data handling to use LibCity's batch dictionary format
4. Integrates all sub-modules (GraphEncoder, TransformerModel, IntervalEmbedding)
5. Includes loss functions (DCL, GPS-Route Matching) as class methods
6. Supports compatibility mode with JGRMEncoder for standard POI datasets

Required data_feature keys:
- vocab_size: Number of road segments in the network
- edge_index: Road network adjacency (source, target pairs)
- route_max_len: Maximum route sequence length

Required config parameters:
- road_embed_size: Road segment embedding dimension (default: 128)
- gps_embed_size: GPS feature embedding dimension (default: 128)
- route_embed_size: Route representation dimension (default: 128)
- hidden_size: Hidden layer dimension (default: 256)
- gps_feat_num: Number of GPS point features (default: 8)
- drop_edge_rate: Dropout rate for GAT edges (default: 0.1)
- drop_route_rate: Dropout rate for route encoder (default: 0.1)
- drop_road_rate: Dropout rate for shared transformer (default: 0.1)
- mask_prob: Probability for masking in MLM (default: 0.2)
- mask_length: Length of consecutive masked segments (default: 2)
- tau: Temperature for contrastive loss (default: 0.07)
- mode: Model mode - 'p' for pretrain embedding, 'x' for GAT (default: 'x')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

from libcity.model.abstract_model import AbstractModel

# Small number for numerical stability in contrastive loss
SMALL_NUM = np.log(1e-45)


class GraphEncoder(nn.Module):
    """
    Graph Attention Network (GAT) encoder for road network.

    Uses two GAT layers to encode road segment embeddings based on
    the road network topology.

    Args:
        input_size: Input feature dimension
        output_size: Output feature dimension
    """
    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        try:
            from torch_geometric.nn import GATConv
            self.layer1 = GATConv(input_size, output_size)
            self.layer2 = GATConv(input_size, output_size)
            self.has_gat = True
        except ImportError:
            # Fallback to simple linear layers if torch_geometric not available
            self.layer1 = nn.Linear(input_size, output_size)
            self.layer2 = nn.Linear(output_size, output_size)
            self.has_gat = False
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        if self.has_gat:
            x = self.activation(self.layer1(x, edge_index))
            x = self.activation(self.layer2(x, edge_index))
        else:
            # Fallback: simple MLP without graph structure
            x = self.activation(self.layer1(x))
            x = self.activation(self.layer2(x))
        return x


class TransformerModel(nn.Module):
    """
    Vanilla Transformer Encoder for sequence modeling.

    Args:
        input_size: Input/output dimension
        num_heads: Number of attention heads
        hidden_size: Feed-forward hidden dimension
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            input_size, num_heads, hidden_size, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class IntervalEmbedding(nn.Module):
    """
    Continuous time interval embedding using soft binning.

    Converts continuous time intervals to embeddings using learned
    soft binning with softmax weighting.

    Args:
        num_bins: Number of time bins
        hidden_size: Output embedding dimension
    """
    def __init__(self, num_bins, hidden_size):
        super(IntervalEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output


class DCL:
    """
    Decoupled Contrastive Loss.

    Implements the DCL loss from https://arxiv.org/pdf/2110.06848.pdf

    Args:
        temperature: Temperature for scaling similarities
        weight_fn: Optional weighting function for positive samples
    """
    def __init__(self, temperature=0.1, weight_fn=None):
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat(
            (torch.mm(z1, z1.t()), cross_view_distance), dim=1
        ) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(
            neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False
        )
        return (positive_loss + negative_loss).mean()


class JGRM(AbstractModel):
    """
    JGRM: Joint GPS and Route Modeling for Trajectory Representation Learning

    This model learns joint representations of trajectories from both GPS traces
    and route (road segment) sequences using a dual-branch encoder architecture
    with cross-modal learning objectives.

    The model produces trajectory embeddings that can be used for downstream
    tasks such as trajectory similarity search, travel time estimation, etc.

    Compatibility Mode:
    When used with JGRMEncoder for standard POI datasets (foursquare, gowalla),
    the model operates in compatibility mode where:
    - POI locations are treated as "road segments"
    - Check-in timestamps provide temporal features
    - Synthetic GPS features are generated from location/time data
    - The model can still learn useful trajectory representations
    """

    def __init__(self, config, data_feature):
        super(JGRM, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data feature extraction - support both vocab_size and loc_size
        self.vocab_size = data_feature.get('vocab_size', data_feature.get('loc_size', 6450))
        if 'loc_size' in data_feature and 'vocab_size' not in data_feature:
            # Compatibility mode: loc_size doesn't include padding
            self.vocab_size = data_feature['loc_size']

        edge_index = data_feature.get('edge_index', None)
        if edge_index is not None:
            if isinstance(edge_index, (np.ndarray, list)):
                self.edge_index = torch.tensor(edge_index, dtype=torch.long)
            else:
                self.edge_index = edge_index.clone().detach()
        else:
            # Create self-loop edges if not provided (for compatibility mode)
            num_nodes = self.vocab_size
            self.edge_index = torch.stack([
                torch.arange(num_nodes),
                torch.arange(num_nodes)
            ], dim=0).long()

        # Model hyperparameters from config
        self.route_max_len = config.get('route_max_len', data_feature.get('route_max_len', 100))
        self.road_feat_num = config.get('road_feat_num', 1)
        self.road_embed_size = config.get('road_embed_size', 128)
        self.gps_feat_num = config.get('gps_feat_num', data_feature.get('gps_feat_num', 8))
        self.gps_embed_size = config.get('gps_embed_size', 128)
        self.route_embed_size = config.get('route_embed_size', 128)
        self.hidden_size = config.get('hidden_size', 256)

        # Dropout rates
        self.drop_edge_rate = config.get('drop_edge_rate', 0.1)
        self.drop_route_rate = config.get('drop_route_rate', 0.1)
        self.drop_road_rate = config.get('drop_road_rate', 0.1)

        # Masking parameters for MLM
        self.mask_length = config.get('mask_length', 2)
        self.mask_prob = config.get('mask_prob', 0.2)

        # Contrastive loss temperature
        self.tau = config.get('tau', 0.07)

        # Model mode: 'p' for pretrain (use embeddings), 'x' for graph encoding
        self.mode = config.get('mode', 'x')

        # Loss weights
        self.mlm_loss_weight = config.get('mlm_loss_weight', 1.0)
        self.match_loss_weight = config.get('match_loss_weight', 2.0)

        # Transformer architecture parameters
        self.route_transformer_layers = config.get('route_transformer_layers', 4)
        self.route_transformer_heads = config.get('route_transformer_heads', 8)
        self.shared_transformer_layers = config.get('shared_transformer_layers', 2)
        self.shared_transformer_heads = config.get('shared_transformer_heads', 4)

        # Location prediction head for trajectory location prediction task
        self.loc_size = data_feature.get('loc_size', self.vocab_size)

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""
        # Node embedding for road segments
        self.route_padding_vec = nn.Parameter(
            torch.zeros(1, self.road_embed_size), requires_grad=True
        )
        self.node_embedding = nn.Embedding(self.vocab_size + 1, self.road_embed_size)

        # Time embeddings
        self.minute_embedding = nn.Embedding(1440 + 2, self.route_embed_size)  # 0 is mask, 1441 is pad
        self.week_embedding = nn.Embedding(7 + 2, self.route_embed_size)  # 0 is mask, 8 is pad
        self.delta_embedding = IntervalEmbedding(100, self.route_embed_size)  # -1 is mask

        # Route encoding branch
        self.graph_encoder = GraphEncoder(self.road_embed_size, self.route_embed_size)
        self.position_embedding1 = nn.Embedding(self.route_max_len, self.route_embed_size)
        self.fc1 = nn.Linear(self.route_embed_size, self.hidden_size)
        self.route_encoder = TransformerModel(
            self.hidden_size, self.route_transformer_heads, self.hidden_size,
            self.route_transformer_layers, self.drop_route_rate
        )

        # GPS encoding branch
        self.gps_linear = nn.Linear(self.gps_feat_num, self.gps_embed_size)
        self.gps_intra_encoder = nn.GRU(
            self.gps_embed_size, self.gps_embed_size,
            bidirectional=True, batch_first=True
        )
        self.gps_inter_encoder = nn.GRU(
            self.gps_embed_size, self.gps_embed_size,
            bidirectional=True, batch_first=True
        )

        # Contrastive learning projection heads
        self.gps_proj_head = nn.Linear(2 * self.gps_embed_size, self.hidden_size)
        self.route_proj_head = nn.Linear(self.hidden_size, self.hidden_size)

        # Shared transformer for joint encoding
        self.position_embedding2 = nn.Embedding(self.route_max_len, self.hidden_size)
        self.modal_embedding = nn.Embedding(2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.sharedtransformer = TransformerModel(
            self.hidden_size, self.shared_transformer_heads, self.hidden_size,
            self.shared_transformer_layers, self.drop_road_rate
        )

        # MLM classifier heads
        self.gps_mlm_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.route_mlm_head = nn.Linear(self.hidden_size, self.vocab_size)

        # GPS-Route matching predictor
        self.matching_predictor = nn.Linear(self.hidden_size * 2, 2)

        # Location prediction head for trajectory location prediction task
        self.loc_pred_head = nn.Linear(self.hidden_size, self.loc_size)

        # Queues for momentum contrastive learning (optional)
        self.register_buffer("gps_queue", torch.randn(self.hidden_size, 2048))
        self.register_buffer("route_queue", torch.randn(self.hidden_size, 2048))
        self.gps_queue = F.normalize(self.gps_queue, dim=0)
        self.route_queue = F.normalize(self.route_queue, dim=0)

    def _get_edge_index(self):
        """Get edge index on the correct device with optional dropout."""
        edge_index = self.edge_index.to(self.device)
        return edge_index

    def _extract_jgrm_data(self, batch):
        """
        Extract and format batch data for JGRM processing.

        Handles both direct JGRM format and JGRMEncoder compatibility format.
        JGRMEncoder provides history_* and current_* prefixed data which needs
        to be combined for the model.

        Args:
            batch: Dictionary from dataloader

        Returns:
            route_data: (batch, seq_len, 3) temporal features
            route_assign_mat: (batch, seq_len) route segment indices
            gps_data: (batch, seq_len, gps_feat_num) GPS features
            gps_assign_mat: (batch, seq_len) GPS-to-route assignments
            gps_length: (batch, seq_len) GPS points per segment
            target: (batch,) target location (if available)
        """
        # Check if using JGRMEncoder format (has current_* keys)
        if 'current_route_data' in batch:
            # JGRMEncoder compatibility mode
            # Use current trajectory data (history is context, current is the active sequence)
            route_data = batch['current_route_data']
            route_assign_mat = batch['current_route_assign_mat']
            gps_data = batch['current_gps_data']
            gps_assign_mat = batch['current_gps_assign_mat']
            gps_length = batch['current_gps_length']
            target = batch.get('target', None)
        else:
            # Direct JGRM format
            route_data = batch.get('route_data')
            route_assign_mat = batch.get('route_assign_mat')
            gps_data = batch.get('gps_data')
            gps_assign_mat = batch.get('gps_assign_mat')
            gps_length = batch.get('gps_length')
            target = batch.get('target', None)

        return route_data, route_assign_mat, gps_data, gps_assign_mat, gps_length, target

    def encode_graph(self, drop_rate=0.):
        """
        Encode road network using GAT.

        Args:
            drop_rate: Edge dropout rate

        Returns:
            node_enc: Encoded node embeddings (vocab_size, route_embed_size)
        """
        node_emb = self.node_embedding.weight
        edge_index = self._get_edge_index()

        if drop_rate > 0 and self.training:
            try:
                from torch_geometric.utils import dropout_edge
                edge_index, _ = dropout_edge(edge_index, p=drop_rate)
            except ImportError:
                # Random edge dropout if torch_geometric not available
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
                edge_index = edge_index[:, mask]

        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_route(self, route_data, route_assign_mat, masked_route_assign_mat):
        """
        Encode route sequences using GAT + Transformer.

        Args:
            route_data: Temporal features (batch, seq_len, 3) - [weekday, minute, delta]
            route_assign_mat: Original route segment indices (batch, seq_len)
            masked_route_assign_mat: Masked route segment indices (batch, seq_len)

        Returns:
            route_unpooled: Per-segment representations (batch, seq_len, hidden_size)
            route_pooled: Trajectory-level representation (batch, hidden_size)
        """
        if self.mode == 'p':
            lookup_table = torch.cat(
                [self.node_embedding.weight, self.route_padding_vec], 0
            )
        else:
            node_enc = self.encode_graph(self.drop_edge_rate)
            lookup_table = torch.cat([node_enc, self.route_padding_vec], 0)

        batch_size, max_seq_len = masked_route_assign_mat.size()

        src_key_padding_mask = (route_assign_mat == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)

        route_emb = torch.index_select(
            lookup_table, 0, masked_route_assign_mat.long().view(-1)
        ).view(batch_size, max_seq_len, -1)

        # Time embedding
        if route_data is None:
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            # Clamp values to valid embedding ranges
            week_data = route_data[:, :, 0].long().clamp(0, 8)  # 0-8 valid
            min_data = route_data[:, :, 1].long().clamp(0, 1441)  # 0-1441 valid
            delta_data = route_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # Position embedding - clamp to max length
        seq_len = min(route_emb.shape[1], self.route_max_len)
        position = torch.arange(route_emb.shape[1], device=self.device).long()
        position = position.clamp(0, self.route_max_len - 1)
        pos_emb = position.unsqueeze(0).repeat(route_emb.shape[0], 1)
        pos_emb = self.position_embedding1(pos_emb)

        # Fuse embeddings
        route_emb = route_emb + pos_emb + week_emb + min_emb + delta_emb
        route_emb = self.fc1(route_emb)
        route_enc = self.route_encoder(route_emb, None, src_key_padding_mask)
        route_enc = torch.where(
            torch.isnan(route_enc), torch.zeros_like(route_enc), route_enc
        )

        route_unpooled = route_enc * pool_mask.repeat(1, 1, route_enc.shape[-1])
        route_pooled = route_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1)

        return route_unpooled, route_pooled

    def encode_gps(self, gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length):
        """
        Encode GPS sequences using hierarchical GRU.

        Args:
            gps_data: GPS point features (batch, gps_seq_len, gps_feat_num)
            masked_gps_assign_mat: Masked GPS-to-segment assignment (batch, gps_seq_len)
            masked_route_assign_mat: Masked route segment indices (batch, route_seq_len)
            gps_length: Number of GPS points per segment (batch, route_seq_len)

        Returns:
            gps_unpooled: Per-segment representations (batch, route_seq_len, 2*gps_embed_size)
            gps_pooled: Trajectory-level representation (batch, 2*gps_embed_size)
        """
        gps_data = self.gps_linear(gps_data)

        # Mask GPS features
        gps_src_key_padding_mask = (masked_gps_assign_mat == self.vocab_size)
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(
            1, 1, gps_data.shape[-1]
        )
        masked_gps_data = gps_data * gps_mask_mat

        # Flatten GPS data for intra-segment GRU
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length)
        _, gps_emb = self.gps_intra_encoder(flattened_gps_data)
        gps_emb = gps_emb[-1]  # Use forward direction

        # Stack for inter-segment GRU
        stacked_gps_emb = self.route_stack(gps_emb, route_length)
        gps_emb, _ = self.gps_inter_encoder(stacked_gps_emb)

        route_src_key_padding_mask = (masked_route_assign_mat == self.vocab_size).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1)
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled

    def route_stack(self, gps_emb, route_length):
        """Stack flattened GPS embeddings back to batch format."""
        values = list(route_length.values())
        data_list = []
        for idx in range(len(route_length)):
            start_idx = sum(values[:idx])
            end_idx = sum(values[:idx + 1])
            data = gps_emb[start_idx:end_idx]
            data_list.append(data)
        stacked_gps_emb = rnn_utils.pad_sequence(
            data_list, padding_value=0, batch_first=True
        )
        return stacked_gps_emb

    def gps_flatten(self, gps_data, gps_length):
        """Flatten GPS data by road segments for parallel GRU processing."""
        traj_num, gps_max_len, gps_feat_num = gps_data.shape
        flattened_gps_list = []
        route_index = {}

        for idx in range(traj_num):
            gps_feat = gps_data[idx]
            length_list = gps_length[idx].tolist() if hasattr(gps_length[idx], 'tolist') else gps_length[idx]

            for _idx, length in enumerate(length_list):
                if length != 0:
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    cnt = route_index.get(idx, 0)
                    route_index[idx] = cnt + 1
                    road_feat = gps_feat[int(start_idx):int(end_idx)]
                    flattened_gps_list.append(road_feat)

        if len(flattened_gps_list) == 0:
            # Return dummy tensor if no valid GPS data
            return torch.zeros(1, 1, gps_feat_num, device=gps_data.device), {0: 1}

        flattened_gps_data = rnn_utils.pad_sequence(
            flattened_gps_list, padding_value=0, batch_first=True
        )
        return flattened_gps_data, route_index

    def encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat):
        """
        Joint encoding using shared transformer.

        Combines GPS and route representations with modal embeddings
        and processes through shared transformer for cross-modal learning.
        """
        max_len = torch.max((route_assign_mat != self.vocab_size).int().sum(1)).item()
        max_len = max_len * 2 + 2
        data_list = []
        mask_list = []
        route_length = [
            length[length != self.vocab_size].shape[0]
            for length in route_assign_mat
        ]

        modal_emb0 = self.modal_embedding(torch.tensor(0, device=self.device))
        modal_emb1 = self.modal_embedding(torch.tensor(1, device=self.device))

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]
            route_cls_token = route_traj_rep[i].unsqueeze(0)
            gps_cls_token = gps_traj_rep[i].unsqueeze(0)

            # Position embedding - clamp to max length
            pos_len = min(length + 1, self.route_max_len)
            position = torch.arange(length + 1, device=self.device).long()
            position = position.clamp(0, self.route_max_len - 1)
            pos_emb = self.position_embedding2(position)

            # Update route embedding
            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            modal_emb = modal_emb0.unsqueeze(0).repeat(length + 1, 1)
            route_emb = route_emb + pos_emb + modal_emb
            route_emb = self.fc2(route_emb)

            # Update GPS embedding
            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            modal_emb = modal_emb1.unsqueeze(0).repeat(length + 1, 1)
            gps_emb = gps_emb + pos_emb + modal_emb
            gps_emb = self.fc2(gps_emb)

            data = torch.cat([gps_emb, route_emb], dim=0)
            data_list.append(data)

            mask = torch.tensor([False] * data.shape[0], device=self.device)
            mask_list.append(mask)

        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)

        # Extract GPS and route representations
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack(
            [joint_emb[i, length + 1] for i, length in enumerate(route_length)], dim=0
        )

        gps_road_rep = rnn_utils.pad_sequence(
            [joint_emb[i, 1:length + 1] for i, length in enumerate(route_length)],
            padding_value=0, batch_first=True
        )
        route_road_rep = rnn_utils.pad_sequence(
            [joint_emb[i, length + 2:2 * length + 2] for i, length in enumerate(route_length)],
            padding_value=0, batch_first=True
        )

        return gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep

    def forward(self, batch):
        """
        Forward pass through the JGRM model.

        Args:
            batch: Dictionary containing:
                - 'route_data': Temporal features (batch, seq_len, 3)
                - 'masked_route_assign_mat': Masked route indices (batch, seq_len)
                - 'gps_data': GPS features (batch, gps_len, gps_feat_num)
                - 'masked_gps_assign_mat': Masked GPS assignments (batch, gps_len)
                - 'route_assign_mat': Original route indices (batch, seq_len)
                - 'gps_length': GPS points per segment (batch, seq_len)

        Returns:
            Tuple of representations from both branches and joint encoding
        """
        route_data = batch['route_data'].to(self.device)
        masked_route_assign_mat = batch['masked_route_assign_mat'].to(self.device)
        gps_data = batch['gps_data'].to(self.device)
        masked_gps_assign_mat = batch['masked_gps_assign_mat'].to(self.device)
        route_assign_mat = batch['route_assign_mat'].to(self.device)
        gps_length = batch['gps_length'].to(self.device)

        # Encode both branches
        gps_road_rep, gps_traj_rep = self.encode_gps(
            gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length
        )
        route_road_rep, route_traj_rep = self.encode_route(
            route_data, route_assign_mat, masked_route_assign_mat
        )

        # Joint encoding
        gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep = \
            self.encode_joint(
                route_road_rep, route_traj_rep,
                gps_road_rep, gps_traj_rep,
                route_assign_mat
            )

        return (gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep,
                gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep)

    def get_traj_match_loss(self, gps_traj_rep, route_traj_rep, batch_size):
        """
        Compute GPS-Route matching loss.

        Uses hard negative mining with softmax-weighted sampling.
        """
        gps_traj_rep = F.normalize(gps_traj_rep, dim=1)
        route_traj_rep = F.normalize(route_traj_rep, dim=1)

        sim_g2r = gps_traj_rep @ route_traj_rep.t() / self.tau
        sim_r2g = route_traj_rep @ gps_traj_rep.t() / self.tau

        weight_g2r = F.softmax(sim_g2r, dim=1)
        weight_r2g = F.softmax(sim_r2g, dim=1)

        sim_g2r.fill_diagonal_(0)
        sim_r2g.fill_diagonal_(0)

        # Select negative samples
        route_traj_rep_neg = []
        for i in range(batch_size):
            neg_idx = torch.multinomial(weight_g2r[i], 1).item()
            route_traj_rep_neg.append(route_traj_rep[neg_idx])
        route_traj_rep_neg = torch.stack(route_traj_rep_neg, dim=0)

        gps_traj_rep_neg = []
        for i in range(batch_size):
            neg_idx = torch.multinomial(weight_r2g[i], 1).item()
            gps_traj_rep_neg.append(gps_traj_rep[neg_idx])
        gps_traj_rep_neg = torch.stack(gps_traj_rep_neg, dim=0)

        # Create positive and negative pairs
        pos_pair = torch.cat([route_traj_rep, gps_traj_rep], dim=1)
        neg_pair1 = torch.cat([route_traj_rep_neg, gps_traj_rep], dim=1)
        neg_pair2 = torch.cat([route_traj_rep, gps_traj_rep_neg], dim=1)

        all_pair = torch.cat([pos_pair, neg_pair1, neg_pair2], dim=0)
        pred = self.matching_predictor(all_pair)

        label = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
        ], dim=0)

        loss = F.cross_entropy(pred, label)
        return loss

    def random_mask(self, gps_assign_mat, route_assign_mat, gps_length):
        """
        Apply random masking for MLM training.

        Masks consecutive segments in route and corresponding GPS points.
        """
        col_num = int(route_assign_mat.shape[1] / self.mask_length) + 1
        batch_size = route_assign_mat.shape[0]

        route_mask_pos = torch.empty(
            (batch_size, col_num),
            dtype=torch.float32,
            device=route_assign_mat.device
        ).uniform_(0, 1) < self.mask_prob

        route_mask_pos = torch.stack(
            sum([[col] * self.mask_length for col in route_mask_pos.t()], []), dim=1
        )

        if route_mask_pos.shape[1] > route_assign_mat.shape[1]:
            route_mask_pos = route_mask_pos[:, :route_assign_mat.shape[1]]

        masked_route_assign_mat = route_assign_mat.clone()
        masked_route_assign_mat[route_mask_pos] = self.vocab_size

        # Mask GPS
        masked_gps_assign_mat = gps_assign_mat.clone()
        gps_mask_pos = []
        for idx, row in enumerate(gps_assign_mat):
            route_mask = route_mask_pos[idx]
            length_list = gps_length[idx]
            unpad_mask_pos_list = sum(
                [[mask] * int(length_list[_idx].item()) for _idx, mask in enumerate(route_mask)], []
            )
            pad_mask_pos_list = unpad_mask_pos_list + [torch.tensor(False, device=self.device)] * (
                gps_assign_mat.shape[1] - len(unpad_mask_pos_list)
            )
            pad_mask_pos = torch.stack(pad_mask_pos_list)
            gps_mask_pos.append(pad_mask_pos)
        gps_mask_pos = torch.stack(gps_mask_pos, dim=0)
        masked_gps_assign_mat[gps_mask_pos] = self.vocab_size

        return masked_route_assign_mat, masked_gps_assign_mat, route_mask_pos

    def predict(self, batch):
        """
        Generate location predictions for trajectory location prediction task.

        For trajectory location prediction, this method returns the predicted
        probability distribution over all locations for the next POI.

        Args:
            batch: Dictionary containing trajectory data from JGRMEncoder

        Returns:
            torch.Tensor: Location prediction scores (batch_size, loc_size)
        """
        # Extract data using compatibility helper
        route_data, route_assign_mat, gps_data, gps_assign_mat, gps_length, _ = \
            self._extract_jgrm_data(batch)

        # Move to device
        if route_data is not None:
            route_data = route_data.to(self.device).float()
        route_assign_mat = route_assign_mat.to(self.device).long()
        gps_data = gps_data.to(self.device).float()
        gps_assign_mat = gps_assign_mat.to(self.device).long()
        gps_length = gps_length.to(self.device).long()

        # Clamp indices to valid range
        route_assign_mat = route_assign_mat.clamp(0, self.vocab_size)
        gps_assign_mat = gps_assign_mat.clamp(0, self.vocab_size)

        # Use original data without masking for inference
        with torch.no_grad():
            gps_road_rep, gps_traj_rep = self.encode_gps(
                gps_data, gps_assign_mat, route_assign_mat, gps_length
            )
            route_road_rep, route_traj_rep = self.encode_route(
                route_data, route_assign_mat, route_assign_mat  # No masking
            )

            # Joint encoding
            _, gps_traj_joint_rep, _, route_traj_joint_rep = self.encode_joint(
                route_road_rep, route_traj_rep,
                gps_road_rep, gps_traj_rep,
                route_assign_mat
            )

            # Average joint representations for trajectory embedding
            trajectory_embedding = (gps_traj_joint_rep + route_traj_joint_rep) / 2

            # Predict next location
            loc_scores = self.loc_pred_head(trajectory_embedding)

        return loc_scores

    def calculate_loss(self, batch):
        """
        Calculate combined training loss for trajectory location prediction.

        For trajectory location prediction task, the loss combines:
        1. Location Prediction Loss: Cross-entropy for next POI prediction
        2. GPS-Route Matching Loss: Contrastive matching loss (auxiliary)

        For full JGRM training (representation learning), also includes:
        3. Route MLM Loss: Masked Language Model loss for route segments
        4. GPS MLM Loss: Masked Language Model loss for GPS-derived segments

        Args:
            batch: Dictionary containing trajectory data from JGRMEncoder

        Returns:
            torch.Tensor: Total loss (scalar)
        """
        # Extract data using compatibility helper
        route_data, route_assign_mat, gps_data, gps_assign_mat, gps_length, target = \
            self._extract_jgrm_data(batch)

        # Move to device
        if route_data is not None:
            route_data = route_data.to(self.device).float()
        route_assign_mat = route_assign_mat.to(self.device).long()
        gps_data = gps_data.to(self.device).float()
        gps_assign_mat = gps_assign_mat.to(self.device).long()
        gps_length = gps_length.to(self.device).long()

        # Clamp indices to valid range
        route_assign_mat = route_assign_mat.clamp(0, self.vocab_size)
        gps_assign_mat = gps_assign_mat.clamp(0, self.vocab_size)

        batch_size = route_assign_mat.shape[0]

        # Apply random masking for self-supervised learning
        masked_route_assign_mat, masked_gps_assign_mat, route_mask_pos = self.random_mask(
            gps_assign_mat, route_assign_mat, gps_length
        )

        # Forward pass with masked inputs
        gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
        gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep = \
            self._forward_with_masks(
                route_data, masked_route_assign_mat, gps_data,
                masked_gps_assign_mat, route_assign_mat, gps_length
            )

        # Average joint representations for trajectory embedding
        trajectory_embedding = (gps_traj_joint_rep + route_traj_joint_rep) / 2

        # Location prediction loss (primary task)
        loc_pred_loss = torch.tensor(0.0, device=self.device)
        if target is not None:
            target = target.to(self.device).long()
            loc_scores = self.loc_pred_head(trajectory_embedding)
            loc_pred_loss = F.cross_entropy(loc_scores, target)

        # Project to same space for matching loss
        gps_traj_proj = self.gps_proj_head(gps_traj_rep)
        route_traj_proj = self.route_proj_head(route_traj_rep)

        # GPS-Route Matching Loss (auxiliary)
        match_loss = self.get_traj_match_loss(gps_traj_proj, route_traj_proj, batch_size)

        # Flatten road representations for MLM
        mat2flatten = {}
        y_label = []
        route_length = (route_assign_mat != self.vocab_size).int().sum(1)
        gps_road_list, route_road_list = [], []
        gps_road_joint_list, route_road_joint_list = [], []
        now_flatten_idx = 0

        for i, length in enumerate(route_length):
            length_val = length.item() if hasattr(length, 'item') else int(length)
            if length_val > 0:
                y_label.append(route_assign_mat[i, :length_val])
                gps_road_list.append(gps_road_rep[i, :length_val])
                route_road_list.append(route_road_rep[i, :length_val])
                gps_road_joint_list.append(gps_road_joint_rep[i, :length_val])
                route_road_joint_list.append(route_road_joint_rep[i, :length_val])
                for l in range(length_val):
                    mat2flatten[(i, l)] = now_flatten_idx
                    now_flatten_idx += 1

        # MLM losses (auxiliary)
        mlm_loss = torch.tensor(0.0, device=self.device)
        if len(y_label) > 0 and now_flatten_idx > 0:
            y_label = torch.cat(y_label, dim=0)
            gps_road_rep_flat = torch.cat(gps_road_list, dim=0)
            route_road_rep_flat = torch.cat(route_road_list, dim=0)
            gps_road_joint_rep_flat = torch.cat(gps_road_joint_list, dim=0)
            route_road_joint_rep_flat = torch.cat(route_road_joint_list, dim=0)

            # Get masked positions
            masked_pos = torch.nonzero(route_assign_mat != masked_route_assign_mat)
            masked_pos_list = []
            for pos in masked_pos:
                key = (pos[0].item(), pos[1].item())
                if key in mat2flatten:
                    masked_pos_list.append(mat2flatten[key])

            if len(masked_pos_list) > 0:
                y_label_masked = y_label[masked_pos_list].long()
                y_label_masked = y_label_masked.clamp(0, self.vocab_size - 1)

                # GPS MLM Loss
                gps_mlm_pred = self.gps_mlm_head(gps_road_joint_rep_flat)
                masked_gps_mlm_pred = gps_mlm_pred[masked_pos_list]
                gps_mlm_loss = F.cross_entropy(masked_gps_mlm_pred, y_label_masked)

                # Route MLM Loss
                route_mlm_pred = self.route_mlm_head(route_road_joint_rep_flat)
                masked_route_mlm_pred = route_mlm_pred[masked_pos_list]
                route_mlm_loss = F.cross_entropy(masked_route_mlm_pred, y_label_masked)

                mlm_loss = (gps_mlm_loss + route_mlm_loss) / 2

        # Combined loss
        # Weight location prediction loss higher for trajectory prediction task
        if target is not None:
            total_loss = (
                2.0 * loc_pred_loss +
                0.5 * self.mlm_loss_weight * mlm_loss +
                0.5 * self.match_loss_weight * match_loss
            )
        else:
            # Pure representation learning mode
            total_loss = (
                self.mlm_loss_weight * mlm_loss +
                self.match_loss_weight * match_loss
            )

        return total_loss

    def _forward_with_masks(self, route_data, masked_route_assign_mat, gps_data,
                            masked_gps_assign_mat, route_assign_mat, gps_length):
        """Forward pass with pre-computed masks."""
        gps_road_rep, gps_traj_rep = self.encode_gps(
            gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length
        )
        route_road_rep, route_traj_rep = self.encode_route(
            route_data, route_assign_mat, masked_route_assign_mat
        )
        gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep = \
            self.encode_joint(
                route_road_rep, route_traj_rep,
                gps_road_rep, gps_traj_rep,
                route_assign_mat
            )
        return (gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep,
                gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep)

    def get_embeddings(self, batch):
        """
        Get trajectory embeddings and semantic IDs.

        Useful for representation learning tasks like trajectory similarity.

        Args:
            batch: Dictionary containing trajectory data

        Returns:
            embeddings: Trajectory embeddings (batch_size, hidden_size)
            gps_emb: GPS branch embeddings
            route_emb: Route branch embeddings
        """
        return self.predict(batch)
