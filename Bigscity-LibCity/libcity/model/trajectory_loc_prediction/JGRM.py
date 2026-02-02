"""
JGRM: Joint GPS and Route Multimodal Model for Trajectory Representation Learning

This model is adapted from the original JGRM implementation.

Original paper:
"Joint GPS and Road Representation for Trajectory Modeling"

Key Components:
1. Dual-stream encoder (GPS + Route)
2. GPS Stream: Linear projection + Intra-road GRU + Inter-road GRU
3. Route Stream: Node embedding + Optional GAT graph encoder + Transformer encoder
4. Joint encoding with shared Transformer
5. Multiple prediction heads (MLM, contrastive learning, matching)

Adaptations for LibCity:
- Replaced custom BaseModel with LibCity's AbstractModel
- Adapted __init__() to use LibCity's config and data_feature pattern
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Extracted hyperparameters to config parameters
- Handled device management through LibCity's config
- Preserved queue-based contrastive learning mechanism

Original files:
- repos/JGRM/JGRM.py (JGRMModel, GraphEncoder, TransformerModel, IntervalEmbedding)
- repos/JGRM/basemodel.py (BaseModel)

Dependencies:
- torch-geometric (for GAT layers in GraphEncoder)
- Standard PyTorch modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

try:
    from torch_geometric.utils import dropout_adj
    from torch_geometric.nn import GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Fallback: define placeholder classes
    class GATConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("torch_geometric is required for GraphEncoder. "
                            "Please install it: pip install torch-geometric")

from libcity.model.abstract_model import AbstractModel


# ========================== Graph Encoder (GAT) ==========================

class GraphEncoder(nn.Module):
    """
    Graph Attention Network (GAT) encoder for learning road segment embeddings.

    Uses 2-layer GAT to encode road network topology.

    Args:
        input_size: Input feature dimension (road embedding size)
        output_size: Output feature dimension (route embedding size)
    """

    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for GraphEncoder. "
                            "Please install it: pip install torch-geometric")
        # Update road edge features using GAT
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(output_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        """
        Forward pass through GAT layers.

        Args:
            x: Node features of shape (num_nodes, input_size)
            edge_index: Graph edge indices of shape (2, num_edges)

        Returns:
            Encoded node features of shape (num_nodes, output_size)
        """
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x


# ========================== Transformer Encoder ==========================

class TransformerModel(nn.Module):
    """
    Vanilla Transformer encoder for sequence modeling.

    Args:
        input_size: Input/hidden dimension
        num_heads: Number of attention heads
        hidden_size: Feed-forward hidden dimension
        num_layers: Number of Transformer layers
        dropout: Dropout rate
    """

    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            input_size, num_heads, hidden_size, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        """
        Forward pass through Transformer encoder.

        Args:
            src: Input sequence of shape (batch_size, seq_len, input_size)
            src_mask: Attention mask (optional)
            src_key_padding_mask: Padding mask of shape (batch_size, seq_len)

        Returns:
            Encoded sequence of shape (batch_size, seq_len, input_size)
        """
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


# ========================== Continuous Time Embedding ==========================

class IntervalEmbedding(nn.Module):
    """
    Continuous interval/time embedding using soft binning.

    Converts continuous time intervals to dense embeddings using
    a learnable soft assignment to time bins.

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
        """
        Forward pass.

        Args:
            x: Input time intervals of shape (batch_size, seq_len)

        Returns:
            Time embeddings of shape (batch_size, seq_len, hidden_size)
        """
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output


# ========================== Main JGRM Model ==========================

class JGRM(AbstractModel):
    """
    JGRM: Joint GPS and Route Multimodal Model for Trajectory Representation Learning.

    This model uses a dual-stream architecture combining GPS point sequences
    and route (road segment) sequences for trajectory representation learning.
    The model supports pre-training with masked language modeling (MLM) and
    GPS-route matching prediction tasks.

    Key Features:
    1. Dual-stream processing: GPS stream and Route stream
    2. GPS Stream: Linear projection -> Intra-road GRU -> Inter-road GRU
    3. Route Stream: Node embedding + GAT graph encoder + Transformer encoder
    4. Joint encoding with shared Transformer
    5. Queue-based contrastive learning for representation alignment

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including vocab sizes, graph data, etc.

    Required config parameters:
        - route_max_len: Maximum route sequence length (default: 100)
        - road_feat_num: Road feature dimension (default: 1)
        - road_embed_size: Road embedding dimension (default: 128)
        - gps_feat_num: GPS feature dimension (default: 8)
        - gps_embed_size: GPS embedding dimension (default: 128)
        - route_embed_size: Route embedding dimension (default: 128)
        - hidden_size: Hidden/output dimension (default: 256)
        - drop_edge_rate: Edge dropout rate for graph (default: 0.1)
        - drop_route_rate: Dropout rate for route encoder (default: 0.1)
        - drop_road_rate: Dropout rate for shared encoder (default: 0.1)
        - mode: Encoding mode - 'p' for plain embedding, 'x' for graph encoding (default: 'x')
        - mask_prob: Probability of masking for MLM (default: 0.2)
        - mask_length: Length of masked spans (default: 2)
        - tau: Temperature for contrastive learning (default: 0.07)
        - mlm_loss_weight: Weight for MLM loss (default: 1.0)
        - match_loss_weight: Weight for matching loss (default: 2.0)
        - route_transformer_layers: Number of route Transformer layers (default: 4)
        - route_transformer_heads: Number of route Transformer heads (default: 8)
        - shared_transformer_layers: Number of shared Transformer layers (default: 2)
        - shared_transformer_heads: Number of shared Transformer heads (default: 4)

    Required data_feature:
        - loc_size / vocab_size: Number of road segments (vocabulary size)
        - edge_index: Road network graph edge indices (optional)
        - pretrained_road_embed: Pre-trained road embeddings (optional)
    """

    def __init__(self, config, data_feature):
        super(JGRM, self).__init__(config, data_feature)

        self.device = config.get('device', 'cpu')

        # Data dimensions from data_feature
        self.vocab_size = data_feature.get('loc_size', data_feature.get('vocab_size', 10000))

        # Model hyperparameters from config
        self.route_max_len = config.get('route_max_len', 100)
        self.road_feat_num = config.get('road_feat_num', 1)
        self.road_embed_size = config.get('road_embed_size', 128)
        self.gps_feat_num = config.get('gps_feat_num', 8)
        self.gps_embed_size = config.get('gps_embed_size', 128)
        self.route_embed_size = config.get('route_embed_size', 128)
        self.hidden_size = config.get('hidden_size', 256)

        self.drop_edge_rate = config.get('drop_edge_rate', 0.1)
        self.drop_route_rate = config.get('drop_route_rate', 0.1)
        self.drop_road_rate = config.get('drop_road_rate', 0.1)

        self.mode = config.get('mode', 'x')  # 'p' for plain, 'x' for graph encoding
        self.mask_prob = config.get('mask_prob', 0.2)
        self.mask_length = config.get('mask_length', 2)
        self.tau = config.get('tau', 0.07)

        self.mlm_loss_weight = config.get('mlm_loss_weight', 1.0)
        self.match_loss_weight = config.get('match_loss_weight', 2.0)

        # Transformer configuration
        self.route_transformer_layers = config.get('route_transformer_layers', 4)
        self.route_transformer_heads = config.get('route_transformer_heads', 8)
        self.shared_transformer_layers = config.get('shared_transformer_layers', 2)
        self.shared_transformer_heads = config.get('shared_transformer_heads', 4)

        # Queue size for contrastive learning
        self.queue_size = config.get('queue_size', 2048)

        # Graph data from data_feature
        edge_index = data_feature.get('edge_index', None)
        if edge_index is not None:
            if isinstance(edge_index, torch.Tensor):
                self.register_buffer('edge_index', edge_index.long())
            else:
                self.register_buffer('edge_index', torch.tensor(edge_index).long())
        else:
            # Create a simple self-loop graph as default
            self.register_buffer('edge_index', torch.stack([
                torch.arange(self.vocab_size),
                torch.arange(self.vocab_size)
            ]).long())

        # Build model components
        self._build_model(data_feature)

        # Loss functions
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.match_criterion = nn.CrossEntropyLoss()

    def _build_model(self, data_feature):
        """Build all model components."""

        # ===================== Node Embedding =====================
        # Padding vector for masked/padding positions
        self.route_padding_vec = nn.Parameter(
            torch.zeros(1, self.road_embed_size), requires_grad=True
        )

        # Road segment (node) embedding
        self.node_embedding = nn.Embedding(self.vocab_size, self.road_embed_size)

        # Initialize with pre-trained embeddings if available
        pretrained_embed = data_feature.get('pretrained_road_embed', None)
        if pretrained_embed is not None:
            if isinstance(pretrained_embed, torch.Tensor):
                self.node_embedding.weight.data.copy_(pretrained_embed)
            else:
                self.node_embedding.weight.data.copy_(torch.tensor(pretrained_embed))

        # ===================== Time Embeddings =====================
        # Minute of day embedding (0 is mask position)
        self.minute_embedding = nn.Embedding(1440 + 1, self.route_embed_size)
        # Day of week embedding (0 is mask position)
        self.week_embedding = nn.Embedding(7 + 1, self.route_embed_size)
        # Continuous interval embedding (-1 is mask position)
        self.delta_embedding = IntervalEmbedding(100, self.route_embed_size)

        # ===================== Route Encoding =====================
        # Graph encoder (GAT)
        if HAS_TORCH_GEOMETRIC and self.mode != 'p':
            self.graph_encoder = GraphEncoder(self.road_embed_size, self.route_embed_size)
        else:
            self.graph_encoder = None

        # Position embedding for route sequence
        self.position_embedding1 = nn.Embedding(self.route_max_len, self.route_embed_size)
        # Feed-forward for fusing route and time
        self.fc1 = nn.Linear(self.route_embed_size, self.hidden_size)
        # Route Transformer encoder
        self.route_encoder = TransformerModel(
            self.hidden_size,
            self.route_transformer_heads,
            self.hidden_size,
            self.route_transformer_layers,
            self.drop_route_rate
        )

        # ===================== GPS Encoding =====================
        # GPS feature projection
        self.gps_linear = nn.Linear(self.gps_feat_num, self.gps_embed_size)
        # Intra-road GRU (within each road segment)
        self.gps_intra_encoder = nn.GRU(
            self.gps_embed_size, self.gps_embed_size,
            bidirectional=True, batch_first=True
        )
        # Inter-road GRU (across road segments)
        self.gps_inter_encoder = nn.GRU(
            self.gps_embed_size, self.gps_embed_size,
            bidirectional=True, batch_first=True
        )

        # ===================== Projection Heads =====================
        # Contrastive learning projection heads
        self.gps_proj_head = nn.Linear(2 * self.gps_embed_size, self.hidden_size)
        self.route_proj_head = nn.Linear(self.hidden_size, self.hidden_size)

        # ===================== Shared Transformer =====================
        # Position embedding for joint encoding
        self.position_embedding2 = nn.Embedding(self.route_max_len, self.hidden_size)
        # Modal embedding (0 for GPS, 1 for Route)
        self.modal_embedding = nn.Embedding(2, self.hidden_size)
        # Position transform
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # Shared Transformer encoder
        self.sharedtransformer = TransformerModel(
            self.hidden_size,
            self.shared_transformer_heads,
            self.hidden_size,
            self.shared_transformer_layers,
            self.drop_road_rate
        )

        # ===================== MLM Prediction Heads =====================
        # GPS stream MLM head
        self.gps_mlm_head = nn.Linear(self.hidden_size, self.vocab_size)
        # Route stream MLM head
        self.route_mlm_head = nn.Linear(self.hidden_size, self.vocab_size)

        # ===================== Matching Prediction =====================
        # Binary matching predictor (match / not match)
        self.matching_predictor = nn.Linear(self.hidden_size * 2, 2)

        # Contrastive learning queues
        self.register_buffer("gps_queue", F.normalize(torch.randn(self.hidden_size, self.queue_size), dim=0))
        self.register_buffer("route_queue", F.normalize(torch.randn(self.hidden_size, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, gps_feat, route_feat):
        """Update the queues for contrastive learning."""
        batch_size = gps_feat.shape[0]

        ptr = int(self.queue_ptr)

        # Handle case where batch_size doesn't evenly divide queue_size
        if ptr + batch_size > self.queue_size:
            remaining = self.queue_size - ptr
            self.gps_queue[:, ptr:] = gps_feat[:remaining].T
            self.route_queue[:, ptr:] = route_feat[:remaining].T
            ptr = 0
            gps_feat = gps_feat[remaining:]
            route_feat = route_feat[remaining:]
            batch_size = gps_feat.shape[0]

        if batch_size > 0:
            self.gps_queue[:, ptr:ptr + batch_size] = gps_feat.T
            self.route_queue[:, ptr:ptr + batch_size] = route_feat.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def encode_graph(self, drop_rate=0.):
        """
        Encode road network graph using GAT.

        Args:
            drop_rate: Edge dropout rate

        Returns:
            Encoded node embeddings of shape (vocab_size, route_embed_size)
        """
        if self.graph_encoder is None:
            return self.node_embedding.weight

        node_emb = self.node_embedding.weight
        if HAS_TORCH_GEOMETRIC:
            edge_index = dropout_adj(self.edge_index, p=drop_rate)[0]
        else:
            edge_index = self.edge_index
        node_enc = self.graph_encoder(node_emb, edge_index)
        return node_enc

    def encode_route(self, route_data, route_assign_mat, masked_route_assign_mat):
        """
        Encode route (road segment sequence) using Transformer.

        Args:
            route_data: Time features (batch_size, seq_len, 3) - [week, minute, delta]
            route_assign_mat: Original road indices (batch_size, seq_len)
            masked_route_assign_mat: Masked road indices for MLM (batch_size, seq_len)

        Returns:
            route_unpooled: Road-level representations (batch_size, seq_len, hidden_size)
            route_pooled: Trajectory-level representation (batch_size, hidden_size)
        """
        # Get lookup table (node embeddings)
        if self.mode == 'p' or self.graph_encoder is None:
            lookup_table = torch.cat([self.node_embedding.weight, self.route_padding_vec], 0)
        else:
            node_enc = self.encode_graph(self.drop_edge_rate)
            lookup_table = torch.cat([node_enc, self.route_padding_vec], 0)

        batch_size, max_seq_len = masked_route_assign_mat.size()

        # Create padding mask (padding position = vocab_size)
        src_key_padding_mask = (route_assign_mat == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 for padding

        # Look up road embeddings
        route_emb = torch.index_select(
            lookup_table, 0, masked_route_assign_mat.int().view(-1)
        ).view(batch_size, max_seq_len, -1)

        # Time embedding
        if route_data is None:
            # Use mean embeddings when no time features (for node evaluation)
            week_emb = self.week_embedding.weight.detach()[1:].mean(dim=0)
            min_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
            delta_emb = self.minute_embedding.weight.detach()[1:].mean(dim=0)
        else:
            week_data = route_data[:, :, 0].long()
            min_data = route_data[:, :, 1].long()
            delta_data = route_data[:, :, 2].float()
            week_emb = self.week_embedding(week_data)
            min_emb = self.minute_embedding(min_data)
            delta_emb = self.delta_embedding(delta_data)

        # Position embedding
        position = torch.arange(route_emb.shape[1]).long().to(route_emb.device)
        pos_emb = position.unsqueeze(0).repeat(route_emb.shape[0], 1)
        pos_emb = self.position_embedding1(pos_emb)

        # Fuse embeddings
        route_emb = route_emb + pos_emb + week_emb + min_emb + delta_emb
        route_emb = self.fc1(route_emb)
        route_enc = self.route_encoder(route_emb, None, src_key_padding_mask)

        # Handle NaN values
        route_enc = torch.where(
            torch.isnan(route_enc),
            torch.full_like(route_enc, 0),
            route_enc
        )

        route_unpooled = route_enc * pool_mask.repeat(1, 1, route_enc.shape[-1])

        # Mean pooling (avoid division by zero)
        route_pooled = route_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1)

        return route_unpooled, route_pooled

    def encode_gps(self, gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length):
        """
        Encode GPS point sequence using hierarchical GRU.

        Args:
            gps_data: GPS features (batch_size, gps_max_len, gps_feat_num)
            masked_gps_assign_mat: Masked GPS-to-road assignments (batch_size, gps_max_len)
            masked_route_assign_mat: Masked road indices (batch_size, route_max_len)
            gps_length: List of GPS point counts per road segment

        Returns:
            gps_unpooled: Road-level GPS representations (batch_size, route_max_len, 2*gps_embed_size)
            gps_pooled: Trajectory-level GPS representation (batch_size, 2*gps_embed_size)
        """
        # Project GPS features
        gps_data = self.gps_linear(gps_data)

        # Create mask for GPS points
        gps_src_key_padding_mask = (masked_gps_assign_mat == self.vocab_size)
        gps_mask_mat = (1 - gps_src_key_padding_mask.int()).unsqueeze(-1).repeat(1, 1, gps_data.shape[-1])
        masked_gps_data = gps_data * gps_mask_mat

        # Flatten GPS data for parallel intra-road GRU processing
        flattened_gps_data, route_length = self.gps_flatten(masked_gps_data, gps_length)

        # Intra-road encoding
        _, gps_emb = self.gps_intra_encoder(flattened_gps_data)
        gps_emb = gps_emb[-1]  # Take forward direction output

        # Stack back to batch form for inter-road encoding
        stacked_gps_emb = self.route_stack(gps_emb, route_length)

        # Inter-road encoding
        gps_emb, _ = self.gps_inter_encoder(stacked_gps_emb)

        # Create pooling mask
        route_src_key_padding_mask = (masked_route_assign_mat == self.vocab_size).transpose(0, 1)
        route_pool_mask = (1 - route_src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        # Mean pooling
        gps_pooled = gps_emb.sum(1) / route_pool_mask.sum(1).clamp(min=1)
        gps_unpooled = gps_emb

        return gps_unpooled, gps_pooled

    def gps_flatten(self, gps_data, gps_length):
        """
        Flatten GPS data by road segments for parallel processing.

        Args:
            gps_data: GPS features (batch_size, gps_max_len, gps_feat_num)
            gps_length: List of GPS point counts per road segment for each trajectory

        Returns:
            flattened_gps_data: (total_roads, max_pts_per_road, gps_feat_num)
            route_index: Dict mapping trajectory index to road count
        """
        traj_num, gps_max_len, gps_feat_num = gps_data.shape
        flattened_gps_list = []
        route_index = {}

        for idx in range(traj_num):
            gps_feat = gps_data[idx]
            length_list = gps_length[idx]

            # Iterate over each road segment in the trajectory
            for _idx, length in enumerate(length_list):
                if length != 0:
                    start_idx = sum(length_list[:_idx])
                    end_idx = start_idx + length_list[_idx]
                    cnt = route_index.get(idx, 0)
                    route_index[idx] = cnt + 1
                    road_feat = gps_feat[start_idx:end_idx]
                    flattened_gps_list.append(road_feat)

        if len(flattened_gps_list) == 0:
            # Handle empty case
            return torch.zeros(1, 1, gps_feat_num, device=gps_data.device), {0: 1}

        flattened_gps_data = rnn_utils.pad_sequence(
            flattened_gps_list, padding_value=0, batch_first=True
        )

        return flattened_gps_data, route_index

    def route_stack(self, gps_emb, route_length):
        """
        Stack flattened GPS embeddings back to batch form.

        Args:
            gps_emb: Flattened GPS embeddings (total_roads, emb_size)
            route_length: Dict mapping trajectory index to road count

        Returns:
            stacked_gps_emb: (batch_size, max_route_len, emb_size)
        """
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

    def encode_joint(self, route_road_rep, route_traj_rep, gps_road_rep, gps_traj_rep, route_assign_mat):
        """
        Joint encoding of GPS and Route representations using shared Transformer.

        Args:
            route_road_rep: Route road-level representations (batch_size, seq_len, hidden_size)
            route_traj_rep: Route trajectory-level representations (batch_size, hidden_size)
            gps_road_rep: GPS road-level representations (batch_size, seq_len, 2*gps_embed_size)
            gps_traj_rep: GPS trajectory-level representations (batch_size, 2*gps_embed_size)
            route_assign_mat: Road indices for determining sequence lengths

        Returns:
            Tuple of (gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep)
        """
        # Calculate actual route lengths
        route_length = [length[length != self.vocab_size].shape[0] for length in route_assign_mat]
        max_len = max(route_length) * 2 + 2

        data_list = []
        mask_list = []

        # Get modal embeddings
        modal_emb0 = self.modal_embedding(torch.tensor(0, device=route_road_rep.device))
        modal_emb1 = self.modal_embedding(torch.tensor(1, device=route_road_rep.device))

        for i, length in enumerate(route_length):
            route_road_token = route_road_rep[i][:length]
            gps_road_token = gps_road_rep[i][:length]

            # Project GPS representation to match hidden_size
            gps_road_token = self.gps_proj_head(gps_road_token)

            route_cls_token = route_traj_rep[i].unsqueeze(0)
            gps_cls_token = self.gps_proj_head(gps_traj_rep[i].unsqueeze(0))

            # Position embedding
            position = torch.arange(length + 1).long().to(route_road_rep.device)
            pos_emb = self.position_embedding2(position)

            # Update route embedding with position and modal embeddings
            route_emb = torch.cat([route_cls_token, route_road_token], dim=0)
            modal_emb = modal_emb1.unsqueeze(0).repeat(length + 1, 1)
            route_emb = route_emb + pos_emb + modal_emb
            route_emb = self.fc2(route_emb)

            # Update GPS embedding with position and modal embeddings
            gps_emb = torch.cat([gps_cls_token, gps_road_token], dim=0)
            modal_emb = modal_emb0.unsqueeze(0).repeat(length + 1, 1)
            gps_emb = gps_emb + pos_emb + modal_emb
            gps_emb = self.fc2(gps_emb)

            # Concatenate GPS and Route sequences
            data = torch.cat([gps_emb, route_emb], dim=0)
            data_list.append(data)

            # Create mask (False = not masked)
            mask = torch.tensor([False] * data.shape[0], device=route_road_rep.device)
            mask_list.append(mask)

        # Pad sequences
        joint_data = rnn_utils.pad_sequence(data_list, padding_value=0, batch_first=True)
        mask_mat = rnn_utils.pad_sequence(mask_list, padding_value=True, batch_first=True)

        # Forward through shared Transformer
        joint_emb = self.sharedtransformer(joint_data, None, mask_mat)

        # Extract representations:
        # Position 0: gps_traj_rep, Position length+1: route_traj_rep
        gps_traj_rep = joint_emb[:, 0]
        route_traj_rep = torch.stack(
            [joint_emb[i, length + 1] for i, length in enumerate(route_length)], dim=0
        )

        # Extract road-level representations
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
        Forward pass.

        Args:
            batch: LibCity Batch object containing trajectory data.
                Required keys:
                    - 'route_data': Time features (batch_size, seq_len, 3)
                    - 'route_assign_mat': Original road indices (batch_size, seq_len)
                    - 'masked_route_assign_mat': Masked road indices (batch_size, seq_len)
                    - 'gps_data': GPS features (batch_size, gps_max_len, gps_feat_num)
                    - 'masked_gps_assign_mat': Masked GPS assignments (batch_size, gps_max_len)
                    - 'gps_length': GPS point counts per road (list of lists)

        Returns:
            Tuple containing all encoded representations and prediction heads:
            (gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep,
             gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep)
        """
        # Extract data from batch
        route_data = batch.get('route_data', batch.get('current_tim', None))
        route_assign_mat = batch['route_assign_mat'] if 'route_assign_mat' in batch.data else batch['current_loc']
        masked_route_assign_mat = batch.get('masked_route_assign_mat', route_assign_mat)
        gps_data = batch['gps_data'] if 'gps_data' in batch.data else batch.get('gps_features', None)
        masked_gps_assign_mat = batch.get('masked_gps_assign_mat', route_assign_mat)
        gps_length = batch.get('gps_length', None)

        # Convert to tensors if needed
        if route_data is not None:
            if not isinstance(route_data, torch.Tensor):
                route_data = torch.FloatTensor(route_data)
            route_data = route_data.to(self.device)

        if not isinstance(route_assign_mat, torch.Tensor):
            route_assign_mat = torch.LongTensor(route_assign_mat)
        route_assign_mat = route_assign_mat.to(self.device)

        if not isinstance(masked_route_assign_mat, torch.Tensor):
            masked_route_assign_mat = torch.LongTensor(masked_route_assign_mat)
        masked_route_assign_mat = masked_route_assign_mat.to(self.device)

        if gps_data is not None:
            if not isinstance(gps_data, torch.Tensor):
                gps_data = torch.FloatTensor(gps_data)
            gps_data = gps_data.to(self.device)

        if not isinstance(masked_gps_assign_mat, torch.Tensor):
            masked_gps_assign_mat = torch.LongTensor(masked_gps_assign_mat)
        masked_gps_assign_mat = masked_gps_assign_mat.to(self.device)

        # Handle gps_length - convert to list of lists if needed
        if gps_length is not None:
            if isinstance(gps_length, torch.Tensor):
                gps_length = gps_length.tolist()

        # Encode GPS stream
        if gps_data is not None and gps_length is not None:
            gps_road_rep, gps_traj_rep = self.encode_gps(
                gps_data, masked_gps_assign_mat, masked_route_assign_mat, gps_length
            )
        else:
            # Fallback: create dummy GPS representations
            batch_size = route_assign_mat.shape[0]
            seq_len = route_assign_mat.shape[1]
            gps_road_rep = torch.zeros(batch_size, seq_len, 2 * self.gps_embed_size, device=self.device)
            gps_traj_rep = torch.zeros(batch_size, 2 * self.gps_embed_size, device=self.device)

        # Encode Route stream
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

    def predict(self, batch):
        """
        Prediction method for LibCity.

        For trajectory representation learning, this returns the joint trajectory
        representations that can be used for downstream tasks.

        Args:
            batch: Input batch dictionary

        Returns:
            Joint trajectory representations (batch_size, hidden_size * 2)
            combining GPS and Route trajectory-level representations
        """
        outputs = self.forward(batch)
        gps_traj_joint_rep = outputs[5]  # gps_traj_joint_rep
        route_traj_joint_rep = outputs[7]  # route_traj_joint_rep

        # Concatenate GPS and Route trajectory representations
        traj_rep = torch.cat([gps_traj_joint_rep, route_traj_joint_rep], dim=-1)

        return traj_rep

    def calculate_loss(self, batch):
        """
        Calculate combined training loss.

        Loss components:
        1. GPS MLM loss: Predict masked road segments from GPS stream
        2. Route MLM loss: Predict masked road segments from Route stream
        3. Matching loss: Predict if GPS and Route representations match

        Combined loss = (gps_mlm + route_mlm + 2 * matching) / 3

        Args:
            batch: LibCity Batch object containing:
                - Trajectory data for forward pass
                - 'route_labels': Ground truth road segment labels for MLM
                - 'match_labels': Binary labels for matching (0=match, 1=not match)

        Returns:
            Combined loss tensor
        """
        outputs = self.forward(batch)

        gps_road_rep = outputs[0]
        route_road_rep = outputs[2]
        gps_road_joint_rep = outputs[4]
        gps_traj_joint_rep = outputs[5]
        route_road_joint_rep = outputs[6]
        route_traj_joint_rep = outputs[7]

        # Get labels from batch
        route_labels = batch.get('route_labels', None)
        match_labels = batch.get('match_labels', None)

        # ===================== MLM Loss =====================
        gps_mlm_loss = torch.tensor(0.0, device=self.device)
        route_mlm_loss = torch.tensor(0.0, device=self.device)

        if route_labels is not None:
            if not isinstance(route_labels, torch.Tensor):
                route_labels = torch.LongTensor(route_labels)
            route_labels = route_labels.to(self.device)

            # GPS MLM prediction
            gps_mlm_logits = self.gps_mlm_head(gps_road_joint_rep)
            gps_mlm_loss = self.mlm_criterion(
                gps_mlm_logits.view(-1, self.vocab_size),
                route_labels.view(-1)
            )

            # Route MLM prediction
            route_mlm_logits = self.route_mlm_head(route_road_joint_rep)
            route_mlm_loss = self.mlm_criterion(
                route_mlm_logits.view(-1, self.vocab_size),
                route_labels.view(-1)
            )

        # ===================== Matching Loss =====================
        matching_loss = torch.tensor(0.0, device=self.device)

        if match_labels is not None:
            if not isinstance(match_labels, torch.Tensor):
                match_labels = torch.LongTensor(match_labels)
            match_labels = match_labels.to(self.device)

            # Concatenate GPS and Route trajectory representations
            combined_rep = torch.cat([gps_traj_joint_rep, route_traj_joint_rep], dim=-1)
            match_logits = self.matching_predictor(combined_rep)
            matching_loss = self.match_criterion(match_logits, match_labels)
        else:
            # Contrastive matching using queue
            # Normalize representations
            gps_feat = F.normalize(gps_traj_joint_rep, dim=-1)
            route_feat = F.normalize(route_traj_joint_rep, dim=-1)

            # Compute similarity with queue
            batch_size = gps_feat.shape[0]

            # Positive pairs: GPS and Route from same trajectory
            pos_sim = torch.sum(gps_feat * route_feat, dim=-1, keepdim=True) / self.tau

            # Negative pairs: GPS with Route queue
            # Clone the queue buffers to avoid inplace operation errors during backward pass
            # (the queues are modified by _dequeue_and_enqueue after the forward pass)
            neg_sim_gps = torch.mm(gps_feat, self.route_queue.clone()) / self.tau

            # Negative pairs: Route with GPS queue
            neg_sim_route = torch.mm(route_feat, self.gps_queue.clone()) / self.tau

            # Contrastive loss (InfoNCE)
            logits_gps = torch.cat([pos_sim, neg_sim_gps], dim=-1)
            logits_route = torch.cat([pos_sim, neg_sim_route], dim=-1)

            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            loss_gps = self.match_criterion(logits_gps, labels)
            loss_route = self.match_criterion(logits_route, labels)

            matching_loss = (loss_gps + loss_route) / 2

            # Update queues
            self._dequeue_and_enqueue(gps_feat.detach(), route_feat.detach())

        # ===================== Combined Loss =====================
        total_loss = (
            self.mlm_loss_weight * gps_mlm_loss +
            self.mlm_loss_weight * route_mlm_loss +
            self.match_loss_weight * matching_loss
        ) / (2 * self.mlm_loss_weight + self.match_loss_weight)

        return total_loss

    def get_trajectory_representation(self, batch):
        """
        Get trajectory representations for downstream tasks.

        This method provides a clean interface for extracting trajectory
        embeddings after pre-training.

        Args:
            batch: Input batch dictionary

        Returns:
            Dict containing various representations:
                - 'gps_traj': GPS trajectory-level representation
                - 'route_traj': Route trajectory-level representation
                - 'joint_gps_traj': Joint GPS trajectory representation
                - 'joint_route_traj': Joint Route trajectory representation
                - 'combined': Concatenated joint representations
        """
        with torch.no_grad():
            outputs = self.forward(batch)

        return {
            'gps_traj': outputs[1],
            'route_traj': outputs[3],
            'joint_gps_traj': outputs[5],
            'joint_route_traj': outputs[7],
            'combined': torch.cat([outputs[5], outputs[7]], dim=-1)
        }
