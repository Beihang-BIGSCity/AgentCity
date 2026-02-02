"""
MDTI: Multi-modal Dual Transformer for Travel Time Estimation

This model implements a multi-modal architecture combining:
1. Grid-based spatial representations (using Grid Transformer + GAT)
2. Road network representations (using Road GNN + Road Transformer)
3. Cross-modal fusion (using Inter Transformer)

The model supports two modes:
1. Pre-training mode: Contrastive learning + Masked Language Modeling
2. TTE (Travel Time Estimation) mode: Fine-tuning for travel time prediction

Original paper source: repos/MDTI
Adapted for LibCity framework.

Key adaptations:
- Inherits from AbstractTrafficStateModel
- Uses LibCity's config.get() pattern for parameters
- Implements predict() and calculate_loss() methods
- Handles data through LibCity's batch dictionary format
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

# Import supporting modules
from libcity.model.eta.mdti_modules.GridTrm import GridTrm
from libcity.model.eta.mdti_modules.RoadTrm import RoadTrm
from libcity.model.eta.mdti_modules.InterTrm import InterTrm
from libcity.model.eta.mdti_modules.RoadGNN import RoadGNN
from libcity.model.eta.mdti_modules.GridToGraph import GridToGraph
from libcity.model.eta.mdti_modules.GridConv import GridConv

try:
    from torch_geometric.nn import GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not installed. MDTI requires torch_geometric.")


def create_unified_mask(road_traj, mask_ratio=0.15):
    """
    Create BERT-style unified random mask - all batches use the same random positions.

    Args:
        road_traj: [batch_size, seq_len] Original road trajectory
        mask_ratio: float, Mask ratio (default: 15%)

    Returns:
        mask_positions: [num_mask] Masked position indices
        original_tokens: [batch_size * num_mask] Original tokens at masked positions
    """
    batch_size, seq_len = road_traj.shape
    device = road_traj.device

    # Generate random mask positions (shared across all batches)
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_positions = torch.randperm(seq_len, device=device)[:num_mask]
    mask_positions = mask_positions.sort()[0]  # Sort for easier debugging

    # Extract original tokens at masked positions
    original_tokens = road_traj[:, mask_positions]  # [batch_size, num_mask]
    original_tokens = original_tokens.reshape(-1)  # [batch_size * num_mask]

    return mask_positions, original_tokens


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])


class GraphEncoder(nn.Module):
    """Simple GAT-based graph encoder."""

    def __init__(self, input_size, output_size):
        super(GraphEncoder, self).__init__()
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("GraphEncoder requires torch_geometric.")
        self.layer1 = GATConv(input_size, output_size)
        self.layer2 = GATConv(output_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.activation(self.layer1(x, edge_index))
        x = self.activation(self.layer2(x, edge_index))
        return x


class MDTI(AbstractTrafficStateModel):
    """
    MDTI: Multi-modal Dual Transformer for Travel Time Estimation.

    This model combines grid-based and road network representations for
    accurate travel time estimation using multi-modal learning.

    Required config parameters:
        - hidden_emb_dim: Hidden embedding dimension (default: 256)
        - out_emb_dim: Output embedding dimension (default: 256)
        - pe_dropout: Positional encoding dropout (default: 0.1)
        - grid_in_channel: Grid input channels (default: 3)
        - grid_out_channel: Grid output channels (default: 64)
        - grid_H: Grid height (default: 114)
        - grid_W: Grid width (default: 52)
        - grid_trm_head: Grid transformer heads (default: 4)
        - grid_trm_layer: Grid transformer layers (default: 2)
        - grid_trm_dropout: Grid transformer dropout (default: 0.1)
        - grid_ffn_dim: Grid FFN dimension (default: hidden_emb_dim * 4)
        - road_type: Number of road types (default: 8)
        - road_trm_head: Road transformer heads (default: 4)
        - road_trm_layer: Road transformer layers (default: 4)
        - road_trm_dropout: Road transformer dropout (default: 0.1)
        - road_ffn_dim: Road FFN dimension (default: hidden_emb_dim * 4)
        - g_fea_size: Road graph feature size (from data_feature)
        - g_dim_per_layer: Road GNN dimensions per layer
        - g_heads_per_layer: Road GNN heads per layer
        - g_num_layers: Number of Road GNN layers
        - g_dropout: Road GNN dropout (default: 0.1)
        - inter_trm_head: Inter transformer heads (default: 2)
        - inter_trm_layer: Inter transformer layers (default: 2)
        - inter_trm_dropout: Inter transformer dropout (default: 0.1)
        - inter_ffn_dim: Inter FFN dimension (default: out_emb_dim * 4)
        - road_num: Number of road segments (from data_feature)
        - mask_ratio: Mask ratio for MLM pre-training (default: 0.15)
        - mode: 'pretrain' or 'tte' (default: 'tte')
        - tuning_all: Whether to tune all parameters in TTE mode (default: True)

    Required data_feature:
        - road_num: Number of road segments
        - g_fea_size: Road graph feature size
    """

    def __init__(self, config, data_feature):
        super(MDTI, self).__init__(config, data_feature)

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("MDTI requires torch_geometric. "
                            "Install it with: pip install torch_geometric")

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Model hyperparameters
        self.hidden_emb_dim = config.get('hidden_emb_dim', 256)
        self.out_emb_dim = config.get('out_emb_dim', 256)
        self.pe_dropout = config.get('pe_dropout', 0.1)

        # Grid encoder parameters
        self.grid_in_channel = config.get('grid_in_channel', 3)
        self.grid_out_channel = config.get('grid_out_channel', 64)
        self.grid_H = config.get('grid_H', 114)
        self.grid_W = config.get('grid_W', 52)
        self.grid_trm_head = config.get('grid_trm_head', 4)
        self.grid_trm_layer = config.get('grid_trm_layer', 2)
        self.grid_trm_dropout = config.get('grid_trm_dropout', 0.1)
        self.grid_ffn_dim = config.get('grid_ffn_dim', self.hidden_emb_dim * 4)

        # Road encoder parameters
        self.road_type = config.get('road_type', 8)
        self.road_trm_head = config.get('road_trm_head', 4)
        self.road_trm_layer = config.get('road_trm_layer', 4)
        self.road_trm_dropout = config.get('road_trm_dropout', 0.1)
        self.road_ffn_dim = config.get('road_ffn_dim', self.hidden_emb_dim * 4)

        # Road GNN parameters
        self.g_fea_size = data_feature.get('g_fea_size', config.get('g_fea_size', 64))
        self.g_dim_per_layer = config.get('g_dim_per_layer', [self.hidden_emb_dim] * 3)
        self.g_heads_per_layer = config.get('g_heads_per_layer', [4, 4, 4])
        self.g_num_layers = config.get('g_num_layers', 3)
        self.g_dropout = config.get('g_dropout', 0.1)

        # Inter transformer parameters
        self.inter_trm_head = config.get('inter_trm_head', 2)
        self.inter_trm_layer = config.get('inter_trm_layer', 2)
        self.inter_trm_dropout = config.get('inter_trm_dropout', 0.1)
        self.inter_ffn_dim = config.get('inter_ffn_dim', self.out_emb_dim * 4)

        # Road vocabulary parameters
        self.road_num = data_feature.get('road_num', config.get('road_num', 10000))
        self.road_special_tokens = config.get('road_special_tokens', {
            'padding_token': 0,
            'cls_token': 1,
            'mask_token': 2,
        })
        self.grid_special_tokens = config.get('grid_special_tokens', {
            'padding_token': 0,
            'cls_token': 1,
        })

        # Training parameters
        self.mask_ratio = config.get('mask_ratio', 0.15)
        self.mode = config.get('mode', 'tte')  # 'pretrain' or 'tte'
        self.tuning_all = config.get('tuning_all', True)

        # Loss weights
        self.w_cl = config.get('w_cl', 1.0)
        self.w_mlm = config.get('w_mlm', 0.7)

        # Build model components
        self._build_model()

        # Extract global grid_image from data_feature (stored once, not per-trajectory)
        # This avoids memory issues caused by storing grid_image for each trajectory
        if 'grid_image' in data_feature:
            grid_image_data = data_feature.get('grid_image')
            self.register_buffer('global_grid_image',
                                torch.FloatTensor(grid_image_data))
            # Log the grid image shape for debugging
            print(f"[MDTI] Loaded global grid_image from data_feature: "
                  f"shape {self.global_grid_image.shape}")
        else:
            # Fallback: create a minimal grid image
            self.register_buffer('global_grid_image',
                                torch.zeros(self.grid_H, self.grid_W, 3))

    def _build_model(self):
        """Build all model components."""
        # Positional encoding
        self.pe = PositionalEncoding(self.hidden_emb_dim, self.pe_dropout)

        # Grid encoder components
        self.grid_cls_token = nn.Parameter(torch.randn(self.hidden_emb_dim))
        self.grid_padding_token = nn.Parameter(torch.zeros(self.hidden_emb_dim), requires_grad=False)

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Grid to Graph converter
        self.grid_to_graph = GridToGraph(self.grid_in_channel, self.grid_H, self.grid_W)

        # Grid GAT encoder
        self.grid_gat = GraphEncoder(self.grid_out_channel, 128)

        # Grid GAT output projection
        if self.grid_out_channel != self.hidden_emb_dim:
            self.grid_gat_proj_to_transformer = nn.Linear(128, self.hidden_emb_dim)
        else:
            self.grid_gat_proj_to_transformer = nn.Identity()

        # Grid convolution (alternative to GAT)
        self.grid_conv = GridConv(self.grid_in_channel, self.grid_out_channel)

        # Grid raw projection
        self.grid_raw_proj = nn.Linear(self.grid_in_channel, self.grid_out_channel)

        # Prompt fusion projection (if using external prompts)
        if self.hidden_emb_dim != 768:
            self.prompt_fusion_proj = nn.Linear(768, self.hidden_emb_dim)
        else:
            self.prompt_fusion_proj = nn.Identity()

        # Grid transformer
        self.grid_enc = GridTrm(
            self.hidden_emb_dim,
            self.grid_ffn_dim,
            self.grid_trm_head,
            self.grid_trm_layer,
            self.grid_trm_dropout,
        )

        # Fusion linear for grid features (embedding + 4 additional features)
        self.fusion_linear = nn.Linear(self.hidden_emb_dim + 4, self.hidden_emb_dim)

        # Road encoder components
        self.week_emb_layer = nn.Embedding(7 + 1, self.hidden_emb_dim, padding_idx=0)
        self.minute_emb_layer = nn.Embedding(1440 + 1, self.hidden_emb_dim, padding_idx=0)

        self.road_cls_token = nn.Parameter(torch.randn(self.hidden_emb_dim))
        self.road_padding_token = nn.Parameter(torch.zeros(self.hidden_emb_dim), requires_grad=False)
        self.road_mask_token = nn.Parameter(torch.randn(self.hidden_emb_dim))

        # Road GNN
        self.road_emb_layer = RoadGNN(
            self.g_fea_size,
            self.g_dim_per_layer,
            self.g_heads_per_layer,
            self.g_num_layers,
            self.g_dropout
        )

        # Road type embedding
        self.type_emb_layer = nn.Embedding(self.road_type + 1, self.hidden_emb_dim, padding_idx=0)

        # Road transformer
        self.road_enc = RoadTrm(
            self.hidden_emb_dim,
            self.road_ffn_dim,
            self.road_trm_head,
            self.road_trm_layer,
            self.road_trm_dropout,
        )

        # Inter-modal fusion layer
        self.inter_layer = InterTrm(
            self.out_emb_dim,
            self.inter_ffn_dim,
            self.inter_trm_head,
            self.inter_trm_layer,
            self.inter_trm_dropout,
        )

        # Contrastive learning projection layers
        self.grid_cl_linear = nn.Linear(self.hidden_emb_dim, self.out_emb_dim)
        self.road_cl_linear = nn.Linear(self.hidden_emb_dim, self.out_emb_dim)

        # Temperature parameter for contrastive learning
        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # MLM prediction head
        self.road_mlm_linear = nn.Linear(
            self.out_emb_dim,
            self.road_num + len(self.road_special_tokens)
        )

        # TTE prediction head
        self.tte_proj = nn.Sequential(
            nn.Linear(self.out_emb_dim, self.out_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.out_emb_dim // 2, 1)
        )

    def _encode_road(self, road_data):
        """
        Encode road trajectory data.

        Args:
            road_data: Dictionary containing road data

        Returns:
            road_seq_emb: Road sequence embeddings after transformer
            road_traj_emb: Road trajectory embedding ([CLS] token)
            g_emb: Graph node embeddings
            road_padding_mask: Padding mask
        """
        g_input_feature = road_data['g_input_feature']
        g_edge_index = road_data['g_edge_index']
        road_traj = road_data['road_traj']
        road_weeks = road_data['road_weeks']
        road_minutes = road_data['road_minutes']
        road_type = road_data['road_type']

        # Temporal embeddings
        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)

        # Road graph embedding
        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])

        # Road sequence embedding
        road_seq_emb = g_emb[road_traj] + road_weeks_emb + road_minutes_emb
        road_seq_emb = self.pe(road_seq_emb)

        # Road type embedding
        road_type_emb = self.pe(self.type_emb_layer(road_type))

        # Padding mask
        road_padding_mask = road_traj > 0

        # Road transformer encoding
        road_seq_emb = self.road_enc(
            src=road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask
        )

        # Project to output dimension
        road_seq_emb = self.road_cl_linear(road_seq_emb)
        road_traj_emb = road_seq_emb[:, 0]  # [CLS] token

        return road_seq_emb, road_traj_emb, g_emb, road_padding_mask, road_weeks_emb, road_minutes_emb, road_type_emb

    def _encode_grid(self, grid_data):
        """
        Encode grid trajectory data.

        Args:
            grid_data: Dictionary containing grid data

        Returns:
            grid_seq_emb: Grid sequence embeddings after transformer
            grid_traj_emb: Grid trajectory embedding ([CLS] token)
            grid_padding_mask: Padding mask
        """
        # Use global grid_image (stored in model, not in batch) to save memory
        # The grid_image is shared across all trajectories
        grid_image = self.global_grid_image
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']

        # Get batch size from grid_traj
        batch_size = grid_traj.shape[0]

        # Handle image dimensions - expand to batch size
        # global_grid_image is (H, W, C), need (B, H, W, C)
        if grid_image.dim() == 3:
            grid_image = grid_image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        grid_image = grid_image.permute(0, 3, 1, 2)  # (B, C, H, W)
        grid_image = self.transform(grid_image)

        # Build graph from grid
        grid_graph_batch = self.grid_to_graph(grid_image)

        # Prepare GAT input
        B, C, H, W = grid_image.shape
        grid_image_flat = grid_image.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Project to GAT input dimension
        grid_feat_flat = self.grid_raw_proj(grid_image_flat)

        # GAT encoding
        edge_index = grid_graph_batch.edge_index
        grid_node_emb = self.grid_gat(grid_feat_flat, edge_index)
        grid_node_emb = self.grid_gat_proj_to_transformer(grid_node_emb)

        # Build full grid embedding with special tokens
        full_grid_emb = torch.cat([
            self.grid_padding_token.unsqueeze(0),
            self.grid_cls_token.unsqueeze(0),
            grid_node_emb
        ], dim=0)

        # Grid sequence embedding
        grid_seq_emb = full_grid_emb[grid_traj]
        grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb
        grid_seq_emb = self.pe(grid_seq_emb)

        # Padding mask
        grid_padding_mask = grid_traj > 0

        # Grid transformer encoding
        grid_seq_emb = self.grid_enc(grid_seq_emb, grid_padding_mask)
        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)
        grid_traj_emb = grid_seq_emb[:, 0]

        return grid_seq_emb, grid_traj_emb, grid_padding_mask

    def _extract_grid_data(self, batch):
        """
        Extract grid data from batch using dictionary-style access.

        LibCity's BatchPAD objects don't support .get() method, so we use
        direct dictionary access batch['key'] instead.

        Args:
            batch: Input batch dictionary with individual tensor fields

        Returns:
            grid_data: Dictionary containing grid trajectory data
        """
        return {
            'grid_traj': batch['grid_traj'],
            'grid_time_emb': batch['grid_time_emb'],
            'grid_feature': batch['grid_feature']
        }

    def _extract_road_data(self, batch):
        """
        Extract road data from batch using dictionary-style access.

        For g_input_feature and g_edge_index, we take the first element
        since the road network is the same for all trajectories.

        LibCity's BatchPAD objects don't support .get() method, so we use
        direct dictionary access batch['key'] instead.

        Args:
            batch: Input batch dictionary with individual tensor fields

        Returns:
            road_data: Dictionary containing road trajectory data
        """
        # For no_pad types, batch returns a list of tensors - take first one
        g_input_feature = batch['g_input_feature']
        g_edge_index = batch['g_edge_index']

        # Handle list of tensors (take first since road network is shared)
        if isinstance(g_input_feature, list):
            g_input_feature = g_input_feature[0]
        if isinstance(g_edge_index, list):
            g_edge_index = g_edge_index[0]

        return {
            'road_traj': batch['road_traj'],
            'road_weeks': batch['road_weeks'],
            'road_minutes': batch['road_minutes'],
            'road_type': batch['road_type'],
            'g_input_feature': g_input_feature,
            'g_edge_index': g_edge_index
        }

    def forward_pretrain(self, batch):
        """
        Forward pass for pre-training mode.

        Computes contrastive loss and masked language modeling loss.

        Args:
            batch: Dictionary containing grid_data, road_data

        Returns:
            cl_loss: Contrastive learning loss
            mlm_loss: Masked language modeling loss
            mlm_prediction: MLM prediction logits
        """
        # Extract grid and road data from batch using dictionary access
        # LibCity's BatchPAD objects don't support .get() method
        grid_data = self._extract_grid_data(batch)
        road_data = self._extract_road_data(batch)

        # Encode road
        road_seq_emb, road_traj_emb, g_emb, road_padding_mask, road_weeks_emb, road_minutes_emb, road_type_emb = \
            self._encode_road(road_data)

        # Encode grid
        grid_seq_emb, grid_traj_emb, grid_padding_mask = self._encode_grid(grid_data)

        # Contrastive loss
        road_e = F.normalize(road_traj_emb, dim=-1)
        grid_e = F.normalize(grid_traj_emb, dim=-1)
        logits = torch.matmul(grid_e, road_e.T) / self.temp

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        cl_loss = (loss_i + loss_t) / 2

        # MLM loss
        road_traj = road_data['road_traj']
        mask_positions, original_tokens = create_unified_mask(road_traj, mask_ratio=self.mask_ratio)

        # Create masked trajectory
        mask_road_traj = road_traj.clone()
        mask_road_traj[:, mask_positions] = self.road_special_tokens['mask_token']

        # Encode masked road trajectory
        mask_road_seq_emb = g_emb[mask_road_traj] + road_weeks_emb + road_minutes_emb
        mask_road_seq_emb = self.pe(mask_road_seq_emb)

        mask_road_seq_emb = self.road_enc(
            src=mask_road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask,
        )
        mask_road_seq_emb = self.road_cl_linear(mask_road_seq_emb)

        # Cross-modal fusion
        fusion_seq_emb = self.inter_layer(
            src=mask_road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )

        # MLM prediction
        mlm_prediction = fusion_seq_emb[:, mask_positions]
        mlm_prediction = mlm_prediction.reshape(-1, mlm_prediction.size(-1))
        mlm_prediction = self.road_mlm_linear(mlm_prediction)

        # MLM loss
        mlm_loss = F.cross_entropy(mlm_prediction, original_tokens)

        return cl_loss, mlm_loss, mlm_prediction

    def forward_tte(self, batch):
        """
        Forward pass for TTE (Travel Time Estimation) mode.

        Args:
            batch: Dictionary containing grid_data, road_data

        Returns:
            pred: Predicted travel time tensor
        """
        # Extract grid and road data from batch using dictionary access
        # LibCity's BatchPAD objects don't support .get() method
        grid_data = self._extract_grid_data(batch)
        road_data = self._extract_road_data(batch)

        # Encode road
        road_seq_emb, _, _, road_padding_mask, _, _, _ = self._encode_road(road_data)

        # Encode grid
        grid_seq_emb, _, grid_padding_mask = self._encode_grid(grid_data)

        # Cross-modal fusion
        fusion_seq_emb = self.inter_layer(
            src=road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )

        # Get [CLS] token embedding
        fusion_traj_emb = fusion_seq_emb[:, 0]

        # TTE prediction
        pred = self.tte_proj(fusion_traj_emb)

        return pred

    def forward(self, batch):
        """
        Forward pass dispatcher based on mode.

        Args:
            batch: Input batch dictionary

        Returns:
            Model output (mode-dependent)
        """
        if self.mode == 'pretrain':
            return self.forward_pretrain(batch)
        else:
            return self.forward_tte(batch)

    def predict(self, batch):
        """
        Predict travel time for a batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Predicted travel times [batch]
        """
        pred = self.forward_tte(batch)
        return pred.squeeze(-1)

    def calculate_loss(self, batch):
        """
        Calculate loss based on mode.

        For pre-training: contrastive + MLM loss
        For TTE: MSE loss

        Args:
            batch: Input batch dictionary with 'travel_time' as ground truth

        Returns:
            Total loss tensor
        """
        if self.mode == 'pretrain':
            cl_loss, mlm_loss, _ = self.forward_pretrain(batch)
            total_loss = self.w_cl * cl_loss + self.w_mlm * mlm_loss
            return total_loss
        else:
            pred = self.forward_tte(batch)

            # Get ground truth travel time using dictionary access
            # LibCity's BatchPAD objects don't support .get() method
            travel_time = batch['travel_time']
            if travel_time.dim() == 1:
                travel_time = travel_time.unsqueeze(1)

            # MSE loss
            loss = F.mse_loss(pred, travel_time)
            return loss
