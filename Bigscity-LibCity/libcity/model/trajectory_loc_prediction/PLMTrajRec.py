"""
PLMTrajRec: Pre-trained Language Model based Trajectory Recovery

This model is adapted from the original PTR (Pre-trained Trajectory Recovery) implementation
for the LibCity framework.

Key Components:
1. LearnableFourierPositionalEncoding: Fourier feature-based positional encoding for GPS coordinates
2. TemporalPositionalEncoding: Standard sinusoidal positional encoding for temporal sequences
3. ReprogrammingLayer: Multi-head attention layer for trajectory reprogramming
4. Decoder: Dual-head decoder for road segment ID and movement rate prediction
5. spatialTemporalConv: Spatial-temporal convolution for road condition processing
6. BERT backbone with LoRA fine-tuning

Adaptations for LibCity:
- Unified all sub-models into a single PLMTrajRec class
- Adapted batch input format to LibCity's trajectory batch dictionary
- Implemented predict() and calculate_loss() methods following LibCity conventions
- Extracted hyperparameters from config dict
- Extracted data features from data_feature dict
- Made BERT model path configurable via config parameter
- Added fallback for missing BERT model (uses simple transformer)

Original files:
- repos/PLMTrajRec/model/model.py (PTR, Decoder, ReprogrammingLayer)
- repos/PLMTrajRec/model/layer.py (LearnableFourierPositionalEncoding, TemporalPositionalEncoding, BERT)
- repos/PLMTrajRec/model/loss_fn.py (Custom accuracy metrics)

Required config parameters:
    - hidden_dim: Hidden dimension for embeddings (default: 512)
    - conv_kernel: Kernel size for trajectory convolution (default: 9)
    - soft_traj_num: Number of soft trajectory prompts (default: 128)
    - road_candi: Whether to use road candidate information (default: true)
    - dropout: Dropout rate (default: 0.3)
    - lambda1: Weight for rate prediction loss (default: 10)
    - bert_model_path: Path to BERT model (default: 'bert-base-uncased')
    - use_lora: Whether to use LoRA fine-tuning (default: true)
    - lora_r: LoRA rank (default: 8)
    - lora_alpha: LoRA alpha parameter (default: 32)
    - lora_dropout: LoRA dropout rate (default: 0.01)
    - default_keep_ratio: Default GPS keep ratio for prompt (default: 0.125)

Required data_feature parameters:
    - id_size: Number of road segment IDs
    - grid_size: Grid size for spatial encoding (optional)
    - time_slots: Number of time slots (optional)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libcity.model.abstract_model import AbstractModel

# Attempt to import BERT-related libraries
try:
    from transformers import BertModel, BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Attempt to import LoRA from peft
try:
    from peft import LoraModel, LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# ============================================================================
# Helper Layers (from layer.py)
# ============================================================================

class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Features for positional encoding.

    Reference: https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)

    Computes the Fourier feature positional encoding of a multi-dimensional position.
    Used for encoding GPS coordinates (latitude, longitude).

    Args:
        input_dim: Input dimension (2 for lat/lng)
        hidden_dim: Output hidden dimension
    """

    def __init__(self, input_dim, hidden_dim):
        super(LearnableFourierPositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projection for Fourier features
        self.Wr = nn.Linear(self.input_dim, self.hidden_dim // 2, bias=False)

        # MLP for final projection
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        """
        Produce positional encodings from x.

        Args:
            x: Tensor of shape [batch, seq_len, 2] representing GPS coordinates

        Returns:
            Positional encoding tensor of shape [batch, seq_len, hidden_dim]
        """
        B, T, F = x.shape
        # Step 1: Compute Fourier features
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        fourier_features = 1 / np.sqrt(self.hidden_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2: Apply MLP
        Y = self.mlp(fourier_features)
        return Y


class TemporalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for temporal sequences.

    Args:
        d_model: Model dimension
        dropout: Dropout probability
        max_len: Maximum sequence length (default: 2000)
        lookup_index: Optional lookup index for specific positions
    """

    def __init__(self, d_model, dropout, max_len=2000, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Positional encoding tensor (batch_size, seq_len, d_model)
        """
        if self.lookup_index is not None:
            return self.dropout(self.pe[:, :, self.lookup_index, :].detach())
        else:
            return self.dropout(self.pe[:, :x.size(1), :].detach())


# ============================================================================
# Model Components (from model.py)
# ============================================================================

class ReprogrammingLayer(nn.Module):
    """
    Reprogramming Layer for trajectory embedding transformation.

    Uses cross-attention mechanism to reprogram trajectory embeddings
    using learned soft prompts.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_keys: Key dimension (default: d_model // n_heads)
        attention_dropout: Dropout for attention weights
    """

    def __init__(self, d_model, n_heads, d_keys=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Forward pass for reprogramming.

        Args:
            target_embedding: Target embeddings [batch, seq_len, d_model]
            source_embedding: Source embeddings (soft prompts) [num_prompts, d_model]
            value_embedding: Value embeddings [num_prompts, d_model]

        Returns:
            Reprogrammed embeddings [batch, seq_len, d_model]
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """Cross-attention based reprogramming."""
        B, L, H, E = target_embedding.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Decoder(nn.Module):
    """
    Dual-head Decoder for road segment ID and movement rate prediction.

    Args:
        id_size: Number of road segment IDs
        hidden_dim: Hidden dimension
    """

    def __init__(self, id_size, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.id_size = id_size

        # Road ID prediction head
        self.road_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.id_size + 1)  # +1 for padding/unknown
        )

        # Movement rate prediction head
        self.rate_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, hidden, prompt_length):
        """
        Decode hidden states into road ID and rate predictions.

        Args:
            hidden: Hidden states from BERT [batch, prompt_len + seq_len, hidden_dim]
            prompt_length: Length of prompt tokens to skip

        Returns:
            road_ID: Log softmax probabilities [seq_len, batch, id_size + 1]
            road_rate: Sigmoid rates [seq_len, batch, 1]
        """
        # Skip prompt tokens
        hidden = hidden[:, prompt_length:, ]
        road_ID = F.log_softmax(self.road_fc(hidden), dim=-1)
        road_rate = torch.sigmoid(self.rate_fc(hidden))

        return road_ID, road_rate


class TrajConv(nn.Module):
    """
    1D Convolution layer for trajectory processing.

    Args:
        hidden_dim: Hidden dimension
        kernel_size: Convolution kernel size
    """

    def __init__(self, hidden_dim, kernel_size):
        super(TrajConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def forward(self, x):
        """
        Apply 1D convolution to trajectory.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Convolved tensor [batch, seq_len, hidden_dim]
        """
        x = x.permute(0, 2, 1)  # [batch, hidden_dim, seq_len]
        traj_x = self.conv(x)
        return traj_x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]


class SpatialTemporalConv(nn.Module):
    """
    Spatial-Temporal Convolution for road condition processing.

    Processes road condition matrix through spatial 2D convolution
    followed by temporal 1D convolution.

    Args:
        in_channel: Input channels
        base_channel: Base hidden channels
    """

    def __init__(self, in_channel, base_channel):
        super(SpatialTemporalConv, self).__init__()
        self.start_conv = nn.Conv2d(in_channel, base_channel, 1, 1, 0)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channel)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(base_channel, base_channel, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, road_condition):
        """
        Process road condition matrix.

        Args:
            road_condition: Road condition tensor [T, N, N]

        Returns:
            Processed tensor [T, N, N, hidden_dim]
        """
        T, N, _ = road_condition.shape
        _start = self.start_conv(road_condition.unsqueeze(1))  # [T, 1, N, N] -> [T, F, N, N]
        spatialConv = self.spatial_conv(_start)  # [T, F, N, N]
        spatial_reshape = spatialConv.reshape(T, -1, N * N).permute(2, 1, 0)  # [N*N, F, T]
        temporalConv = self.temporal_conv(spatial_reshape)
        conv_res = temporalConv.reshape(N, N, -1, T).permute(3, 2, 0, 1)  # [T, F, N, N]
        return (_start + conv_res).permute(0, 2, 3, 1)  # [T, N, N, F]


class BERTWrapper(nn.Module):
    """
    BERT wrapper with optional LoRA fine-tuning.

    Args:
        bert_model_path: Path to pre-trained BERT model
        use_lora: Whether to apply LoRA fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        hidden_dim: Model hidden dimension (for projection if different from BERT's hidden size)
    """

    def __init__(self, bert_model_path='bert-base-uncased', use_lora=True,
                 lora_r=8, lora_alpha=32, lora_dropout=0.01, hidden_dim=512):
        super(BERTWrapper, self).__init__()

        self.use_lora = use_lora
        self.bert_model_path = bert_model_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.hidden_dim = hidden_dim

        # Initialize projection layers as None (will be set if needed)
        self.input_projection = None
        self.output_projection = None

        if HAS_TRANSFORMERS:
            try:
                self.model = BertModel.from_pretrained(bert_model_path)
                self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
                self.bert_available = True

                # Get BERT's hidden size (768 for bert-base-uncased)
                bert_hidden_size = self.model.config.hidden_size

                # Add projection layers if hidden_dim != bert_hidden_size
                if hidden_dim != bert_hidden_size:
                    self.input_projection = nn.Linear(hidden_dim, bert_hidden_size)
                    self.output_projection = nn.Linear(bert_hidden_size, hidden_dim)

                if use_lora and HAS_PEFT:
                    self.model = self._apply_lora(self.model)
            except Exception as e:
                print(f"Warning: Failed to load BERT model from {bert_model_path}: {e}")
                print("Using fallback transformer encoder.")
                self.bert_available = False
        else:
            print("Warning: transformers library not available. Using fallback transformer encoder.")
            self.bert_available = False

        if not self.bert_available:
            self._build_fallback_encoder()

    def _apply_lora(self, model):
        """Apply LoRA fine-tuning to BERT model."""
        if not HAS_PEFT:
            return model

        lora_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=self.lora_dropout,
        )
        return LoraModel(model, lora_config, 'bert_lora')

    def _build_fallback_encoder(self):
        """Build a fallback transformer encoder if BERT is not available."""
        # Use model's hidden_dim for consistency
        self.fallback_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )

    def forward(self, x, padding_mask):
        """
        Forward pass through BERT or fallback encoder.

        Args:
            x: Input embeddings [batch, seq_len, hidden_dim]
            padding_mask: Attention mask [batch, seq_len]

        Returns:
            Encoder hidden states [batch, seq_len, hidden_dim]
        """
        if self.bert_available:
            # Project input to BERT's expected dimension if needed
            if self.input_projection is not None:
                x = self.input_projection(x)

            encoder_hidden = self.model(
                inputs_embeds=x,
                attention_mask=padding_mask,
                output_hidden_states=True
            ).hidden_states[-1]

            # Project output back to model's hidden dimension if needed
            if self.output_projection is not None:
                encoder_hidden = self.output_projection(encoder_hidden)

            return encoder_hidden
        else:
            # Use fallback encoder
            # Convert padding mask to attention mask (True = masked)
            attn_mask = (padding_mask == 0)
            output = self.fallback_encoder(x, src_key_padding_mask=attn_mask)
            return output


class SimpleBERTTokenEmbedding(nn.Module):
    """
    Simple embedding layer to replace BERT token embedding lookup.

    Used for MASK and PAD tokens when BERT is not available.

    Args:
        hidden_dim: Embedding dimension
    """

    def __init__(self, hidden_dim):
        super(SimpleBERTTokenEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        # Learnable MASK and PAD token embeddings
        self.mask_embedding = nn.Parameter(torch.randn(hidden_dim))
        self.pad_embedding = nn.Parameter(torch.randn(hidden_dim))

    def get_mask_token(self):
        return self.mask_embedding

    def get_pad_token(self):
        return self.pad_embedding


# ============================================================================
# Main PLMTrajRec Model
# ============================================================================

class PLMTrajRec(AbstractModel):
    """
    PLMTrajRec: Pre-trained Language Model based Trajectory Recovery

    This model uses a pre-trained BERT model with LoRA fine-tuning for
    sparse trajectory recovery. It predicts road segment IDs and movement
    rates for each point in a trajectory.

    The model uses:
    1. Learnable Fourier Features for GPS coordinate encoding
    2. Soft trajectory prompts for task specification
    3. ReprogrammingLayer for trajectory transformation
    4. BERT backbone for sequence modeling
    5. Dual-head decoder for road ID and rate prediction

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data features including road ID vocabulary size

    Required config parameters:
        - hidden_dim: Hidden dimension (default: 512)
        - conv_kernel: Convolution kernel size (default: 9)
        - soft_traj_num: Number of soft prompts (default: 128)
        - road_candi: Use road candidates (default: True)
        - dropout: Dropout rate (default: 0.3)
        - lambda1: Rate loss weight (default: 10)
        - bert_model_path: BERT model path (default: 'bert-base-uncased')
        - use_lora: Use LoRA fine-tuning (default: True)
        - lora_r, lora_alpha, lora_dropout: LoRA parameters
        - default_keep_ratio: GPS keep ratio for prompt (default: 0.125)

    Required data_feature:
        - id_size: Number of road segment IDs (default: 2505)
    """

    def __init__(self, config, data_feature):
        super(PLMTrajRec, self).__init__(config, data_feature)

        # Device
        self.device = config.get('device', 'cpu')

        # Model dimensions
        self.hidden_dim = config.get('hidden_dim', 512)
        self.conv_kernel = config.get('conv_kernel', 9)
        self.soft_traj_num = config.get('soft_traj_num', 128)
        self.road_candi = config.get('road_candi', True)
        self.dropout = config.get('dropout', 0.3)

        # Loss weight
        self.lambda1 = config.get('lambda1', 10)

        # BERT configuration
        self.bert_model_path = config.get('bert_model_path', 'bert-base-uncased')
        self.use_lora = config.get('use_lora', True)
        self.lora_r = config.get('lora_r', 8)
        self.lora_alpha = config.get('lora_alpha', 32)
        self.lora_dropout = config.get('lora_dropout', 0.01)

        # Prompt configuration
        self.default_keep_ratio = config.get('default_keep_ratio', 0.125)

        # Data dimensions from data_feature
        self.id_size = data_feature.get('id_size', config.get('id_size', 2505))
        self.grid_size = data_feature.get('grid_size', config.get('grid_size', 64))

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""

        # Learnable Fourier Positional Encoding for GPS coordinates
        self.LFF = LearnableFourierPositionalEncoding(2, self.hidden_dim)

        # Input embedding layer
        self.input_embed = nn.Linear(2, self.hidden_dim)

        # Spatial-Temporal Convolution for road conditions
        self.ST_conv = SpatialTemporalConv(1, self.hidden_dim)

        # Input projection layers
        self.global_input_protext = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.local_input_protext = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Token embedding for MASK and PAD
        self.token_embed = SimpleBERTTokenEmbedding(self.hidden_dim)

        # Road segment embedding
        self.road_embed = nn.Parameter(
            torch.randn(self.id_size, self.hidden_dim),
            requires_grad=True
        )

        # Temporal positional encoding
        self.position_embed = TemporalPositionalEncoding(self.hidden_dim, self.dropout)

        # Input fusion layer for road candidates
        if self.road_candi:
            self.input_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Prompt processing layer
        self.prompt_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Trajectory convolution
        self.Traj_conv = TrajConv(self.hidden_dim, self.conv_kernel)

        # Time delay embeddings
        self.forward_delay = nn.Linear(1, self.hidden_dim)
        self.backward_delay = nn.Linear(1, self.hidden_dim)

        # Local feature fusion
        self.local_cat_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # BERT backbone
        self.bert = BERTWrapper(
            bert_model_path=self.bert_model_path,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            hidden_dim=self.hidden_dim
        )

        # Decoder for road ID and rate prediction
        self.Decoder = Decoder(self.id_size, self.hidden_dim)

        # Learnable MASK token for unknown positions
        self.learnable_mask_token = nn.Parameter(
            torch.randn(1, self.hidden_dim),
            requires_grad=True
        )

        # Soft trajectory prompts
        self.soft_traj_prompt = nn.Parameter(
            torch.randn(self.soft_traj_num, self.hidden_dim),
            requires_grad=True
        )

        # Reprogramming layer
        self.ReprogrammingLayer = ReprogrammingLayer(self.hidden_dim, 8)
        self.ReprogrammingLayer_cat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Time prompt embedding
        self.time_prompt_embed = nn.Linear(2, self.hidden_dim)

        # Road condition merge layer
        self.road_condition_merge = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.road_out = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize learnable parameters."""
        stdv = 1. / math.sqrt(self.soft_traj_prompt.shape[1])
        self.soft_traj_prompt.data.uniform_(-stdv, stdv)

        stdv1 = 1. / math.sqrt(self.learnable_mask_token.shape[1])
        self.learnable_mask_token.data.uniform_(-stdv1, stdv1)

    def GPS_road_embed(self, src_lat, src_lng, mask_index, padd_index, src_candi_id, learned_mask_prompt=None):
        """
        Compute GPS and road candidate embeddings.

        Args:
            src_lat: Source latitudes [batch, seq_len]
            src_lng: Source longitudes [batch, seq_len]
            mask_index: Mask for unknown positions [batch, seq_len]
            padd_index: Padding mask [batch, seq_len]
            src_candi_id: Road candidate IDs [batch, seq_len, num_candidates]
            learned_mask_prompt: Learned mask prompt embeddings (optional)

        Returns:
            src_input: Combined GPS and road embeddings [batch, seq_len, hidden_dim]
        """
        src_data = torch.cat((src_lat.unsqueeze(-1), src_lng.unsqueeze(-1)), -1)
        lff = self.LFF(src_data)  # Fourier encoding for observed points

        B, T, _ = lff.shape

        # Project GPS embeddings
        src_gps_hidden = self.global_input_protext(lff)

        # Get token embeddings
        MASK_token = self.token_embed.get_mask_token().to(self.device)
        PAD_token = self.token_embed.get_pad_token().to(self.device)

        # Replace unknown positions with learned mask prompt or MASK token
        if learned_mask_prompt is not None:
            src_gps_hidden[mask_index == 1] = learned_mask_prompt[mask_index == 1]
        else:
            src_gps_hidden[mask_index == 1] = MASK_token

        # Replace padding positions
        src_gps_hidden[padd_index == 1] = PAD_token

        src_input = src_gps_hidden

        # Process road candidates if available
        if self.road_candi and src_candi_id is not None:
            # Handle both sparse index format [batch, seq_len] and dense one-hot format [batch, seq_len, id_size]
            if src_candi_id.dim() == 2:
                # Sparse format: use embedding lookup instead of matmul
                src_road_canid = self.road_embed[src_candi_id.long()]  # [batch, seq_len, hidden_dim]
            else:
                # Dense format (original behavior): weighted average of multiple candidates
                src_road_canid = torch.matmul(src_candi_id, self.road_embed)
                candi_road = src_candi_id.sum(2).unsqueeze(-1)
                src_road_canid = src_road_canid / (candi_road + 1e-6)

            # Replace mask positions
            if learned_mask_prompt is not None:
                src_road_canid[mask_index == 1] = learned_mask_prompt[mask_index == 1]
            else:
                src_road_canid[mask_index == 1] = MASK_token

            src_road_canid[padd_index == 1] = PAD_token

            # Concatenate and project
            src_input = self.input_layer(torch.cat((src_gps_hidden, src_road_canid), -1))
            src_input = self.Traj_conv(src_input)

        return src_input

    def local_model(self, src_input, forward_delta_t, backward_delta_t, forward_index, backward_index, mask_index, padd_index):
        """
        Local interpolation model using forward/backward neighbors.

        Args:
            src_input: Input embeddings [batch, seq_len, hidden_dim]
            forward_delta_t: Time delta to forward neighbor [batch, seq_len]
            backward_delta_t: Time delta to backward neighbor [batch, seq_len]
            forward_index: Index of forward neighbor [batch, seq_len]
            backward_index: Index of backward neighbor [batch, seq_len]
            mask_index: Mask for unknown positions [batch, seq_len]
            padd_index: Padding mask [batch, seq_len]

        Returns:
            local_out: Interpolated embeddings [batch, seq_len, hidden_dim]
        """
        B, T, C = src_input.shape

        forward_lff = src_input.clone()
        backward_lff = src_input.clone()

        # Get forward and backward embeddings for masked positions
        mask = mask_index.bool()
        forward_lff[mask] = forward_lff[torch.arange(B, device=self.device)[:, None], forward_index][mask]
        backward_lff[mask] = backward_lff[torch.arange(B, device=self.device)[:, None], backward_index][mask]

        # Compute time-based weights
        forward_delta = torch.exp(-F.relu(self.forward_delay(forward_delta_t.unsqueeze(-1))))
        backward_delta = torch.exp(-F.relu(self.backward_delay(backward_delta_t.unsqueeze(-1))))

        # Weighted interpolation
        local_out = (forward_delta * forward_lff + backward_delta * backward_lff) / (forward_delta + backward_delta + 1e-6)

        return local_out

    def mask_prompt(self, road_condition, road_condition_xyt_index, forward_delta_t, backward_delta_t, forward_index, backward_index, mask_index, padd_index):
        """
        Generate mask prompt embeddings using road conditions and time deltas.

        Args:
            road_condition: Road condition matrix [T, N, N]
            road_condition_xyt_index: XYT indices for trajectory [batch, seq_len, 3]
            forward_delta_t: Time delta to forward neighbor [batch, seq_len]
            backward_delta_t: Time delta to backward neighbor [batch, seq_len]
            forward_index: Index of forward neighbor [batch, seq_len]
            backward_index: Index of backward neighbor [batch, seq_len]
            mask_index: Mask for unknown positions [batch, seq_len]
            padd_index: Padding mask [batch, seq_len]

        Returns:
            out: Mask prompt embeddings [batch, seq_len, hidden_dim]
        """
        B, T = forward_delta_t.shape

        # Time embedding from forward and backward deltas
        times_embed = self.time_prompt_embed(
            torch.cat((forward_delta_t.unsqueeze(-1), backward_delta_t.unsqueeze(-1)), -1)
        ) + self.learnable_mask_token

        # Process road conditions if available
        if road_condition is not None:
            road_condition_conv = self.ST_conv(road_condition)  # [T, N, N, hidden_dim]

            x = road_condition_xyt_index[:, :, 0]
            y = road_condition_xyt_index[:, :, 1]
            t = road_condition_xyt_index[:, :, 2]
            trajectory_road_condition = road_condition_conv[t, x, y]  # [batch, seq_len, hidden_dim]

            # Interpolate road conditions for masked positions
            forward_lff = trajectory_road_condition.clone()
            backward_lff = trajectory_road_condition.clone()

            mask = mask_index.bool()
            forward_lff[mask] = forward_lff[torch.arange(B, device=self.device)[:, None], forward_index][mask]
            backward_lff[mask] = backward_lff[torch.arange(B, device=self.device)[:, None], backward_index][mask]

            forward_delta = torch.exp(-F.relu(self.forward_delay(forward_delta_t.unsqueeze(-1))))
            backward_delta = torch.exp(-F.relu(self.backward_delay(backward_delta_t.unsqueeze(-1))))

            road_condition_out = (forward_delta * forward_lff + backward_delta * backward_lff) / (forward_delta + backward_delta + 1e-6)
            road_condition_out = self.road_out(road_condition_out)

            out = torch.cat((road_condition_out, times_embed), -1)
            out = self.road_condition_merge(out)
        else:
            out = times_embed

        return out

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: LibCity Batch object containing trajectory data.
                Required keys:
                    - 'src_lat': Source latitudes [batch, seq_len]
                    - 'src_lng': Source longitudes [batch, seq_len]
                    - 'mask_index': Mask for unknown positions [batch, seq_len]
                    - 'padd_index': Padding mask [batch, seq_len]
                    - 'traj_length': Trajectory lengths [batch]
                Optional keys:
                    - 'src_candi_id': Road candidate IDs [batch, seq_len, num_candidates]
                    - 'prompt_token': Pre-computed prompt tokens [batch, prompt_len, hidden_dim]
                    - 'road_condition': Road condition matrix [T, N, N]
                    - 'road_condition_xyt_index': XYT indices [batch, seq_len, 3]
                    - 'forward_delta_t': Time delta to forward neighbor [batch, seq_len]
                    - 'backward_delta_t': Time delta to backward neighbor [batch, seq_len]
                    - 'forward_index': Forward neighbor index [batch, seq_len]
                    - 'backward_index': Backward neighbor index [batch, seq_len]
                    - 'keep_ratio': GPS keep ratio for prompt generation

        Returns:
            outputs_id: Road ID log probabilities [seq_len, batch, id_size + 1]
            outputs_rate: Movement rate predictions [seq_len, batch, 1]
        """
        # Extract required data from batch
        src_lat = batch['src_lat'].to(self.device)
        src_lng = batch['src_lng'].to(self.device)
        mask_index = batch['mask_index'].to(self.device)
        padd_index = batch['padd_index'].to(self.device)
        traj_length = batch['traj_length']

        if isinstance(traj_length, torch.Tensor):
            traj_length = traj_length.tolist()

        B, T = src_lat.shape

        # Optional inputs
        try:
            src_candi_id = batch['src_candi_id'].to(self.device).float()
        except KeyError:
            src_candi_id = None

        # Get or generate prompt tokens
        try:
            prompt_token = batch['prompt_token'].to(self.device)
        except KeyError:
            # Generate default prompt tokens using soft prompts
            prompt_token = self.soft_traj_prompt.unsqueeze(0).expand(B, -1, -1)

        prompt_token = self.prompt_layer(prompt_token)
        _, prompt_length, _ = prompt_token.shape

        # Get time delta information
        try:
            forward_delta_t = batch['forward_delta_t'].to(self.device)
            backward_delta_t = batch['backward_delta_t'].to(self.device)
        except KeyError:
            forward_delta_t = torch.zeros(B, T, device=self.device)
            backward_delta_t = torch.zeros(B, T, device=self.device)

        try:
            forward_index = batch['forward_index'].to(self.device)
            backward_index = batch['backward_index'].to(self.device)
        except KeyError:
            forward_index = torch.zeros(B, T, dtype=torch.long, device=self.device)
            backward_index = torch.zeros(B, T, dtype=torch.long, device=self.device)

        # Get road condition information
        try:
            road_condition = batch['road_condition'].to(self.device).float()
        except KeyError:
            road_condition = None

        try:
            road_condition_xyt_index = batch['road_condition_xyt_index'].to(self.device)
        except KeyError:
            road_condition_xyt_index = None

        # Generate learned mask prompt
        learned_mask_prompt = self.mask_prompt(
            road_condition, road_condition_xyt_index,
            forward_delta_t, backward_delta_t,
            forward_index, backward_index,
            mask_index, padd_index
        )

        # Compute GPS and road embeddings
        src_input = self.GPS_road_embed(src_lat, src_lng, mask_index, padd_index, src_candi_id, learned_mask_prompt)

        # Apply reprogramming layer
        src_input = self.ReprogrammingLayer(src_input, self.soft_traj_prompt, self.soft_traj_prompt)

        # Concatenate prompt tokens
        src_input = torch.cat((prompt_token, src_input), dim=1)

        # Add positional encoding
        pe = self.position_embed(src_input)
        src_input = src_input + pe

        # Create padding mask for BERT
        _padding_mask = torch.arange(T, device=self.device).unsqueeze(0) < torch.tensor(traj_length, device=self.device).unsqueeze(1)
        _padding_mask = _padding_mask.float()

        prompt_mask = torch.ones((B, prompt_length), device=self.device)
        _padding_mask = torch.cat((prompt_mask, _padding_mask), 1)

        # Forward through BERT
        bert_out = self.bert(src_input, _padding_mask)

        # Decode to road ID and rate predictions
        outputs_id, outputs_rate = self.Decoder(bert_out, prompt_length)

        # Apply output mask
        outputs_id_mask = torch.arange(T, device=self.device)[None, :, None] < torch.tensor(traj_length, device=self.device)[:, None, None]
        outputs_rate_mask = torch.arange(T, device=self.device)[None, :, None] < torch.tensor(traj_length, device=self.device)[:, None, None]

        outputs_id = outputs_id * outputs_id_mask
        outputs_rate = outputs_rate * outputs_rate_mask

        # Permute to [seq_len, batch, ...] format
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate

    def predict(self, batch):
        """
        Prediction method for LibCity.

        Args:
            batch: Input batch dictionary

        Returns:
            scores: Location prediction scores [batch, id_size + 1]
        """
        outputs_id, outputs_rate = self.forward(batch)

        # outputs_id shape: [seq_len, batch, id_size + 1]
        # For next-location prediction, return scores for all positions
        # Permute to [batch, seq_len, id_size + 1]
        outputs_id = outputs_id.permute(1, 0, 2)

        # Get trajectory lengths to extract last valid position predictions
        traj_length = batch['traj_length']
        if isinstance(traj_length, torch.Tensor):
            traj_length = traj_length.cpu().tolist()

        B = outputs_id.shape[0]
        device = outputs_id.device

        # Extract predictions for the last valid position of each sequence
        last_indices = torch.tensor([l - 1 for l in traj_length], device=device, dtype=torch.long)
        batch_indices = torch.arange(B, device=device, dtype=torch.long)

        # Get last position predictions: [batch, id_size + 1]
        scores = outputs_id[batch_indices, last_indices, :]

        return scores

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: LibCity Batch object containing:
                - trajectory data for forward pass
                - 'target_road_id': Target road segment IDs [batch, seq_len]
                - 'target_rate': Target movement rates [batch, seq_len]

        Returns:
            loss: Combined loss tensor (ID loss + lambda1 * rate loss)
        """
        outputs_id, outputs_rate = self.forward(batch)

        # Get targets
        trg_id = batch['target_road_id'].to(self.device)
        trg_rate = batch['target_rate'].to(self.device)
        traj_length = batch['traj_length']

        if isinstance(traj_length, torch.Tensor):
            traj_length = traj_length.tolist()

        B = trg_id.shape[0]
        T = trg_id.shape[1]

        # Permute outputs back to [batch, seq_len, ...] for loss computation
        outputs_id = outputs_id.permute(1, 0, 2)  # [batch, seq_len, id_size + 1]
        outputs_rate = outputs_rate.permute(1, 0, 2)  # [batch, seq_len, 1]

        # Create mask for valid positions
        valid_mask = torch.arange(T, device=self.device).unsqueeze(0) < torch.tensor(traj_length, device=self.device).unsqueeze(1)

        # Road ID loss (NLL loss)
        # Reshape for loss computation
        outputs_id_flat = outputs_id.reshape(-1, outputs_id.shape[-1])
        trg_id_flat = trg_id.reshape(-1).long()

        # Apply mask
        valid_mask_flat = valid_mask.reshape(-1)
        outputs_id_valid = outputs_id_flat[valid_mask_flat]
        trg_id_valid = trg_id_flat[valid_mask_flat]

        id_loss = F.nll_loss(outputs_id_valid, trg_id_valid)

        # Rate loss (MSE loss)
        outputs_rate_flat = outputs_rate.reshape(-1)
        trg_rate_flat = trg_rate.reshape(-1)

        outputs_rate_valid = outputs_rate_flat[valid_mask_flat]
        trg_rate_valid = trg_rate_flat[valid_mask_flat]

        rate_loss = F.mse_loss(outputs_rate_valid, trg_rate_valid)

        # Combined loss
        total_loss = id_loss + self.lambda1 * rate_loss

        return total_loss
