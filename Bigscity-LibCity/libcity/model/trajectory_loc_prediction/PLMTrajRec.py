# coding: utf-8
"""
PLMTrajRec: Pre-trained Language Model for Trajectory Recovery

This module adapts the PTR (Pre-trained Trajectory Recovery) model from:
    PLMTrajRec: https://github.com/xxx/PLMTrajRec (original repository)

Key adaptations made for LibCity:
1. Inherited from AbstractModel as per LibCity convention for trajectory prediction
2. Adapted data handling to use LibCity's batch dictionary format
3. Implemented predict() and calculate_loss() methods following LibCity conventions
4. Made BERT model path configurable through config parameters
5. Integrated road condition data loading from data_feature

Required data_feature keys:
- id_size: Number of road segment IDs
- mbr: Minimum Bounding Rectangle (dict with min_lat, max_lat, min_lng, max_lng)
- road_condition: Road condition flow data (numpy array or tensor)

Required config parameters:
- bert_model_path: Path to BERT model (default: 'bert-base-uncased')
- hidden_dim: Hidden dimension size (default: 512)
- conv_kernel: Kernel size for trajectory convolution (default: 9)
- soft_traj_num: Number of soft trajectory tokens (default: 128)
- road_candi: Whether to use road candidate features (default: True)
- dropout: Dropout rate (default: 0.3)
- lambda1: Weight for rate prediction loss (default: 10)
- use_lora: Whether to use LoRA for BERT fine-tuning (default: True)
- lora_r: LoRA attention dimension (default: 8)
- lora_alpha: LoRA alpha scaling parameter (default: 32)
- lora_dropout: LoRA dropout rate (default: 0.01)

Original Model Components:
- LearnableFourierPositionalEncoding: Learnable Fourier features for GPS encoding
- TemporalPositionalEncoding: Temporal positional encoding
- ReprogrammingLayer: Multi-head attention based reprogramming
- spatialTemporalConv: Spatial-temporal convolution for road conditions
- BERT with LoRA: Pre-trained language model for sequence encoding
- Decoder: FC layers for road ID and rate prediction
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_model import AbstractModel

# Try to import BERT and LoRA dependencies
try:
    from transformers import BertModel, BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraModel, LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def init_weights(module):
    """
    Keras-style initialization for RNN weights.
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in module.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in module.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in module.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)


class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Features for GPS positional encoding.
    Reference: https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)

    Computes positional encoding for multi-dimensional positions (lat, lng).
    """
    def __init__(self, input_dim, hidden_dim):
        super(LearnableFourierPositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wr = nn.Linear(self.input_dim, self.hidden_dim // 2, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, seq_len, 2) representing GPS coordinates
        Returns:
            positional encoding of shape (batch, seq_len, hidden_dim)
        """
        B, T, F = x.shape
        # Compute Fourier features
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        fourier_features = 1 / np.sqrt(self.hidden_dim) * torch.cat([cosines, sines], dim=-1)
        # Compute projected Fourier features
        output = self.mlp(fourier_features)
        return output


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding using sinusoidal functions.
    """
    def __init__(self, d_model, dropout, max_len=2000, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)  # (1, T_max, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, T, F_in)
        Returns:
            positional encoding of shape (batch_size, T, F_out)
        """
        if self.lookup_index is not None:
            output = x + self.pe[:, :, self.lookup_index, :]
        else:
            output = x + self.pe[:, :x.size(1), :]
        return self.dropout(output.detach())


class ReprogrammingLayer(nn.Module):
    """
    Multi-head attention based reprogramming layer.
    Used for adapting trajectory embeddings using soft prompts.
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
        B, L, H, E = target_embedding.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class SpatialTemporalConv(nn.Module):
    """
    Spatial-temporal convolution for road condition encoding.
    Processes grid-based road condition data with spatial and temporal convolutions.
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
        Args:
            road_condition: tensor of shape (T, N, N) - road condition grids over time
        Returns:
            tensor of shape (T, N, N, F) - encoded road conditions
        """
        T, N, _ = road_condition.shape
        _start = self.start_conv(road_condition.unsqueeze(1))  # (T, base_channel, N, N)
        spatial_out = self.spatial_conv(_start)  # (T, F, N, N)
        spatial_reshape = spatial_out.reshape(T, -1, N * N).permute(2, 1, 0)  # (N*N, F, T)
        temporal_out = self.temporal_conv(spatial_reshape)
        conv_res = temporal_out.reshape(N, N, -1, T).permute(3, 2, 0, 1)  # (T, F, N, N)
        return (_start + conv_res).permute(0, 2, 3, 1)  # (T, N, N, F)


class TrajConv(nn.Module):
    """
    1D convolution along trajectory dimension.
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
        Args:
            x: tensor of shape (B, T, F)
        Returns:
            tensor of shape (B, T, F)
        """
        x = x.permute(0, 2, 1)
        traj_x = self.conv(x)
        return traj_x.permute(0, 2, 1)


class Decoder(nn.Module):
    """
    Decoder for predicting road segment IDs and movement ratios.
    """
    def __init__(self, id_size, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.id_size = id_size

        self.road_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.id_size + 1)
        )
        self.rate_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, hidden, prompt_length):
        """
        Args:
            hidden: encoder output of shape (B, prompt_len + T, hidden_dim)
            prompt_length: length of prompt tokens to skip
        Returns:
            road_ID: log softmax probabilities of shape (B, T, id_size+1)
            road_rate: sigmoid activated rates of shape (B, T, 1)
        """
        hidden = hidden[:, prompt_length:, ]
        road_ID = F.log_softmax(self.road_fc(hidden), dim=-1)
        road_rate = torch.sigmoid(self.rate_fc(hidden))
        return road_ID, road_rate


class BERTEncoder(nn.Module):
    """
    BERT encoder with optional LoRA fine-tuning.
    Wraps HuggingFace BERT model with LoRA adapters.
    Includes projection layers to handle dimension mismatch between model hidden_dim and BERT hidden size.
    """
    def __init__(self, model_path, input_dim=512, use_lora=True, lora_r=8, lora_alpha=32, lora_dropout=0.01):
        super(BERTEncoder, self).__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for BERTEncoder. "
                            "Install with: pip install transformers")

        self.model_path = model_path
        self.use_lora = use_lora
        self.input_dim = input_dim

        # Load BERT model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
        self.model = BertModel.from_pretrained(model_path)

        # Get BERT's hidden size (typically 768 for bert-base)
        self.bert_hidden_size = self.model.config.hidden_size

        # Projection layers to handle dimension mismatch
        # Project from input_dim (e.g., 512) to BERT hidden size (768)
        self.to_bert_dim = nn.Linear(input_dim, self.bert_hidden_size)
        # Project from BERT hidden size (768) back to input_dim (e.g., 512)
        self.from_bert_dim = nn.Linear(self.bert_hidden_size, input_dim)

        # Apply LoRA if enabled
        if use_lora and HAS_PEFT:
            lora_config = LoraConfig(
                task_type="SEQ_2_SEQ_LM",
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=lora_dropout,
            )
            self.lora_model = LoraModel(self.model, lora_config, 'bert')
        else:
            self.lora_model = self.model

    def forward(self, x, padding_mask):
        """
        Args:
            x: input embeddings of shape (B, T, input_dim)
            padding_mask: attention mask of shape (B, T)
        Returns:
            encoder output of shape (B, T, input_dim)
        """
        # Project input to BERT dimension
        x_bert = self.to_bert_dim(x)  # (B, T, bert_hidden_size)

        encoder_hidden = self.lora_model(
            inputs_embeds=x_bert,
            attention_mask=padding_mask,
            output_hidden_states=True
        ).hidden_states[-1]  # (B, T, bert_hidden_size)

        # Project output back to input dimension
        output = self.from_bert_dim(encoder_hidden)  # (B, T, input_dim)
        return output

    def get_token_embedding(self, token):
        """Get embedding for a special token (MASK or PAD)."""
        bert_token = self.model.state_dict()['embeddings.word_embeddings.weight']
        if token == "MASK":
            tokens_tensor = torch.tensor([103])
        elif token == "PAD":
            tokens_tensor = torch.tensor([0])
        else:
            raise ValueError(f"Unknown token: {token}")
        return bert_token[tokens_tensor]


class PLMTrajRec(AbstractModel):
    """
    PLMTrajRec: Pre-trained Language Model for Trajectory Recovery

    This model uses BERT with LoRA for sparse trajectory recovery task.
    It predicts road segment IDs and movement ratios for each trajectory point.

    The model architecture includes:
    1. Learnable Fourier Positional Encoding for GPS coordinates
    2. Road candidate embedding with attention
    3. Spatial-temporal convolution for road conditions
    4. BERT encoder with LoRA for sequence modeling
    5. Decoder for road ID and rate prediction
    """

    def __init__(self, config, data_feature):
        super(PLMTrajRec, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data feature parameters
        self.id_size = data_feature.get('id_size', 2505)
        self.mbr = data_feature.get('mbr', {
            'min_lat': 30.655,
            'max_lat': 30.727,
            'min_lng': 104.043,
            'max_lng': 104.129
        })

        # Model hyperparameters from config
        self.hidden_dim = config.get('hidden_dim', 512)
        self.conv_kernel = config.get('conv_kernel', 9)
        self.soft_traj_num = config.get('soft_traj_num', 128)
        self.road_candi = config.get('road_candi', True)
        self.dropout = config.get('dropout', 0.3)
        self.lambda1 = config.get('lambda1', 10)

        # BERT configuration
        self.bert_model_path = config.get('bert_model_path', 'bert-base-uncased')
        self.use_lora = config.get('use_lora', True)
        self.lora_r = config.get('lora_r', 8)
        self.lora_alpha = config.get('lora_alpha', 32)
        self.lora_dropout = config.get('lora_dropout', 0.01)

        # Keep ratio configuration (for different sampling rates)
        self.default_keep_ratio = config.get('default_keep_ratio', 0.125)

        # Load road condition data
        road_condition = data_feature.get('road_condition', None)
        if road_condition is not None:
            if isinstance(road_condition, np.ndarray):
                road_condition = torch.from_numpy(road_condition).float()
            self.register_buffer('road_condition', road_condition)
        else:
            # Create dummy road condition if not provided
            self.register_buffer('road_condition', torch.zeros(24, 64, 64))

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build all model components."""
        # Learnable Fourier Positional Encoding for GPS
        self.LFF = LearnableFourierPositionalEncoding(2, self.hidden_dim)

        # Input embedding
        self.input_embed = nn.Linear(2, self.hidden_dim)

        # Spatial-temporal convolution for road conditions
        self.ST_conv = SpatialTemporalConv(1, self.hidden_dim)

        # Projection layers
        self.global_input_project = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.local_input_project = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Road embedding (learnable)
        self.road_embed = nn.Parameter(
            torch.randn(self.id_size, self.hidden_dim),
            requires_grad=True
        )

        # Positional encoding
        self.position_embed = TemporalPositionalEncoding(self.hidden_dim, self.dropout)

        # Road candidate input layer
        if self.road_candi:
            self.input_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Prompt layer
        self.prompt_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Trajectory convolution
        self.Traj_conv = TrajConv(self.hidden_dim, self.conv_kernel)

        # Forward and backward delay layers
        self.forward_delay = nn.Linear(1, self.hidden_dim)
        self.backward_delay = nn.Linear(1, self.hidden_dim)

        # Local concatenation layer
        self.local_cat_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # BERT encoder with dimension projection (hidden_dim -> BERT 768 -> hidden_dim)
        self.bert = BERTEncoder(
            self.bert_model_path,
            input_dim=self.hidden_dim,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )

        # Create learnable MASK and PAD tokens matching model hidden_dim
        # (don't use BERT tokens to avoid dimension mismatch - BERT uses 768, model uses hidden_dim)
        self.MASK_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.PAD_token = nn.Parameter(torch.zeros(self.hidden_dim))
        nn.init.normal_(self.MASK_token, std=0.02)
        nn.init.normal_(self.PAD_token, std=0.02)

        # Decoder
        self.Decoder = Decoder(self.id_size, self.hidden_dim)

        # Learnable mask token
        self.learnable_mask_token = nn.Parameter(
            torch.randn(1, self.hidden_dim),
            requires_grad=True
        )

        # Soft trajectory prompt
        self.soft_traj_prompt = nn.Parameter(
            torch.randn(self.soft_traj_num, self.hidden_dim),
            requires_grad=True
        )

        # Reprogramming layer
        self.ReprogrammingLayer = ReprogrammingLayer(self.hidden_dim, 8)
        self.ReprogrammingLayer_cat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Time prompt embedding
        self.time_prompt_embed = nn.Linear(2, self.hidden_dim)

        # Road condition merge
        self.road_condition_merge = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.road_out = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Learnable soft prompts (replacing prompt_token from encoder)
        # This was previously stored in the cache file but is now a model parameter
        self.soft_prompts = nn.Parameter(torch.zeros(self.soft_traj_num, self.hidden_dim))
        nn.init.xavier_uniform_(self.soft_prompts)

        # Loss functions
        self.criterion_ce = nn.NLLLoss()
        self.criterion_reg = nn.MSELoss()

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Reset learnable parameters."""
        stdv = 1. / math.sqrt(self.soft_traj_prompt.shape[1])
        self.soft_traj_prompt.data.uniform_(-stdv, stdv)

        stdv1 = 1. / math.sqrt(self.learnable_mask_token.shape[1])
        self.learnable_mask_token.data.uniform_(-stdv1, stdv1)

    def GPS_road_embed(self, src_lat, src_lng, mask_index, padd_index, src_candi_id, learned_mask_prompt=None):
        """
        Embed GPS coordinates with road candidate information.

        Args:
            src_lat: latitude tensor (B, T)
            src_lng: longitude tensor (B, T)
            mask_index: mask indices (B, T) - 1 for masked positions
            padd_index: padding indices (B, T) - 1 for padded positions
            src_candi_id: road candidate IDs (B, T, id_size)
            learned_mask_prompt: optional learned mask prompts

        Returns:
            src_input: encoded input tensor (B, T, hidden_dim)
        """
        src_data = torch.cat((src_lat.unsqueeze(-1), src_lng.unsqueeze(-1)), -1)
        lff = self.LFF(src_data)  # Fourier encoding for GPS

        B, T, _ = lff.shape

        # Project GPS embeddings
        src_gps_hidden = self.global_input_project(lff)

        # Apply learned mask prompt or MASK token
        if learned_mask_prompt is not None:
            src_gps_hidden[mask_index == 1] = learned_mask_prompt[mask_index == 1]
        else:
            src_gps_hidden[mask_index == 1] = self.MASK_token

        src_gps_hidden[padd_index == 1] = self.PAD_token

        src_input = src_gps_hidden

        if self.road_candi:
            # Embed road candidates
            src_road_canid = torch.matmul(src_candi_id, self.road_embed)
            candi_road = src_candi_id.sum(2).unsqueeze(-1)
            src_road_canid = src_road_canid / (candi_road + 1e-6)

            # Apply mask
            if learned_mask_prompt is not None:
                src_road_canid[mask_index == 1] = learned_mask_prompt[mask_index == 1]
            else:
                src_road_canid[mask_index == 1] = self.MASK_token

            src_road_canid[padd_index == 1] = self.PAD_token

            # Combine GPS and road candidate embeddings
            src_input = self.input_layer(torch.cat((src_gps_hidden, src_road_canid), -1))
            src_input = self.Traj_conv(src_input)

        return src_input

    def mask_prompt(self, road_condition, road_condition_xyt_index, forward_delta_t, backward_delta_t,
                    forward_index, backward_index, mask_index, padd_index):
        """
        Generate learned mask prompts based on road conditions and temporal information.

        Args:
            road_condition: road condition tensor (T, N, N)
            road_condition_xyt_index: indices for road condition lookup (B, T, 3)
            forward_delta_t: forward time deltas (B, T)
            backward_delta_t: backward time deltas (B, T)
            forward_index: forward known point indices (B, T)
            backward_index: backward known point indices (B, T)
            mask_index: mask indices (B, T)
            padd_index: padding indices (B, T)

        Returns:
            out: mask prompt tensor (B, T, hidden_dim)
        """
        B, T = forward_delta_t.shape

        # Time embedding
        times_embed = self.time_prompt_embed(
            torch.cat((forward_delta_t.unsqueeze(-1), backward_delta_t.unsqueeze(-1)), -1)
        ) + self.learnable_mask_token

        # Road condition convolution
        road_condition_conv = self.ST_conv(road_condition)  # (T, N, N, F)

        # Extract trajectory road conditions
        x = road_condition_xyt_index[:, :, 0]
        y = road_condition_xyt_index[:, :, 1]
        t = road_condition_xyt_index[:, :, 2]
        trajectory_road_condition = road_condition_conv[t, x, y]  # (B, T, F)

        # Interpolate for masked positions
        forward_lff = trajectory_road_condition.clone()
        backward_lff = trajectory_road_condition.clone()

        mask = mask_index.bool()
        forward_lff[mask] = forward_lff[torch.arange(B, device=mask.device)[:, None], forward_index.long()][mask]
        backward_lff[mask] = backward_lff[torch.arange(B, device=mask.device)[:, None], backward_index.long()][mask]

        # Temporal weighting
        forward_delta = torch.exp(-F.relu(self.forward_delay(forward_delta_t.unsqueeze(-1))))
        backward_delta = torch.exp(-F.relu(self.backward_delay(backward_delta_t.unsqueeze(-1))))

        road_condition_out = (forward_delta * forward_lff + backward_delta * backward_lff) / (forward_delta + backward_delta)
        road_condition_out = self.road_out(road_condition_out)

        # Merge road condition and time embeddings
        out = torch.cat((road_condition_out, times_embed), -1)
        out = self.road_condition_merge(out)

        return out

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - 'src_lat': source latitude tensor (B, T)
                - 'src_lng': source longitude tensor (B, T)
                - 'mask_index': mask indices (B, T)
                - 'padd_index': padding indices (B, T)
                - 'src_candi_id': sparse road candidate IDs (B, T) - integer indices
                - 'traj_length': trajectory lengths list
                - 'road_condition_xyt_index': road condition indices (B, T, 3)
                - 'forward_delta_t': forward time deltas (B, T)
                - 'backward_delta_t': backward time deltas (B, T)
                - 'forward_index': forward known point indices (B, T)
                - 'backward_index': backward known point indices (B, T)

        Returns:
            outputs_id: road ID predictions (T, B, id_size+1)
            outputs_rate: movement rate predictions (T, B, 1)
        """
        # Extract batch data
        src_lat = batch['src_lat']
        src_lng = batch['src_lng']
        mask_index = batch['mask_index']
        padd_index = batch['padd_index']
        src_candi_id_sparse = batch['src_candi_id']  # (B, T) - sparse integer indices
        traj_length = batch['traj_length']
        # Convert to tensor if it's a list (from 'no_tensor' type)
        if isinstance(traj_length, list):
            traj_length = torch.tensor(traj_length, dtype=torch.long, device=src_lat.device)
        road_condition_xyt_index = batch['road_condition_xyt_index']
        forward_delta_t = batch['forward_delta_t']
        backward_delta_t = batch['backward_delta_t']
        forward_index = batch['forward_index']
        backward_index = batch['backward_index']

        B, T = src_lat.shape

        # Convert sparse src_candi_id to one-hot for road candidate embedding
        # src_candi_id_sparse: (B, T) -> src_candi_id: (B, T, id_size)
        src_candi_id = F.one_hot(src_candi_id_sparse.long(), num_classes=self.id_size).float()

        # Use learnable soft_prompts instead of prompt_token from batch
        # Expand to batch size: (soft_traj_num, hidden_dim) -> (B, soft_traj_num, hidden_dim)
        prompt_token = self.soft_prompts.unsqueeze(0).expand(B, -1, -1)

        # Process prompt tokens
        prompt_token = self.prompt_layer(prompt_token)
        _, prompt_length, _ = prompt_token.shape

        # Generate learned mask prompts
        learned_mask_prompt = self.mask_prompt(
            self.road_condition, road_condition_xyt_index,
            forward_delta_t, backward_delta_t,
            forward_index, backward_index,
            mask_index, padd_index
        )

        # Embed GPS and road candidates
        src_input = self.GPS_road_embed(
            src_lat, src_lng, mask_index, padd_index,
            src_candi_id, learned_mask_prompt
        )

        # Apply reprogramming
        src_input = self.ReprogrammingLayer(
            src_input, self.soft_traj_prompt, self.soft_traj_prompt
        )

        # Concatenate prompt tokens
        src_input = torch.cat((prompt_token, src_input), dim=1)

        # Add positional encoding
        pe = self.position_embed(src_input)
        src_input = src_input + pe

        # Create padding mask for BERT
        _padding_mask = torch.arange(T, device=src_input.device).unsqueeze(0) < traj_length.unsqueeze(1)
        _padding_mask = _padding_mask.float()

        prompt_mask = torch.ones((B, prompt_length), device=src_input.device)
        _padding_mask = torch.cat((prompt_mask, _padding_mask), 1)

        # BERT encoding
        bert_out = self.bert(src_input, _padding_mask)

        # Decode predictions
        outputs_id, outputs_rate = self.Decoder(bert_out, prompt_length)

        # Apply length mask
        outputs_id_mask = torch.arange(T, device=src_input.device)[None, :, None] < traj_length[:, None, None]
        outputs_rate_mask = outputs_id_mask

        outputs_id = outputs_id * outputs_id_mask
        outputs_rate = outputs_rate * outputs_rate_mask

        # Permute to (T, B, F)
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate

    def predict(self, batch):
        """
        Predict road segment IDs and movement ratios.

        Args:
            batch: Dictionary containing input data

        Returns:
            Dictionary with:
                - 'road_id': predicted road segment IDs (T, B)
                - 'road_rate': predicted movement ratios (T, B)
        """
        outputs_id, outputs_rate = self.forward(batch)

        # Get argmax for road IDs
        predicted_road_id = outputs_id.argmax(-1)  # (T, B)
        predicted_rate = outputs_rate.squeeze(-1)  # (T, B)

        return {
            'road_id': predicted_road_id,
            'road_rate': predicted_rate
        }

    def calculate_loss(self, batch):
        """
        Calculate combined loss for road ID classification and rate regression.

        Args:
            batch: Dictionary containing:
                - All forward() inputs
                - 'target_road_id': target road IDs (B, T)
                - 'target_rate': target movement rates (B, T)

        Returns:
            torch.Tensor: Total loss (scalar)
        """
        outputs_id, outputs_rate = self.forward(batch)

        # Get targets
        target_road_id = batch['target_road_id']  # (B, T)
        target_rate = batch['target_rate']  # (B, T)
        traj_length = batch['traj_length']

        # Prepare targets - permute to (T, B)
        trg_id = target_road_id.permute(1, 0).long()
        trg_rate = target_rate.permute(1, 0).unsqueeze(-1)

        # Calculate ID loss
        output_ids_dim = outputs_id.shape[-1]
        output_ids = outputs_id.reshape(-1, output_ids_dim)
        tmp_trg_id = trg_id.reshape(-1)
        loss_ids = self.criterion_ce(output_ids, tmp_trg_id)

        # Calculate rate loss
        loss_rates = self.criterion_reg(outputs_rate, trg_rate) * self.lambda1

        # Combined loss
        total_loss = loss_ids + loss_rates

        return total_loss
