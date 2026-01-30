"""
DiffTraj: A Diffusion-based Trajectory Location Prediction Model
Adapted for LibCity Framework's Trajectory Location Prediction Task

Original Paper: DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model
Source Repository: /home/wangwenrui/shk/AgentCity/repos/DiffTraj

Key Adaptations for LibCity TrajLocPred:
- Inherits from AbstractModel for trajectory location prediction tasks
- Implements forward(), predict(), and calculate_loss() methods
- Converts location IDs to embeddings for diffusion processing
- Uses diffusion to model next location prediction as a denoising task
- Works with LibCity's batch structure: history_loc, current_loc, target

LibCity Batch Format (TrajLocPred):
- batch['history_loc']: Historical location IDs [batch, history_len]
- batch['history_tim']: Historical time encodings [batch, history_len]
- batch['current_loc']: Current trajectory location IDs [batch, seq_len]
- batch['current_tim']: Current time encodings [batch, seq_len]
- batch['target']: Target location ID [batch]

Required Config Parameters:
- num_diffusion_timesteps: Number of diffusion steps (default: 100)
- beta_start: Starting beta value (default: 0.0001)
- beta_end: Ending beta value (default: 0.02)
- hidden_size: Hidden dimension size (default: 256)
- loc_emb_size: Location embedding size (default: 256)
- tim_emb_size: Time embedding size (default: 64)
- dropout: Dropout rate (default: 0.1)
- num_layers: Number of transformer layers (default: 4)
- num_heads: Number of attention heads (default: 8)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libcity.model.abstract_model import AbstractModel


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of timesteps
        embedding_dim: Dimension of the embedding

    Returns:
        Tensor of shape [batch, embedding_dim]
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with time embedding injection for diffusion."""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(DiffusionTransformerBlock, self).__init__()

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb, mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            time_emb: [batch, hidden_size]
            mask: Optional attention mask
        """
        # Add time embedding
        time_shift = self.time_proj(time_emb).unsqueeze(1)  # [batch, 1, hidden_size]
        x = x + time_shift

        # Self-attention
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN
        x = self.norm2(x + self.ffn(x))

        return x


class DiffusionDenoiser(nn.Module):
    """
    Denoising network for diffusion-based location prediction.
    Uses a Transformer architecture with time conditioning.
    """

    def __init__(self, loc_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(DiffusionDenoiser, self).__init__()

        self.hidden_size = hidden_size

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection to location logits
        self.output_proj = nn.Linear(hidden_size, loc_size)

        # Noise prediction head (for denoising)
        self.noise_pred = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, t, context=None, mask=None):
        """
        Args:
            x: Noisy embeddings [batch, seq_len, hidden_size]
            t: Timesteps [batch]
            context: Optional context embeddings [batch, context_len, hidden_size]
            mask: Optional attention mask

        Returns:
            Predicted noise [batch, seq_len, hidden_size]
        """
        # Get time embedding
        time_emb = get_timestep_embedding(t, self.hidden_size)
        time_emb = self.time_embed(time_emb)  # [batch, hidden_size]

        # Concatenate context if provided
        if context is not None:
            x = torch.cat([context, x], dim=1)
            if mask is not None:
                context_mask = torch.zeros(mask.size(0), context.size(1),
                                          dtype=torch.bool, device=mask.device)
                mask = torch.cat([context_mask, mask], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, time_emb, mask)

        # Extract only the target embeddings (last position if no context, otherwise last seq_len positions)
        if context is not None:
            x = x[:, context.size(1):, :]

        # Predict noise
        noise = self.noise_pred(x)

        return noise


class DiffTraj(AbstractModel):
    """
    DiffTraj: Diffusion-based Trajectory Location Prediction Model

    Adapts diffusion models for next location prediction by:
    1. Encoding location sequences as embeddings
    2. Applying diffusion noise to target location embeddings
    3. Using a denoiser to predict and remove noise
    4. Outputting location predictions via classification

    This adaptation uses diffusion in the embedding space rather than
    on coordinates, making it suitable for discrete location prediction.

    Args:
        config: LibCity configuration object
        data_feature: Data feature dictionary containing:
            - loc_size: Number of locations
            - tim_size: Number of time slots
            - loc_pad: Location padding index
            - tim_pad: Time padding index
    """

    def __init__(self, config, data_feature):
        super(DiffTraj, self).__init__(config, data_feature)

        self.device = config.get('device', torch.device('cpu'))

        # Data feature parameters
        self.loc_size = data_feature.get('loc_size', 1000)
        self.tim_size = data_feature.get('tim_size', 48)
        self.loc_pad = data_feature.get('loc_pad', self.loc_size - 1)
        self.tim_pad = data_feature.get('tim_pad', self.tim_size - 1)

        # Model architecture parameters
        self.hidden_size = config.get('hidden_size', 256)
        self.loc_emb_size = config.get('loc_emb_size', 256)
        self.tim_emb_size = config.get('tim_emb_size', 64)
        self.num_layers = config.get('num_layers', 4)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)

        # Diffusion parameters
        self.num_timesteps = config.get('num_diffusion_timesteps', 100)
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.beta_schedule = config.get('beta_schedule', 'linear')

        # Inference parameters
        self.inference_steps = config.get('inference_steps', 10)
        self.evaluate_method = config.get('evaluate_method', 'prob')

        # Build model components
        self._build_embeddings()
        self._build_denoiser()
        self._build_diffusion_schedule()

    def _build_embeddings(self):
        """Build embedding layers for locations and times."""
        self.loc_embedding = nn.Embedding(
            self.loc_size, self.loc_emb_size, padding_idx=self.loc_pad
        )
        self.tim_embedding = nn.Embedding(
            self.tim_size, self.tim_emb_size, padding_idx=self.tim_pad
        )

        # Project combined embeddings to hidden size
        input_size = self.loc_emb_size + self.tim_emb_size
        self.input_proj = nn.Linear(input_size, self.hidden_size)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.hidden_size, dropout=self.dropout)

        # Context encoder for history
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=2
        )

        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.loc_size)

    def _build_denoiser(self):
        """Build the diffusion denoiser network."""
        self.denoiser = DiffusionDenoiser(
            loc_size=self.loc_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

    def _build_diffusion_schedule(self):
        """Build the diffusion noise schedule."""
        if self.beta_schedule == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == 'cosine':
            steps = self.num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, self.num_timesteps, steps) / self.num_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

        self.register_buffer('betas', betas)

        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # For sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # For q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # For sampling
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))

        # Posterior coefficients
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def _extract(self, a, t, x_shape):
        """Extract values from a tensor at timesteps t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x0: Clean embeddings [batch, seq_len, hidden_size]
            t: Timesteps [batch]
            noise: Optional pre-generated noise

        Returns:
            Noisy embeddings x_t and the noise
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        x_t = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    def _encode_sequence(self, loc, tim):
        """
        Encode location and time sequences to embeddings.

        Args:
            loc: Location IDs [batch, seq_len]
            tim: Time IDs [batch, seq_len]

        Returns:
            Encoded embeddings [batch, seq_len, hidden_size]
        """
        loc_emb = self.loc_embedding(loc)  # [batch, seq_len, loc_emb_size]
        tim_emb = self.tim_embedding(tim)  # [batch, seq_len, tim_emb_size]

        combined = torch.cat([loc_emb, tim_emb], dim=-1)  # [batch, seq_len, loc_emb_size + tim_emb_size]
        hidden = self.input_proj(combined)  # [batch, seq_len, hidden_size]
        hidden = self.pos_encoding(hidden)

        return hidden

    def _get_context(self, batch):
        """
        Get context embeddings from history and current trajectory.

        Args:
            batch: LibCity batch dictionary

        Returns:
            Context embeddings [batch, context_len, hidden_size]
        """
        # Encode current trajectory
        current_loc = batch['current_loc']
        current_tim = batch['current_tim']
        current_emb = self._encode_sequence(current_loc, current_tim)

        # Encode history if available
        if 'history_loc' in batch and batch['history_loc'] is not None:
            history_loc = batch['history_loc']
            history_tim = batch['history_tim']

            # Handle different history formats
            if history_loc.dim() == 2:
                history_emb = self._encode_sequence(history_loc, history_tim)
                context = torch.cat([history_emb, current_emb], dim=1)
            else:
                # history_loc might be a list or 3D tensor
                context = current_emb
        else:
            context = current_emb

        # Apply context encoder
        context = self.context_encoder(context)

        return context

    def forward(self, batch):
        """
        Forward pass for training (predicts noise and computes denoised embeddings).

        Args:
            batch: Dictionary containing:
                - 'current_loc': Current trajectory locations [batch, seq_len]
                - 'current_tim': Current trajectory times [batch, seq_len]
                - 'history_loc': Historical locations [batch, history_len]
                - 'history_tim': Historical times [batch, history_len]
                - 'target': Target location ID [batch]

        Returns:
            Location logits [batch, loc_size]
        """
        # Get context from history and current trajectory
        context = self._get_context(batch)

        # Get target location embedding (what we want to predict)
        target = batch['target']

        # Create target embedding for diffusion
        # Use the last context position as a starting point and add target info
        batch_size = target.size(0)

        # Initialize target embedding from learned location embedding
        target_emb = self.loc_embedding(target).unsqueeze(1)  # [batch, 1, loc_emb_size]

        # Project to hidden size (pad with zeros for time component since we don't have target time)
        target_tim = torch.zeros(batch_size, 1, self.tim_emb_size, device=target.device)
        target_combined = torch.cat([target_emb, target_tim], dim=-1)
        target_hidden = self.input_proj(target_combined)  # [batch, 1, hidden_size]

        # Sample random timesteps for diffusion
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=target.device)

        # Add noise to target embedding
        noisy_target, noise = self.q_sample(target_hidden, t)

        # Predict noise using denoiser with context
        predicted_noise = self.denoiser(noisy_target, t, context=context)

        # Compute denoised embedding
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, noisy_target.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, noisy_target.shape
        )
        denoised = (noisy_target - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

        # Project to location logits
        logits = self.output_proj(denoised.squeeze(1))  # [batch, loc_size]

        return logits, predicted_noise, noise

    def predict(self, batch):
        """
        Generate location predictions using reverse diffusion.

        Args:
            batch: Dictionary containing trajectory data

        Returns:
            Log probabilities for each location [batch, loc_size]
        """
        self.eval()

        # Get context
        context = self._get_context(batch)
        batch_size = context.size(0)

        # Start from pure noise
        x = torch.randn(batch_size, 1, self.hidden_size, device=self.device)

        # Reverse diffusion with fewer steps for efficiency
        step_size = max(1, self.num_timesteps // self.inference_steps)

        for t in reversed(range(0, self.num_timesteps, step_size)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.denoiser(x, t_batch, context=context)

            # Compute denoised embedding
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]

            # DDPM sampling step
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t])
            else:
                noise = torch.zeros_like(x)
                sigma = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (self.betas[t] / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + sigma * noise

        # Project final embedding to logits
        logits = self.output_proj(x.squeeze(1))  # [batch, loc_size]
        score = F.log_softmax(logits, dim=-1)

        if self.evaluate_method == 'sample':
            # Build pos_neg_index for sample-based evaluation
            if 'neg_loc' in batch:
                pos_neg_index = torch.cat(
                    (batch['target'].unsqueeze(1), batch['neg_loc']), dim=1
                )
                score = torch.gather(score, 1, pos_neg_index)

        return score

    def calculate_loss(self, batch):
        """
        Calculate training loss combining diffusion loss and classification loss.

        Args:
            batch: Dictionary containing trajectory and target data

        Returns:
            Combined loss tensor
        """
        logits, predicted_noise, target_noise = self.forward(batch)

        # Diffusion loss (MSE between predicted and actual noise)
        diffusion_loss = F.mse_loss(predicted_noise, target_noise)

        # Classification loss (cross-entropy for location prediction)
        target = batch['target']
        classification_loss = F.cross_entropy(logits, target)

        # Combined loss with weighting
        loss_weight = 0.5  # Balance between diffusion and classification
        total_loss = loss_weight * diffusion_loss + (1 - loss_weight) * classification_loss

        return total_loss
