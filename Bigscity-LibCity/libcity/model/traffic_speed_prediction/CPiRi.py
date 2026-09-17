from logging import getLogger
import os

import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class _FallbackTemporalEncoder(nn.Module):
    """Lightweight temporal encoder used when external Sundial is unavailable."""

    def __init__(self, in_dim: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, in_dim]
        return self.net(x)


class CPiRi(AbstractTrafficStateModel):
    """
    CPiRi adaptation for LibCity traffic speed prediction.

    Original CPiRi design highlights:
    1) Frozen Sundial temporal encoder on each node-wise sequence
    2) Spatial Transformer over node representations
    3) Direct multi-step output

    This adapter keeps the same high-level pipeline and adds a robust fallback:
    - If external Sundial dependencies/checkpoint are unavailable, a lightweight
      temporal encoder is used so the model still runs in LibCity.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.config = config
        self._logger = getLogger()

        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = int(data_feature.get('output_dim', config.get('output_dim', 1)))
        self.input_window = int(config.get('input_window', 12))
        self.output_window = int(config.get('output_window', 12))
        self.input_dim = int(config.get('input_dim', 1))
        self._scaler = data_feature.get('scaler')

        # Keep original argument names for compatibility with CPiRi configs.
        self.revin = bool(config.get('revin', False))
        self.from_pretrain = bool(config.get('from_pretrain', True))
        self.spital_num_layers = int(config.get('spital_num_layers', config.get('spatial_num_layers', 4)))

        # Runtime behavior knobs.
        self.use_external_sundial = bool(config.get('use_external_sundial', False))
        self.external_sundial_arch_dir = config.get('external_sundial_arch_dir', '')
        self.external_sundial_ckpt = config.get('external_sundial_ckpt', '')

        self.hidden_size = int(config.get('hidden_size', 256))
        self.dropout = float(config.get('dropout', 0.3))
        self.null_val = float(config.get('null_val', 0.0))

        self._build_temporal_encoder()
        self._build_spatial_module()
        self.output_proj = nn.Linear(self.hidden_size, self.output_window * self.output_dim)

    @staticmethod
    def _get_valid_nhead(hidden_size: int, nhead: int) -> int:
        if hidden_size % nhead == 0:
            return nhead
        # pick the largest valid divisor <= nhead, fallback to 1
        for h in range(min(nhead, hidden_size), 0, -1):
            if hidden_size % h == 0:
                return h
        return 1

    def _build_spatial_module(self):
        nhead = self._get_valid_nhead(self.hidden_size, int(self.config.get('num_attention_heads', 8)))
        ff_dim = int(self.config.get('intermediate_size', self.hidden_size * 4))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
            activation='gelu',
            dropout=self.dropout,
        )
        self.spital_att = nn.TransformerEncoder(
            layer,
            num_layers=self.spital_num_layers,
            enable_nested_tensor=False,
            norm=nn.LayerNorm(self.hidden_size),
        )

    def _build_temporal_encoder(self):
        """
        Try external Sundial-backed temporal encoder, otherwise fallback encoder.
        """
        self.temporal_encoder_type = 'fallback'
        temporal_in_dim = self.input_window * self.input_dim

        if self.use_external_sundial:
            # External path is optional and should never break model construction.
            # We only enable it when dependency and files are valid.
            try:
                self._init_external_sundial_encoder()
                self.temporal_encoder_type = 'external_sundial'
                return
            except Exception as e:
                self._logger.warning(
                    'CPiRi external Sundial init failed; fallback encoder will be used. reason=%s', e
                )

        self.temporal_encoder = _FallbackTemporalEncoder(
            in_dim=temporal_in_dim,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        # Optional warm-start for fallback temporal encoder.
        fallback_ckpt = self.config.get('temporal_encoder_ckpt', None)
        if isinstance(fallback_ckpt, str) and fallback_ckpt:
            if os.path.exists(fallback_ckpt):
                try:
                    state = torch.load(fallback_ckpt, map_location='cpu')
                    if isinstance(state, dict) and 'state_dict' in state:
                        state = state['state_dict']
                    self.temporal_encoder.load_state_dict(state, strict=False)
                    self._logger.info('Loaded fallback temporal encoder checkpoint: %s', fallback_ckpt)
                except Exception as e:
                    self._logger.warning('Failed to load temporal_encoder_ckpt=%s reason=%s', fallback_ckpt, e)
            else:
                self._logger.warning('temporal_encoder_ckpt does not exist: %s', fallback_ckpt)

        if bool(self.config.get('freeze_temporal_encoder', self.from_pretrain)):
            self.temporal_encoder.requires_grad_(False)

    def _init_external_sundial_encoder(self):
        """
        Best-effort external Sundial initialization.
        NOTE: This path is intentionally optional and guarded.
        """
        if not self.external_sundial_arch_dir:
            raise FileNotFoundError('external_sundial_arch_dir is not provided')

        # Local import to avoid hard dependency when not used.
        from safetensors.torch import load_model  # type: ignore
        from baselines.Sundial.arch.configuration_sundial import SundialConfig  # type: ignore
        from baselines.Sundial.arch.modeling_sundial import SundialForPrediction  # type: ignore

        config, _ = SundialConfig.from_pretrained(
            pretrained_model_name_or_path=self.external_sundial_arch_dir,
            return_unused_kwargs=True,
            torch_dtype='float32',
        )
        model = SundialForPrediction(config)

        if self.from_pretrain:
            if not self.external_sundial_ckpt or (not os.path.exists(self.external_sundial_ckpt)):
                raise FileNotFoundError('external_sundial_ckpt is missing')
            load_model(model, self.external_sundial_ckpt)
            model.requires_grad_(False)

        self.hidden_size = int(config.hidden_size)
        self.temporal_encoder = model

        # In original CPiRi, Sundial backbone is frozen except spatial module.
        if bool(self.config.get('freeze_temporal_encoder', True)):
            self.temporal_encoder.requires_grad_(False)

    def _encode_temporal(self, x_flat: torch.Tensor, train_mode: bool) -> torch.Tensor:
        """
        x_flat: [B*N, input_window*input_dim]
        returns: [B*N, hidden_size]
        """
        if self.temporal_encoder_type == 'external_sundial':
            # Keep close to original CPiRi behavior.
            hidden = self.temporal_encoder.model(x_flat).last_hidden_state[:, -1, :]

            # Original CPiRi samples from flow head (1 sample during train, 20-mean during eval).
            # In this LibCity adaptation we only need node embeddings at this stage, so return hidden.
            return hidden

        # Fallback encoder path.
        return self.temporal_encoder(x_flat)

    def forward(self, batch):
        """
        Args:
            batch['X']: [B, input_window, N, C]
        Returns:
            [B, output_window, N, output_dim]
        """
        history_data = batch['X']
        input_data = history_data[..., :self.input_dim]

        batch_size, _, num_nodes, _ = input_data.shape
        x = input_data.transpose(1, 2).contiguous().view(batch_size * num_nodes, -1)  # [B*N, L*C]

        if self.revin:
            means = x.mean(dim=1, keepdim=True).detach()
            stdev = x.std(dim=1, keepdim=True, unbiased=False).detach()
            stdev = torch.where(stdev > 1e-2, stdev, torch.ones_like(stdev))
            x = (x - means) / stdev

        hidden = self._encode_temporal(x, train_mode=self.training)  # [B*N, H]
        hidden = self.spital_att(hidden.view(batch_size, num_nodes, -1)).reshape(batch_size * num_nodes, -1)

        pred = self.output_proj(hidden)  # [B*N, output_window*output_dim]
        pred = pred.view(batch_size, num_nodes, self.output_window, self.output_dim).transpose(1, 2).contiguous()

        if self.revin:
            # Broadcast node-wise revin statistics over output horizon and output channels.
            means = means.view(batch_size, num_nodes, 1, 1).transpose(1, 2)
            stdev = stdev.view(batch_size, num_nodes, 1, 1).transpose(1, 2)
            pred = pred * stdev + means

        return pred

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y'][..., :self.output_dim]
        y_pred = self.predict(batch)[..., :self.output_dim]

        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true)
            y_pred = self._scaler.inverse_transform(y_pred)

        return loss.masked_mae_torch(y_pred, y_true, null_val=self.null_val)
