"""
ConvTimeNet Model for LibCity Traffic Speed Prediction

Adapted from: https://github.com/Lionel-Luo/ConvTimeNet
Paper: "ConvTimeNet: A Deep Hierarchical Fully Convolutional Model for Multivariate Time Series Analysis"

Original Files:
- TSForecasting/models/ConvTimeNet.py
- TSForecasting/layers/ConvTimeNet_backbone.py
- TSForecasting/layers/Patch_layers.py
- TSForecasting/layers/RevIN.py

Key Adaptations:
1. Inherits from AbstractTrafficStateModel
2. Reshapes LibCity's [batch, seq_len, num_nodes, features] to [batch, seq_len, num_nodes * features]
3. Implements predict() and calculate_loss() methods
4. Uses LibCity's config and data_feature pattern
5. All layers consolidated into single file
6. Device handling adapted for LibCity framework

Hyperparameters (from config):
- input_window: Input sequence length (default: 12)
- output_window: Prediction length (default: 12)
- e_layers: Number of encoder layers (default: 6)
- d_model: Model dimension (default: 64)
- d_ff: Feed-forward dimension (default: 256)
- patch_ks: Patch kernel size (default: 32)
- patch_sd: Patch stride ratio (default: 0.5)
- dw_ks: List of kernel sizes per layer (default: [9, 11, 15, 21, 29, 39])
- dropout: Dropout rate (default: 0.1)
- head_dropout: Head dropout (default: 0.0)
- revin: Use RevIN normalization (default: True)
- affine: RevIN affine transformation (default: True)
- subtract_last: Subtract last for RevIN (default: False)
- enable_res_param: Enable learnable residual parameter (default: True)
- re_param: Enable re-parameterization (default: True)
- re_param_kernel: Re-parameterization kernel size (default: 3)
- padding_patch: Patch padding ('end' or None, default: 'end')
- deformable: Use deformable patching (default: True)
"""

from logging import getLogger
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


# ====================== RevIN Layer ======================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series
    From: https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ====================== Utility Functions ======================
def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


def zero_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)


# ====================== Deformable Patch Layers ======================
class BoxCoder(nn.Module):
    def __init__(self, patch_count, patch_stride, patch_size, seq_len, channels, device='cuda:0'):
        super().__init__()
        self.device = device

        self.seq_len = seq_len
        self.channels = channels
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.patch_stride = patch_stride

        self._generate_anchor(device=device)

    def _generate_anchor(self, device="cuda:0"):
        anchors = []
        self.S_bias = (self.patch_size - 1) / 2

        for i in range(self.patch_count):
            x = i * self.patch_stride + 0.5 * (self.patch_size - 1)
            anchors.append(x)

        anchors = torch.as_tensor(anchors, device=device)
        self.register_buffer("anchor", anchors)

    def forward(self, boxes):
        self.bound = self.decode(boxes)
        points = self.meshgrid(self.bound)
        return points, self.bound

    def decode(self, rel_codes):
        boxes = self.anchor

        dx = rel_codes[:, :, :, 0]
        ds = torch.relu(rel_codes[:, :, :, 1] + self.S_bias)

        pred_boxes = torch.zeros_like(rel_codes)
        ref_x = boxes.view(1, boxes.shape[0], 1)

        pred_boxes[:, :, :, 0] = (dx + ref_x - ds)
        pred_boxes[:, :, :, 1] = (dx + ref_x + ds)
        pred_boxes /= (self.seq_len - 1)

        pred_boxes = pred_boxes.clamp_(min=0., max=1.)

        return pred_boxes

    def meshgrid(self, boxes):
        B, patch_count, C = boxes.shape[0], boxes.shape[1], boxes.shape[2]
        channel_boxes = torch.zeros((boxes.shape[0], boxes.shape[1], 2)).to(self.device)
        channel_boxes[:, :, 1] = 1.0
        xs = boxes.view(B * patch_count, C, 2)
        xs = torch.nn.functional.interpolate(xs, size=self.patch_size, mode='linear', align_corners=True)
        ys = torch.nn.functional.interpolate(channel_boxes, size=self.channels, mode='linear', align_corners=True)

        xs = xs.view(B, patch_count, C, self.patch_size, 1)
        ys = ys.unsqueeze(3).expand(B, patch_count, C, self.patch_size).unsqueeze(-1)

        grid = torch.stack([xs, ys], dim=-1)
        return grid


class OffsetPredictor(nn.Module):
    def __init__(self, in_feats, patch_size, stride, use_zero_init=True):
        super().__init__()
        self.stride = stride
        self.channel = in_feats
        self.patch_size = patch_size

        self.offset_predictor = nn.Sequential(
            nn.Conv1d(1, 64, patch_size, stride=stride, padding=0),
            nn.GELU(),
            nn.Conv1d(64, 2, 1, 1, padding=0)
        )

        if use_zero_init:
            self.offset_predictor.apply(zero_init)

    def forward(self, X):
        patch_X = X.unsqueeze(1).permute(0, 1, 3, 2)
        patch_X = F.unfold(patch_X, kernel_size=(self.patch_size, self.channel), stride=self.stride).permute(0, 2, 1)

        B, patch_count = patch_X.shape[0], patch_X.shape[1]
        patch_X = patch_X.contiguous().view(B, patch_count, self.patch_size, self.channel)
        patch_X = patch_X.permute(0, 1, 3, 2)

        patch_X = patch_X.contiguous().view(B * patch_count * self.channel, 1, self.patch_size)

        pred_offset = self.offset_predictor(patch_X)
        pred_offset = pred_offset.view(B, patch_count, self.channel, 2).contiguous()

        return pred_offset


class DepatchSampling(nn.Module):
    def __init__(self, in_feats, seq_len, patch_size, stride, device='cuda:0'):
        super(DepatchSampling, self).__init__()
        self.in_feats = in_feats
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.device = device

        self.patch_count = (seq_len - patch_size) // stride + 1

        self.dropout = nn.Dropout(0.1)

        self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride)

        self.box_coder = BoxCoder(self.patch_count, stride, patch_size, self.seq_len, in_feats, device=device)

    def get_sampling_location(self, X):
        pred_offset = self.offset_predictor(X)
        sampling_locations, bound = self.box_coder(pred_offset)
        return sampling_locations, bound

    def forward(self, X, return_bound=False):
        img = X.unsqueeze(1)
        B = img.shape[0]

        sampling_locations, bound = self.get_sampling_location(X)
        sampling_locations = sampling_locations.view(B, self.patch_count * self.in_feats, self.patch_size, 2)

        sampling_locations = (sampling_locations - 0.5) * 2
        output = F.grid_sample(img, sampling_locations, align_corners=True)
        output = output.view(B, self.patch_count, self.in_feats, self.patch_size)
        output = output.permute(0, 2, 1, 3).contiguous()
        return output


# ====================== ConvTimeNet Encoder Layers ======================
class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)


class ConvEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 256, kernel_size: int = 9, dropout: float = 0.1,
                 activation: str = "gelu", enable_res_param=True, norm='batch', re_param=True,
                 small_ks=3, device='cuda:0'):
        super(ConvEncoderLayer, self).__init__()

        self.norm_tp = norm
        self.re_param = re_param

        if not re_param:
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, 1, 'same', groups=d_model)
        else:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, small_ks, stride=1, padding='same', groups=d_model)
            self.DW_infer = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)

        self.dw_act = get_activation_fn(activation)

        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1, 1),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, 1, 1)
        )

        # Add & Norm
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def _get_merged_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        module_output.weight = torch.nn.Parameter(
            module_output.weight + F.pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0)
        )
        module_output.bias = torch.nn.Parameter(module_output.bias + self.DW_conv_small.bias)
        self.DW_infer = module_output

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Deep-wise Conv Layer
        if not self.re_param:
            out_x = self.DW_conv(src)
        else:
            if self.training:
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                out_x = large_out + small_out
            else:
                self._get_merged_param()
                out_x = self.DW_infer(src)

        src2 = self.dw_act(out_x)

        src = self.sublayerconnect1(src, src2)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src

        # Position-wise Conv Feed-Forward
        src2 = self.ff(src)
        src2 = self.sublayerconnect2(src, src2)

        # Norm
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2

        return src2


class ConvEncoder(nn.Module):
    def __init__(self, kernel_size, d_model, d_ff=None,
                 norm='batch', dropout=0., activation='gelu',
                 enable_res_param=True, n_layers=3, re_param=True, re_param_kernel=3, device='cuda:0'):
        super().__init__()

        self.layers = nn.ModuleList([
            ConvEncoderLayer(
                d_model, d_ff=d_ff, kernel_size=kernel_size[i], dropout=dropout,
                activation=activation, enable_res_param=enable_res_param, norm=norm,
                re_param=re_param, small_ks=re_param_kernel, device=device
            ) for i in range(n_layers)
        ])

    def forward(self, src: Tensor):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class ConviEncoder(nn.Module):
    """Channel-independent encoder"""
    def __init__(self, patch_num, patch_len, kernel_size=[11, 15, 21, 29, 39, 51], n_layers=6, d_model=128,
                 d_ff=256, norm='batch', dropout=0., act="gelu", enable_res_param=True,
                 re_param=True, re_param_kernel=3, device='cuda:0'):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input embedding
        self.W_P = nn.Linear(patch_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = ConvEncoder(
            kernel_size, d_model, d_ff=d_ff, norm=norm, dropout=dropout,
            activation=act, enable_res_param=enable_res_param, n_layers=n_layers,
            re_param=re_param, re_param_kernel=re_param_kernel, device=device
        )

    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)  # [bs, nvars, patch_num, d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [bs*nvars, patch_num, d_model]

        # Encoder
        z = self.encoder(u.permute(0, 2, 1)).permute(0, 2, 1)  # [bs*nvars, patch_num, d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # [bs, nvars, patch_num, d_model]
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num]

        return z


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.n_vars = n_vars

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# ====================== ConvTimeNet Backbone ======================
class ConvTimeNet_backbone(nn.Module):
    def __init__(self, c_in: int, seq_len: int, context_window: int, target_window: int,
                 patch_len: int, stride: int, n_layers: int = 6, dw_ks=[9, 11, 15, 21, 29, 39],
                 d_model=64, d_ff: int = 256, norm: str = 'batch', dropout: float = 0., act: str = "gelu",
                 head_dropout=0, padding_patch=None, head_type='flatten', revin=True, affine=True,
                 subtract_last=False, deformable=True, enable_res_param=True, re_param=True,
                 re_param_kernel=3, device='cuda:0'):
        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching / Deformable Patching
        self.deformable = deformable
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        seq_len_padded = (patch_num - 1) * self.stride + self.patch_len
        if deformable:
            self.deformable_sampling = DepatchSampling(c_in, seq_len_padded, self.patch_len, self.stride, device=device)

        # Backbone
        self.backbone = ConviEncoder(
            patch_num=patch_num, patch_len=patch_len, kernel_size=dw_ks,
            n_layers=n_layers, d_model=d_model, d_ff=d_ff, norm=norm,
            dropout=dropout, act=act, enable_res_param=enable_res_param,
            re_param=re_param, re_param_kernel=re_param_kernel, device=device
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == 'flatten':
            self.head = Flatten_Head(self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        else:
            raise ValueError(f'No such head: {head_type}')

    def forward(self, z):  # z: [bs, nvars, seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        if not self.deformable:
            z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            z = self.deformable_sampling(z)

        z = z.permute(0, 1, 3, 2)  # z: [bs, nvars, patch_len, patch_num]

        # model
        z = self.backbone(z)  # z: [bs, nvars, d_model, patch_num]
        z = self.head(z)  # z: [bs, nvars, target_window]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z


# ====================== Main ConvTimeNet Model for LibCity ======================
class ConvTimeNet(AbstractTrafficStateModel):
    """
    ConvTimeNet model adapted for LibCity traffic speed prediction.

    Data format transformation:
    - LibCity input: [batch, seq_len, num_nodes, features]
    - ConvTimeNet expects: [batch, channels, seq_len] where channels = num_nodes * features
    - Output: [batch, pred_len, num_nodes, output_dim]
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))

        # Data features
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self._scaler = self.data_feature.get('scaler')

        # Model parameters from config
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.input_dim = self.data_feature.get('feature_dim', config.get('input_dim', 1))
        self.output_dim = config.get('output_dim', 1)

        # ConvTimeNet specific parameters
        self.e_layers = config.get('e_layers', 6)
        self.d_model = config.get('d_model', 64)
        self.d_ff = config.get('d_ff', 256)
        self.patch_ks = config.get('patch_ks', 32)
        self.patch_sd = config.get('patch_sd', 0.5)
        self.dropout = config.get('dropout', 0.1)
        self.head_dropout = config.get('head_dropout', 0.0)

        # Kernel sizes for each layer
        default_dw_ks = [9, 11, 15, 21, 29, 39]
        self.dw_ks = config.get('dw_ks', default_dw_ks[:self.e_layers])

        # Ensure dw_ks has correct length for e_layers
        if len(self.dw_ks) < self.e_layers:
            # Extend with last value if not enough
            self.dw_ks = self.dw_ks + [self.dw_ks[-1]] * (self.e_layers - len(self.dw_ks))
        elif len(self.dw_ks) > self.e_layers:
            self.dw_ks = self.dw_ks[:self.e_layers]

        # RevIN parameters
        self.revin = config.get('revin', True)
        self.affine = config.get('affine', True)
        self.subtract_last = config.get('subtract_last', False)

        # Other parameters
        self.padding_patch = config.get('padding_patch', 'end')
        self.deformable = config.get('deformable', True)
        self.enable_res_param = config.get('enable_res_param', True)
        self.re_param = config.get('re_param', True)
        self.re_param_kernel = config.get('re_param_kernel', 3)
        self.norm = config.get('norm', 'batch')
        self.act = config.get('act', 'gelu')
        self.head_type = config.get('head_type', 'flatten')

        # Calculate total channels (num_nodes * input_dim for traffic data)
        self.c_in = self.num_nodes * self.input_dim

        # Calculate stride from ratio
        self.stride = max(1, int(self.patch_ks * self.patch_sd))

        # Build model
        self._build_model()

        self._logger.info(
            f"ConvTimeNet initialized with: num_nodes={self.num_nodes}, "
            f"input_window={self.input_window}, output_window={self.output_window}, "
            f"c_in={self.c_in}, d_model={self.d_model}, e_layers={self.e_layers}, "
            f"patch_ks={self.patch_ks}, stride={self.stride}, dw_ks={self.dw_ks}"
        )

    def _build_model(self):
        """Build the ConvTimeNet model"""
        self.model = ConvTimeNet_backbone(
            c_in=self.c_in,
            seq_len=self.input_window,
            context_window=self.input_window,
            target_window=self.output_window,
            patch_len=self.patch_ks,
            stride=self.stride,
            n_layers=self.e_layers,
            dw_ks=self.dw_ks,
            d_model=self.d_model,
            d_ff=self.d_ff,
            norm=self.norm,
            dropout=self.dropout,
            act=self.act,
            head_dropout=self.head_dropout,
            padding_patch=self.padding_patch,
            head_type=self.head_type,
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            deformable=self.deformable,
            enable_res_param=self.enable_res_param,
            re_param=self.re_param,
            re_param_kernel=self.re_param_kernel,
            device=self.device
        )

    def forward(self, batch):
        """
        Forward pass

        Args:
            batch: dict with 'X' key containing input tensor
                   X shape: [batch, input_window, num_nodes, input_dim]

        Returns:
            output: [batch, output_window, num_nodes, output_dim]
        """
        x = batch['X']  # [batch, input_window, num_nodes, input_dim]
        batch_size = x.shape[0]

        # Extract only the feature dimensions needed (first input_dim features)
        x = x[..., :self.input_dim]  # [batch, input_window, num_nodes, input_dim]

        # Reshape from [batch, seq_len, num_nodes, features] to [batch, num_nodes * features, seq_len]
        # First reshape to [batch, seq_len, num_nodes * features]
        x = x.reshape(batch_size, self.input_window, -1)  # [batch, input_window, num_nodes * input_dim]

        # Permute to [batch, channels, seq_len] as expected by ConvTimeNet
        x = x.permute(0, 2, 1)  # [batch, num_nodes * input_dim, input_window]

        # Apply model
        x = self.model(x)  # [batch, num_nodes * input_dim, output_window]

        # Permute back to [batch, output_window, channels]
        x = x.permute(0, 2, 1)  # [batch, output_window, num_nodes * input_dim]

        # Reshape back to [batch, output_window, num_nodes, input_dim]
        x = x.reshape(batch_size, self.output_window, self.num_nodes, self.input_dim)

        # Select only output_dim features
        x = x[..., :self.output_dim]

        return x  # [batch, output_window, num_nodes, output_dim]

    def predict(self, batch):
        """
        Predict for a batch

        Args:
            batch: dict with 'X' key

        Returns:
            predictions: [batch, output_window, num_nodes, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate training loss

        Args:
            batch: dict with 'X' and 'y' keys

        Returns:
            loss: scalar tensor
        """
        y_true = batch['y']  # [batch, output_window, num_nodes, features]
        y_predicted = self.predict(batch)

        # Apply inverse scaler for proper loss calculation
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        return loss.masked_mae_torch(y_predicted, y_true, null_val=0.0)
