import math
import random
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from logging import getLogger
from timm.models.vision_transformer import trunc_normal_
try:
    from positional_encodings.torch_encodings import PositionalEncoding2D
except ImportError:
    # Fallback if positional_encodings package is not available
    class PositionalEncoding2D(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.channels = channels

        def forward(self, tensor):
            """Simple 2D positional encoding fallback"""
            batch_size, num_nodes, num_patches, num_feat = tensor.shape
            # Create simple learnable positional embeddings
            pos_enc = torch.zeros_like(tensor)
            for i in range(num_patches):
                pos_enc[:, :, i, :] = torch.sin(torch.arange(num_feat, device=tensor.device).float() * i / 10.0)
            return pos_enc

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


# ============================================================================
# Supporting Modules
# ============================================================================

class MaskGenerator(nn.Module):
    """Mask generator for MAE pre-training."""

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = patch_size
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, 1),
            stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): shape [B, N, C, L]

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)  # B, N, C, L, 1
        # B*N, C, L, 1
        long_term_history = long_term_history.reshape(batch_size*num_nodes, num_feat, len_time_series, 1)
        # B*N, d, P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)  # B, N, d, P
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self):
        super().__init__()

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].

        Returns:
            tuple: (output sequence, positional encoding matrix)
        """
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        tp_enc_2d = PositionalEncoding2D(num_feat).to(input_data.device)
        pos_encoding = tp_enc_2d(input_data)
        input_data = input_data + pos_encoding
        return input_data, pos_encoding


class TransformerLayers(nn.Module):
    """Transformer encoder layers."""

    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.contiguous()
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output


class Mask(nn.Module):
    """Masked Autoencoder for spatial or temporal features."""

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout,
                 mask_ratio, encoder_depth, decoder_depth, spatial=False, mode="forecasting"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial = spatial
        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None

        # encoder specifics
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding()
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=False):
        """
        Args:
            long_term_history (torch.Tensor): shape [B, N, C, L]
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states
            list: unmasked token index
            list: masked token index
        """
        batch_size, num_nodes, _, _ = long_term_history.shape

        # patchify and embed input
        patches = self.patch_embedding(long_term_history)  # B, N, d, P
        patches = patches.transpose(-1, -2)  # B, N, P, d
        batch_size, num_nodes, num_time, num_dim = patches.shape

        # positional embedding
        patches, self.pos_mat = self.positional_encoding(patches)  # B, N, P, d

        unmasked_token_index, masked_token_index = None, None
        encoder_input = patches  # B, N, P, d

        if self.spatial:
            encoder_input = encoder_input.transpose(-2, -3)  # B, P, N, d

        hidden_states_unmasked = self.encoder(encoder_input)

        if self.spatial:
            hidden_states_unmasked = hidden_states_unmasked.transpose(-2, -3)  # B, N, P, d

        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(
            batch_size, num_nodes, -1, self.embed_dim)  # B, N, P, d

        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data (torch.Tensor): shape [B, L, N, C]

        Returns:
            torch.Tensor: hidden states with shape [B, N, P, d]
        """
        # reshape: [B, L, N, C] -> [B, N, C, L]
        history_data = history_data.permute(0, 2, 3, 1)

        # encoding without mask for forecasting
        hidden_states_full, _, _ = self.encoding(history_data, mask=False)
        return hidden_states_full


# ============================================================================
# GraphWaveNet Components
# ============================================================================

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        A = A.to(x.device)
        if len(A.shape) == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0),
                                   stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNet(nn.Module):
    """Graph WaveNet backend for forecasting."""

    def __init__(self, num_nodes, supports, dropout=0.3, gcn_bool=True, addaptadj=True,
                 aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, **kwargs):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.in_dim = in_dim

        # Hidden state fusion layers
        self.fc_his_t = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_his_s = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels, kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout,
                                        support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels,
                                   kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim,
                                   kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input, hidden_states):
        """
        Args:
            input (torch.Tensor): shape [B, L, N, C]
            hidden_states (torch.Tensor): shape [B, N, out_len, d*2]

        Returns:
            torch.Tensor: prediction with shape [B, N, P]
        """
        # reshape input: [B, L, N, C] -> [B, C, N, L]
        input = input.transpose(1, 3)

        # feed forward
        input = nn.functional.pad(input, (1, 0, 0, 0))
        input = input[:, :self.in_dim, :, :]
        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # Add hidden states
        hidden_states_t = self.fc_his_t(hidden_states[:, :, :, :96])  # B, N, out_len, 256
        hidden_states_t = hidden_states_t.permute(0, 3, 1, 2)  # B, 256, N, out_len
        skip = skip + hidden_states_t

        hidden_states_s = self.fc_his_s(hidden_states[:, :, :, 96:])  # B, N, out_len, 256
        hidden_states_s = hidden_states_s.permute(0, 3, 1, 2)  # B, 256, N, out_len
        skip = skip + hidden_states_s

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # reshape output: [B, P, N, 1] -> [B, N, P]
        x = x.squeeze(-1).transpose(1, 2)
        return x


# ============================================================================
# Main STDMAE Model
# ============================================================================

class STDMAE(AbstractTrafficStateModel):
    """Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting

    Adapted for LibCity framework.
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # Extract data features
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.adj_mx = data_feature.get('adj_mx')
        self.device = config.get('device', torch.device('cpu'))

        # Extract config parameters
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.seq_len = config.get('seq_len', 864)  # Long sequence length for MAE

        # MAE parameters
        self.patch_size = config.get('patch_size', 12)
        self.embed_dim = config.get('embed_dim', 96)
        self.num_heads = config.get('num_heads', 4)
        self.mlp_ratio = config.get('mlp_ratio', 4)
        self.dropout = config.get('dropout', 0.1)
        self.mask_ratio = config.get('mask_ratio', 0.75)
        self.encoder_depth = config.get('encoder_depth', 4)
        self.decoder_depth = config.get('decoder_depth', 1)

        # GraphWaveNet parameters
        self.gcn_bool = config.get('gcn_bool', True)
        self.addaptadj = config.get('addaptadj', True)
        self.residual_channels = config.get('residual_channels', 32)
        self.dilation_channels = config.get('dilation_channels', 32)
        self.skip_channels = config.get('skip_channels', 256)
        self.end_channels = config.get('end_channels', 512)
        self.kernel_size = config.get('kernel_size', 2)
        self.blocks = config.get('blocks', 4)
        self.layers = config.get('layers', 2)

        # Pre-trained model paths
        self.pre_trained_tmae_path = config.get('pre_trained_tmae_path', None)
        self.pre_trained_smae_path = config.get('pre_trained_smae_path', None)

        # Initialize TMAE (Temporal Masked Autoencoder)
        mask_args = {
            'patch_size': self.patch_size,
            'in_channel': 1,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'mask_ratio': self.mask_ratio,
            'encoder_depth': self.encoder_depth,
            'decoder_depth': self.decoder_depth,
            'spatial': False,
            'mode': 'forecasting'
        }
        self.tmae = Mask(**mask_args)

        # Initialize SMAE (Spatial Masked Autoencoder)
        mask_args['spatial'] = True
        self.smae = Mask(**mask_args)

        # Prepare supports for GraphWaveNet
        supports = []
        if self.adj_mx is not None:
            supports = [torch.tensor(self.adj_mx, dtype=torch.float32).to(self.device)]

        # Initialize GraphWaveNet backend
        backend_args = {
            'num_nodes': self.num_nodes,
            'supports': supports,
            'dropout': self.dropout,
            'gcn_bool': self.gcn_bool,
            'addaptadj': self.addaptadj,
            'aptinit': None,
            'in_dim': self.feature_dim,
            'out_dim': self.output_window,
            'residual_channels': self.residual_channels,
            'dilation_channels': self.dilation_channels,
            'skip_channels': self.skip_channels,
            'end_channels': self.end_channels,
            'kernel_size': self.kernel_size,
            'blocks': self.blocks,
            'layers': self.layers
        }
        self.backend = GraphWaveNet(**backend_args)

        # Load pre-trained models if paths are provided
        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        """Load pre-trained TMAE and SMAE models and freeze their parameters."""
        if self.pre_trained_tmae_path is not None:
            try:
                checkpoint_dict = torch.load(self.pre_trained_tmae_path, map_location=self.device)
                self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])
                # Freeze parameters
                for param in self.tmae.parameters():
                    param.requires_grad = False
                self._logger.info(f"Loaded pre-trained TMAE from {self.pre_trained_tmae_path}")
            except Exception as e:
                self._logger.warning(f"Failed to load pre-trained TMAE: {e}")

        if self.pre_trained_smae_path is not None:
            try:
                checkpoint_dict = torch.load(self.pre_trained_smae_path, map_location=self.device)
                self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
                # Freeze parameters
                for param in self.smae.parameters():
                    param.requires_grad = False
                self._logger.info(f"Loaded pre-trained SMAE from {self.pre_trained_smae_path}")
            except Exception as e:
                self._logger.warning(f"Failed to load pre-trained SMAE: {e}")

    def forward(self, batch):
        """
        Args:
            batch (dict): batch data with keys:
                - 'X': short-term history with shape [B, L, N, C]
                - 'X_ext' (optional): long-term history with shape [B, L_long, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, output_window, N, output_dim]
        """
        history_data = batch['X']  # [B, L, N, C]

        # Get long history data (if available) or use history_data
        # Note: LibCity's Batch class doesn't implement __contains__, so we use try/except
        try:
            long_history_data = batch['X_ext']
            if long_history_data is None:
                long_history_data = history_data
        except KeyError:
            long_history_data = history_data

        batch_size, _, num_nodes, _ = history_data.shape

        # Extract features from TMAE and SMAE
        # Use only the first feature (typically traffic speed)
        long_data_input = long_history_data[..., [0]]  # [B, L, N, 1]

        # Get hidden states from temporal and spatial MAE
        hidden_states_t = self.tmae(long_data_input)  # [B, N, P, d]
        hidden_states_s = self.smae(long_data_input)  # [B, N, P, d]

        # Concatenate temporal and spatial hidden states
        hidden_states = torch.cat((hidden_states_t, hidden_states_s), -1)  # [B, N, P, d*2]

        # Use the last patch's features
        out_len = 1
        hidden_states = hidden_states[:, :, -out_len:, :]  # [B, N, 1, d*2]

        # Forecast using GraphWaveNet backend
        y_hat = self.backend(history_data, hidden_states)  # [B, N, output_window]

        # Reshape to match expected output format
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)  # [B, output_window, N, 1]

        return y_hat

    def predict(self, batch):
        """
        Prediction interface for LibCity.

        Args:
            batch (dict): batch data

        Returns:
            torch.Tensor: predictions
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training.

        Args:
            batch (dict): batch data with keys 'X' and 'y'

        Returns:
            torch.Tensor: loss value
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)

        # Inverse transform if scaler is available
        if self._scaler is not None:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # Use masked MAE loss
        return loss.masked_mae_torch(y_predicted, y_true, 0)
