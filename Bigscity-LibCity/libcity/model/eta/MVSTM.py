import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def normalize(data, mean, std):
    """Normalize data using mean and standard deviation."""
    return (data - mean) / std


def unnormalize(data, mean, std):
    """Unnormalize data using mean and standard deviation."""
    return data * std + mean


class MVSTM(AbstractTrafficStateModel):
    """
    Multi-View Spatial-Temporal Model (MVSTM) for Travel Time Estimation.

    This model combines multiple views of travel data:
    - Spatial view: Link embeddings for road segments
    - Temporal view: Time slice embeddings and weekday embeddings
    - Contextual view: Driver embeddings and weather embeddings

    Architecture:
    - Embedding layers for links, drivers, time slices, weekdays, and weather
    - LSTM for processing variable-length link sequences
    - MLP for final travel time prediction

    Original paper implementation adapted from DIDI Travel Time Estimation challenge.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        data_feature (dict): Data feature dictionary containing vocabulary sizes and statistics
    """

    def __init__(self, config, data_feature):
        super(MVSTM, self).__init__(config, data_feature)

        self.config = config
        self.data_feature = data_feature
        self.device = config.get('device', torch.device('cpu'))

        # Get vocabulary sizes from data_feature or use config defaults
        self.num_links = data_feature.get('num_links', config.get('num_links', 50000))
        self.num_drivers = data_feature.get('num_drivers', config.get('num_drivers', 10000))
        self.num_time_slices = data_feature.get('num_time_slices', config.get('num_time_slices', 288))
        self.num_weekdays = 7
        self.num_weather_types = data_feature.get('num_weather_types', config.get('num_weather_types', 5))

        # Embedding dimensions from config
        self.link_emb_dim = config.get('link_emb_dim', 20)
        self.driver_emb_dim = config.get('driver_emb_dim', 20)
        self.slice_emb_dim = config.get('slice_emb_dim', 20)
        self.weekday_emb_dim = config.get('weekday_emb_dim', 3)
        self.weather_emb_dim = config.get('weather_emb_dim', 3)

        # LSTM parameters
        self.lstm_hidden_dim = config.get('lstm_hidden_dim', 128)
        self.lstm_num_layers = config.get('lstm_num_layers', 1)

        # MLP parameters
        self.mlp_hidden_dims = config.get('mlp_hidden_dims', [256, 128])

        # Number of numerical features: dist, simple_eta, low_temp, high_temp
        self.num_numerical_features = config.get('num_numerical_features', 4)

        # Initialize embedding layers
        # Link embedding: +1 for unknown/padding link
        self.link_emb = nn.Embedding(self.num_links + 1, self.link_emb_dim, padding_idx=0)

        # Time slice embedding (288 slices for 5-minute intervals in a day)
        self.slice_emb = nn.Embedding(self.num_time_slices, self.slice_emb_dim)

        # Driver embedding
        self.driver_emb = nn.Embedding(self.num_drivers, self.driver_emb_dim)

        # Weekday embedding (0-6)
        self.weekday_emb = nn.Embedding(self.num_weekdays, self.weekday_emb_dim)

        # Weather embedding
        self.weather_emb = nn.Embedding(self.num_weather_types, self.weather_emb_dim)

        # LSTM input: link_emb (20) + link_time (1) + link_current_status (1) + link_ratio (1) = 23
        lstm_input_dim = self.link_emb_dim + 3  # link_time, link_current_status, link_ratio

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Calculate MLP input dimension
        # LSTM output (128) + numerical (4) + slice_emb (20) + driver_emb (20) + weekday_emb (3) = 175
        mlp_input_dim = (
            self.lstm_hidden_dim +
            self.num_numerical_features +
            self.slice_emb_dim +
            self.driver_emb_dim +
            self.weekday_emb_dim
        )

        # Build MLP layers
        mlp_layers = []
        in_dim = mlp_input_dim
        for hidden_dim in self.mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.LeakyReLU(inplace=True))
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

        # Normalization statistics from data_feature
        self.eta_mean = data_feature.get('eta_mean', 6.553886963677842)
        self.eta_std = data_feature.get('eta_std', 0.5905307292899195)
        self.dist_mean = data_feature.get('dist_mean', 8.325948361544423)
        self.dist_std = data_feature.get('dist_std', 0.6799133140855674)
        self.simple_eta_mean = data_feature.get('simple_eta_mean', 6.453206241137908)
        self.simple_eta_std = data_feature.get('simple_eta_std', 0.5758803681400783)
        self.high_temp_mean = data_feature.get('high_temp_mean', 31.84375)
        self.high_temp_std = data_feature.get('high_temp_std', 1.6975971069426339)
        self.low_temp_mean = data_feature.get('low_temp_mean', 26.46875)
        self.low_temp_std = data_feature.get('low_temp_std', 0.9348922063532245)
        self.link_time_min = data_feature.get('link_time_min', 0.0)
        self.link_time_max = data_feature.get('link_time_max', 2949.12)

        # Whether to use log transformation for target
        self.use_log_transform = config.get('use_log_transform', True)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name and 'emb' not in name:
                nn.init.xavier_uniform_(param)

    def _prepare_batch(self, batch):
        """
        Prepare batch data for forward pass.

        This method handles the conversion from LibCity batch format to the
        format expected by the model.

        Args:
            batch (dict): LibCity batch dictionary

        Returns:
            dict: Prepared data dictionary
        """
        # Extract features from batch
        # The batch format should follow LibCity ETA conventions

        prepared = {}

        # Link sequence features
        # Expected: batch contains 'link_ids', 'link_time', 'link_ratio', 'link_current_status'
        if 'link_ids' in batch.data:
            prepared['link_ids'] = batch['link_ids']  # (batch, seq_len)
        elif 'link_cross_start' in batch.data:
            # Use average of start and end link embeddings if available
            prepared['link_cross_start'] = batch['link_cross_start']
            prepared['link_cross_end'] = batch.get('link_cross_end', batch['link_cross_start'])
        else:
            raise KeyError("Batch must contain 'link_ids' or 'link_cross_start'")

        # Link-level features
        if 'link_time' in batch.data:
            link_time = batch['link_time']
            # Normalize link time
            prepared['link_time'] = (link_time - self.link_time_min) / (self.link_time_max - self.link_time_min + 1e-8)
        elif 'link_cross_time' in batch.data:
            prepared['link_time'] = batch['link_cross_time']
        else:
            raise KeyError("Batch must contain 'link_time' or 'link_cross_time'")

        if 'link_current_status' in batch.data:
            prepared['link_current_status'] = batch['link_current_status'] / 5.0
        elif 'link_cross_current_status' in batch.data:
            prepared['link_current_status'] = batch['link_cross_current_status']
        else:
            # Default to ones if not available
            prepared['link_current_status'] = torch.ones_like(prepared['link_time'])

        if 'link_ratio' in batch.data:
            prepared['link_ratio'] = batch['link_ratio']
        elif 'link_cross_ratio' in batch.data:
            prepared['link_ratio'] = batch['link_cross_ratio']
        else:
            # Default to ones if not available
            prepared['link_ratio'] = torch.ones_like(prepared['link_time'])

        # Sequence lengths for pack_padded_sequence
        # Squeeze to convert [batch, 1] to [batch] for pack_padded_sequence
        if 'link_len' in batch.data:
            prepared['link_len'] = batch['link_len'].squeeze(-1)
        elif 'link_cross_len' in batch.data:
            prepared['link_len'] = batch['link_cross_len'].squeeze(-1)
        else:
            # Compute from non-zero elements
            if 'link_ids' in batch.data:
                prepared['link_len'] = (batch['link_ids'] != 0).sum(dim=1).float()
            else:
                prepared['link_len'] = (batch['link_cross_start'] != 0).sum(dim=1).float()

        # Order-level numerical features
        # Squeeze to convert [batch, 1] to [batch] for scalar features
        if 'dist' in batch.data:
            # Apply log transform and normalize
            dist = batch['dist'].squeeze(-1)
            if self.use_log_transform:
                dist = torch.log(dist.clamp(min=1e-8))
            prepared['dist'] = normalize(dist, self.dist_mean, self.dist_std)
        else:
            prepared['dist'] = torch.zeros(batch['link_ids'].size(0) if 'link_ids' in batch.data
                                           else batch['link_cross_start'].size(0),
                                           device=self.device)

        if 'simple_eta' in batch.data:
            simple_eta = batch['simple_eta'].squeeze(-1)
            if self.use_log_transform:
                simple_eta = torch.log(simple_eta.clamp(min=1e-8))
            prepared['simple_eta'] = normalize(simple_eta, self.simple_eta_mean, self.simple_eta_std)
        else:
            prepared['simple_eta'] = torch.zeros_like(prepared['dist'])

        if 'high_temp' in batch.data:
            prepared['high_temp'] = normalize(batch['high_temp'].squeeze(-1), self.high_temp_mean, self.high_temp_std)
        elif 'hightemp' in batch.data:
            prepared['high_temp'] = normalize(batch['hightemp'].squeeze(-1), self.high_temp_mean, self.high_temp_std)
        else:
            prepared['high_temp'] = torch.zeros_like(prepared['dist'])

        if 'low_temp' in batch.data:
            prepared['low_temp'] = normalize(batch['low_temp'].squeeze(-1), self.low_temp_mean, self.low_temp_std)
        elif 'lowtemp' in batch.data:
            prepared['low_temp'] = normalize(batch['lowtemp'].squeeze(-1), self.low_temp_mean, self.low_temp_std)
        else:
            prepared['low_temp'] = torch.zeros_like(prepared['dist'])

        # Categorical features
        # Squeeze to convert [batch, 1] to [batch] for embedding lookups
        if 'driver_id' in batch.data:
            prepared['driver_id'] = batch['driver_id'].squeeze(-1).long()
        elif 'uid' in batch.data:
            prepared['driver_id'] = batch['uid'].squeeze(-1).long()
        else:
            prepared['driver_id'] = torch.zeros(prepared['dist'].size(0), dtype=torch.long, device=self.device)

        if 'slice_id' in batch.data:
            prepared['slice_id'] = batch['slice_id'].squeeze(-1).long()
        elif 'timeid' in batch.data:
            prepared['slice_id'] = batch['timeid'].squeeze(-1).long()
        else:
            prepared['slice_id'] = torch.zeros(prepared['dist'].size(0), dtype=torch.long, device=self.device)

        if 'weekday' in batch.data:
            prepared['weekday'] = batch['weekday'].squeeze(-1).long()
        elif 'weekid' in batch.data:
            prepared['weekday'] = batch['weekid'].squeeze(-1).long()
        else:
            prepared['weekday'] = torch.zeros(prepared['dist'].size(0), dtype=torch.long, device=self.device)

        if 'weather' in batch.data:
            prepared['weather'] = batch['weather'].squeeze(-1).long()
        else:
            prepared['weather'] = torch.zeros(prepared['dist'].size(0), dtype=torch.long, device=self.device)

        return prepared

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (dict): Input batch dictionary

        Returns:
            torch.Tensor: Predicted travel time (normalized, in log scale if use_log_transform=True)
        """
        # Prepare batch data
        x = self._prepare_batch(batch)

        # Get link embeddings
        if 'link_ids' in x:
            x_link = self.link_emb(x['link_ids'])  # (batch, seq_len, link_emb_dim)
        elif 'link_cross_start' in x:
            x_link_start = self.link_emb(x['link_cross_start'])
            x_link_end = self.link_emb(x['link_cross_end'])
            x_link = (x_link_start + x_link_end) / 2
        else:
            x_link = self.link_emb(batch['link_cross_start'])

        # Get link-level features
        x_link_time = x['link_time'].unsqueeze(-1)  # (batch, seq_len, 1)
        x_link_status = x['link_current_status'].unsqueeze(-1)  # (batch, seq_len, 1)
        x_link_ratio = x['link_ratio'].unsqueeze(-1)  # (batch, seq_len, 1)

        # Concatenate LSTM input features
        x_lstm = torch.cat([
            x_link,
            x_link_time,
            x_link_status,
            x_link_ratio
        ], dim=-1)  # (batch, seq_len, lstm_input_dim)

        # Get sequence lengths and ensure they are valid
        link_len = x['link_len'].cpu()
        # Safety check: squeeze if still has extra dimension
        if link_len.dim() > 1:
            link_len = link_len.squeeze(-1)
        # Ensure minimum length of 1 to avoid pack_padded_sequence errors
        link_len = link_len.clamp(min=1).long()

        # Pack padded sequence for efficient LSTM processing
        packed = pack_padded_sequence(
            x_lstm,
            link_len,
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward pass
        lstm_output, (ht, ct) = self.lstm(packed)

        # Use final hidden state
        # ht shape: (num_layers * num_directions, batch, hidden_dim)
        ht = ht[-1]  # Take last layer, shape: (batch, hidden_dim)

        # Get embeddings for categorical features
        x_slice = self.slice_emb(x['slice_id'])  # (batch, slice_emb_dim)
        x_driver = self.driver_emb(x['driver_id'])  # (batch, driver_emb_dim)
        x_weekday = self.weekday_emb(x['weekday'])  # (batch, weekday_emb_dim)

        # Stack numerical features
        x_num = torch.stack([
            x['simple_eta'],
            x['dist'],
            x['low_temp'],
            x['high_temp']
        ], dim=-1)  # (batch, 4)

        # Concatenate all features for MLP
        x_combined = torch.cat([
            ht,
            x_num,
            x_slice,
            x_driver,
            x_weekday
        ], dim=-1)  # (batch, mlp_input_dim)

        # MLP forward pass
        output = self.mlp(x_combined)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)

    def predict(self, batch):
        """
        Generate predictions for a batch.

        Args:
            batch (dict): Input batch dictionary

        Returns:
            torch.Tensor: Predicted travel time in original scale (seconds)
        """
        output = self.forward(batch)

        # Convert from normalized log scale to original scale
        if self.use_log_transform:
            # Unnormalize and then exp
            output = output * self.eta_std + self.eta_mean
            output = torch.exp(output)
        else:
            output = unnormalize(output, self.eta_mean, self.eta_std)

        return output

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Uses L1 loss (MAE) on the normalized log-scale predictions,
        following the original implementation.

        Args:
            batch (dict): Input batch dictionary containing target 'time' or 'eta'

        Returns:
            torch.Tensor: Scalar loss value
        """
        output = self.forward(batch)

        # Get target travel time
        # Squeeze to convert [batch, 1] to [batch] for loss computation
        if 'eta' in batch.data:
            target = batch['eta'].squeeze(-1)
        elif 'time' in batch.data:
            target = batch['time'].squeeze(-1)
        else:
            raise KeyError("Batch must contain 'eta' or 'time' as target")

        # Transform target to match output scale
        if self.use_log_transform:
            target = torch.log(target.clamp(min=1e-8))
            target = normalize(target, self.eta_mean, self.eta_std)
        else:
            target = normalize(target, self.eta_mean, self.eta_std)

        # L1 loss (MAE) as used in original implementation
        loss_value = F.l1_loss(output, target)

        return loss_value
