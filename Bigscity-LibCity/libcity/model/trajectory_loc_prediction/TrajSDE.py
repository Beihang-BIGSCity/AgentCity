# coding: utf-8
"""
TrajSDE - Trajectory Prediction using Neural Stochastic Differential Equations

This module adapts the TrajSDE model for LibCity framework.
TrajSDE uses Neural SDEs for uncertainty-aware trajectory prediction.

Original repository: /home/wangwenrui/shk/AgentCity/repos/TrajSDE

Key components:
- PredictionModelSDENet: Main PyTorch Lightning model (from model_base_mix_sde.py)
- LocalEncoderSDESepPara2: SDE-based temporal encoding (from encoders/enc_hivt_nusargo_sde_sep2.py)
- SDEDecoder: SDE-based future prediction (from decoders/dec_hivt_nusargo_sde.py)
- GlobalInteractor: Aggregator for global context (from aggregators/agg_hivt.py)

Dependencies:
- torchsde: For SDE integration
- torch-geometric: For graph neural network operations
- pytorch-lightning: Original model uses PyTorch Lightning

Challenges:
1. Data format mismatch: TrajSDE uses TemporalData (torch_geometric.data.Data subclass)
   while LibCity uses custom Batch objects
2. TrajSDE expects trajectory data with:
   - x: Historical trajectory positions [num_nodes, timesteps, 2]
   - positions: Absolute positions [num_nodes, timesteps, 2]
   - edge_index: Graph connectivity [2, num_edges]
   - Lane graph information (lane_positions, lane_vectors, etc.)
   - Agent indices for prediction targets
3. LibCity trajectory data format:
   - current_loc: Location indices [batch_size, seq_len]
   - current_tim: Time indices [batch_size, seq_len]
   - target: Target location index [batch_size]
   - uid: User IDs [batch_size]
4. TrajSDE is designed for vehicle trajectory prediction (nuScenes, Argoverse)
   while LibCity trajectory_loc_prediction is for POI/location prediction

This is a MINIMAL SKELETON for testing purposes.
Full implementation requires:
- Proper data format conversion between LibCity and TrajSDE
- Coordinate space handling (discrete locations vs continuous coordinates)
- Graph construction from trajectory data
- Lane graph generation (or disable lane-based features)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_model import AbstractModel

# Add TrajSDE repo to Python path
TRAJSDE_REPO_PATH = '/home/wangwenrui/shk/AgentCity/repos/TrajSDE'
if TRAJSDE_REPO_PATH not in sys.path:
    sys.path.insert(0, TRAJSDE_REPO_PATH)

try:
    # Import TrajSDE components
    from models.model_base_mix_sde import PredictionModelSDENet
    from models.utils.util import TemporalData
    # Note: The actual encoder/decoder/aggregator are loaded dynamically by PredictionModelSDENet
    # We'll handle this differently in our adapter
except ImportError as e:
    print(f"Warning: Failed to import TrajSDE components: {e}")
    print(f"Make sure {TRAJSDE_REPO_PATH} exists and contains the required modules")
    PredictionModelSDENet = None
    TemporalData = None


class TrajSDE(AbstractModel):
    """
    TrajSDE adapter for LibCity framework.

    This is a MINIMAL working skeleton that demonstrates the structure.
    Full implementation requires substantial data format conversion.

    Current limitations:
    - Does not handle discrete location to continuous coordinate conversion
    - Missing graph construction from trajectory sequences
    - Lane graph features are disabled
    - Simplified batch processing

    For testing and development purposes only.
    """

    def __init__(self, config, data_feature):
        super(TrajSDE, self).__init__(config, data_feature)

        # Device configuration
        self.device = config.get('device', 'cpu')

        # Data dimensions from LibCity data_feature
        self.loc_size = data_feature.get('loc_size', 1000)
        self.uid_size = data_feature.get('uid_size', 100)
        self.tim_size = data_feature.get('tim_size', 24)

        # Model hyperparameters
        self.embed_dim = config.get('embed_dim', 64)
        self.num_modes = config.get('num_modes', 6)  # Number of trajectory modes
        self.historical_steps = config.get('historical_steps', 21)
        self.future_steps = config.get('future_steps', 60)
        self.hidden_size = config.get('hidden_size', 128)

        # Check if TrajSDE components are available
        self.use_native_trajsde = config.get('use_native_trajsde', False) and PredictionModelSDENet is not None

        if self.use_native_trajsde:
            print("WARNING: Native TrajSDE integration is experimental and incomplete.")
            print("Data format conversion from LibCity to TrajSDE format is not fully implemented.")
            self._build_native_trajsde_model(config)
        else:
            # Build a simplified placeholder model for testing
            print("Building simplified TrajSDE-inspired model (not using native TrajSDE components)")
            self._build_simplified_model()

    def _build_native_trajsde_model(self, config):
        """
        Build the native TrajSDE model.

        NOTE: This is incomplete as it requires:
        1. Proper configuration dictionary matching TrajSDE's expected format
        2. Loading encoder/decoder/aggregator modules dynamically
        3. Loss and metric configuration

        For a complete implementation, you would need to:
        - Create a config dict matching the YAML format in repos/TrajSDE/configs/
        - Handle dynamic module loading
        - Adapt LibCity batch data to TemporalData format
        """
        raise NotImplementedError(
            "Native TrajSDE model integration requires:\n"
            "1. Configuration matching TrajSDE YAML format\n"
            "2. Dynamic encoder/decoder/aggregator loading\n"
            "3. Data format conversion from LibCity Batch to TemporalData\n"
            "4. Coordinate space mapping (discrete locations -> continuous coords)\n"
            "5. Graph construction from trajectory sequences\n"
            "\nThis is beyond a minimal skeleton. Use simplified model instead."
        )

    def _build_simplified_model(self):
        """
        Build a simplified SDE-inspired model for testing.

        This creates a basic neural network that mimics some TrajSDE concepts
        but works directly with LibCity's discrete location format.

        Architecture:
        - Location embeddings
        - Time embeddings
        - Simple GRU encoder (inspired by TrajSDE's temporal encoding)
        - Linear decoder (simplified version of SDE decoder)
        """
        # Embeddings for discrete locations and times
        self.loc_embedding = nn.Embedding(self.loc_size, self.embed_dim)
        self.tim_embedding = nn.Embedding(self.tim_size, self.embed_dim)

        # Simple temporal encoder (inspired by SDE encoder concept)
        # In TrajSDE, this would be the LocalEncoderSDESepPara2 with SDE integration
        self.encoder = nn.GRU(
            input_size=self.embed_dim * 2,  # loc + time embeddings
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Decoder (simplified version of SDEDecoder)
        # In TrajSDE, this would use SDE integration for uncertainty estimation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.loc_size)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_batch(self, batch):
        """
        Prepare batch data from LibCity format.

        LibCity batch typically contains:
        - current_loc: [batch_size, seq_len] - sequence of location indices
        - current_tim: [batch_size, seq_len] - sequence of time indices
        - target: [batch_size] - target location index
        - uid: [batch_size] - user IDs

        For native TrajSDE, this would need to be converted to TemporalData format with:
        - x: [num_nodes, timesteps, 2] - trajectory positions
        - positions: [num_nodes, timesteps, 2] - absolute positions
        - edge_index: [2, num_edges] - graph connectivity
        - padding_mask, bos_mask, etc.
        """
        loc = batch['current_loc']  # [batch_size, seq_len]
        tim = batch['current_tim']  # [batch_size, seq_len]

        # Get batch dimensions
        batch_size, seq_len = loc.shape

        # Handle variable length sequences if needed
        if hasattr(batch, 'get_origin_len'):
            loc_len = batch.get_origin_len('current_loc')
        else:
            loc_len = torch.full((batch_size,), seq_len, dtype=torch.long)

        return {
            'location': loc.to(self.device),
            'time': tim.to(self.device),
            'length': loc_len,
            'target': batch['target'].to(self.device)
        }

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch: LibCity batch object

        Returns:
            torch.Tensor: Location prediction scores [batch_size, loc_size]
        """
        # Prepare batch data
        batch_data = self._prepare_batch(batch)

        loc = batch_data['location']  # [batch_size, seq_len]
        tim = batch_data['time']      # [batch_size, seq_len]

        # Get embeddings
        loc_emb = self.loc_embedding(loc)  # [batch_size, seq_len, embed_dim]
        tim_emb = self.tim_embedding(tim)  # [batch_size, seq_len, embed_dim]

        # Concatenate embeddings
        x = torch.cat([loc_emb, tim_emb], dim=-1)  # [batch_size, seq_len, embed_dim*2]

        # Encode trajectory
        # In TrajSDE, this would use SDE-based encoding with uncertainty
        output, hidden = self.encoder(x)  # output: [batch_size, seq_len, hidden_size]

        # Get the last output (or use hidden state)
        # We use the last timestep's output for prediction
        last_output = output[:, -1, :]  # [batch_size, hidden_size]

        # Decode to location predictions
        # In TrajSDE, this would use SDE integration to predict multiple future positions
        score = self.decoder(last_output)  # [batch_size, loc_size]

        return score

    def predict(self, batch):
        """
        Predict next location.

        Args:
            batch: LibCity batch containing input data

        Returns:
            torch.Tensor: Location prediction scores [batch_size, loc_size]
        """
        self.eval()
        with torch.no_grad():
            score = self.forward(batch)
            # Return log softmax for compatibility with NLLLoss evaluation
            return F.log_softmax(score, dim=-1)

    def calculate_loss(self, batch):
        """
        Calculate training loss.

        Args:
            batch: LibCity batch containing input data and targets

        Returns:
            torch.Tensor: Loss value (scalar)
        """
        # Forward pass
        score = self.forward(batch)

        # Get target
        batch_data = self._prepare_batch(batch)
        target = batch_data['target']

        # Calculate cross-entropy loss
        # In TrajSDE, loss includes:
        # 1. L2 loss for trajectory positions
        # 2. Differential BCE loss for mode classification
        # 3. Multiple metrics (ADE, FDE, MR)
        loss = self.criterion(score, target)

        return loss


# ============================================================================
# DOCUMENTATION: How to implement full TrajSDE integration
# ============================================================================

"""
FULL IMPLEMENTATION ROADMAP:

1. DATA FORMAT CONVERSION (Most Critical):
   -----------------------------------------
   LibCity uses discrete location indices, TrajSDE uses continuous coordinates.

   Required steps:
   a) Create location coordinate mapping:
      - Add 'loc_coords' to data_feature: dict mapping loc_id -> (x, y) coordinates
      - Or generate synthetic coordinates based on location co-occurrence patterns

   b) Convert LibCity Batch to TemporalData:
      ```python
      def _convert_to_temporal_data(self, batch):
          # Extract batch data
          current_loc = batch['current_loc']  # [batch_size, seq_len]
          current_tim = batch['current_tim']  # [batch_size, seq_len]

          # Map discrete locations to continuous coordinates
          # This requires location coordinate data
          positions = self.loc_coords[current_loc]  # [batch_size, seq_len, 2]

          # Compute velocities (x_{t} - x_{t-1})
          x = positions[:, 1:] - positions[:, :-1]  # [batch_size, seq_len-1, 2]

          # Build graph connectivity
          # For trajectory prediction, typically use k-NN or radius-based graphs
          edge_index = self._build_spatial_graph(positions)

          # Create TemporalData object
          data = TemporalData(
              x=x.permute(1, 0, 2),  # [seq_len-1, batch_size, 2]
              positions=positions.permute(1, 0, 2),  # [seq_len, batch_size, 2]
              edge_index=edge_index,
              padding_mask=self._create_padding_mask(batch),
              bos_mask=self._create_bos_mask(batch),
              # ... other required fields
          )
          return data
      ```

2. GRAPH CONSTRUCTION:
   --------------------
   TrajSDE uses spatial graphs for agent-agent interaction.

   Required steps:
   a) Implement k-NN graph builder:
      ```python
      def _build_spatial_graph(self, positions, k=10):
          # Build k-NN graph based on spatial proximity
          # positions: [batch_size, seq_len, 2]
          # Returns: edge_index [2, num_edges]
          pass
      ```

   b) Handle lane graph features:
      - TrajSDE uses lane centerlines for map context
      - For POI prediction, could use road network or POI categories
      - Or disable lane features (set has_goal=False, skip AL encoder)

3. MODEL CONFIGURATION:
   ---------------------
   Create proper config matching TrajSDE's YAML format.

   Example:
   ```python
   trajsde_config = {
       'encoder': {
           'file_path': f'{TRAJSDE_REPO_PATH}/models/encoders/enc_hivt_nusargo_sde_sep2.py',
           'module_name': 'LocalEncoderSDESepPara2',
           'kwargs': {
               'embed_dim': 64,
               'num_heads': 8,
               'dropout': 0.1,
               # ... other params
           }
       },
       'decoder': { ... },
       'aggregator': { ... },
       'losses': [ ... ],
       'metrics': [ ... ],
   }
   ```

4. LOSS AND METRICS:
   ------------------
   TrajSDE uses:
   - L2 loss for trajectory coordinates
   - DiffBCE for mode classification
   - ADE (Average Displacement Error)
   - FDE (Final Displacement Error)
   - MR (Miss Rate)

   For discrete locations, adapt to:
   - CrossEntropy for location prediction
   - Top-k accuracy metrics
   - Distance-based metrics (if coordinates available)

5. TRAINING INTEGRATION:
   ----------------------
   TrajSDE uses PyTorch Lightning. For LibCity:
   - Extract training logic from PredictionModelSDENet
   - Implement in standard PyTorch (as done in AbstractModel)
   - Handle optimizer/scheduler configuration

6. DEPENDENCIES:
   --------------
   Required packages:
   - torchsde: pip install torchsde
   - torch-geometric: pip install torch-geometric
   - Check repos/TrajSDE/env.yml for exact versions

7. TESTING:
   ---------
   Start with synthetic data:
   ```python
   # Test with dummy continuous trajectory data
   test_batch = {
       'current_loc': torch.randint(0, 100, (32, 20)),  # batch=32, seq=20
       'current_tim': torch.randint(0, 24, (32, 20)),
       'target': torch.randint(0, 100, (32,)),
   }
   model = TrajSDE(config, data_feature)
   output = model.predict(test_batch)
   loss = model.calculate_loss(test_batch)
   ```

8. ALTERNATIVE APPROACH:
   ---------------------
   If full integration is too complex, consider:

   a) Use TrajSDE concepts without native components:
      - Implement simplified SDE encoder using torchsde
      - Use GRU with noise injection for uncertainty
      - Multi-mode prediction with mixture of experts

   b) Hybrid approach:
      - Use TrajSDE's encoder/decoder as-is
      - Add discrete location mapping layer on top
      - Train end-to-end with location classification loss

RECOMMENDED STARTING POINT:
---------------------------
1. First implement location coordinate mapping (if coordinates available)
2. Test with simplified model (current implementation)
3. Gradually add TrajSDE components one by one
4. Start with encoder only, using simple decoder
5. Add full SDE decoder once data pipeline works
6. Finally integrate multi-modal prediction
"""
