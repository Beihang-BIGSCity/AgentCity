# HetETA Encoder Migration Summary

## Overview

This document describes the HetETAEncoder implementation that bridges trajectory data from LibCity's ETADataset with the HetETA model's expected traffic state format.

## Problem Addressed

The HetETA model expects traffic state format (batch with 'X' and 'y' keys) but was receiving trajectory format. The encoder converts:
- **Input**: Trajectory data (GPS points with timestamps)
- **Output**: Spatiotemporal grid format `X[batch, time_steps, num_nodes, features]`

## Files Created/Modified

### Created Files

1. **Encoder**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/heteta_encoder.py`

### Modified Files

1. **Encoder Registry**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/eta_encoder/__init__.py`
   - Added `HetETAEncoder` import and export

2. **Task Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
   - Updated HetETA to use `HetETAEncoder`

3. **Model Config**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/eta/HetETA.json`
   - Added `num_nodes`, `feature_dim`, `output_dim`, `per_period`, `grid_size` parameters

4. **HetETA Model**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/eta/HetETA.py`
   - Updated `predict()` method to handle ETA format with `_predict_eta()` and `_predict_traffic_state()`
   - Updated `calculate_loss()` to handle both ETA and traffic state formats

## Architecture

### Encoder Flow

```
Trajectory Data (GPS points)
         |
         v
+-----------------+
|  HetETAEncoder  |
+-----------------+
         |
         v
+------------------------------------+
| 1. Extract Link Speeds             |
|    - Map coordinates to nodes      |
|    - Calculate segment speeds      |
|    - Aggregate per-link distances  |
+------------------------------------+
         |
         v
+------------------------------------+
| 2. Create Temporal Features        |
|    - Weekly patterns (4 steps)     |
|    - Daily patterns (4 steps)      |
|    - Recent patterns (4 steps)     |
+------------------------------------+
         |
         v
+------------------------------------+
| 3. Output Batch Format             |
|    - X: [B, 12, N, 1]              |
|    - link_distances: [B, N]        |
|    - time: ground truth (seconds)  |
+------------------------------------+
```

### Model Flow

```
+------------------------------------+
|           HetETA Model             |
+------------------------------------+
         |
         v
+------------------------------------+
| 1. Speed Prediction                |
|    - ST Conv Blocks (weeks/days/   |
|      recent)                       |
|    - Multi-head attention          |
|    - Heterogeneous graph conv      |
+------------------------------------+
         |
         v
+------------------------------------+
| 2. ETA Calculation                 |
|    - pred_speed: [B, N]            |
|    - ETA = sum(dist / speed)       |
+------------------------------------+
         |
         v
     ETA Output: [B, 1]
```

## Key Implementation Details

### Coordinate to Node Mapping

The encoder uses grid discretization to map GPS coordinates to node IDs:

```python
def _coord_to_node(self, lon, lat):
    # Normalize to [0, 1] range
    norm_lon = (lon - self.lon_min) / (self.lon_max - self.lon_min)
    norm_lat = (lat - self.lat_min) / (self.lat_max - self.lat_min)

    # Map to grid
    grid_x = int(norm_lon * self.grid_size)
    grid_y = int(norm_lat * self.grid_size)

    # Convert to linear index
    node_id = grid_y * self.grid_size + grid_x
    return min(node_id, self.num_nodes - 1)
```

### Temporal Feature Generation

Since historical speed data is not available, the encoder generates synthetic temporal patterns:

- **Weekly patterns**: Base speed with weekly variation
- **Daily patterns**: Base speed with daily rush hour variation
- **Recent patterns**: Current observed speeds with small noise

### ETA Calculation

```python
def _predict_eta(self, batch):
    pred_speed = self.forward({'X': x})  # [B, 1, N, 1]

    # ETA = sum(distance / speed)
    time_per_link = link_distances / pred_speed
    eta = time_per_link.sum(dim=1)  # [B, 1]

    return eta
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 4 | Number of recent time steps |
| `days` | 4 | Number of daily pattern time steps |
| `weeks` | 4 | Number of weekly pattern time steps |
| `num_nodes` | 100 | Number of road links/nodes |
| `feature_dim` | 1 | Number of features per node (speed) |
| `output_dim` | 1 | Output dimension |
| `per_period` | 5 | Minutes per time period |
| `grid_size` | 10 | Grid size for coordinate mapping |

## Output Format

The encoder outputs the following fields (matching ETAExecutor expectations):

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `X` | float | [T, N, F] | Spatiotemporal speed features |
| `y` | float | [1, N, 1] | Target speeds |
| `link_distances` | float | [N] | Distance per link (meters) |
| `link_ids` | int | variable | List of link IDs in route |
| `time` | float | [1] | Ground truth travel time (seconds) |
| `traj_len` | int | [1] | Trajectory length |
| `traj_id` | int | [1] | Trajectory ID |
| `uid` | int | [1] | User ID |
| `weekid` | int | [1] | Day of week (1-7) |
| `timeid` | int | [1] | Minute of day (1-1440) |
| `start_timestamp` | int | [1] | Unix timestamp |

## Limitations and Assumptions

1. **Simplified Link Identification**: Uses grid-based coordinate mapping instead of actual road network links

2. **Synthetic Historical Data**: Generates temporal patterns based on heuristics since actual historical speed data is not available

3. **Adjacency Matrix**: Built from observed trajectory transitions rather than actual road network topology

4. **Speed Estimation**: Uses GPS-derived speeds which may have noise from GPS errors

## Usage

```python
# In config file or command line
config = {
    "model": "HetETA",
    "eta_encoder": "HetETAEncoder",
    "num_nodes": 100,
    "seq_len": 4,
    "days": 4,
    "weeks": 4,
    # ... other parameters
}
```

## Future Improvements

1. Integrate with actual road network data for proper link identification
2. Use historical traffic speed data if available
3. Implement proper multi-relation adjacency matrices for heterogeneous graph
4. Add support for map-matched trajectories with known link sequences
