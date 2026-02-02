# JGRM Encoder Migration Summary

## Overview

This document describes the custom trajectory encoder created for the JGRM (Joint GPS and Route Modeling) model to enable compatibility with standard LibCity trajectory datasets.

## Problem Statement

JGRM was originally designed for trajectory representation learning with actual GPS traces and road network data. It requires specialized data fields that standard LibCity trajectory encoders do not provide:

- `route_data`: Temporal features per road segment (weekday, minute, delta_time)
- `route_assign_mat`: Road segment sequence indices
- `gps_data`: GPS point features (8 dimensions per point)
- `gps_assign_mat`: GPS-to-road segment assignments
- `gps_length`: Number of GPS points per road segment

Standard POI datasets (foursquare_tky, foursquare_nyc, gowalla) only provide discrete check-in locations and timestamps.

## Solution

### 1. JGRMEncoder

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/trajectory_encoder/jgrm_encoder.py`

The JGRMEncoder creates a compatibility layer that adapts standard trajectory data to JGRM's expected format:

#### Data Mappings

| JGRM Field | Source Data | Mapping Strategy |
|------------|-------------|------------------|
| `route_assign_mat` | Location IDs | POI IDs treated as "road segments" |
| `route_data` | Timestamps | [weekday, minute_of_day, delta_time] |
| `gps_data` | Location + Time | 8D synthetic features (lat, lng, temporal, velocity) |
| `gps_assign_mat` | Location IDs | 1-to-1 mapping (each point maps to itself) |
| `gps_length` | N/A | Set to 1 per location |
| `edge_index` | Trajectories | Built from location transition co-occurrence |

#### GPS Feature Construction (8 dimensions)

1. Normalized latitude
2. Normalized longitude
3. Sin(hour)
4. Cos(hour)
5. Sin(day_of_week)
6. Cos(day_of_week)
7. Speed proxy (distance/time from previous point)
8. Heading proxy (direction from previous point)

#### Graph Construction

The encoder builds an edge_index for the GAT by tracking location transitions across all trajectories. Each transition (src_loc, dst_loc) becomes an edge in the graph.

### 2. JGRM Model Updates

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`

The JGRM model was updated to support compatibility mode:

#### Key Changes

1. **Data Extraction Helper**: Added `_extract_jgrm_data()` method to handle both direct JGRM format and JGRMEncoder format

2. **Location Prediction Head**: Added `loc_pred_head` for trajectory location prediction task

3. **Flexible Initialization**: Support for both `vocab_size` and `loc_size` in data_feature

4. **Edge Index Fallback**: Creates self-loop edges if edge_index not provided

5. **Value Clamping**: Added clamping for embedding indices to prevent out-of-bounds errors

6. **Combined Loss**: Updated `calculate_loss()` to include location prediction loss when target is available

### 3. Configuration Updates

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`

Updated JGRM entry to use JGRMEncoder:

```json
"JGRM": {
    "dataset_class": "TrajectoryDataset",
    "executor": "TrajLocPredExecutor",
    "evaluator": "TrajLocPredEvaluator",
    "traj_encoder": "JGRMEncoder"
}
```

### 4. Encoder Registration

**File**: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/trajectory_encoder/__init__.py`

Added import and export for JGRMEncoder.

## Usage

### Running JGRM with Standard Dataset

```bash
python run_model.py --task traj_loc_pred --model JGRM --dataset foursquare_tky
```

### Required Config Parameters

The following parameters can be set in config or model JSON:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `route_max_len` | 100 | Maximum route sequence length |
| `gps_feat_num` | 8 | Number of GPS point features |
| `hidden_size` | 256 | Hidden layer dimension |
| `road_embed_size` | 128 | Road segment embedding dimension |
| `gps_embed_size` | 128 | GPS feature embedding dimension |
| `route_embed_size` | 128 | Route representation dimension |

## Limitations and Assumptions

1. **Simplified GPS Mapping**: Each POI is treated as a single "road segment" with one GPS point, which is a simplification of the original JGRM design

2. **Synthetic Features**: GPS features are synthesized from location IDs and timestamps rather than actual GPS coordinates (unless geo file is available)

3. **Graph Sparsity**: The location transition graph may be sparser than a real road network graph

4. **Representation Quality**: The learned representations may differ from those trained on actual GPS/road network data, but should still capture meaningful trajectory patterns

## File Locations

- Encoder: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/data/dataset/trajectory_encoder/jgrm_encoder.py`
- Model: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/model/trajectory_loc_prediction/JGRM.py`
- Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/model/traj_loc_pred/JGRM.json`
- Task Config: `/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/libcity/config/task_config.json`
