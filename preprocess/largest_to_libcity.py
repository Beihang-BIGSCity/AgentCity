#!/usr/bin/env python3
"""
Convert LargeST dataset to LibCity atomic file format.

Source: LargeST - A Benchmark Dataset for Large-Scale Traffic Forecasting
        /home/wangwenrui/shk/AgentCity/datasets/LargeST/ca_his_raw_2017.h5
Target: Bigscity-LibCity/raw_data/LARGEST/

Dataset Information:
- Shape: (105,120 timesteps × 8,600 sensors)
- Time Range: 2017-01-01 00:00:00 to 2017-12-31 23:55:00
- Interval: 5 minutes
- Feature: Traffic speed (float64)
- Missing data: ~8.24% NaN values

Usage:
    python preprocess/largest_to_libcity.py
"""

import os
import json
import h5py
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Configuration
SOURCE_FILE = "/home/wangwenrui/shk/AgentCity/datasets/LargeST/ca_his_raw_2017.h5"
TARGET_DIR = "/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/raw_data/LARGEST"
DATASET_NAME = "LARGEST"

# Preprocessing parameters
MAX_SPEED = 120.0  # mph - cap outliers
MIN_SENSOR_COVERAGE = 0.5  # Filter sensors with >50% missing data
KNN_NEIGHBORS = 10  # Number of neighbors for k-NN graph construction

def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)
    print(f"Target directory: {path}")

def load_source_data():
    """Load the LargeST HDF5 file."""
    print("\n" + "="*80)
    print("Loading LargeST dataset...")
    print("="*80)

    with h5py.File(SOURCE_FILE, 'r') as f:
        # Load sensor IDs
        sensor_ids_raw = f['t/axis0'][:]
        sensor_ids = [sid.decode('utf-8') for sid in sensor_ids_raw]

        # Load timestamps
        timestamps_ns = f['t/axis1'][:]
        timestamps = pd.to_datetime(timestamps_ns, unit='ns')

        # Load traffic data matrix [time, sensors]
        print(f"Loading data matrix (this may take a while)...")
        data_matrix = f['t/block0_values'][:]

    print(f"Data shape: {data_matrix.shape}")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    print(f"Number of sensors: {len(sensor_ids)}")
    print(f"Data type: {data_matrix.dtype}")

    return data_matrix, timestamps, sensor_ids

def filter_and_clean_data(data_matrix, sensor_ids):
    """Filter sensors with too much missing data and cap outliers."""
    print("\n" + "="*80)
    print("Filtering and cleaning data...")
    print("="*80)

    num_timesteps, num_sensors = data_matrix.shape

    # Calculate missing data ratio per sensor
    nan_ratio = np.isnan(data_matrix).sum(axis=0) / num_timesteps
    valid_sensors = nan_ratio < (1 - MIN_SENSOR_COVERAGE)

    print(f"Sensors before filtering: {num_sensors}")
    print(f"Sensors with >{int((1-MIN_SENSOR_COVERAGE)*100)}% missing data: {(~valid_sensors).sum()}")
    print(f"Sensors after filtering: {valid_sensors.sum()}")

    # Filter data
    data_clean = data_matrix[:, valid_sensors]
    sensor_ids_clean = [sensor_ids[i] for i, valid in enumerate(valid_sensors) if valid]

    # Cap outliers
    print(f"\nData range before capping: [{np.nanmin(data_clean):.2f}, {np.nanmax(data_clean):.2f}]")
    outliers = np.sum(data_clean > MAX_SPEED)
    print(f"Values exceeding {MAX_SPEED} mph: {outliers} ({outliers/data_clean.size*100:.4f}%)")

    data_clean = np.clip(data_clean, 0, MAX_SPEED)
    print(f"Data range after capping: [{np.nanmin(data_clean):.2f}, {np.nanmax(data_clean):.2f}]")

    # Report missing values
    total_values = data_clean.size
    nan_values = np.isnan(data_clean).sum()
    print(f"\nMissing values (NaN): {nan_values} / {total_values} ({nan_values/total_values*100:.2f}%)")

    return data_clean, sensor_ids_clean

def create_geo_file(sensor_ids, output_path):
    """Create the .geo file with node information."""
    print("\n" + "="*80)
    print("Creating .geo file...")
    print("="*80)

    geo_records = []
    for geo_id, sensor_id in enumerate(sensor_ids):
        # Since we don't have coordinates, use empty list as per PEMSD7 format
        geo_records.append({
            'geo_id': geo_id,
            'type': 'Point',
            'coordinates': '[]'
        })

    geo_df = pd.DataFrame(geo_records)
    geo_df.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(f"Number of nodes: {len(geo_df)}")
    print(f"Sample:\n{geo_df.head()}")

def create_knn_graph(num_nodes, k):
    """Create a k-NN graph based on sensor indices (temporal/spatial proximity approximation)."""
    print("\n" + "="*80)
    print(f"Creating k-NN graph (k={k})...")
    print("="*80)

    edges = []

    # For each node, connect to k nearest neighbors based on index
    # This assumes sensors with similar indices are spatially close
    for i in range(num_nodes):
        # Calculate distances to all other nodes (using index distance as proxy)
        distances = np.abs(np.arange(num_nodes) - i)

        # Get k+1 nearest (including itself), then exclude itself
        nearest_indices = np.argsort(distances)[1:k+1]

        for j in nearest_indices:
            # Calculate distance based on index difference
            dist = abs(i - j) / num_nodes  # Normalize to [0, 1]
            edges.append((i, j, dist))

    print(f"Number of directed edges: {len(edges)}")
    return edges

def create_rel_file(num_nodes, output_path):
    """Create the .rel file with edge/relationship information."""
    print("\n" + "="*80)
    print("Creating .rel file...")
    print("="*80)

    # Create k-NN graph
    edges = create_knn_graph(num_nodes, KNN_NEIGHBORS)

    rel_records = []
    for rel_id, (origin, dest, cost) in enumerate(edges):
        rel_records.append({
            'rel_id': rel_id,
            'type': 'geo',
            'origin_id': origin,
            'destination_id': dest,
            'cost': cost
        })

    rel_df = pd.DataFrame(rel_records)
    rel_df.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(f"Number of edges: {len(rel_df)}")
    print(f"Average degree: {len(rel_df) / num_nodes:.2f}")
    print(f"Sample:\n{rel_df.head()}")

def create_dyna_file(data_matrix, timestamps, output_path):
    """Create the .dyna file with time-series data."""
    print("\n" + "="*80)
    print("Creating .dyna file...")
    print("="*80)

    num_timesteps, num_sensors = data_matrix.shape
    total_records = num_timesteps * num_sensors

    print(f"Total records to create: {total_records:,}")
    print(f"This may take a while...")

    # Write incrementally to avoid memory issues
    batch_size = 1000  # Process 1k timesteps at a time
    dyna_id = 0
    first_batch = True

    with open(output_path, 'w') as f:
        # Write header
        f.write("dyna_id,type,time,entity_id,traffic_speed\n")

        for t_start in tqdm(range(0, num_timesteps, batch_size), desc="Writing batches"):
            t_end = min(t_start + batch_size, num_timesteps)

            # Build batch in memory
            batch_lines = []
            for t in range(t_start, t_end):
                timestamp_str = timestamps[t].strftime('%Y-%m-%dT%H:%M:%SZ')

                for entity_id in range(num_sensors):
                    speed = data_matrix[t, entity_id]

                    # Convert NaN to 0.0 for LibCity compatibility
                    if np.isnan(speed):
                        speed = 0.0

                    batch_lines.append(f"{dyna_id},state,{timestamp_str},{entity_id},{speed}\n")
                    dyna_id += 1

            # Write batch to file
            f.writelines(batch_lines)

    print(f"\nCreated {output_path}")
    print(f"Number of records: {total_records:,}")

    # Read first few lines to show sample
    print(f"\nReading sample...")
    sample_df = pd.read_csv(output_path, nrows=10)
    print(f"Sample:\n{sample_df}")

def create_config_file(num_nodes, output_path):
    """Create the config.json file."""
    print("\n" + "="*80)
    print("Creating config.json...")
    print("="*80)

    config = {
        "geo": {
            "including_types": ["Point"],
            "Point": {}
        },
        "rel": {
            "including_types": ["geo"],
            "geo": {
                "cost": "num"
            }
        },
        "dyna": {
            "including_types": ["state"],
            "state": {
                "entity_id": "geo_id",
                "traffic_speed": "num"
            }
        },
        "info": {
            "data_col": ["traffic_speed"],
            "weight_col": "cost",
            "data_files": [DATASET_NAME],
            "geo_file": DATASET_NAME,
            "rel_file": DATASET_NAME,
            "output_dim": 1,
            "time_intervals": 300,
            "init_weight_inf_or_zero": "zero",
            "set_weight_link_or_dist": "dist",
            "calculate_weight_adj": False,
            "weight_adj_epsilon": 0.1,
            "num_nodes": num_nodes
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created {output_path}")
    print(f"Configuration:")
    print(json.dumps(config, indent=2))

def create_readme(sensor_ids_original, sensor_ids_filtered, data_matrix, output_path):
    """Create README file with dataset information."""
    print("\n" + "="*80)
    print("Creating README...")
    print("="*80)

    readme_content = f"""# LargeST Dataset - LibCity Format

## Source Information
- **Dataset**: LargeST - A Benchmark Dataset for Large-Scale Traffic Forecasting
- **Region**: California (CA)
- **Year**: 2017
- **Source File**: ca_his_raw_2017.h5

## Conversion Information
- **Conversion Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Script**: preprocess/largest_to_libcity.py

## Data Statistics

### Original Dataset
- **Sensors**: {len(sensor_ids_original)}
- **Timesteps**: {data_matrix.shape[0]:,}
- **Time Range**: 2017-01-01 00:00:00 to 2017-12-31 23:55:00
- **Interval**: 5 minutes (300 seconds)

### Filtered Dataset
- **Sensors**: {len(sensor_ids_filtered)} (filtered sensors with >{int((1-MIN_SENSOR_COVERAGE)*100)}% missing data)
- **Timesteps**: {data_matrix.shape[0]:,}
- **Total Records**: {data_matrix.shape[0] * len(sensor_ids_filtered):,}

### Preprocessing Applied
1. **Sensor Filtering**: Removed sensors with >{int((1-MIN_SENSOR_COVERAGE)*100)}% missing data
2. **Outlier Capping**: Capped speeds at {MAX_SPEED} mph
3. **Missing Value Handling**: NaN values converted to 0.0
4. **Graph Construction**: k-NN graph with k={KNN_NEIGHBORS} based on sensor indices

## Files

### {DATASET_NAME}.geo
- **Format**: CSV
- **Columns**: geo_id, type, coordinates
- **Rows**: {len(sensor_ids_filtered)}
- **Description**: Node/sensor information. Coordinates are empty as spatial data not available.

### {DATASET_NAME}.rel
- **Format**: CSV
- **Columns**: rel_id, type, origin_id, destination_id, cost
- **Rows**: ~{len(sensor_ids_filtered) * KNN_NEIGHBORS}
- **Description**: k-NN graph edges. Cost is normalized index distance.

### {DATASET_NAME}.dyna
- **Format**: CSV
- **Columns**: dyna_id, type, time, entity_id, traffic_speed
- **Rows**: {data_matrix.shape[0] * len(sensor_ids_filtered):,}
- **Description**: Time-series traffic speed data (mph).

### config.json
- **Format**: JSON
- **Description**: Dataset configuration for LibCity framework.

## Notes

1. **No Spatial Coordinates**: The original LargeST dataset does not include sensor coordinates. The .geo file uses empty coordinate lists.

2. **Graph Structure**: Since no adjacency matrix was provided, a k-NN graph is constructed based on sensor index proximity. This is an approximation and may not reflect true spatial relationships.

3. **Missing Values**: ~8.24% of values were NaN in the original dataset. These have been converted to 0.0 for LibCity compatibility. Consider using masked loss functions during training.

4. **Data Scale**: This is a large-scale dataset. The .dyna file contains over 900 million records and may require significant memory and processing time.

5. **Recommended Usage**:
   - Consider subsampling sensors for initial experiments
   - Use batch loading strategies
   - Memory-mapped arrays may be beneficial

## Citation

If you use this dataset, please cite the original LargeST paper:

```
@article{{largest2023,
  title={{LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting}},
  author={{[Authors]}},
  journal={{[Conference/Journal]}},
  year={{2023}}
}}
```

## Contact

For issues or questions:
- Dataset location: /home/wangwenrui/shk/AgentCity/Bigscity-LibCity/raw_data/LARGEST/
- Conversion date: {datetime.now().strftime('%Y-%m-%d')}
"""

    with open(output_path, 'w') as f:
        f.write(readme_content)

    print(f"Created {output_path}")

def main():
    """Main conversion pipeline."""
    print("="*80)
    print("LargeST to LibCity Conversion")
    print("="*80)
    print(f"Source: {SOURCE_FILE}")
    print(f"Target: {TARGET_DIR}")

    # Create target directory
    ensure_dir(TARGET_DIR)

    # Load source data
    data_matrix, timestamps, sensor_ids = load_source_data()

    # Filter and clean data
    data_clean, sensor_ids_clean = filter_and_clean_data(data_matrix, sensor_ids)

    # Create atomic files
    create_geo_file(
        sensor_ids_clean,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.geo")
    )

    create_rel_file(
        len(sensor_ids_clean),
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.rel")
    )

    create_dyna_file(
        data_clean,
        timestamps,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.dyna")
    )

    create_config_file(
        len(sensor_ids_clean),
        os.path.join(TARGET_DIR, "config.json")
    )

    create_readme(
        sensor_ids,
        sensor_ids_clean,
        data_clean,
        os.path.join(TARGET_DIR, "README.md")
    )

    print("\n" + "="*80)
    print("Conversion Complete!")
    print("="*80)
    print(f"\nOutput files in: {TARGET_DIR}")
    print(f"- {DATASET_NAME}.geo")
    print(f"- {DATASET_NAME}.rel")
    print(f"- {DATASET_NAME}.dyna")
    print(f"- config.json")
    print(f"- README.md")
    print("\nDataset ready for use with LibCity framework!")

if __name__ == "__main__":
    main()
