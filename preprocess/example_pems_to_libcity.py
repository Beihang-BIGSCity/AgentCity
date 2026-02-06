#!/usr/bin/env python3
"""
Example: Convert PEMS-BAY style dataset to LibCity atomic file format.

This is a template script for converting traffic speed/flow datasets.
The dataset-converter agent will generate similar scripts based on the
specific source data format.

Source Format (typical PEMS-BAY):
- metr-la.h5 or pems-bay.h5: HDF5 file with 'df' key containing DataFrame
- adj_mx.pkl: Pickle file with adjacency matrix and sensor IDs
- sensor_locations.csv (optional): Sensor coordinates

Target: Bigscity-LibCity/raw_data/<dataset_name>/
- <dataset_name>.geo: Sensor locations
- <dataset_name>.rel: Sensor relationships (from adjacency matrix)
- <dataset_name>.dyna: Time series data
- config.json: Dataset configuration

Usage:
    python preprocess/example_pems_to_libcity.py
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# Configuration - Modify these for your dataset
# ============================================================================
DATASET_NAME = "PEMS_BAY_NEW"  # Name for the converted dataset
SOURCE_DIR = "./datasets/PEMS-BAY"  # Where raw data was downloaded
TARGET_DIR = f"./Bigscity-LibCity/raw_data/{DATASET_NAME}"

# Source file names (modify based on actual downloaded files)
DATA_FILE = "pems-bay.h5"  # HDF5 file with speed data
ADJ_FILE = "adj_mx_bay.pkl"  # Adjacency matrix file
SENSOR_FILE = None  # Optional: sensor_locations.csv

# Time configuration
TIME_START = "2017-01-01T00:00:00Z"  # Dataset start time
TIME_INTERVAL = 300  # Sampling interval in seconds (5 minutes = 300)

# Data column configuration
DATA_COLUMNS = ["traffic_speed"]  # Columns to include in .dyna file

# ============================================================================


def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def load_h5_data(filepath):
    """Load time series data from HDF5 file."""
    import h5py

    # Try pandas HDF5 format first
    try:
        df = pd.read_hdf(filepath)
        print(f"Loaded HDF5 with pandas: shape {df.shape}")
        return df
    except Exception:
        pass

    # Try h5py format
    with h5py.File(filepath, 'r') as f:
        print(f"HDF5 keys: {list(f.keys())}")
        # Common key names
        for key in ['df', 'data', 'speed', 'block0_values']:
            if key in f:
                data = f[key][:]
                print(f"Loaded key '{key}' with shape {data.shape}")
                return data

    raise ValueError(f"Could not load data from {filepath}")


def load_adjacency_matrix(filepath):
    """Load adjacency matrix from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Handle different pickle formats
    if isinstance(data, tuple) or isinstance(data, list):
        # Format: (sensor_ids, sensor_id_to_ind, adj_mx)
        sensor_ids = data[0]
        adj_mx = data[2]
        return sensor_ids, adj_mx
    elif isinstance(data, dict):
        # Format: {'sensor_ids': [...], 'adj': [...]}
        sensor_ids = data.get('sensor_ids', data.get('ids', None))
        adj_mx = data.get('adj', data.get('adj_mx', data.get('matrix', None)))
        return sensor_ids, adj_mx
    elif isinstance(data, np.ndarray):
        # Just the matrix
        return None, data
    else:
        raise ValueError(f"Unknown adjacency format: {type(data)}")


def load_sensor_locations(filepath):
    """Load sensor locations from CSV file."""
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        return df
    return None


def create_geo_file(sensor_ids, sensor_locations, output_path):
    """Create the .geo file with node information."""
    records = []

    for idx, sensor_id in enumerate(sensor_ids):
        record = {
            'geo_id': sensor_id,
            'type': 'Point',
        }

        # Add coordinates if available
        if sensor_locations is not None:
            loc = sensor_locations[sensor_locations['sensor_id'] == sensor_id]
            if len(loc) > 0:
                lat = loc['latitude'].values[0]
                lon = loc['longitude'].values[0]
                record['coordinates'] = json.dumps([lon, lat])
            else:
                record['coordinates'] = '[]'
        else:
            # Placeholder coordinates
            record['coordinates'] = '[]'

        records.append(record)

    geo_df = pd.DataFrame(records)
    geo_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(geo_df)} nodes")
    return geo_df


def create_rel_file(sensor_ids, adj_matrix, output_path):
    """Create the .rel file with edge/relationship information."""
    records = []
    rel_id = 0

    n_nodes = len(sensor_ids)

    for i in range(n_nodes):
        for j in range(n_nodes):
            weight = adj_matrix[i, j]
            # Only include non-zero edges (or all for dense representation)
            if weight > 0:
                records.append({
                    'rel_id': rel_id,
                    'type': 'geo',
                    'origin_id': sensor_ids[i],
                    'destination_id': sensor_ids[j],
                    'cost': float(weight)
                })
                rel_id += 1

    rel_df = pd.DataFrame(records)
    rel_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(rel_df)} edges")
    return rel_df


def create_dyna_file(data, sensor_ids, output_path, start_time=None, interval=300):
    """Create the .dyna file with time-series data."""
    records = []
    dyna_id = 0

    # Parse start time
    if start_time is None:
        start_time = datetime(2017, 1, 1)
    elif isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00').replace('+00:00', ''))

    # Handle DataFrame vs numpy array
    if isinstance(data, pd.DataFrame):
        timestamps = data.index
        values = data.values
        if data.columns.dtype == 'int64':
            sensor_ids = data.columns.tolist()
    else:
        values = data
        n_times = values.shape[0]
        timestamps = [start_time + timedelta(seconds=interval * i) for i in range(n_times)]

    n_times, n_nodes = values.shape[0], values.shape[1]

    print(f"Creating dyna file: {n_times} timesteps x {n_nodes} nodes")

    for t_idx in range(n_times):
        if isinstance(timestamps[t_idx], datetime):
            time_str = timestamps[t_idx].strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            time_str = str(timestamps[t_idx])

        for n_idx in range(n_nodes):
            entity_id = sensor_ids[n_idx] if sensor_ids else n_idx
            value = values[t_idx, n_idx]

            records.append({
                'dyna_id': dyna_id,
                'type': 'state',
                'time': time_str,
                'entity_id': entity_id,
                'traffic_speed': float(value)
            })
            dyna_id += 1

    dyna_df = pd.DataFrame(records)
    dyna_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(dyna_df)} records")
    return dyna_df


def create_config_file(output_path, n_nodes, data_columns, time_interval):
    """Create the config.json file."""
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
            }
        },
        "info": {
            "data_col": data_columns,
            "weight_col": "cost",
            "data_files": [DATASET_NAME],
            "geo_file": DATASET_NAME,
            "rel_file": DATASET_NAME,
            "output_dim": len(data_columns),
            "time_intervals": time_interval,
            "init_weight_inf_or_zero": "inf",
            "set_weight_link_or_dist": "dist",
            "calculate_weight_adj": True,
            "weight_adj_epsilon": 0.1
        }
    }

    # Add data columns to dyna config
    for col in data_columns:
        config["dyna"]["state"][col] = "num"

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created {output_path}")
    return config


def main():
    """Main conversion pipeline."""
    print("=" * 60)
    print(f"Converting {DATASET_NAME} to LibCity format")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print()

    # Ensure target directory exists
    ensure_dir(TARGET_DIR)

    # Load source data
    print("Loading source data...")

    # Load time series
    data_path = os.path.join(SOURCE_DIR, DATA_FILE)
    if os.path.exists(data_path):
        data = load_h5_data(data_path)
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load adjacency matrix
    adj_path = os.path.join(SOURCE_DIR, ADJ_FILE)
    if os.path.exists(adj_path):
        sensor_ids, adj_matrix = load_adjacency_matrix(adj_path)
        if sensor_ids is None:
            sensor_ids = list(range(adj_matrix.shape[0]))
    else:
        print(f"Warning: Adjacency file not found: {adj_path}")
        n_nodes = data.shape[1] if isinstance(data, np.ndarray) else data.shape[1]
        sensor_ids = list(range(n_nodes))
        adj_matrix = np.eye(n_nodes)  # Identity matrix as placeholder

    # Load sensor locations (optional)
    sensor_locations = None
    if SENSOR_FILE:
        sensor_path = os.path.join(SOURCE_DIR, SENSOR_FILE)
        sensor_locations = load_sensor_locations(sensor_path)

    print(f"Loaded {len(sensor_ids)} sensors")
    print()

    # Create atomic files
    print("Creating LibCity atomic files...")

    create_geo_file(
        sensor_ids,
        sensor_locations,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.geo")
    )

    create_rel_file(
        sensor_ids,
        adj_matrix,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.rel")
    )

    create_dyna_file(
        data,
        sensor_ids,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.dyna"),
        start_time=TIME_START,
        interval=TIME_INTERVAL
    )

    create_config_file(
        os.path.join(TARGET_DIR, "config.json"),
        n_nodes=len(sensor_ids),
        data_columns=DATA_COLUMNS,
        time_interval=TIME_INTERVAL
    )

    print()
    print("=" * 60)
    print("Conversion complete!")
    print(f"Dataset saved to: {TARGET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
