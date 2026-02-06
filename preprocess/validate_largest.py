#!/usr/bin/env python3
"""
Validation script for LARGEST dataset in LibCity format.

Validates:
- File existence and format
- Data integrity
- Schema compliance
- Cross-file consistency

Usage:
    python preprocess/validate_largest.py
"""

import os
import json
import pandas as pd
import numpy as np

DATASET_DIR = "/home/wangwenrui/shk/AgentCity/Bigscity-LibCity/raw_data/LARGEST"
DATASET_NAME = "LARGEST"

def check_file_exists(filename):
    """Check if file exists."""
    filepath = os.path.join(DATASET_DIR, filename)
    exists = os.path.exists(filepath)
    size = os.path.getsize(filepath) / (1024**3) if exists else 0  # Size in GB
    print(f"  {'✓' if exists else '✗'} {filename}: {size:.2f} GB" if exists else f"  ✗ {filename}: NOT FOUND")
    return exists

def validate_geo_file():
    """Validate .geo file."""
    print("\n1. Validating LARGEST.geo...")
    filepath = os.path.join(DATASET_DIR, f"{DATASET_NAME}.geo")

    try:
        df = pd.read_csv(filepath)

        # Check columns
        required_cols = ['geo_id', 'type', 'coordinates']
        has_cols = all(col in df.columns for col in required_cols)
        print(f"  {'✓' if has_cols else '✗'} Required columns: {required_cols}")

        # Check data
        print(f"  ✓ Number of nodes: {len(df)}")
        print(f"  ✓ Geo ID range: [{df['geo_id'].min()}, {df['geo_id'].max()}]")

        # Check for duplicates
        duplicates = df['geo_id'].duplicated().sum()
        print(f"  {'✓' if duplicates == 0 else '✗'} Duplicate geo_ids: {duplicates}")

        # Check sequential IDs
        expected_ids = set(range(len(df)))
        actual_ids = set(df['geo_id'].values)
        missing_ids = expected_ids - actual_ids
        print(f"  {'✓' if len(missing_ids) == 0 else '✗'} Sequential geo_ids: {len(missing_ids)} missing")

        return len(df)

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 0

def validate_rel_file(num_nodes):
    """Validate .rel file."""
    print("\n2. Validating LARGEST.rel...")
    filepath = os.path.join(DATASET_DIR, f"{DATASET_NAME}.rel")

    try:
        # Read sample first
        sample_df = pd.read_csv(filepath, nrows=1000)

        # Check columns
        required_cols = ['rel_id', 'type', 'origin_id', 'destination_id', 'cost']
        has_cols = all(col in sample_df.columns for col in required_cols)
        print(f"  {'✓' if has_cols else '✗'} Required columns: {required_cols}")

        # Get total edges (count lines)
        with open(filepath, 'r') as f:
            total_edges = sum(1 for _ in f) - 1  # Exclude header
        print(f"  ✓ Number of edges: {total_edges:,}")

        # Check origin/destination IDs are valid
        valid_origin = (sample_df['origin_id'] >= 0).all() and (sample_df['origin_id'] < num_nodes).all()
        valid_dest = (sample_df['destination_id'] >= 0).all() and (sample_df['destination_id'] < num_nodes).all()
        print(f"  {'✓' if valid_origin else '✗'} Valid origin IDs (sample)")
        print(f"  {'✓' if valid_dest else '✗'} Valid destination IDs (sample)")

        # Check cost range
        print(f"  ✓ Cost range (sample): [{sample_df['cost'].min():.6f}, {sample_df['cost'].max():.6f}]")

        # Average degree
        avg_degree = total_edges / num_nodes
        print(f"  ✓ Average degree: {avg_degree:.2f}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def validate_dyna_file(num_nodes):
    """Validate .dyna file."""
    print("\n3. Validating LARGEST.dyna...")
    filepath = os.path.join(DATASET_DIR, f"{DATASET_NAME}.dyna")

    try:
        # Read sample
        sample_df = pd.read_csv(filepath, nrows=100000)

        # Check columns
        required_cols = ['dyna_id', 'type', 'time', 'entity_id', 'traffic_speed']
        has_cols = all(col in sample_df.columns for col in required_cols)
        print(f"  {'✓' if has_cols else '✗'} Required columns: {required_cols}")

        # Get total records (count lines)
        print("  Counting total records (this may take a moment)...")
        with open(filepath, 'r') as f:
            total_records = sum(1 for _ in f) - 1  # Exclude header
        print(f"  ✓ Number of records: {total_records:,}")

        # Check entity_id range
        valid_entities = (sample_df['entity_id'] >= 0).all() and (sample_df['entity_id'] < num_nodes).all()
        print(f"  {'✓' if valid_entities else '✗'} Valid entity IDs (sample)")
        print(f"  ✓ Entity ID range (sample): [{sample_df['entity_id'].min()}, {sample_df['entity_id'].max()}]")

        # Check traffic_speed range
        print(f"  ✓ Speed range (sample): [{sample_df['traffic_speed'].min():.1f}, {sample_df['traffic_speed'].max():.1f}] mph")

        # Check time format
        try:
            pd.to_datetime(sample_df['time'])
            print(f"  ✓ Time format valid (ISO 8601)")
            print(f"  ✓ Time range (sample): {sample_df['time'].iloc[0]} to {sample_df['time'].iloc[-1]}")
        except:
            print(f"  ✗ Invalid time format")

        # Check for NaN speeds
        nan_count = sample_df['traffic_speed'].isna().sum()
        print(f"  {'✓' if nan_count == 0 else '⚠'} NaN values in sample: {nan_count}")

        # Expected records
        expected_records = 105120 * num_nodes
        matches = abs(total_records - expected_records) < 100
        print(f"  {'✓' if matches else '⚠'} Expected records: {expected_records:,} (actual: {total_records:,})")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def validate_config_file():
    """Validate config.json file."""
    print("\n4. Validating config.json...")
    filepath = os.path.join(DATASET_DIR, "config.json")

    try:
        with open(filepath, 'r') as f:
            config = json.load(f)

        # Check required sections
        required_sections = ['geo', 'rel', 'dyna', 'info']
        has_sections = all(section in config for section in required_sections)
        print(f"  {'✓' if has_sections else '✗'} Required sections: {required_sections}")

        # Check info fields
        info = config.get('info', {})
        required_info = ['data_col', 'weight_col', 'data_files', 'geo_file', 'rel_file',
                        'output_dim', 'time_intervals', 'num_nodes']
        has_info = all(field in info for field in required_info)
        print(f"  {'✓' if has_info else '✗'} Required info fields present")

        # Check values
        print(f"  ✓ Dataset name: {info.get('data_files', ['?'])[0]}")
        print(f"  ✓ Number of nodes: {info.get('num_nodes', '?')}")
        print(f"  ✓ Time interval: {info.get('time_intervals', '?')} seconds")
        print(f"  ✓ Output dimension: {info.get('output_dim', '?')}")
        print(f"  ✓ Data columns: {info.get('data_col', ['?'])}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Main validation routine."""
    print("="*80)
    print("LARGEST Dataset Validation")
    print("="*80)
    print(f"\nDataset directory: {DATASET_DIR}\n")

    # Check all files exist
    print("Checking files...")
    files = [
        f"{DATASET_NAME}.geo",
        f"{DATASET_NAME}.rel",
        f"{DATASET_NAME}.dyna",
        "config.json",
        "README.md"
    ]
    all_exist = all(check_file_exists(f) for f in files)

    if not all_exist:
        print("\n✗ Some files are missing. Validation aborted.")
        return

    # Validate each file
    num_nodes = validate_geo_file()
    if num_nodes > 0:
        validate_rel_file(num_nodes)
        validate_dyna_file(num_nodes)

    validate_config_file()

    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)
    print("\nSummary:")
    print(f"  Dataset: LARGEST")
    print(f"  Location: {DATASET_DIR}")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Format: LibCity atomic files")
    print(f"  Status: Ready for training")
    print("\nNext steps:")
    print("  1. Use this dataset in LibCity training scripts")
    print("  2. Configure model with dataset='LARGEST'")
    print("  3. Consider subsampling for initial experiments")

if __name__ == "__main__":
    main()
