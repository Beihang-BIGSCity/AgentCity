#!/usr/bin/env python3
"""
Example: Convert trajectory/check-in dataset to LibCity atomic file format.

This is a template script for converting trajectory location prediction datasets
(e.g., Foursquare, Gowalla, Brightkite check-in data).

Source Format (typical check-in dataset):
- checkins.csv: user_id, timestamp, latitude, longitude, venue_id
- venues.csv or pois.csv: venue_id, latitude, longitude, category, name

Target: Bigscity-LibCity/raw_data/<dataset_name>/
- <dataset_name>.geo: POI/venue information
- <dataset_name>.usr: User information
- <dataset_name>.dyna: Check-in records (trajectory type)
- config.json: Dataset configuration

Usage:
    python preprocess/example_trajectory_to_libcity.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# Configuration - Modify these for your dataset
# ============================================================================
DATASET_NAME = "foursquare_nyc_new"  # Name for the converted dataset
SOURCE_DIR = "./datasets/foursquare_nyc"  # Where raw data was downloaded
TARGET_DIR = f"./Bigscity-LibCity/raw_data/{DATASET_NAME}"

# Source file names (modify based on actual downloaded files)
CHECKIN_FILE = "dataset_TSMC2014_NYC.txt"  # Check-in records
VENUE_FILE = None  # Optional: separate venue file

# Column mapping for check-in file
# Map source column names to standard names
CHECKIN_COLUMNS = {
    'user_id': 0,  # Column index or name for user ID
    'venue_id': 1,  # Column index or name for venue/POI ID
    'category_id': 2,  # Column index or name for category ID (optional)
    'category_name': 3,  # Column index or name for category name (optional)
    'latitude': 4,  # Column index or name for latitude
    'longitude': 5,  # Column index or name for longitude
    'timezone_offset': 6,  # Column index or name for timezone offset (optional)
    'utc_time': 7,  # Column index or name for UTC timestamp
}

# File format
SEPARATOR = '\t'  # Separator for CSV/TSV file
HAS_HEADER = False  # Whether file has header row

# Distance upper bound for POI matching (in km)
DISTANCE_UPPER = 30.0

# ============================================================================


def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def parse_timestamp(ts):
    """Parse timestamp to ISO format."""
    if pd.isna(ts):
        return None

    if isinstance(ts, (int, float)):
        # Unix timestamp
        return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%SZ')

    if isinstance(ts, str):
        # Try various formats
        formats = [
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%a %b %d %H:%M:%S %z %Y',  # Twitter format
            '%Y/%m/%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts, fmt).strftime('%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                continue

        # Try pandas parsing as last resort
        try:
            return pd.to_datetime(ts).strftime('%Y-%m-%dT%H:%M:%SZ')
        except Exception:
            pass

    return str(ts)


def load_checkin_data(filepath, columns_map, separator='\t', has_header=False):
    """Load check-in data from file."""
    # Determine header setting
    header = 0 if has_header else None

    df = pd.read_csv(filepath, sep=separator, header=header, encoding='utf-8',
                     on_bad_lines='skip')

    print(f"Loaded {len(df)} check-ins from {filepath}")
    print(f"Columns: {list(df.columns)}")

    # Rename columns based on mapping
    rename_map = {}
    for std_name, col_ref in columns_map.items():
        if col_ref is not None:
            if isinstance(col_ref, int):
                if col_ref < len(df.columns):
                    rename_map[df.columns[col_ref]] = std_name
            else:
                if col_ref in df.columns:
                    rename_map[col_ref] = std_name

    df = df.rename(columns=rename_map)
    print(f"Renamed columns: {list(df.columns)}")

    return df


def create_geo_file(checkins, output_path):
    """Create the .geo file with POI/venue information."""
    # Extract unique venues with their info
    venue_cols = ['venue_id', 'latitude', 'longitude']
    optional_cols = ['category_id', 'category_name']

    for col in optional_cols:
        if col in checkins.columns:
            venue_cols.append(col)

    venues = checkins[venue_cols].drop_duplicates(subset=['venue_id'])
    venues = venues.reset_index(drop=True)

    # Create geo records
    records = []
    for idx, row in venues.iterrows():
        record = {
            'geo_id': int(idx),  # Re-index from 0
            'type': 'Point',
            'coordinates': json.dumps([float(row['longitude']), float(row['latitude'])]),
        }

        if 'category_id' in row:
            record['venue_category_id'] = row['category_id']
        if 'category_name' in row:
            record['venue_category_name'] = row['category_name']

        records.append(record)

    geo_df = pd.DataFrame(records)
    geo_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(geo_df)} POIs")

    # Return venue_id to geo_id mapping
    venue_to_geo = dict(zip(venues['venue_id'], range(len(venues))))
    return geo_df, venue_to_geo


def create_usr_file(checkins, output_path):
    """Create the .usr file with user information."""
    users = checkins['user_id'].unique()

    # Create user records with re-indexed IDs
    records = [{'usr_id': idx} for idx in range(len(users))]

    usr_df = pd.DataFrame(records)
    usr_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(usr_df)} users")

    # Return user_id to usr_id mapping
    user_to_usr = dict(zip(users, range(len(users))))
    return usr_df, user_to_usr


def create_dyna_file(checkins, venue_to_geo, user_to_usr, output_path):
    """Create the .dyna file with trajectory data."""
    records = []

    # Sort by user and time
    if 'utc_time' in checkins.columns:
        time_col = 'utc_time'
    elif 'timestamp' in checkins.columns:
        time_col = 'timestamp'
    else:
        time_col = checkins.columns[0]  # Fallback

    checkins_sorted = checkins.sort_values(['user_id', time_col])

    for idx, row in checkins_sorted.iterrows():
        # Map to new IDs
        usr_id = user_to_usr.get(row['user_id'])
        geo_id = venue_to_geo.get(row['venue_id'])

        if usr_id is None or geo_id is None:
            continue

        # Parse timestamp
        time_str = parse_timestamp(row.get(time_col, row.get('utc_time', row.get('timestamp'))))

        records.append({
            'dyna_id': len(records),
            'type': 'trajectory',
            'time': time_str,
            'entity_id': usr_id,
            'location': geo_id,
        })

    dyna_df = pd.DataFrame(records)
    dyna_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(dyna_df)} check-ins")
    return dyna_df


def create_config_file(output_path, distance_upper=30.0, has_category=False):
    """Create the config.json file for trajectory dataset."""
    geo_point_config = {}
    if has_category:
        geo_point_config = {
            "venue_category_id": "enum",
            "venue_category_name": "enum"
        }

    config = {
        "geo": {
            "including_types": ["Point"],
            "Point": geo_point_config
        },
        "usr": {
            "properties": {}
        },
        "dyna": {
            "including_types": ["trajectory"],
            "trajectory": {
                "entity_id": "usr_id",
                "location": "geo_id"
            }
        },
        "info": {
            "distance_upper": distance_upper
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
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

    # Load check-in data
    print("Loading source data...")
    checkin_path = os.path.join(SOURCE_DIR, CHECKIN_FILE)

    if not os.path.exists(checkin_path):
        raise FileNotFoundError(f"Check-in file not found: {checkin_path}")

    checkins = load_checkin_data(
        checkin_path,
        CHECKIN_COLUMNS,
        separator=SEPARATOR,
        has_header=HAS_HEADER
    )

    print(f"Loaded {len(checkins)} check-ins")
    print(f"Unique users: {checkins['user_id'].nunique()}")
    print(f"Unique venues: {checkins['venue_id'].nunique()}")
    print()

    # Create atomic files
    print("Creating LibCity atomic files...")

    # Check if we have category information
    has_category = 'category_id' in checkins.columns or 'category_name' in checkins.columns

    geo_df, venue_to_geo = create_geo_file(
        checkins,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.geo")
    )

    usr_df, user_to_usr = create_usr_file(
        checkins,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.usr")
    )

    create_dyna_file(
        checkins,
        venue_to_geo,
        user_to_usr,
        os.path.join(TARGET_DIR, f"{DATASET_NAME}.dyna")
    )

    create_config_file(
        os.path.join(TARGET_DIR, "config.json"),
        distance_upper=DISTANCE_UPPER,
        has_category=has_category
    )

    print()
    print("=" * 60)
    print("Conversion complete!")
    print(f"Dataset saved to: {TARGET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
