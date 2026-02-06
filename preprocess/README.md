# Dataset Preprocessing Scripts

This directory contains Python scripts for converting external datasets to LibCity atomic file format.

## LibCity Data Format Overview

LibCity uses a set of atomic files to represent different types of spatio-temporal data:

### File Types

| File | Extension | Description |
|------|-----------|-------------|
| Geographic | `.geo` | Node/location information (sensors, POIs, etc.) |
| Relationship | `.rel` | Edge/connection information (distances, adjacency) |
| Dynamic | `.dyna` | Time-series data (traffic speed, check-ins, etc.) |
| User | `.usr` | User information (for trajectory data) |
| Config | `config.json` | Dataset configuration and schema |

### Example: Traffic Speed Dataset

```
raw_data/METR_LA/
├── METR_LA.geo      # 207 sensor locations
├── METR_LA.rel      # 11,753 distance relationships
├── METR_LA.dyna     # 6.9M speed records
└── config.json      # Dataset configuration
```

### Example: Trajectory Dataset

```
raw_data/foursquare_nyc/
├── foursquare_nyc.geo   # 38,333 POI locations
├── foursquare_nyc.usr   # 1,083 users
├── foursquare_nyc.dyna  # 227,428 check-ins
└── config.json          # Dataset configuration
```

## Template Scripts

### Traffic Speed/Flow Datasets
- **example_pems_to_libcity.py**: Template for PEMS-style traffic datasets
  - Input: HDF5 speed data + pickle adjacency matrix
  - Output: `.geo`, `.rel`, `.dyna`, `config.json`

### Trajectory Datasets
- **example_trajectory_to_libcity.py**: Template for check-in/trajectory datasets
  - Input: CSV check-in records
  - Output: `.geo`, `.usr`, `.dyna`, `config.json`

## Creating New Conversion Scripts

When the `dataset-converter` agent creates a new conversion script, it will:

1. Analyze the source data structure
2. Copy the appropriate template
3. Modify the configuration section for the specific dataset
4. Implement any custom parsing logic needed
5. Run and verify the conversion

### Script Structure

```python
#!/usr/bin/env python3
"""
Convert <SourceDataset> to LibCity atomic file format.
"""

# ============================================================================
# Configuration - Dataset-specific settings
# ============================================================================
DATASET_NAME = "my_dataset"
SOURCE_DIR = "./datasets/my_dataset"
TARGET_DIR = f"./Bigscity-LibCity/raw_data/{DATASET_NAME}"
# ... more config ...

# ============================================================================

def load_source_data():
    """Load and parse source files."""
    pass

def create_geo_file(data, output_path):
    """Create .geo file."""
    pass

def create_rel_file(data, output_path):
    """Create .rel file (if applicable)."""
    pass

def create_dyna_file(data, output_path):
    """Create .dyna file."""
    pass

def create_config_file(output_path):
    """Create config.json."""
    pass

def main():
    """Run conversion pipeline."""
    pass

if __name__ == "__main__":
    main()
```

## Usage

```bash
# Run a conversion script
python preprocess/example_pems_to_libcity.py

# Or specify custom paths
python preprocess/my_dataset_to_libcity.py --source ./data --target ./output
```

## Supported Source Formats

The conversion scripts can handle various common formats:

### Time Series Data
- HDF5 (`.h5`, `.hdf5`) - pandas DataFrames or numpy arrays
- NumPy (`.npz`, `.npy`) - speed/flow matrices
- CSV (`.csv`) - with timestamp column
- Pickle (`.pkl`) - serialized data structures

### Graph/Adjacency Data
- Pickle (`.pkl`) - adjacency matrices
- NumPy (`.npz`, `.npy`) - distance matrices
- CSV (`.csv`) - edge lists

### Trajectory Data
- CSV/TSV - check-in records with user, time, location
- JSON - trajectory sequences

## File Format Specifications

### .geo File
```csv
geo_id,type,coordinates[,extra_columns...]
0,Point,"[-118.31828, 34.15497]"
1,Point,"[-118.23799, 34.11621]",venue_category,Coffee Shop
```

### .rel File
```csv
rel_id,type,origin_id,destination_id,cost
0,geo,0,0,0.0
1,geo,0,1,4123.8
```

### .dyna File (state type - traffic)
```csv
dyna_id,type,time,entity_id,traffic_speed
0,state,2012-03-01T00:00:00Z,0,64.375
1,state,2012-03-01T00:05:00Z,0,62.667
```

### .dyna File (trajectory type)
```csv
dyna_id,type,time,entity_id,location
0,trajectory,2012-04-03T14:00:09Z,0,2388
1,trajectory,2012-04-03T14:00:25Z,1,3921
```

### .usr File
```csv
usr_id
0
1
2
```

### config.json
```json
{
  "geo": {
    "including_types": ["Point"],
    "Point": {}
  },
  "rel": {
    "including_types": ["geo"],
    "geo": {"cost": "num"}
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
    "time_intervals": 300,
    "output_dim": 1
  }
}
```

## Notes

- All timestamps should be in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
- Coordinates should be [longitude, latitude] as JSON arrays
- Entity IDs in .dyna files must reference valid geo_ids or usr_ids
- The `info` section in config.json varies by task type
