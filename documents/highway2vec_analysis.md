# Highway2Vec Repository Analysis

## Repository Information
- **URL**: https://github.com/Calychas/highway2vec
- **Cloned to**: /home/wangwenrui/shk/AgentCity/repos/highway2vec
- **Clone Status**: Success
- **Paper**: SIGSPATIAL 2022 - "Representing OpenStreetMap microregions with respect to their road network characteristics"

## Repository Overview

Highway2vec is a road network representation learning method that generates embeddings for geographical microregions (hexagonal grids using H3 indexing) based on their road network characteristics extracted from OpenStreetMap data. The model uses an autoencoder architecture to learn low-dimensional representations of road features.

### Key Concept
The approach divides cities into hexagonal regions (using Uber's H3 spatial indexing system) and extracts road network features from OpenStreetMap within each hexagon. These features are aggregated and normalized, then fed into an autoencoder to generate compact embeddings that capture road network characteristics.

## Directory Structure

```
highway2vec/
├── data/
│   ├── raw/              # Configuration files and raw data
│   ├── generated/        # Generated graph data from OSM
│   ├── processed/        # Processed intermediate data
│   ├── features/         # Feature datasets for training
│   └── runs/             # Model training runs and outputs
├── src/
│   ├── models/
│   │   └── autoencoder.py      # Main model definition
│   ├── tools/
│   │   ├── feature_extraction.py  # Feature engineering
│   │   ├── h3_utils.py           # H3 hexagon utilities
│   │   ├── osmnx_utils.py        # OSM data downloading
│   │   ├── aggregation.py        # Feature aggregation
│   │   ├── configs.py            # Configuration classes
│   │   ├── clustering.py         # Clustering utilities
│   │   ├── dim_reduction.py      # Dimensionality reduction
│   │   ├── vis_utils.py          # Visualization utilities
│   │   └── logger.py             # Logging utilities
│   └── settings.py              # Project settings and paths
├── scripts/
│   ├── download_and_preprocess_data.py  # Data pipeline
│   ├── generate_dataset.py              # Dataset generation
│   └── generate_place.py                # Place-specific data generation
├── notebooks/
│   ├── autoencoder.ipynb        # Training notebook
│   ├── vis_ae.ipynb             # Embedding visualization
│   └── vis_data.ipynb           # Data visualization
├── requirements.txt
├── requirements_freeze.txt
└── README.md
```

## Key Files and Components

### 1. Model Definition

**File**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/src/models/autoencoder.py`

**Main Class**: `LitAutoEncoder`

**Architecture**:
- Encoder: Linear(in_dim → hidden_dim) → ReLU → Linear(hidden_dim → latent_dim)
- Decoder: Linear(latent_dim → hidden_dim) → ReLU → Linear(hidden_dim → in_dim)
- Loss: MSE reconstruction loss
- Framework: PyTorch Lightning

**Key Parameters**:
- `in_dim`: Input feature dimension (determined by selected road features)
- `hidden_dim`: Hidden layer dimension (default: 64)
- `latent_dim`: Embedding dimension (default: 3)
- `lr`: Learning rate (default: 1e-3)

**Methods**:
- `forward()`: Returns latent embeddings from encoder
- `training_step()`, `validation_step()`, `test_step()`: Training loop
- `configure_optimizers()`: Adam optimizer

### 2. Data Processing Pipeline

#### a. Data Download and Preprocessing
**File**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/scripts/download_and_preprocess_data.py`

**Purpose**: Downloads OSM road network data for specified cities and generates H3 hexagonal grids

**Dependencies**:
- OSMnx for downloading OpenStreetMap data
- H3 for hexagonal spatial indexing

#### b. Feature Extraction
**File**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/src/tools/feature_extraction.py`

**Key Functions**:
- `generate_features_for_edges()`: Generates features from OSM road data
- `apply_feature_selection()`: Applies feature selection and mapping
- `normalize_df()`: Normalizes features (global or local normalization)
- `explode_and_pivot()`: Converts categorical features to one-hot encoding

**Feature Categories** (from featureset_transformation_default.jsonc):
1. **oneway**: One-way street indicator (True/False)
2. **lanes**: Number of lanes (1-15)
3. **highway**: Road type (motorway, primary, secondary, tertiary, residential, etc.)
4. **maxspeed**: Speed limit (5-200 km/h in bins)
5. **bridge**: Bridge type indicators
6. **access**: Access restrictions
7. **junction**: Junction type (roundabout, circular)
8. **width**: Road width (1.0-30.0 meters)
9. **tunnel**: Tunnel indicators
10. **surface**: Surface type (paved, asphalt, concrete, unpaved, etc.)
11. **bicycle**: Bicycle access
12. **lit**: Street lighting

**Feature Processing**:
- Features are extracted from OSM tags
- Categorical features are one-hot encoded
- Features can be scaled by road segment length
- Aggregated at hexagon level (sum of all road segments)
- Normalized globally or locally by city

#### c. Dataset Generation
**File**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/scripts/generate_dataset.py`

**Purpose**: Aggregates road features by hexagon and creates training dataset

**Output**: `SpatialDataset` object containing:
- Cities metadata
- Edge-level features
- Hexagon-level aggregated features
- Normalized features for training

### 3. Configuration Files

**Location**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/data/raw/`

**Key Config Files**:
1. `featureset_transformation_default.jsonc`: Defines all possible feature values
2. `featureset_selection_1.jsonc`: Feature selection and merging rules
3. `implicit_maxspeeds.jsonc`: Speed limit defaults by country

**Configuration Classes** (`src/tools/configs.py`):
- `DatasetGenerationConfig`: Dataset generation parameters
- `ExperimentConfig`: Model training parameters

### 4. Spatial Utilities

#### H3 Hexagon Utils
**File**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/src/tools/h3_utils.py`

**Key Functions**:
- `generate_hexagons_for_place()`: Creates H3 hexagonal grid for a region
- `assign_hexagons_to_edges()`: Maps road segments to hexagons
- `get_buffered_place_for_h3()`: Buffers region boundary for edge cases

#### OSMnx Utils
**File**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/src/tools/osmnx_utils.py`

**Key Functions**:
- `generate_data_for_place()`: Downloads OSM data for a place
- `download_and_save_data_for_place()`: Downloads and saves graph data

**Extracted OSM Tags**:
bridge, tunnel, oneway, lanes, ref, name, highway, maxspeed, service, access, area, landuse, width, est_width, junction, surface, footway, bicycle, lit

### 5. Training and Inference

**Primary Interface**: Jupyter notebooks in `notebooks/`

**Training Notebook**: `autoencoder.ipynb`
- Loads dataset from pickle
- Splits train/test data
- Creates PyTorch DataLoader
- Trains LitAutoEncoder with PyTorch Lightning
- Saves trained model and embeddings

**Visualization Notebooks**:
- `vis_data.ipynb`: Visualizes raw data and features
- `vis_ae.ipynb`: Visualizes learned embeddings

## Dependencies

### Core Dependencies (requirements.txt)
```
# Geospatial
GDAL
Fiona
rasterio
geopandas
shapely
h3~=3.7.4
osmnx
networkx
folium
contextily
keplergl

# Deep Learning
torch
torchvision
torchaudio
pytorch-lightning

# Data Processing
pandas
numpy
scikit-learn
swifter
pyarrow

# Visualization
plotly
matplotlib
seaborn
jupyter

# Utilities
tqdm
click
unidecode
json5
```

### Version Information (from requirements_freeze.txt)
- Python: 3.x (specified in .python-version file)
- PyTorch: 1.13.0
- PyTorch Lightning: 1.7.7
- pandas: 1.5.1
- numpy: 1.23.4
- geopandas: 0.12.1
- h3: 3.7.4
- osmnx: 1.2.2
- scikit-learn: 1.1.3

## Model Architecture Details

### Input
- **Type**: Aggregated road network features per hexagon
- **Dimension**: Variable (depends on selected features, typically 100-200 features)
- **Format**: Normalized feature vectors (sum of all road segments in hexagon)

### Architecture
```
Encoder:
  Input (in_dim) → Linear → ReLU → Linear → Latent (latent_dim)

Decoder:
  Latent (latent_dim) → Linear → ReLU → Linear → Reconstruction (in_dim)
```

### Output
- **Embeddings**: Latent vectors of dimension `latent_dim` (default: 3)
- **Purpose**: Low-dimensional representation of road network characteristics

## Data Flow

1. **Download**: OSM data → GraphML/GeoPackage files
2. **H3 Grid**: Generate hexagonal grid for region
3. **Feature Extraction**: Extract road features from OSM tags
4. **Aggregation**: Sum features by hexagon
5. **Normalization**: Normalize features (global or local)
6. **Training**: Feed to autoencoder
7. **Embeddings**: Extract latent vectors for each hexagon

## Model Task

**Task Type**: Road Network Representation Learning / Trajectory Embedding

**Input**: Aggregated road network features (categorical and continuous)
**Output**: Fixed-dimensional embeddings for geographical regions

**Use Cases**:
- Region similarity analysis
- City comparison based on road network
- Clustering of urban areas
- Feature-based search for similar regions

## Migration Considerations for LibCity

### Compatibility
- **Framework**: PyTorch Lightning (compatible with LibCity's PyTorch base)
- **Input Format**: Requires OpenStreetMap data preprocessing
- **Output**: Embeddings (not direct prediction)

### Key Differences from LibCity Models
1. **Data Source**: Uses OpenStreetMap instead of traffic sensor data
2. **Spatial Unit**: H3 hexagons instead of fixed sensors/nodes
3. **Task**: Unsupervised representation learning (not traffic prediction)
4. **Training**: Autoencoder reconstruction (not supervised prediction)

### Required Adaptations
1. **Data Pipeline**: Need to integrate OSM data downloading and H3 hexagon generation
2. **Feature Extraction**: Port feature extraction logic to LibCity dataset classes
3. **Model Class**: Adapt LitAutoEncoder to LibCity's AbstractModel interface
4. **Executor**: Create custom executor for representation learning tasks
5. **Configuration**: Integrate featureset configs into LibCity's config system
6. **Evaluation**: Define appropriate metrics for embedding quality

### Potential LibCity Category
- **Primary**: `road_representation` or `trajectory_embedding`
- **Alternative**: New category for unsupervised spatial representation learning

## Structure Notes

1. **Well-Organized**: Clear separation between data processing, model, and utilities
2. **Configuration-Driven**: Feature selection and transformation via JSON configs
3. **Notebook-Based Workflow**: Training primarily done in Jupyter notebooks
4. **No CLI Interface**: No command-line training script (only notebooks)
5. **External Dependencies**: Heavy reliance on OSMnx and H3 libraries
6. **Data Not Included**: Repository doesn't include pre-processed data (must download)
7. **Archived**: Repository is no longer maintained (see new version: SRAI library)

## Model Class Names

- **Main Model**: `LitAutoEncoder` (PyTorch Lightning module)
- **Dataset Class**: `SpatialDataset` (dataclass for holding all data)
- **Config Classes**: `DatasetGenerationConfig`, `ExperimentConfig`

## Additional Notes

1. **Repository Status**: No longer maintained; authors recommend using the SRAI library
2. **New Version**: Updated pipeline available at https://github.com/Calychas/highway2vec_remaster
3. **Paper Data**: Final run from paper available via Google Drive link in README
4. **Submodules**: No git submodules used
5. **Tests**: No test suite included
6. **Documentation**: Primarily through README and notebooks
