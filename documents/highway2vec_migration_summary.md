## Repository: highway2vec
- **URL**: https://github.com/Calychas/highway2vec
- **Cloned to**: /home/wangwenrui/shk/AgentCity/repos/highway2vec
- **Status**: Successfully cloned

### Key Files
- **Model**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/src/models/autoencoder.py`
- **Training**: Jupyter notebook-based (`notebooks/autoencoder.ipynb`)
- **Config**: JSON-based configs in `/home/wangwenrui/shk/AgentCity/repos/highway2vec/data/raw/` (featureset_transformation_default.jsonc, featureset_selection_1.jsonc)
- **Data Loader**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/scripts/generate_dataset.py`
- **Feature Extraction**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/src/tools/feature_extraction.py`
- **Data Pipeline**: `/home/wangwenrui/shk/AgentCity/repos/highway2vec/scripts/download_and_preprocess_data.py`

### Dependencies
- **Python**: 3.x
- **PyTorch**: 1.13.0
- **PyTorch Lightning**: 1.7.7
- **Key Packages**: 
  - geopandas 0.12.1
  - h3 3.7.4 (Uber's H3 spatial indexing)
  - osmnx 1.2.2 (OpenStreetMap data)
  - pandas 1.5.1
  - numpy 1.23.4
  - scikit-learn 1.1.3
  - networkx
  - shapely

### Structure Notes

**Model Architecture**:
- Simple autoencoder for road network representation learning
- Encoder: Linear → ReLU → Linear (to latent space)
- Decoder: Linear → ReLU → Linear (reconstruction)
- Loss: MSE reconstruction loss
- Main class: `LitAutoEncoder` (PyTorch Lightning)

**Data Processing**:
- Uses OpenStreetMap (OSM) data as input source
- Divides regions into H3 hexagonal grids
- Extracts 12 categories of road features:
  - oneway, lanes, highway type, maxspeed, bridge, access
  - junction, width, tunnel, surface, bicycle, lit
- One-hot encodes categorical features (100-200 total features)
- Aggregates features at hexagon level
- Normalizes globally or locally by city

**Key Workflow**:
1. Download OSM data for cities (using OSMnx)
2. Generate H3 hexagonal grid
3. Extract and one-hot encode road features
4. Aggregate features by hexagon
5. Train autoencoder on aggregated features
6. Extract latent embeddings for each hexagon

**Main Model Class**: `LitAutoEncoder`
- Located in: `src/models/autoencoder.py`
- Parameters: in_dim, hidden_dim (64), latent_dim (3), lr (1e-3)
- Output: Low-dimensional embeddings representing road network characteristics

**Task Type**: Road Network Representation Learning / Unsupervised Embedding

**Unique Aspects**:
- Uses H3 spatial indexing (not common in traffic models)
- OpenStreetMap-based (not traditional traffic sensor data)
- Unsupervised learning (autoencoder reconstruction)
- Focus on road network topology, not traffic flow
- Notebook-based training (no CLI script)

**Migration Challenges**:
1. **Data Source**: Requires OSM data integration (LibCity uses traffic datasets)
2. **Spatial Framework**: H3 hexagons vs LibCity's node/edge framework
3. **Task Mismatch**: Unsupervised embedding vs supervised traffic prediction
4. **Dependencies**: Requires osmnx and h3 libraries (not in LibCity)
5. **No Supervised Labels**: No ground truth for standard evaluation metrics
6. **Training Interface**: Notebook-based, needs CLI/script conversion

**Recommended LibCity Category**: `road_representation` or `trajectory_embedding`

**Additional Notes**:
- Repository is no longer maintained (archived project)
- Authors recommend new version: SRAI library
- Heavy preprocessing pipeline required
- Feature engineering is configuration-driven (JSON configs)
- No test suite or formal evaluation metrics
- Designed for geographic analysis, not traffic prediction
