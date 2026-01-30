# STWave Model Migration to LibCity

## Overview

This document describes the migration of the STWave model to the LibCity framework for traffic speed prediction.

**Original Paper**: "When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks"

**Source Repository**: `./repos/STWave`

**Target Location**: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STWave.py`

## Model Architecture

STWave is a spatio-temporal graph neural network that uses wavelet decomposition to disentangle traffic signals into:
- **Low-frequency component**: Represents long-term trends
- **High-frequency component**: Represents short-term fluctuations

The model processes these components through a dual encoder architecture:
1. **Temporal Attention** for low-frequency signals
2. **Temporal Convolutional Network (TCN)** for high-frequency signals
3. **Sparse Spatial Attention** for both components
4. **Adaptive Fusion** to combine the outputs

## Key Components Ported

| Component | Original Location | Description |
|-----------|------------------|-------------|
| STWave | models.py:302 | Main model class |
| Dual_Encoder | models.py:201 | Dual path encoder |
| Sparse_Spatial_Attention | models.py:78 | Sparse attention with spectral info |
| TemporalAttention | models.py:10 | Causal temporal attention |
| Adaptive_Fusion | models.py:133 | Low-high frequency fusion |
| TemporalConvNet | models.py:165 | Dilated causal convolutions |
| Chomp1d | models.py:8 | Padding removal for causal conv |
| TemEmbedding | models.py:19 | Temporal feature embedding |
| FeedForward | models.py:38 | Multi-layer perceptron |

## Key Adaptations

### 1. Base Class Inheritance
```python
class STWave(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
```

### 2. Graph Feature Computation

The original model requires pre-computed graph features:
- `localadj`: Nearest neighbors based on shortest path distances
- `spawave`: Eigenvalues/eigenvectors of spatial Laplacian
- `temwave`: Eigenvalues/eigenvectors of temporal adjacency

**Adaptation**: These are now computed automatically from `adj_mx`:

```python
def compute_localadj(adj):
    # Uses Dijkstra's algorithm to find nearest neighbors

def compute_spawave(adj, dims):
    # Computes Laplacian eigendecomposition
```

### 3. Data Format Handling

**Original**: Expects separate XL, XH, TE tensors
**Adapted**: Extracts from LibCity batch format

```python
def forward(self, batch):
    xl, xh, te = self._extract_temporal_features(batch)
    hat_y, _ = self._forward(xl, xh, te)
    return hat_y
```

### 4. Wavelet Decomposition

Uses PyWavelets to disentangle signals:
```python
def disentangle(self, x, w, j):
    coef = pywt.wavedec(x_np, w, level=j)
    # Separate low and high frequency coefficients
    xl = pywt.waverec(coefl, w)
    xh = pywt.waverec(coefh, w)
    return xl, xh
```

### 5. Loss Function

Combined loss from main prediction and low-frequency prediction:
```python
def calculate_loss(self, batch):
    main_loss = loss.masked_mae_torch(y_predicted_inv, y_true_inv, null_val=0.0)
    lf_loss = loss.masked_mae_torch(hat_y_l_inv, YL, null_val=0.0)
    return main_loss + lf_loss
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| heads | 8 | Number of attention heads |
| dims | 16 | Dimension per head |
| layers | 2 | Number of dual encoder layers |
| samples | 1 | Sampling factor for sparse attention |
| wave | "coif1" | Wavelet type for decomposition |
| level | 1 | Wavelet decomposition level |
| input_window | 12 | Input sequence length |
| output_window | 12 | Output sequence length |
| time_intervals | 300 | Time interval in seconds (5 min) |

## Files Modified

1. **Created**: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/STWave.py`
   - Complete model implementation adapted for LibCity

2. **Modified**: `Bigscity-LibCity/libcity/model/traffic_speed_prediction/__init__.py`
   - Added import and registration

3. **Modified**: `Bigscity-LibCity/libcity/config/task_config.json`
   - Changed dataset_class from "STWaveDataset" to "TrafficStatePointDataset"

## Dependencies

- PyTorch
- NumPy
- SciPy (for sparse matrix operations and eigendecomposition)
- PyWavelets (pywt) for wavelet decomposition

## Usage

```python
# Example configuration
config = {
    "model": "STWave",
    "dataset": "METR_LA",
    "heads": 8,
    "dims": 16,
    "layers": 2,
    "wave": "coif1",
    "level": 1,
    "input_window": 12,
    "output_window": 12
}
```

## Limitations and Notes

1. **Temporal Adjacency**: The original model computes temporal adjacency using DTW (Dynamic Time Warping). For efficiency, the adapted version uses spatial adjacency as a fallback. For better accuracy, consider pre-computing DTW-based temporal adjacency.

2. **Temporal Features**: The model expects temporal features (time_of_day, day_of_week) in the input. If not available, default sequential indices are used.

3. **Memory Usage**: The sparse spatial attention reduces memory compared to full attention but still requires graph eigendecomposition during initialization.

## Testing

Run the model with:
```bash
cd Bigscity-LibCity
python run_model.py --task traffic_state_pred --model STWave --dataset METR_LA
```
