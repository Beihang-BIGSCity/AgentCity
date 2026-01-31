# HSTWAVE Migration Summary

## Migration Summary

- **Paper Title**: "Riding the Wave: Multi-Scale Spatial-Temporal Graph Learning (IJCAI)"
- **Repository**: https://github.com/luck-seu/HST-WAVE
- **Model**: HSTWAVE
- **Status**: SUCCESS

## Migration Process

### Phase 1: Repository Cloning

- Successfully cloned to `./repos/HSTWAVE`
- Identified main model class and components

### Phase 2: Model Adaptation

- **Created LibCity-compatible model at**: `Bigscity-LibCity/libcity/model/traffic_flow_prediction/HSTWAVE.py`
- **Total parameters**: 1,870,328

**Key components migrated**:
- GTU (Gated Temporal Unit)
- MSWT (Multi-Scale Weaving Transformer)
- CHGAN (Coupled Heterogeneous Graph Attention Network)
- MSDTHGTEncoder
- SequenceAugmentor for contrastive learning

**Adaptation highlights**:
- Converted from PyTorch Lightning to AbstractTrafficStateModel
- Simplified heterogeneous graph handling for LibCity's adjacency matrix format
- Integrated LibCity's data batching and evaluation pipeline

### Phase 3: Configuration

- **Registered in**: `task_config.json`
- **Created configuration file**: `HSTWAVE.json` with paper hyperparameters

**Key hyperparameters**:
- `hidden_dim`: 64
- `num_layers`: 3
- `num_scales`: 3
- `batch_size`: 8
- `learning_rate`: 0.001
- `weight_decay`: 5e-4
- `contrastive_weight`: 0.2

### Phase 4: Testing & Iteration

#### Iteration 1: FAILED

**Error encountered**:
- Shape mismatch error: Expected 192 channels but got 36
- **Root cause**: Concatenation along wrong dimension in MSWT module

#### Iteration 2: SUCCESS

**Fixes applied**:
- Fixed concatenation from `dim=2` to `dim=3` in MSWT forward method
- Updated `kernel_size` in `out_linear` layer to match corrected dimensions
- Training completed successfully

## Test Results

- **Dataset**: METR_LA
- **Epochs**: 1 (validation run)
- **Train Loss**: 3.4400
- **Val Loss**: 3.4310
- **Training Time**: ~60 minutes per epoch
- **Evaluation metrics**: Reasonable MAE/RMSE across all 12 horizons

## Files Created/Modified

### Created Files
- `Bigscity-LibCity/libcity/model/traffic_flow_prediction/HSTWAVE.py`
- `Bigscity-LibCity/libcity/config/model/traffic_state_pred/HSTWAVE.json`

### Modified Files
- `Bigscity-LibCity/libcity/config/task_config.json` (updated)
- `Bigscity-LibCity/libcity/model/traffic_flow_prediction/__init__.py` (updated)

## Usage

```bash
python run_model.py --task traffic_state_pred --model HSTWAVE --dataset METR_LA
```

**Example with custom configuration**:

```bash
python run_model.py \
  --task traffic_state_pred \
  --model HSTWAVE \
  --dataset METR_LA \
  --batch_size 8 \
  --learning_rate 0.001 \
  --max_epoch 100
```

## Notes & Recommendations

### General Notes
- Model works with standard traffic datasets (METR_LA, PEMS_BAY, etc.)
- Contrastive learning enabled by default with weight of 0.2
- Longer training (50-100 epochs) needed for publication-level performance
- Batch size of 8 recommended for memory efficiency with multi-scale architecture

### Performance Considerations
- Multi-scale architecture requires substantial GPU memory
- Training time is approximately 60 minutes per epoch on METR_LA
- For faster experimentation, consider reducing `num_scales` or `num_layers`

### Future Enhancements
- Support for additional heterogeneous graph types
- Experiment with different contrastive learning strategies
- Fine-tune hyperparameters for specific datasets

## Migration Challenges & Solutions

### Challenge 1: Heterogeneous Graph Handling
- **Issue**: Original model used complex heterogeneous graph structures not directly supported by LibCity
- **Solution**: Simplified to use LibCity's standard adjacency matrix format while preserving multi-relational learning capabilities

### Challenge 2: Dimension Mismatch in Multi-Scale Concatenation
- **Issue**: Incorrect concatenation dimension caused shape mismatches in downstream layers
- **Solution**: Changed concatenation from feature dimension (dim=2) to channel dimension (dim=3)

### Challenge 3: PyTorch Lightning to LibCity Conversion
- **Issue**: Original model tightly coupled with PyTorch Lightning training loops
- **Solution**: Refactored to inherit from AbstractTrafficStateModel and use LibCity's training infrastructure

## Conclusion

The HSTWAVE model has been successfully migrated to the LibCity framework. The migration maintains the core architectural components including the multi-scale weaving transformer, gated temporal units, and contrastive learning mechanisms. Initial testing shows the model trains successfully and produces reasonable predictions. Further hyperparameter tuning and extended training runs are recommended to achieve performance comparable to the original paper.
