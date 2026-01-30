# UniST Grid Dimension Quick Reference

## For Common Datasets

| Dataset | Nodes | Recommended Grid | Total Cells | Padding |
|---------|-------|------------------|-------------|---------|
| METR_LA | 207 | 15x15 | 225 | 18 |
| PEMSD3 | 358 | 19x19 | 361 | 3 |
| PEMSD4 | 307 | 18x18 | 324 | 17 |
| PEMSD7 | 883 | 30x30 | 900 | 17 |
| PEMSD8 | 170 | 13x13 | 169 | -1* |
| PEMS_BAY | 325 | 18x18 | 324 | -1* |
| LOS_LOOP | 207 | 15x15 | 225 | 18 |

\* Negative padding means the dataset has more nodes than grid cells. In these cases, increase grid size by 1.

## Calculation Method

1. **Square Root**: Find `sqrt(num_nodes)`
2. **Round Up**: Take ceiling of the square root
3. **Verify**: Ensure `H * W >= num_nodes`
4. **Balance**: Try to keep the grid as square as possible

Example for 207 nodes:
- sqrt(207) ≈ 14.38
- Round up: 15
- Grid: 15x15 = 225 nodes
- Padding: 225 - 207 = 18 extra cells

## Python Helper

```python
import math

def calculate_grid_dims(num_nodes):
    """Calculate optimal grid dimensions for given number of nodes."""
    sqrt_n = math.ceil(math.sqrt(num_nodes))

    # Try square grid first
    if sqrt_n * sqrt_n >= num_nodes:
        return sqrt_n, sqrt_n

    # If not perfect square, try rectangular
    height = sqrt_n
    width = math.ceil(num_nodes / height)

    return height, width

# Example usage
num_nodes = 207
h, w = calculate_grid_dims(num_nodes)
print(f"For {num_nodes} nodes: {h}x{w} = {h*w} (padding: {h*w - num_nodes})")
```

## Memory Impact

Grid size affects memory usage:

| Grid Size | Small Model | Middle Model | Large Model |
|-----------|-------------|--------------|-------------|
| 10x10 | ~2GB | ~5GB | ~12GB |
| 15x15 | ~3GB | ~8GB | ~18GB |
| 20x20 | ~5GB | ~12GB | ~28GB |
| 30x30 | ~10GB | ~25GB | ~60GB |

*Estimates based on batch_size=16, input_window=12, output_window=12
