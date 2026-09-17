"""
Profile PatchTST with synthetic data (no dataset loading).
Metrics: Trainable Params, FLOPs, Peak GPU Memory (train), Inference Time.
"""
import os, sys, time, json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libcity.model.traffic_speed_prediction.PatchTST import PatchTST


class ForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, X):
        return self.model({'X': X})


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt(n):
    if n is None:
        return 'N/A'
    if n >= 1e9:
        return f'{n / 1e9:.3f}G'
    if n >= 1e6:
        return f'{n / 1e6:.3f}M'
    if n >= 1e3:
        return f'{n / 1e3:.3f}K'
    return f'{n:.0f}'


class FakeScaler:
    def inverse_transform(self, x):
        return x


class DictConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---- Config ---- #
    batch_size = 2
    input_window = 1038
    output_window = 12
    num_nodes = 1       # Use 1 node to fit in GPU; params are node-independent
    input_dim = 1
    output_dim = 1

    # ---- Build config (shared params, device varies) ---- #
    model_params = {
        'input_window': input_window,
        'output_window': output_window,
        'output_dim': output_dim,
        'e_layers': 10,
        'n_heads': 4,
        'd_model': 2180,
        'd_ff': 8720,
        'patch_len': 16,
        'stride': 1,
        'dropout': 0.3,
        'fc_dropout': 0.3,
        'head_dropout': 0.0,
        'revin': 1,
        'affine': 0,
        'subtract_last': 0,
        'decomposition': 0,
        'kernel_size': 25,
        'individual': 0,
        'max_seq_len': 1024,
        'padding_patch': 'end',
        'pe': 'zeros',
        'learn_pe': True,
        'attn_dropout': 0.0,
        'res_attention': True,
        'pre_norm': False,
        'norm': 'BatchNorm',
    }

    config = DictConfig({**model_params, 'device': device})
    config_cpu = DictConfig({**model_params, 'device': torch.device('cpu')})

    data_feature = {
        'num_nodes': num_nodes,
        'feature_dim': input_dim,
        'scaler': FakeScaler(),
    }

    # ---- Build model ---- #
    model = PatchTST(config, data_feature).to(device)
    model.eval()

    # ---- Synthetic batch ---- #
    X = torch.randn(batch_size, input_window, num_nodes, input_dim, device=device)
    y = torch.randn(batch_size, output_window, num_nodes, output_dim, device=device)
    batch = {'X': X, 'y': y}

    print(f'Input shape: {tuple(X.shape)}')
    print(f'patch_num (real seq_len): {model.model.backbone.patch_num}')

    # ---- 1. Trainable Params ---- #
    trainable_params = count_trainable_params(model)

    # ---- 2. FLOPs (on CPU to avoid GPU OOM) ---- #
    from thop import profile as thop_profile
    model_cpu = PatchTST(config_cpu, data_feature).eval()
    wrapped = ForwardWrapper(model_cpu)
    X_cpu = torch.randn(1, input_window, num_nodes, input_dim)
    with torch.no_grad():
        macs, _ = thop_profile(wrapped, inputs=(X_cpu,), verbose=False)
    flops = macs * 2
    del model_cpu, wrapped, X_cpu

    # ---- 3. Peak GPU Memory (train: fwd + bwd + optim step) ---- #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    # warmup
    optimizer.zero_grad()
    loss = model.calculate_loss(batch)
    loss.backward()
    optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # measured step
    optimizer.zero_grad()
    loss = model.calculate_loss(batch)
    loss.backward()
    optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        peak_train_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        peak_train_mb = None

    # ---- 4. Inference Time ---- #
    model.eval()
    with torch.no_grad():
        _ = model.predict(batch)  # warmup
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model.predict(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        infer_ms = (time.perf_counter() - t0) * 1000

    avg_sample_ms = infer_ms / batch_size

    # ---- 5. Print ---- #
    peak_str = f'{peak_train_mb:.2f} MB' if peak_train_mb is not None else 'CPU'
    print(f'\n===== PROFILE RESULTS =====')
    print(f'| Model | Trainable Params | Inference Throughput (FLOPs) | Peak GPU Memory (train) | Inference Time |')
    print(f'|-------|------------------|------------------------------|-------------------------|----------------|')
    print(f'| PatchTST '
          f'| {fmt(trainable_params)} ({trainable_params:,}) '
          f'| {fmt(flops)} ({fmt(macs)} MACs) '
          f'| {peak_str} '
          f'| {avg_sample_ms:.3f} ms/sample ({infer_ms:.3f} ms/batch) |')
    print(f'===========================')
    print(f'\nConfig: d_model={config["d_model"]}, d_ff={config["d_ff"]}, '
          f'e_layers={config["e_layers"]}, n_heads={config["n_heads"]}, '
          f'patch_len={config["patch_len"]}, stride={config["stride"]}, '
          f'input_window={input_window}, output_window={output_window}, '
          f'batch_size={batch_size}, num_nodes={num_nodes}')


if __name__ == '__main__':
    main()
