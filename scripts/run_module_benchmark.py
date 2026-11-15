"""Run repeatable cross-dataset module experiments using existing LibCity models.

Raw timestamps are split before windows are constructed. All variants share the
same data, optimizer, budget and validation-only checkpoint selection. Keeping
the time series on the GPU avoids materializing overlapping windows on disk.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIBCITY = ROOT / "Bigscity-LibCity"
sys.path.insert(0, str(LIBCITY))
# Compatibility required by the vendored LibCity loader with recent NumPy.
for name, value in (("float", float), ("int", int), ("bool", bool)):
    if name not in np.__dict__:
        setattr(np, name, value)

VARIANTS = ("concat", "add", "weighted", "temporal_only", "spatial_only", "native_gru")
TARGETS = {"METR_LA": "traffic_speed", "PEMS_BAY": "traffic_speed",
           "PEMSD4": "traffic_flow", "PEMSD7": "traffic_flow"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def load_class(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def protocol(args):
    return {"version": 1, "epochs": args.epochs, "batch_size": args.batch_size,
            "hidden_dim": 32, "learning_rate": 0.01, "lr_milestones": [5],
            "lr_gamma": 0.1, "optimizer": "Adam", "clip_grad_norm": 5,
            "input_window": 12, "output_window": 12,
            "split": [0.6, 0.2, 0.2], "split_before_windowing": True,
            "scaler": "standard; fit on training raw timestamps only, including zeros",
            "features": "one target channel + time_in_day", "null_value": 0,
            "checkpoint_selection": "minimum validation masked_MAE over all elements",
            "primary_metric": "arithmetic mean of 12 per-horizon masked_MAE",
            "seeds": [0, 1, 2], "variants": list(VARIANTS),
            "precision": "float32; TF32 disabled", "shuffle_train": True,
            "pad_last_batch": False, "deterministic_cudnn": True,
            "historical_results_comparable": False}


def prepare(dataset_name, device, out):
    from libcity.config import ConfigParser
    from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
    from libcity.utils import StandardScaler

    config = ConfigParser("traffic_state_pred", "STDiffGRU", dataset_name, other_args={
        "gpu": True, "gpu_id": device.index, "data_col": [TARGETS[dataset_name]],
        "output_dim": 1, "cache_dataset": False, "add_time_in_day": True,
        "add_day_in_week": False, "load_external": True})
    dataset = TrafficStatePointDataset(config)
    raw = dataset._load_dyna(dataset_name)
    assert raw.shape[-1] == 1 and np.isfinite(raw).all()
    data = dataset._add_external_information(raw).astype(np.float32)
    assert data.shape[-1] == 2
    total = len(data)
    cuts = [0, int(total * 0.6), int(total * 0.8), total]
    mean = float(data[:cuts[1], :, 0].mean(dtype=np.float64))
    std = float(data[:cuts[1], :, 0].std(dtype=np.float64))
    assert std > 0
    data[..., 0] = (data[..., 0] - mean) / std
    series = torch.as_tensor(data, device=device)
    # unfold returns [samples, nodes, features, window]; keep a strided view.
    windows = [series[a:b].unfold(0, 24, 1).permute(0, 3, 1, 2)
               for a, b in zip(cuts[:-1], cuts[1:])]
    assert all(len(w) > 0 for w in windows)
    assert cuts[1] - 1 < cuts[1] and cuts[2] - 1 < cuts[2]
    feature = {"scaler": StandardScaler(mean, std), "adj_mx": dataset.adj_mx,
               "num_nodes": dataset.num_nodes, "feature_dim": 2, "output_dim": 1}
    info = {"dataset": dataset_name, "target": TARGETS[dataset_name],
            "nodes": dataset.num_nodes, "timesteps": total, "interval_minutes": 5,
            "raw_split_boundaries": cuts, "window_counts": [len(w) for w in windows],
            "train_mean": mean, "train_std": std,
            "null_ratio": float((raw == 0).mean()),
            "graph_nonzero": int(np.count_nonzero(dataset.adj_mx)),
            "timestamp_ranges": [[str(dataset.timesolts[a]), str(dataset.timesolts[b - 1])]
                                 for a, b in zip(cuts[:-1], cuts[1:])]}
    assert np.count_nonzero(dataset.adj_mx) > 0
    write_json(out / f"{dataset_name}_profile.json", info)
    print(json.dumps({"event": "data_ready", **info}), flush=True)
    return config, feature, windows, info


def make_model(config, feature, variant):
    from libcity.model.traffic_speed_prediction.STDiffGRU import STDiffGRU
    from libcity.model.traffic_speed_prediction.RNN import RNN
    cfg = dict(config.config)
    cfg.update(hidden_dim=32, hidden_size=32, num_layers=1, dropout=0.0,
               num_diffusion_steps=2, input_window=12, output_window=12,
               spatial_module="none" if variant == "temporal_only" else "diffusion_gcn",
               temporal_module="none" if variant == "spatial_only" else "gru",
               rnn_type="GRU", teacher_forcing_ratio=0, null_value=0)
    model = RNN(cfg, feature) if variant == "native_gru" else STDiffGRU(cfg, feature)
    if variant in ("add", "weighted"):
        name = "AddFusion" if variant == "add" else "WeightedFusion"
        model.fusion = load_class(f"data/modlib/code/fusion/{variant}.py", name)(hidden_dim=32)
    return model.to(config["device"])


def batches(windows, batch_size, shuffle=False):
    order = torch.randperm(len(windows), device=windows.device) if shuffle else torch.arange(
        len(windows), device=windows.device)
    for indices in order.split(batch_size):
        sample = windows[indices]
        yield {"X": sample[:, :12].contiguous(), "y": sample[:, 12:].contiguous()}


@torch.no_grad()
def score(model, windows, batch_size, scaler, device):
    model.eval()
    sums = torch.zeros(4, 12, device=device, dtype=torch.float64)
    for batch in batches(windows, batch_size):
        pred = scaler.inverse_transform(model.predict(batch)[..., :1])
        true = scaler.inverse_transform(batch["y"][..., :1])
        true = torch.where(true.abs() < 1e-4, 0, true)
        mask = true.ne(0)
        error = (pred - true).double()
        dims = (0, 2, 3)
        sums[0] += (error.abs() * mask).sum(dims)
        sums[1] += (error.square() * mask).sum(dims)
        sums[2] += (error.abs() / true.double().abs().clamp_min(1e-12) * mask).sum(dims)
        sums[3] += mask.sum(dims)
    assert bool((sums[3] > 0).all())
    mae, rmse, mape = sums[0] / sums[3], (sums[1] / sums[3]).sqrt(), sums[2] / sums[3]
    return {"masked_mae": mae.mean().item(), "masked_rmse": rmse.mean().item(),
            "masked_mape": mape.mean().item(), "pooled_masked_mae": (sums[0].sum() / sums[3].sum()).item(),
            "mae_by_horizon": mae.tolist(), "rmse_by_horizon": rmse.tolist(),
            "mape_by_horizon": mape.tolist(), "valid_counts_by_horizon": sums[3].tolist()}


def run_one(args, dataset_name, config, feature, windows, variant, seed):
    name = f"{dataset_name}_{variant}_s{seed}"
    result_file = args.output / "runs" / f"{name}.json"
    signature = hashlib.sha256(json.dumps(protocol(args), sort_keys=True).encode()).hexdigest()
    if result_file.exists():
        old = json.loads(result_file.read_text())
        if old.get("protocol_sha256") != signature:
            raise ValueError(f"protocol differs from existing result {result_file}")
        print(f"SKIP {name}", flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(config, feature, variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    train_windows, valid_windows, test_windows = windows
    history, best_state, best_val, best_epoch = [], None, float("inf"), -1
    start = time.monotonic()
    print(f"START {name} params={sum(p.numel() for p in model.parameters())}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in batches(train_windows, args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            train_loss = model.calculate_loss(batch)
            if not bool(torch.isfinite(train_loss)):
                raise FloatingPointError(f"non-finite loss in {name}")
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += train_loss.item() * len(batch["X"])
            count += len(batch["X"])
        validation = score(model, valid_windows, args.batch_size, feature["scaler"], config["device"])
        val = validation["pooled_masked_mae"]
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count,
                        "val_masked_mae": val, "lr": optimizer.param_groups[0]["lr"]})
        if val < best_val:
            best_val, best_epoch = val, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        print(f"EPOCH {name} {epoch + 1}/{args.epochs} train={loss_sum/count:.5f} val={val:.5f} elapsed={time.monotonic()-start:.1f}s", flush=True)
    model.load_state_dict(best_state)
    test = score(model, test_windows, args.batch_size, feature["scaler"], config["device"])
    elapsed = time.monotonic() - start
    checkpoint = args.output / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    result = {"name": name, "dataset": dataset_name, "variant": variant, "seed": seed,
              "status": "success", "protocol_sha256": signature,
              "parameters": sum(p.numel() for p in model.parameters()),
              "epochs": args.epochs, "best_epoch": best_epoch, "best_val_masked_mae": best_val,
              "wall_seconds": elapsed, "gpu": torch.cuda.get_device_name(config["device"]),
              "device_index": config["device"].index, "history": history, "test": test,
              "checkpoint": str(checkpoint.relative_to(ROOT))}
    if variant == "weighted":
        result["fusion_weights"] = model.fusion.branch_weight.softmax(0).detach().cpu().tolist()
    write_json(result_file, result)
    print(json.dumps({"event": "complete", "name": name, "mae": test["masked_mae"], "seconds": elapsed}), flush=True)
    del optimizer, model, best_state
    torch.cuda.empty_cache()


def self_check():
    from libcity.model import loss
    from libcity.utils import StandardScaler
    torch.manual_seed(7)
    truth = torch.rand(9, 12, 4, 1) * 10
    truth[::2, :, 0] = 0
    pred = truth + torch.randn_like(truth)
    class Fixed(torch.nn.Module):
        def predict(self, batch):
            return batch["X"]
    # Here X holds predictions and y holds truth, so exercise score's actual reducer.
    windows = torch.cat((pred, truth), dim=1)
    actual = score(Fixed(), windows, 4, StandardScaler(0, 1), torch.device("cpu"))
    expected = [loss.masked_mae_torch(pred[:, h], truth[:, h].clone(), 0).item() for h in range(12)]
    np.testing.assert_allclose(actual["mae_by_horizon"], expected, rtol=2e-6)
    expected_rmse = [loss.masked_rmse_torch(pred[:, h], truth[:, h].clone(), 0).item() for h in range(12)]
    np.testing.assert_allclose(actual["rmse_by_horizon"], expected_rmse, rtol=2e-6)
    series = torch.arange(120).view(60, 2, 1)
    w = series[:36].unfold(0, 24, 1).permute(0, 3, 1, 2)
    assert len(w) == 13 and torch.equal(w[0, :12], series[:12])
    assert torch.equal(w[-1, 12:], series[24:36])
    print("PASS: streaming masked metrics match LibCity; window alignment and split containment")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=TARGETS, default=list(TARGETS))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output", type=Path, default=ROOT / "result/module_benchmark_20261026")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    os.chdir(LIBCITY)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = protocol(args)
    manifest["source_sha256"] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__), LIBCITY / "libcity/model/traffic_speed_prediction/STDiffGRU.py",
                  LIBCITY / "libcity/model/traffic_speed_prediction/RNN.py",
                  ROOT / "data/modlib/code/fusion/add.py", ROOT / "data/modlib/code/fusion/weighted.py"]}
    manifest["torch_version"] = torch.__version__
    # Each GPU worker has its own manifest; the protocol itself is identical.
    write_json(args.output / f"manifest_gpu{args.gpu}.json", manifest)
    for dataset_name in args.datasets:
        config, feature, windows, info = prepare(dataset_name, device, args.output)
        for seed in (0, 1, 2):
            for variant in VARIANTS:
                run_one(args, dataset_name, config, feature, windows, variant, seed)
        del windows, feature, config
        torch.cuda.empty_cache()
    print("WORKER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
