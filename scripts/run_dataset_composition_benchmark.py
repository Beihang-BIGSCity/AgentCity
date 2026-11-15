"""Benchmark the actual dataset-conditioned AgentCity compositions and ablations.

Run from any directory with the LibCity Python interpreter. Architecture choices
come exclusively from each completed composition artifact and its registered
model/config. Only the common training/data protocol and branch-disable switches
are supplied here; complete source baselines retain their native architecture.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_module_benchmark as common

ROOT, LIBCITY = common.ROOT, common.LIBCITY
TASKS = {"PEMS_BAY": "compose_pems_bay_20261027",
         "PEMSD4": "compose_pemsd4_20261027", "PEMSD7": "compose_pemsd7_20261027"}
COMPOSED_VARIANTS = ("full", "temporal_only", "spatial_only")
BASELINES = ("STGCN", "GWNET", "DCRNN", "AGCRN", "GRU", "LSTM")
FAMILIES = ("encoder", "spatial", "temporal", "fusion", "head")
GRAPH_FIELDS = ("geo_file", "rel_file", "weight_col", "bidir_adj_mx",
                "init_weight_inf_or_zero", "set_weight_link_or_dist",
                "calculate_weight_adj", "weight_adj_epsilon", "distance_inverse")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def artifact_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    return LIBCITY / path


def composition_artifacts(dataset):
    task_dir = ROOT / "data/adaptation" / TASKS[dataset]
    path = task_dir / "composition.json"
    composition = json.loads(path.read_text())
    if composition.get("dataset") != dataset:
        raise ValueError(f"{path}: composition dataset does not match {dataset}")
    name = composition["model_name"]
    if name == "STDiffGRU" or not name.startswith("AC"):
        raise ValueError(f"Expected a newly generated AC composition, received {name}")
    modules = composition.get("modules", {})
    if not all(isinstance(modules.get(f), dict) and modules[f].get("id") for f in FAMILIES):
        raise ValueError(f"{path}: all five selected module ids must be recorded")
    generated = composition.get("generated_files", composition.get("generated_artifacts", {}))
    model_file = artifact_path(generated.get("model_file", composition.get(
        "model_file", f"libcity/model/traffic_speed_prediction/{name}.py")))
    config_file = artifact_path(generated.get("model_config", composition.get(
        "model_config", f"libcity/config/model/{composition.get('task', 'traffic_state_pred')}/{name}.json")))
    for f in (model_file, config_file):
        if not f.is_file():
            raise FileNotFoundError(f"Composition artifact is missing: {f}")
    return composition, path, model_file, config_file


def protocol(args):
    value = common.protocol(args)
    value.update(version=2, experiment="actual_AgentCity_dataset_compositions",
                 seeds=list(args.seeds), variants=list(args.variants),
                 baselines=list(args.baselines),
                 architecture_policy="preserve generated module choices/depth/config; disable one branch only",
                 baseline_policy="complete registered source model, native architecture and training loss",
                 temporal_conv_backend=getattr(args, "temporal_conv_backend", "native"),
                 baseline_learning_rate=args.baseline_learning_rate or "native model config default",
                 features="actual dataset composition input_contract: target with optional time_in_day",
                 composed_optimizer={"name": "Adam", "learning_rate": 0.01, "milestones": [5], "gamma": 0.1, "clip_grad_norm": 5},
                 source_optimizer="native config optimizer/LR/scheduler/clipping; LR can be explicitly overridden",
                 training_loss="native model.calculate_loss; DCRNN receives cumulative batches_seen",
                 comparison="same raw split/windows/scaler/budget/validation criterion as module_benchmark_20261026; task-specific graph; source-model optimizer differences recorded separately")
    return value


def input_contract(config, composition):
    """Resolve the generated input width without changing encoder architecture."""
    target = common.TARGETS[composition["dataset"]]
    if config.get("data_col") != [target] or config.get("output_dim") != 1:
        raise ValueError("Only the task's single target channel/output is supported")
    external = bool(config.get("load_external", False))
    time_in_day = bool(config.get("add_time_in_day", False))
    day_in_week = bool(config.get("add_day_in_week", False))
    if external and (day_in_week or config.get("normal_external", False)):
        raise ValueError("Only target plus optional unscaled time_in_day is supported")
    ext_file = config.get("ext_file", composition["dataset"])
    if external and (LIBCITY / "raw_data" / composition["dataset"] / f"{ext_file}.ext").exists():
        raise ValueError("An additional external file exists; refusing to silently drop unknown external channels")
    channels = [target] + (["time_in_day"] if external and time_in_day else [])
    width = len(channels)
    contracts = [composition.get("data_contract", {}), composition.get("input", {})]
    for declared in contracts:
        if isinstance(declared.get("input_channels"), int) and declared["input_channels"] != width:
            raise ValueError("Composition input_channels differs from generated config")
        if "feature_dim" in declared and declared["feature_dim"] != width:
            raise ValueError("Composition feature_dim differs from generated config input contract")
        if "ext_dim" in declared and declared["ext_dim"] != width - 1:
            raise ValueError("Composition ext_dim differs from generated config input contract")
        for key in ("load_external", "add_time_in_day", "add_day_in_week"):
            if key in declared and bool(declared[key]) != bool(config.get(key, False)):
                raise ValueError(f"Composition {key} differs from generated config")
        for key in ("input_shape", "x_shape"):
            if key in declared and declared[key][-1] != width:
                raise ValueError("Composition input shape differs from generated config")
    encoder_width = composition["modules"]["encoder"].get("ctr_args", {}).get("input_dim")
    if isinstance(encoder_width, str) and encoder_width.strip().lower().startswith("auto"):
        encoder_width = None
    if encoder_width is not None and encoder_width != width:
        raise ValueError("Selected encoder input_dim differs from actual input contract")
    contract = {"feature_dim": width, "ext_dim": width - 1, "output_dim": 1,
                "channels": channels, "load_external": external,
                "add_time_in_day": time_in_day, "add_day_in_week": day_in_week,
                "normal_external": bool(config.get("normal_external", False))}
    contract["sha256"] = canonical_sha(contract)
    return contract


def apply_input_contract(windows, feature, profile, contract, output, dataset):
    width = contract["feature_dim"]
    if width not in (1, 2) or any(w.shape[-1] < width for w in windows):
        raise ValueError("Prepared data cannot satisfy the generated input contract")
    path = output / "input_contracts" / f"{dataset}.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError(f"Input contract changed; refusing mixed-input resume: {path}")
    common.write_json(path, contract)
    feature.update(feature_dim=width, ext_dim=contract["ext_dim"])
    profile["input_contract"] = contract
    common.write_json(output / f"{dataset}_profile.json", profile)
    return [window[..., :width] for window in windows]


def apply_task_graph(config, feature, profile, output, dataset):
    """Use the generated model's geo/rel contract without reloading raw .dyna."""
    from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
    point = TrafficStatePointDataset(config)  # Constructor reads only .geo/.rel.
    adj = np.ascontiguousarray(point.adj_mx, dtype=np.float32)
    expected = (feature["num_nodes"], feature["num_nodes"])
    if adj.shape != expected or not np.isfinite(adj).all() or not np.count_nonzero(adj):
        raise ValueError(f"Invalid task adjacency for {dataset}: shape={adj.shape}, expected={expected}")
    description = {"shape": list(adj.shape), "dtype": str(adj.dtype)}
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode() + adj.tobytes()).hexdigest()
    graph = {"source": "actual generated model ConfigParser -> TrafficStatePointDataset constructor",
             "requested_config": {key: config.get(key) for key in GRAPH_FIELDS},
             "resolved_config": {key: getattr(point, key) for key in GRAPH_FIELDS},
             "adj_sha256": digest, "nonzero": int(np.count_nonzero(adj)), **description}
    path = output / "graphs" / f"{dataset}.json"
    if path.exists() and json.loads(path.read_text()) != graph:
        raise ValueError(f"Task graph changed; refusing mixed-graph resume: {path}")
    common.write_json(path, graph)
    feature["adj_mx"] = adj
    profile.update(graph_nonzero=graph["nonzero"], task_graph=graph)
    common.write_json(output / f"{dataset}_profile.json", profile)
    print(json.dumps({"event": "task_graph_ready", "dataset": dataset,
                      "adj_sha256": digest, "graph_config": graph["resolved_config"]}), flush=True)
    return graph


def make_config(dataset, name, device, variant="full", generated_config=None, shared_input=None):
    from libcity.config import ConfigParser
    # These fields control the data contract and training budget, not structure.
    overrides = {"gpu": device.type == "cuda", "gpu_id": device.index or 0,
                 "data_col": [common.TARGETS[dataset]], "output_dim": 1,
                 "input_window": 12, "output_window": 12,
                 "cache_dataset": False, "scaler": "standard", "null_value": 0}
    if shared_input is not None:
        overrides.update({key: shared_input[key] for key in (
            "load_external", "add_time_in_day", "add_day_in_week", "normal_external")})
    if variant == "temporal_only":
        overrides["spatial_module"] = "none"
    elif variant == "spatial_only":
        overrides["temporal_module"] = "none"
    elif variant == "native_gru":
        # Match the explicitly documented native GRU in the previous benchmark.
        name = "GRU"  # ConfigParser derives rnn_type from this name.
        overrides.update(rnn_type="GRU", hidden_size=32, num_layers=1,
                         dropout=0.0, teacher_forcing_ratio=0)
    if generated_config is not None:
        generated = json.loads(Path(generated_config).read_text())
        if generated.get("data_col", [common.TARGETS[dataset]]) != [common.TARGETS[dataset]] or generated.get("output_dim", 1) != 1:
            raise ValueError("Generated target contract must be preserved; expected one task target")
        if generated.get("hidden_dim") != 32:
            raise ValueError("Generated hidden_dim must be 32; refusing to alter selected architecture")
        for key in ("input_window", "output_window"):
            if generated.get(key, 12) != 12:
                raise ValueError(f"Generated {key} must be 12")
        # Load exactly the config named by composition, even if not at the default path.
        # Vendored ConfigParser prepends './' even to absolute strings.
        config_file = os.path.relpath(Path(generated_config).with_suffix(""), Path.cwd())
    else:
        config_file = None
    return ConfigParser("traffic_state_pred", name, dataset,
                        config_file=config_file, other_args=overrides)


def instantiate(config, feature):
    from libcity.utils import get_model
    model = get_model(config, copy.deepcopy(feature)).to(config["device"])
    if config.get("benchmark_temporal_conv_backend", "native") == "linear":
        from equivalent_temporal_conv import apply_backend
        model = apply_backend(model)
    return model


def model_source(model):
    return Path(inspect.getsourcefile(type(model))).resolve()


def parameter_shapes(model):
    return {name: list(p.shape) for name, p in model.named_parameters()}


def check_prediction(model, batch):
    pred = model.predict(batch)
    expected = (len(batch["X"]), 12, batch["X"].shape[2], 1)
    if tuple(pred.shape) != expected or not bool(torch.isfinite(pred).all()):
        raise ValueError(f"Invalid prediction shape/values: {tuple(pred.shape)}, expected {expected}")
    return pred


class ForbiddenBranch(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("Disabled branch was evaluated: model ignores its ablation switch")


def family_modules(model, family):
    """Support the generated single-branch and depth-preserving ModuleList forms."""
    attribute = family
    for candidate in (family + "_layers_list", family + "_layers"):
        if isinstance(getattr(model, candidate, None), torch.nn.ModuleList):
            attribute = candidate
            break
    module = getattr(model, attribute, None)
    if isinstance(module, torch.nn.ModuleList):
        return attribute, list(module)
    return attribute, [module] if isinstance(module, torch.nn.Module) else []


@torch.no_grad()
def validate_ablation(full, ablated, variant, batch, modules):
    """Verify real information removal, whether unused parameters remain or not."""
    disabled = "spatial" if variant == "temporal_only" else "temporal"
    retained = "temporal" if disabled == "spatial" else "spatial"
    for family in FAMILIES:
        _, components = family_modules(full, family)
        if not components:
            raise ValueError(f"Full model must expose its selected {family} module")
        expected_class = modules[family].get("class")
        for component in components:
            if expected_class and type(component).__name__ != expected_class:
                raise ValueError(f"Selected {family} class differs from composition: {type(component).__name__} != {expected_class}")
        _, kept = family_modules(ablated, family)
        if family != disabled and [type(x) for x in kept] != [type(x) for x in components]:
            raise ValueError(f"Ablation changed the retained {family} module class/depth")
    disabled_attribute, full_disabled = family_modules(full, disabled)
    actual_attribute, actual_disabled = family_modules(ablated, disabled)
    if actual_attribute != disabled_attribute:
        raise ValueError("Ablation changed the branch container contract")
    full_shapes, actual_shapes = parameter_shapes(full), parameter_shapes(ablated)
    expected_removed = {k: v for k, v in full_shapes.items() if not k.startswith(disabled_attribute + ".")}
    if actual_shapes == full_shapes and actual_disabled:
        mode = "zero_information_keep_parameters"
        if [type(x) for x in actual_disabled] != [type(x) for x in full_disabled]:
            raise ValueError("Ablation changed disabled branch classes/depth")
    elif actual_shapes == expected_removed and not actual_disabled:
        mode = "zero_information_remove_parameters"
    else:
        raise ValueError("Ablation changed parameters outside the disabled branch")
    disabled_parameters = sum(p.numel() for module in full_disabled for p in module.parameters())
    total = sum(p.numel() for p in full.parameters())
    params = sum(p.numel() for p in ablated.parameters())
    expected_difference = 0 if mode == "zero_information_keep_parameters" else disabled_parameters
    if total - params != expected_difference:
        raise ValueError("Ablation allocated parameter count differs from declared mode")
    full.eval()
    ablated.eval()
    def observe(model):
        calls = {family: [0] * len(family_modules(model, family)[1]) for family in ("spatial", "temporal")}
        fusions = family_modules(model, "fusion")[1]
        inputs = [[] for _ in fusions]
        handles = []
        for family in calls:
            for index, component in enumerate(family_modules(model, family)[1]):
                def count_call(module, args, output, family=family, index=index):
                    calls[family][index] += 1
                handles.append(component.register_forward_hook(count_call))
        for index, fusion in enumerate(fusions):
            def capture_input(module, args, index=index):
                inputs[index].append(args)
            handles.append(fusion.register_forward_pre_hook(capture_input))
        try:
            prediction = check_prediction(model, batch).clone()
        finally:
            for handle in handles:
                handle.remove()
        return prediction, calls, inputs
    _, full_calls, _ = observe(full)
    if any(not counts or not all(counts) for counts in full_calls.values()):
        raise ValueError("Full model did not execute both selected branches at every layer")
    before, ablated_calls, fusion_inputs = observe(ablated)
    if any(ablated_calls[disabled]) or not all(ablated_calls[retained]):
        raise ValueError("Disabled branch was evaluated or retained branch was skipped")
    slot = 0 if disabled == "spatial" else 1
    for layer in fusion_inputs:
        if not layer or any(len(args) < 2 for args in layer):
            raise ValueError("Fusion must receive explicit spatial and temporal representations at every layer")
        if any(torch.count_nonzero(args[slot]).item() != 0 for args in layer):
            raise ValueError("Disabled branch representation is not all-zero at fusion")
    original = getattr(ablated, actual_attribute, None)
    trap = torch.nn.ModuleList([ForbiddenBranch() for _ in full_disabled]) if isinstance(original, torch.nn.ModuleList) else ForbiddenBranch()
    setattr(ablated, actual_attribute, trap)
    try:
        after = check_prediction(ablated, batch)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    finally:
        setattr(ablated, actual_attribute, original)
    # cuDNN recurrent layers retain backward state only in training mode.
    previous_mode = ablated.training
    try:
        ablated.train()
        with torch.enable_grad():
            ablated.zero_grad(set_to_none=True)
            objective = ablated.calculate_loss(batch) if hasattr(ablated, "calculate_loss") else ablated.predict(batch).square().mean()
            objective.backward()
    finally:
        ablated.train(previous_mode)
    if any(p.grad is not None and torch.count_nonzero(p.grad).item() for module in actual_disabled for p in module.parameters()):
        raise ValueError("Disabled branch received nonzero gradients")
    if not any(p.grad is not None and torch.count_nonzero(p.grad).item()
               for module in family_modules(ablated, retained)[1] for p in module.parameters()):
        raise ValueError("Retained branch received no nonzero gradients")
    ablated.zero_grad(set_to_none=True)
    if parameter_shapes(full) != full_shapes or parameter_shapes(ablated) != actual_shapes:
        raise ValueError("Forward dynamically created trainable parameters")
    return {"disabled_branch": disabled, "retained_branch": retained,
            "parameters": params, "allocated_parameters": params, "full_parameters": total,
            "active_parameters": total - disabled_parameters, "disabled_parameters": disabled_parameters,
            "removed_parameters": total - params, "ablation_mode": mode,
            "full_branch_calls": full_calls, "ablated_branch_calls": ablated_calls,
            "fusion_layers_checked": len(fusion_inputs), "zero_at_fusion": True,
            "disabled_gradients_zero": True, "retained_gradients_nonzero": True,
            "disabled_branch_trap_passed": True, "shape_passed": True}


def freeze_sources(out, dataset, composition, composition_path, source, config_path, model):
    """Immutable snapshots: resume fails if live architecture or source changes."""
    files = {"composition.json": composition_path, "model.py": source, "model_config.json": config_path,
             "benchmark.py": Path(__file__), "data_and_metrics.py": Path(common.__file__)}
    backend = getattr(model, "_benchmark_execution_backend", None)
    if backend:
        files["execution_backend.py"] = Path(backend["source"])
    task_dir = composition_path.parent
    for filename in ("plan.json", "dataset_profile.json", "data_integration.json",
                     "run_report.json", "tuning.json", "verdict.json"):
        if (task_dir / filename).is_file():
            files[filename] = task_dir / filename
    for family, selected in composition["modules"].items():
        library = selected.get("library_path")
        if library and (ROOT / "data/modlib" / library).is_file():
            files[f"library_{family}.py"] = ROOT / "data/modlib" / library
    # Imported module implementation files are part of the architecture too.
    for family in FAMILIES:
        for index, component in enumerate(family_modules(model, family)[1]):
            imported = Path(inspect.getsourcefile(type(component))).resolve()
            if imported != source:
                files[f"implementation_{family}_{index}.py"] = imported
    frozen = out / "frozen" / dataset
    hashes = {}
    for filename, origin in files.items():
        destination = frozen / filename
        data = origin.read_bytes()
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f"Frozen source changed; use a new output directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[filename] = {"origin": str(origin.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()}
    return hashes


def train_loss(model, batch, batches_seen):
    # DCRNN's native curriculum learning requires an integer global batch count.
    if type(model).__name__ == "DCRNN":
        return model.calculate_loss(batch, batches_seen=batches_seen)
    return model.calculate_loss(batch)


def check_native_interface(model, sample):
    """Validate the vendored implementation without altering its source/layers."""
    try:
        model.eval()
        with torch.no_grad():
            check_prediction(model, sample)
        model.train()
        value = train_loss(model, sample, 0)
        if not bool(torch.isfinite(value)):
            raise ValueError("Native training loss is non-finite")
        value.backward()
    except RuntimeError as exc:
        if type(model).__name__ == "GWNET" and "conv1d" in str(exc) and "Expected 2D" in str(exc):
            raise NotImplementedError(
                "The vendored complete GWNET passes a four-dimensional tensor to Conv1d; "
                "this baseline cannot run without an explicit upstream-source fix. "
                "No replacement architecture or silent source patch was applied.") from exc
        raise


def optimizer_for(model, config, variant, baseline_learning_rate=None):
    if not variant.startswith("source_"):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
        return optimizer, scheduler, {"name": "Adam", "learning_rate": 0.01, "milestones": [5],
                                      "gamma": 0.1, "clip_grad_norm": 5}
    rate = baseline_learning_rate or config["learning_rate"]
    learner = config.get("learner", "adam").lower()
    settings = {"name": learner, "learning_rate": rate, "weight_decay": config.get("weight_decay", 0),
                "eps": config.get("lr_epsilon", 1e-8), "lr_decay": config.get("lr_decay", False),
                "clip_grad_norm": config.get("max_grad_norm", 1) if config.get("clip_grad_norm", False) else None}
    kw = {"lr": rate, "eps": settings["eps"], "weight_decay": settings["weight_decay"]}
    if learner == "adam":
        settings["betas"] = [config.get("lr_beta1", 0.9), config.get("lr_beta2", 0.999)]
        optimizer = torch.optim.Adam(model.parameters(), betas=tuple(settings["betas"]), **kw)
    elif learner == "rmsprop":
        settings.update(alpha=config.get("lr_alpha", 0.99), momentum=config.get("lr_momentum", 0))
        optimizer = torch.optim.RMSprop(model.parameters(), alpha=settings["alpha"], momentum=settings["momentum"], **kw)
    else:
        raise NotImplementedError(f"Native baseline optimizer {learner!r} is not supported; refusing a substitute")
    scheduler = None
    if settings["lr_decay"]:
        settings.update(scheduler=config["lr_scheduler"].lower(), gamma=config.get("lr_decay_ratio", 0.1))
        if settings["scheduler"] == "multisteplr":
            settings["milestones"] = config["steps"]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings["milestones"], gamma=settings["gamma"])
        elif settings["scheduler"] == "steplr":
            settings["step_size"] = config["step_size"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings["step_size"], gamma=settings["gamma"])
        else:
            raise NotImplementedError(f"Native baseline scheduler {settings['scheduler']!r} is not supported")
    return optimizer, scheduler, settings


def run_one(args, dataset, config, feature, windows, variant, seed, evidence):
    name = f"{dataset}_{variant}_s{seed}"
    result_file = args.output / "runs" / f"{name}.json"
    signature = canonical_sha({"protocol": protocol(args), "evidence": evidence,
                               "resolved_config": config.config})
    if result_file.exists():
        old = json.loads(result_file.read_text())
        if old.get("run_sha256") != signature or old.get("status") != "success":
            raise ValueError(f"Existing result differs from live source/protocol: {result_file}")
        checkpoint = ROOT / old["checkpoint"]
        if not checkpoint.is_file() or sha256(checkpoint) != old.get("checkpoint_sha256"):
            raise ValueError(f"Completed checkpoint is missing or changed: {checkpoint}")
        print(f"SKIP {name}", flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config["device"].type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = instantiate(config, feature)
    parameters = parameter_shapes(model)
    count = sum(p.numel() for p in model.parameters())
    ablation = evidence.get("ablation_audit", {}).get(variant, {})
    if ablation and ablation["allocated_parameters"] != count:
        raise ValueError("Training model allocation differs from verified ablation")
    optimizer, scheduler, optimizer_settings = optimizer_for(model, config, variant, args.baseline_learning_rate)
    history, best_state, best_val, best_epoch, batches_seen = [], None, float("inf"), -1, 0
    start = time.monotonic()
    print(f"START {name} model={type(model).__name__} params={count}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        loss_sum, samples = 0.0, 0
        for batch in common.batches(windows[0], args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            value = train_loss(model, batch, batches_seen)
            if not bool(torch.isfinite(value)):
                raise FloatingPointError(f"Non-finite training loss: {name}")
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_settings["clip_grad_norm"] or float("inf"), error_if_nonfinite=True)
            optimizer.step()
            batches_seen += 1
            loss_sum += value.item() * len(batch["X"])
            samples += len(batch["X"])
        validation = common.score(model, windows[1], args.batch_size, feature["scaler"], config["device"])
        val = validation["pooled_masked_mae"]
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / samples,
                        "val_masked_mae": val, "validation": validation,
                        "lr": optimizer.param_groups[0]["lr"], "batches_seen": batches_seen})
        common.write_json(args.output / "history" / f"{name}.json", history)
        if val < best_val:
            best_val, best_epoch = val, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler is not None:
            scheduler.step()
        print(f"EPOCH {name} {epoch+1}/{args.epochs} train={loss_sum/samples:.5f} val={val:.5f} elapsed={time.monotonic()-start:.1f}s", flush=True)
    if parameter_shapes(model) != parameters:
        raise ValueError("Training dynamically changed model parameters")
    model.load_state_dict(best_state)
    test = common.score(model, windows[2], args.batch_size, feature["scaler"], config["device"])
    checkpoint = args.output / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    result = {"name": name, "dataset": dataset, "variant": variant, "seed": seed,
              "status": "success", "model_name": type(model).__name__, "run_sha256": signature,
              "protocol_sha256": canonical_sha(protocol(args)), "evidence": evidence,
              "resolved_config": {k: str(v) if isinstance(v, torch.device) else v for k, v in config.config.items()},
              "parameters": count, "parameter_shapes": parameters, "epochs": args.epochs,
              "allocated_parameters": count, "active_parameters": ablation.get("active_parameters", count),
              "disabled_parameters": ablation.get("disabled_parameters", 0),
              "removed_parameters": ablation.get("removed_parameters", 0),
              "ablation_mode": ablation.get("ablation_mode", "full_model"),
              "training_loss_type": "masked_MSE; native quick one-step training" if variant == "source_STGCN" else "masked_MAE; native calculate_loss",
              "optimizer": optimizer_settings,
              "best_epoch": best_epoch, "best_val_masked_mae": best_val,
              "wall_seconds": time.monotonic() - start, "history": history, "test": test,
              "gpu": torch.cuda.get_device_name(config["device"]) if config["device"].type == "cuda" else "CPU",
              "device_index": config["device"].index,
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint)}
    common.write_json(result_file, result)
    print(json.dumps({"event": "complete", "name": name, "mae": test["masked_mae"]}), flush=True)
    del optimizer, model, best_state
    if config["device"].type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--variants", nargs="*", choices=(*COMPOSED_VARIANTS, "native_gru"), default=list(COMPOSED_VARIANTS))
    parser.add_argument("--baselines", nargs="*", choices=BASELINES, default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-learning-rate", type=float, default=None,
                        help="Optional source-model LR override; otherwise use each native model config default")
    parser.add_argument("--temporal-conv-backend", choices=("native", "linear"), default="native",
                        help="Opt-in mathematically equivalent FP32 temporal-convolution lowering")
    parser.add_argument("--output", type=Path, default=ROOT / "result/dataset_composition_20261027")
    parser.add_argument("--self-check", action="store_true", help="CPU unit/metric checks only; no raw data/training")
    parser.add_argument("--check-only", action="store_true", help="Check actual generated models and selected baselines on small tensors")
    args = parser.parse_args()
    if args.self_check:
        common.self_check()
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_dataset_composition_benchmark.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if args.epochs < 1 or args.batch_size < 1 or not (args.variants or args.baselines):
        parser.error("Positive epochs/batch size and at least one variant or baseline required")
    os.chdir(LIBCITY)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cpu" if args.check_only else f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.output = args.output.resolve()
    manifest = {"protocol": protocol(args), "benchmark_sha256": sha256(__file__),
                "common_sha256": sha256(common.__file__), "torch_version": torch.__version__}
    for dataset in args.datasets:
        composition, cp, source, cfg_path = composition_artifacts(dataset)
        config = make_config(dataset, composition["model_name"], device, generated_config=cfg_path)
        config["benchmark_temporal_conv_backend"] = args.temporal_conv_backend
        task_input = input_contract(config, composition)
        if args.check_only:
            from libcity.utils import StandardScaler
            nodes = 7
            width = task_input["feature_dim"]
            feature = {"num_nodes": nodes, "feature_dim": width, "ext_dim": width - 1, "output_dim": 1,
                       "adj_mx": np.ones((nodes, nodes), dtype=np.float32), "scaler": StandardScaler(30, 10)}
            sample = {"X": torch.randn(2, 12, nodes, width), "y": torch.randn(2, 12, nodes, width)}
            windows = None
        else:
            _, feature, windows, profile = common.prepare(dataset, device, args.output)
            windows = apply_input_contract(windows, feature, profile, task_input, args.output, dataset)
            task_graph = apply_task_graph(config, feature, profile, args.output, dataset)
            feature["num_batches"] = (len(windows[0]) + args.batch_size - 1) // args.batch_size
            sample = next(common.batches(windows[0], 2))
        full = instantiate(config, feature)
        if model_source(full) != source.resolve():
            raise ValueError("Registered model source differs from composition.generated_files.model_file")
        frozen = freeze_sources(args.output, dataset, composition, cp, source, cfg_path, full)
        audit = {}
        configs = {"full": config}
        for variant in ("temporal_only", "spatial_only"):
            configs[variant] = make_config(dataset, composition["model_name"], device, variant, cfg_path)
            configs[variant]["benchmark_temporal_conv_backend"] = args.temporal_conv_backend
            ablated = instantiate(configs[variant], feature)
            audit[variant] = validate_ablation(full, ablated, variant, sample, composition["modules"])
            del ablated
        common.write_json(args.output / "audits" / f"{dataset}_ablation.json", audit)
        evidence = {"composition_sha256": sha256(cp), "source_sha256": sha256(source),
                    "config_sha256": sha256(cfg_path), "frozen_sources": frozen,
                    "input_contract": task_input,
                    "ablation_audit": audit,
                    "execution_backend": getattr(full, "_benchmark_execution_backend", {"name": "native"}),
                    "selected_modules": composition["modules"], "depth": composition.get("depth"),
                    "pipeline_task_id": TASKS[dataset]}
        if not args.check_only:
            evidence["task_graph"] = task_graph
        del full
        jobs = [(variant, configs[variant], evidence) for variant in args.variants if variant in COMPOSED_VARIANTS]
        for name in (["native_gru"] if "native_gru" in args.variants else []) + args.baselines:
            native = make_config(dataset, "RNN" if name == "native_gru" else name, device, name,
                                 shared_input=task_input)
            native["benchmark_temporal_conv_backend"] = args.temporal_conv_backend
            native_model = instantiate(native, feature)
            check_native_interface(native_model, sample)
            src = model_source(native_model)
            frozen_native = args.output / "frozen" / dataset / f"baseline_{name}.py"
            if frozen_native.exists() and sha256(frozen_native) != sha256(src):
                raise ValueError(f"Baseline source changed: {src}")
            frozen_native.write_bytes(src.read_bytes())
            default_config = LIBCITY / "libcity/config/model/traffic_state_pred" / f"{native['model']}.json"
            frozen_default = frozen_native.with_suffix(".json")
            if frozen_default.exists() and sha256(frozen_default) != sha256(default_config):
                raise ValueError(f"Baseline default config changed: {default_config}")
            frozen_default.write_bytes(default_config.read_bytes())
            native_evidence = {"baseline_source": str(src.relative_to(ROOT)), "source_sha256": sha256(src),
                               "execution_backend": getattr(native_model, "_benchmark_execution_backend", {"name": "native"}),
                               "input_contract": task_input,
                               "default_config_sha256": sha256(default_config),
                               "architecture": ("complete RNN.py GRU, hidden_size=32/num_layers=1, actual task input"
                                                if name == "native_gru" else "complete unmodified registered model with native defaults"),
                               "training_loss": "native calculate_loss; STGCN defaults quick one-step MSE" if name == "STGCN" else "native calculate_loss"}
            if not args.check_only:
                native_evidence["task_graph"] = task_graph
            jobs.append((name if name == "native_gru" else f"source_{name}", native, native_evidence))
            del native_model
        if args.check_only:
            print(json.dumps({"event": "check_passed", "dataset": dataset, "audit": audit,
                              "jobs": [name for name, _, _ in jobs]}), flush=True)
            continue
        common.write_json(args.output / f"manifest_{dataset}_gpu{args.gpu}.json", {**manifest, "evidence": evidence})
        for seed in args.seeds:
            for variant, run_config, run_evidence in jobs:
                run_one(args, dataset, run_config, feature, windows, variant, seed, run_evidence)
        del windows, feature
        torch.cuda.empty_cache()
    print("WORKER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
