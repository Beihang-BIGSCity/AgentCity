"""Audit saved experiment evidence without training or re-running predictions.

Each dataset is checked against its own executed frozen trainer. The raw-data
files are rehashed once, and every completed checkpoint is loaded on CPU with
weights_only=True. A partial matrix always produces a failing verification.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("PEMS_BAY", "PEMSD4", "PEMSD7")
TARGETS = {"PEMS_BAY": "traffic_speed", "PEMSD4": "traffic_flow", "PEMSD7": "traffic_flow"}
FEATURES = ("cv_global", "cv_node", "spatial_heterogeneity", "rho1", "sampling_interval_min", "node_count")
FAMILIES = ("encoder", "spatial", "temporal", "fusion", "head")
CORE = ("full", "temporal_only", "spatial_only")
MANIFEST_CORE = ("protocol", "benchmark_sha256", "common_sha256", "torch_version", "evidence")
WORKER_FIELDS = ("worker_scope_seeds", "worker_device", "worker_script_sha256",
                 "worker_executes_original_frozen_trainer")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def expected_matrix(plan):
    require(plan.get("matrix_finalized") is True, "experiment matrix is not finalized")
    variants = plan["expected_variants"]
    require(set(variants) == set(DATASETS), "unexpected datasets in finalized matrix")
    seeds = plan["evaluation"]["seeds"]
    require(seeds == [0, 1, 2], "expected three seeds 0/1/2")
    for dataset, names in variants.items():
        require(len(names) == len(set(names)), f"duplicate variants: {dataset}")
        require(set(CORE) <= set(names), f"missing core ablations: {dataset}")
    result = {(d, v, s) for d, names in variants.items() for v in names for s in seeds}
    require(len(result) == plan["expected_new_training_runs"], "declared run count differs from matrix")
    return result


def worker_manifest_context(directory, dataset):
    """Permit scheduling-only worker manifests over one identical science context."""
    paths = sorted(Path(directory).glob(f"manifest_{dataset}_gpu*.json"))
    require(paths, f"missing executed trainer manifest: {dataset}")
    shared, workers = None, {}
    for path in paths:
        manifest = read_json(path)
        require(set(MANIFEST_CORE) <= set(manifest), f"missing scientific manifest fields: {path.name}")
        require(not (set(manifest) - set(MANIFEST_CORE) - set(WORKER_FIELDS)),
                f"unrecognized manifest fields could alter execution: {path.name}")
        core = {key: manifest[key] for key in MANIFEST_CORE}
        if shared is None:
            shared = core
        for key in MANIFEST_CORE:
            require(canonical_sha(core[key]) == canonical_sha(shared[key]),
                    f"scientific worker manifest mismatch: {key} in {path.name}")
        device = int(path.stem.rsplit("_gpu", 1)[1])
        require(device not in workers, f"duplicate worker device manifest: {device}")
        require(manifest.get("worker_device", device) == device, f"worker device disagrees with manifest filename: {path.name}")
        scope = manifest.get("worker_scope_seeds", core["protocol"]["seeds"])
        require(isinstance(scope, list) and scope and len(scope) == len(set(scope))
                and set(scope) <= set(core["protocol"]["seeds"]), f"invalid worker seed scope: {path.name}")
        if "worker_executes_original_frozen_trainer" in manifest:
            require(manifest["worker_executes_original_frozen_trainer"] is True,
                    f"worker does not declare the original frozen trainer: {path.name}")
        if "worker_script_sha256" in manifest:
            candidates = list((Path(directory) / "source").glob("*.py"))
            require(any(sha256(candidate) == manifest["worker_script_sha256"] for candidate in candidates),
                    f"worker dispatcher source SHA is missing or changed: {path.name}")
        workers[device] = {"path": path.name, "seeds": scope,
                           "scheduling": {k: manifest[k] for k in WORKER_FIELDS if k in manifest}}
    return shared, workers


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, label):
    require(finite_number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
            f"{label}: saved value {actual!r} differs from {expected!r}")


def verify_metrics(metrics):
    for series, headline in (("mae_by_horizon", "masked_mae"),
                             ("rmse_by_horizon", "masked_rmse"),
                             ("mape_by_horizon", "masked_mape")):
        values = metrics[series]
        require(len(values) == 12 and all(finite_number(v) and v >= 0 for v in values),
                f"{series}: expected 12 finite nonnegative horizon values")
        close(metrics[headline], sum(values) / 12, headline)
    counts = metrics["valid_counts_by_horizon"]
    require(len(counts) == 12 and all(finite_number(v) and v > 0 and v == int(v) for v in counts),
            "invalid masked target counts")
    pooled = sum(m * n for m, n in zip(metrics["mae_by_horizon"], counts)) / sum(counts)
    close(metrics["pooled_masked_mae"], pooled, "pooled_masked_mae")
    return counts


def verify_history(record):
    require(record["epochs"] == 15, "run does not have the required 15-epoch budget")
    history = record["history"]
    require([h["epoch"] for h in history] == list(range(1, 16)), "incomplete/duplicate epoch history")
    for entry in history:
        require(all(finite_number(entry[k]) for k in ("train_loss", "val_masked_mae", "lr")),
                "non-finite train/validation history")
        verify_metrics(entry["validation"])
        close(entry["val_masked_mae"], entry["validation"]["pooled_masked_mae"], "validation selector")
    best = min(history, key=lambda h: h["val_masked_mae"])
    require(record["best_epoch"] == best["epoch"], "checkpoint was not selected by minimum validation loss")
    close(record["best_val_masked_mae"], best["val_masked_mae"], "best validation value")


def verify_checkpoint(record, checkpoint):
    import torch
    require(sha256(checkpoint) == record["checkpoint_sha256"], "checkpoint SHA mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and state, "checkpoint is not a nonempty state_dict")
    shapes = record["parameter_shapes"]
    require(isinstance(shapes, dict) and shapes, "missing named parameter shapes")
    for name, shape in shapes.items():
        require(name in state and list(state[name].shape) == shape, f"parameter shape mismatch: {name}")
        require(state[name].dtype == torch.float32, f"parameter precision changed: {name}")
    for name, value in state.items():
        require(isinstance(value, torch.Tensor), f"non-tensor checkpoint entry: {name}")
        require(bool(torch.isfinite(value).all()), f"non-finite checkpoint tensor: {name}")
    count = sum(math.prod(shape) for shape in shapes.values())
    require(count == record["parameters"] == record["allocated_parameters"], "allocated parameter count mismatch")
    active, disabled = record["active_parameters"], record["disabled_parameters"]
    require(0 < active <= count and 0 <= disabled <= count and active + disabled == count,
            "active/disabled parameter accounting mismatch")
    require(record["removed_parameters"] == 0, "study requires information ablation without removing parameters")
    if record["variant"] in ("temporal_only", "spatial_only"):
        require(disabled > 0 and record["ablation_mode"] == "zero_information_keep_parameters",
                "ablation is not the declared keep-parameters information ablation")
    else:
        require(active == count and disabled == 0 and record["ablation_mode"] == "full_model",
                "full/baseline parameter accounting is inconsistent")
    return state


def module_ids(composition):
    return {family: composition["modules"][family]["id"] for family in FAMILIES}


def method_hashes(source, class_name):
    result = {}
    for cls in ast.parse(source).body:
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name in ("__init__", "forward"):
                    if method.body and isinstance(method.body[0], ast.Expr):
                        expression = method.body[0].value
                        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                            method.body = method.body[1:]
                    result[method.name] = hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()
    return result


def pipeline_log_evidence(log_path, composition, frozen_source):
    """Extract original tool writes, rather than relying on revised timestamps."""
    choices, first_run, first_source, first_profile = [], None, None, None
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if "Executing tool: " not in line:
            continue
        event = line.split("Executing tool: ", 1)[1]
        if " with args: " not in event:
            continue
        name, payload = event.split(" with args: ", 1)
        timestamp = line[:19]
        if "run_adaptation" in name and first_run is None:
            first_run = timestamp
        try:
            arguments = json.loads(payload)
        except (ValueError, TypeError):
            continue
        path = arguments.get("file_path", arguments.get("path", ""))
        content = arguments.get("content", "")
        if name != "write_file":
            continue
        if path.endswith("/composition.json"):
            try:
                choices.append({"timestamp": timestamp, "modules": module_ids(json.loads(content))})
            except (KeyError, ValueError):
                continue
        if path.endswith(f"/{composition['model_name']}.py") and first_source is None:
            first_source = {"timestamp": timestamp,
                            "methods": method_hashes(content, composition["model_name"])}
        if path.endswith("/dataset_profile.json") and first_profile is None:
            try:
                candidate = json.loads(content)
                if all(k in candidate for k in FEATURES):
                    first_profile = {k: candidate[k] for k in FEATURES}
            except ValueError:
                continue
    require(choices and first_run, "missing original composition write or run_adaptation event in pipeline log")
    require(choices[0]["timestamp"] < first_run, "composition was not recorded before pilot execution")
    require(all(x["modules"] == module_ids(composition) for x in choices), "module ids changed during the observed pipeline")
    result = {"first_composition": choices[0], "first_run_adaptation": first_run,
              "selected_modules_unchanged": True, "initial_profile_features": first_profile,
              "first_model_write": first_source, "log_sha256": sha256(log_path)}
    if first_source:
        require(first_source["timestamp"] < first_run, "initial generated source follows pilot execution")
        result["initial_vs_frozen_core_method_ast_equal"] = (
            first_source["methods"] == method_hashes(frozen_source.read_text(), composition["model_name"]))
    return result


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def reconstruct_graph(dataset, frozen, graph):
    """Read only .geo/.rel through the actual vendored graph loader; no .dyna."""
    import numpy as np
    for name, value in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    sys.path.insert(0, str(ROOT / "Bigscity-LibCity"))
    from libcity.config import ConfigParser
    from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset
    composition = read_json(frozen / "composition.json")
    with working_directory(ROOT / "Bigscity-LibCity"):
        config_file = os.path.relpath((frozen / "model_config").resolve(), Path.cwd())
        config = ConfigParser("traffic_state_pred", composition["model_name"], dataset,
                              config_file=config_file, other_args={"gpu": False, "gpu_id": 0})
        dataset_object = TrafficStatePointDataset(config)
    adjacency = np.ascontiguousarray(dataset_object.adj_mx, dtype=np.float32)
    description = {"shape": list(adjacency.shape), "dtype": str(adjacency.dtype)}
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode() + adjacency.tobytes()).hexdigest()
    require(digest == graph["adj_sha256"], "reconstructed task graph SHA differs from recorded graph")
    require(np.isfinite(adjacency).all() and int(np.count_nonzero(adjacency)) == graph["nonzero"],
            "graph finiteness/nonzero count mismatch")
    for key, value in graph["resolved_config"].items():
        require(getattr(dataset_object, key) == value, f"graph preprocessing mismatch: {key}")
    return adjacency


class ExperimentVerifier:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.errors, self.warnings, self.verified_runs = [], [], []
        self.pipeline_audits, self.dataset_state = {}, {}
        self.raw_fingerprints_verified = 0

    def attempt(self, label, callback):
        try:
            return callback()
        except Exception as exc:
            self.errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return None

    def verify_pipeline(self, dataset):
        frozen = self.directory / "frozen" / dataset
        task = ROOT / "data/adaptation" / f"compose_{dataset.lower()}_20261027"
        composition = read_json(frozen / "composition.json")
        live = read_json(task / "composition.json")
        require(composition["dataset"] == dataset and composition["model_name"].startswith("AC")
                and composition["model_name"] != "STDiffGRU", "not a dataset-generated AC composition")
        require(module_ids(composition) == module_ids(live), "live/frozen selected module ids differ")
        profile = read_json(frozen / "dataset_profile.json")
        retrieval_file = frozen / "retrieval_before_planning.json"
        if not retrieval_file.exists():
            retrieval_file = task / "retrieval_before_planning.json"
        retrieval = read_json(retrieval_file)
        query = retrieval["query"]
        for key in FEATURES:
            require(finite_number(profile[key]) and profile[key] == query[key], f"profile/retrieval feature changed: {key}")
            require(read_json(task / "dataset_profile.json")[key] == query[key], f"live profile changed: {key}")
        require(query["profile_partition"] == "training" and query["profile_train_rate"] == 0.6,
                "retrieval query was not restricted to training 0.6")
        require(profile["profile_partition"] == "training" and profile["profile_train_rate"] == 0.6
                and profile["training_timesteps"] == query["training_timesteps"],
                "saved profile training partition differs from retrieval input")
        require(query["training_timesteps"] == int(query["timesteps"] * 0.6), "training-only profile extent mismatch")
        require(query["data_col"] == TARGETS[dataset], "profile target differs from task target")
        contract = read_json(self.directory / "input_contracts" / f"{dataset}.json")
        require(contract["channels"] == [TARGETS[dataset]] and contract["feature_dim"] == 1
                and contract["output_dim"] == 1, "expected final generated single-channel contract")
        require(canonical_sha({k: v for k, v in contract.items() if k != "sha256"}) == contract["sha256"],
                "input contract SHA mismatch")
        data = read_json(self.directory / f"{dataset}_profile.json")
        total = data["timesteps"]
        cuts = [0, int(total * 0.6), int(total * 0.8), total]
        require(data["raw_split_boundaries"] == cuts, "benchmark does not split raw timestamps at 60/20/20")
        require(data["window_counts"] == [b - a - 23 for a, b in zip(cuts[:-1], cuts[1:])],
                "benchmark windows cross splits or have the wrong 12+12 span")
        require(finite_number(data["train_mean"]) and finite_number(data["train_std"]) and data["train_std"] > 0,
                "invalid training-only scaler")
        graph = read_json(self.directory / "graphs" / f"{dataset}.json")
        adjacency = reconstruct_graph(dataset, frozen, graph)
        audit = pipeline_log_evidence(self.directory / "pipeline_logs" / f"{dataset}.log", composition, frozen / "model.py")
        if audit["initial_profile_features"]:
            require(all(audit["initial_profile_features"][k] == query[k] for k in FEATURES),
                    "first profile written in pipeline differs from pre-planning retrieval")
        if audit.get("initial_vs_frozen_core_method_ast_equal") is False:
            self.warnings.append(f"{dataset}: generated methods were revised for pilot plumbing; module ids were unchanged")
        audit.update(dataset=dataset, status="pass", profile_query_features_unchanged=True,
                     retrieval_sha256=sha256(retrieval_file), composition_sha256=sha256(frozen / "composition.json"),
                     model_sha256=sha256(frozen / "model.py"), input_contract=contract, task_graph=graph,
                     scope="pipeline chronology/profile and saved-source audit; pilot metrics are not raw-time benchmark metrics")
        name = composition["model_name"]
        registered = ROOT / "Bigscity-LibCity/libcity/model/traffic_speed_prediction" / f"{name}.py"
        require(sha256(registered) == sha256(frozen / "model.py"), "registered generated source differs from frozen model")
        generated_config = ROOT / "Bigscity-LibCity/libcity/config/model/traffic_state_pred" / f"{name}.json"
        require(sha256(generated_config) == sha256(frozen / "model_config.json"), "registered generated config differs from frozen config")
        package = registered.parent / "__init__.py"
        require(any(isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[-1] == name
                    and any(a.name == name for a in n.names)
                    for n in ast.parse(package.read_text()).body), "generated model is not exported by its package")
        task_config = read_json(ROOT / "Bigscity-LibCity/libcity/config/task_config.json")["traffic_state_pred"]
        require(name in task_config["allowed_model"] and task_config[name]["dataset_class"] == "TrafficStatePointDataset",
                "generated model is not registered for point data")
        for filename in ("plan.json", "run_report.json", "tuning.json", "verdict.json", "data_integration.json", "composition.json"):
            if sha256(task / filename) != sha256(frozen / filename):
                self.warnings.append(f"{dataset}/{filename}: live artifact differs from executed frozen snapshot; snapshot retained")
        manifest, workers = worker_manifest_context(self.directory, dataset)
        require(sha256(frozen / "benchmark.py") == manifest["benchmark_sha256"], "executed trainer snapshot hash mismatch")
        require(sha256(frozen / "data_and_metrics.py") == manifest["common_sha256"], "executed data/metric snapshot hash mismatch")
        protocol = manifest["protocol"]
        require(protocol["split"] == [0.6, 0.2, 0.2] and protocol["split_before_windowing"] is True
                and protocol["epochs"] == 15 and protocol["batch_size"] == 128
                and protocol["null_value"] == 0, "formal benchmark protocol mismatch")
        audit["worker_manifests"] = workers
        audit["shared_manifest_scientific_sha256"] = canonical_sha(manifest)
        self.pipeline_audits[dataset] = audit
        write_json(self.directory / "pipeline_evidence_audits" / f"{dataset}.json", audit)
        self.dataset_state[dataset] = {"composition": composition, "contract": contract, "graph": graph,
                                       "adjacency": adjacency, "profile": data, "protocol": protocol,
                                       "workers": workers, "counts": None, "full_shapes": {}}

    def verify_sources(self, dataset, record):
        evidence = record["evidence"]
        frozen = self.directory / "frozen" / dataset
        if record["variant"] in CORE:
            for filename, info in evidence["frozen_sources"].items():
                require(sha256(frozen / filename) == info["sha256"], f"executed snapshot hash mismatch: {filename}")
            for key, filename in (("source_sha256", "model.py"), ("config_sha256", "model_config.json"),
                                  ("composition_sha256", "composition.json")):
                require(evidence[key] == sha256(frozen / filename), f"{key} does not match this dataset's snapshot")
            require(module_ids({"modules": evidence["selected_modules"]}) == module_ids(self.dataset_state[dataset]["composition"]),
                    "run module choices differ from generated composition")
            require(record["model_name"] == self.dataset_state[dataset]["composition"]["model_name"], "wrong generated model class")
        else:
            suffix = record["variant"].removeprefix("source_")
            require(sha256(frozen / f"baseline_{suffix}.py") == evidence["source_sha256"], "baseline source snapshot mismatch")
            require(sha256(ROOT / evidence["baseline_source"]) == evidence["source_sha256"], "registered baseline source changed")
            require(sha256(frozen / f"baseline_{suffix}.json") == evidence["default_config_sha256"], "baseline config snapshot mismatch")
        backend = evidence.get("execution_backend", {"name": "native"})
        if backend["name"] != "native":
            require(backend["name"] == "linear_temporal_conv_fp32", "unrecognized execution backend")
            require(sha256(frozen / "execution_backend.py") == backend["source_sha256"], "execution backend snapshot hash mismatch")
            require(backend["precision"] == "unchanged" and backend["parameter_and_state_dict_layout"] == "unchanged",
                    "backend altered precision or parameter layout")
        return backend

    def verify_record(self, path, record):
        dataset = record["dataset"]
        state = self.dataset_state[dataset]
        require(record["status"] == "success", "run did not finish successfully")
        device = record["device_index"]
        require(device in state["workers"], "run device has no corresponding executed worker manifest")
        require(record["seed"] in state["workers"][device]["seeds"], "run seed lies outside this worker's declared scope")
        require(record["protocol_sha256"] == canonical_sha(state["protocol"]), "run protocol SHA differs from executed manifest")
        require(record["run_sha256"] == canonical_sha({"protocol": state["protocol"], "evidence": record["evidence"],
                                                        "resolved_config": record["resolved_config"]}),
                "run signature does not match its protocol/evidence/config")
        expected_name = f"{dataset}_{record['variant']}_s{record['seed']}"
        require(record["name"] == path.stem == expected_name, "run identity/filename mismatch")
        config = record["resolved_config"]
        require(config["dataset"] == dataset and config["data_col"] == [TARGETS[dataset]] and config["output_dim"] == 1,
                "run target differs from dataset target")
        require(config.get("null_value", 0) == 0 and config["input_window"] == config["output_window"] == 12,
                "run window or null-mask contract changed")
        for key in ("load_external", "add_time_in_day", "add_day_in_week", "normal_external"):
            require(bool(config.get(key, False)) == state["contract"][key], f"resolved input flag changed: {key}")
        require(record["evidence"]["input_contract"] == state["contract"], "run input contract differs from dataset")
        require(record["evidence"]["task_graph"] == state["graph"], "run graph differs from dataset")
        verify_history(record)
        counts = verify_metrics(record["test"])
        require(all(n <= state["profile"]["window_counts"][2] * state["profile"]["nodes"] for n in counts),
                "masked target counts exceed test samples")
        if state["counts"] is None:
            state["counts"] = counts
        require(counts == state["counts"], "test target mask/count differs across runs of the same dataset")
        backend = self.verify_sources(dataset, record)
        checkpoint = ROOT / record["checkpoint"]
        tensors = verify_checkpoint(record, checkpoint)
        import numpy as np
        for name in ("adj_mx", "adj"):
            if name in tensors and tuple(tensors[name].shape) == tuple(state["adjacency"].shape):
                require(np.array_equal(tensors[name].numpy(), state["adjacency"]), "checkpoint task adjacency differs")
        for module_path in backend.get("module_paths", []):
            shape = tensors[module_path + ".weight"].shape
            require(len(shape) == 4 and shape[2] == 1 and shape[3] <= 3, "lowered convolution weight shape changed")
        if record["variant"] in CORE:
            require("encoder.proj.weight" in tensors and list(tensors["encoder.proj.weight"].shape) == [32, 1],
                    "checkpoint encoder does not preserve the generated hidden32/C1 contract")
            require(record["parameters"] == sum(math.prod(s) for s in record["parameter_shapes"].values()), "core parameter sum differs")
            expected = state["full_shapes"].setdefault("shapes", record["parameter_shapes"])
            require(record["parameter_shapes"] == expected, "information ablation changed allocated parameter shapes")
            if record["variant"] != "full":
                saved_audit = read_json(self.directory / "audits" / f"{dataset}_ablation.json")[record["variant"]]
                for flag in ("zero_at_fusion", "disabled_gradients_zero", "retained_gradients_nonzero", "disabled_branch_trap_passed", "shape_passed"):
                    require(saved_audit[flag] is True, f"ablation audit failed: {flag}")
                for key in ("allocated_parameters", "active_parameters", "disabled_parameters", "removed_parameters", "ablation_mode"):
                    require(saved_audit[key] == record[key], f"ablation parameter audit mismatch: {key}")
        self.verified_runs.append({"name": record["name"], "checkpoint_sha256": record["checkpoint_sha256"],
                                   "parameters": record["parameters"], "best_epoch": record["best_epoch"],
                                   "masked_mae": record["test"]["masked_mae"], "execution_backend": backend["name"],
                                   "device_index": device, "worker_manifest": state["workers"][device]["path"]})

    def verify_raw_data(self):
        provenance = read_json(self.directory / "provenance.json")
        files = provenance["raw_data_fingerprints"]
        expected = {f"Bigscity-LibCity/raw_data/{d}/{d}.{suffix}" for d in DATASETS for suffix in ("dyna", "geo", "rel")}
        require(set(files) == expected, "raw-data provenance must contain all nine atomic data files")
        for filename, info in files.items():
            path = ROOT / filename
            require(path.stat().st_size == info["bytes"], f"raw file size changed: {filename}")
            require(sha256(path) == info["sha256"], f"raw file SHA changed: {filename}")
            self.raw_fingerprints_verified += 1

    def run(self):
        plan = read_json(self.directory / "experiment_plan.json")
        expected = expected_matrix(plan)
        records = {}
        for path in sorted((self.directory / "runs").glob("*.json")):
            record = self.attempt(path.name, lambda p=path: read_json(p))
            if record is None:
                continue
            key = self.attempt(path.name + "/identity", lambda r=record: (r["dataset"], r["variant"], r["seed"]))
            if key is None:
                continue
            if key in records:
                self.errors.append(f"duplicate matrix identity: {key}")
            records[key] = (path, record)
        missing, extra = sorted(expected - set(records)), sorted(set(records) - expected)
        if missing:
            self.errors.append(f"missing {len(missing)} expected runs")
        if extra:
            self.errors.append(f"unexpected runs: {extra}")
        for dataset in DATASETS:
            self.attempt(f"pipeline/{dataset}", lambda d=dataset: self.verify_pipeline(d))
        for key in sorted(set(records) & expected):
            path, record = records[key]
            if key[0] not in self.dataset_state:
                self.errors.append(f"{record['name']}: dataset evidence unavailable")
                continue
            self.attempt(record["name"], lambda p=path, r=record: self.verify_record(p, r))
        self.attempt("raw_data_fingerprints", self.verify_raw_data)
        passed = not self.errors and len(self.verified_runs) == len(expected)
        return {"passed": passed, "status": "pass" if passed else "incomplete_or_failed",
                "verified_utc": datetime.now(timezone.utc).isoformat(), "verifier_sha256": sha256(__file__),
                "experiment_plan_sha256": sha256(self.directory / "experiment_plan.json"),
                "provenance_sha256": sha256(self.directory / "provenance.json") if (self.directory / "provenance.json").exists() else None,
                "directory": str(self.directory), "expected_count": len(expected), "found_count": len(records),
                "verified_checkpoints": len(self.verified_runs), "raw_data_sha_verified": self.raw_fingerprints_verified,
                "missing": [list(x) for x in missing], "extra": [list(x) for x in extra],
                "errors": self.errors, "warnings": self.warnings, "verified_runs": self.verified_runs,
                "pipeline_evidence_audits": {d: f"pipeline_evidence_audits/{d}.json" for d in self.pipeline_audits},
                "scope": "saved metric arithmetic, validation selection, checkpoint tensors/SHA, raw data SHA, dataset contracts and executed source provenance; no prediction re-inference",
                "version_policy": "each dataset's executed frozen trainer/backend is authoritative; historical trainer hashes may differ from current live source",
                "pilot_policy": "native LibCity pilot sample-window metrics are not pooled with disjoint-raw-time formal benchmark metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "result/dataset_composition_20261027")
    args = parser.parse_args()
    directory = args.directory.resolve()
    verifier = ExperimentVerifier(directory)
    try:
        report = verifier.run()
    except Exception as exc:
        report = {"passed": False, "status": "failed", "errors": [f"{type(exc).__name__}: {exc}"],
                  "missing": [], "extra": [], "verified_checkpoints": len(verifier.verified_runs)}
    write_json(directory / "verification.json", report)
    lines = [f"STATUS: {report['status']}",
             f"Expected: {report.get('expected_count', 'unknown')}; found: {report.get('found_count', 'unknown')}; verified checkpoints: {report['verified_checkpoints']}",
             f"Raw data SHA verified: {report.get('raw_data_sha_verified', 0)}/9"]
    lines += ["MISSING: " + " / ".join(map(str, item)) for item in report.get("missing", [])]
    lines += ["EXTRA: " + " / ".join(map(str, item)) for item in report.get("extra", [])]
    lines += ["ERROR: " + item for item in report["errors"]]
    lines += ["WARNING: " + item for item in report.get("warnings", [])]
    lines.append(report.get("scope", "Verification did not complete."))
    text = "\n".join(lines) + "\n"
    (directory / "verification_final.log").write_text(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
