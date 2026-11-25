import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/persist_mrl_external_v1")
CODE = Path("/home/proxy-function-gap-persist-mrl")
BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_protocol():
    protocol_path = CODE / "frozen_protocol.json"
    expected = (ROOT / "protocol" / "frozen_protocol.sha256").read_text().strip()
    actual = sha256(protocol_path)
    if actual != expected:
        raise RuntimeError(f"Frozen protocol hash mismatch: {actual} != {expected}")
    return json.loads(protocol_path.read_text())


def verify_inputs(protocol):
    files = {"source_xlsx": protocol["source"]["path"]}
    files.update({f"sample_{role}": path for role, path in protocol["sample_exclusion"]["paths"].items()})
    files.update({f"checkpoint_{m['seed']}": m["path"] for m in protocol["model"]["checkpoints"]})
    result = {}
    for name, path in files.items():
        actual = sha256(path)
        if name == "source_xlsx":
            expected = protocol["source"]["sha256"]
        elif name.startswith("sample_"):
            expected = protocol["sample_exclusion"]["sha256"][name[7:]]
        else:
            expected = next(m["sha256"] for m in protocol["model"]["checkpoints"] if f"checkpoint_{m['seed']}" == name)
        if actual != expected:
            raise RuntimeError(f"Input hash mismatch for {name}")
        result[name] = {"path": path, "sha256": actual, "bytes": os.path.getsize(path)}
    return result


def read_sample_sequences(protocol):
    result = {}
    for role, path in protocol["sample_exclusion"]["paths"].items():
        with gzip.open(path, "rt", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            sequences = {row[protocol["sample_exclusion"]["sequence_field"]] for row in rows}
        if not sequences or any(len(x) != 50 or set(x) - set(BASES) for x in sequences):
            raise ValueError(f"Invalid Sample {role} reference")
        result[role] = sequences
    return result


def conflict_types(sequence, sample):
    conflicts = []
    for role, references in sample.items():
        if sequence in references:
            conflicts.append({"role": role, "type": "exact"})
        elif any(len(sequence) == len(ref) and sum(a != b for a, b in zip(sequence, ref)) == 1 for ref in references):
            conflicts.append({"role": role, "type": "hamming1"})
    return conflicts


def prepare(protocol):
    source = protocol["source"]
    frame = pd.read_excel(source["path"], sheet_name=source["sheet"], engine="openpyxl")
    if len(frame) != 233:
        raise ValueError("Expected 233 source constructs")
    rule = protocol["construct_inclusion"]
    selected = frame.loc[frame[rule["group_field"]] == rule["group_value"]].copy()
    if len(selected) != rule["expected_rows_before_exclusion"]:
        raise ValueError("Frozen construct-group count mismatch")
    if selected[source["construct_field"]].nunique() != len(selected):
        raise ValueError("Sequence ID is not unique")
    selected[source["sequence_field"]] = selected[source["sequence_field"]].astype(str).str.upper()
    valid = selected[source["sequence_field"]].map(lambda x: len(x) >= 4 and not set(x) - set(BASES))
    valid &= np.isfinite(pd.to_numeric(selected[source["endpoint_field"]], errors="coerce"))
    if not valid.all():
        raise ValueError("Invalid sequence or endpoint in frozen group")
    if selected[source["sequence_field"]].nunique() != rule["expected_unique_utrs_before_exclusion"]:
        raise ValueError("Frozen unique-UTR count mismatch")
    sample = read_sample_sequences(protocol)
    conflicts = {}
    for sequence in sorted(selected[source["sequence_field"]].unique()):
        found = conflict_types(sequence, sample)
        if found:
            conflicts[sequence] = found
    selected["included"] = ~selected[source["sequence_field"]].isin(conflicts)
    eligible = selected.loc[selected["included"]].copy()
    counts = {"constructs": len(eligible), "utr_clusters": eligible[source["sequence_field"]].nunique()}
    stopped = counts["constructs"] < rule["minimum_eligible_constructs"] or counts["utr_clusters"] < rule["minimum_eligible_utr_clusters"]
    audit = {
        "source_rows": len(frame),
        "frozen_group_rows": len(selected),
        "frozen_group_unique_utrs": selected[source["sequence_field"]].nunique(),
        "excluded_utr_clusters": len(conflicts),
        "excluded_constructs": int((~selected["included"]).sum()),
        "remaining_constructs": counts["constructs"],
        "remaining_utr_clusters": counts["utr_clusters"],
        "remaining_cds": int(eligible["Sequence CDS"].nunique()),
        "remaining_3utr": int(eligible["Sequence 3 UTR"].nunique()),
        "remaining_length_min": int(eligible[source["sequence_field"]].str.len().min()),
        "remaining_length_max": int(eligible[source["sequence_field"]].str.len().max()),
        "exact_50nt_constructs": int((eligible[source["sequence_field"]].str.len() == 50).sum()),
        "conflicts": conflicts,
        "minimum_sample_stop": stopped,
        "labels_used_for_prediction_label_metric": False
    }
    dump_json(audit, ROOT / "protocol" / "inclusion_audit.json")
    excluded = selected.loc[~selected["included"], [source["construct_field"], source["sequence_field"]]]
    excluded.to_csv(ROOT / "protocol" / "excluded_constructs.tsv", sep="\t", index=False)
    columns = [source["construct_field"], source["sequence_field"]]
    eligible[columns].rename(columns={source["construct_field"]: "sequence_id", source["sequence_field"]: "utr"}).to_csv(ROOT / "protocol" / "eligible_sequences.tsv", sep="\t", index=False)
    eligible[[source["construct_field"], source["sequence_field"], source["endpoint_field"]]].rename(columns={source["construct_field"]: "sequence_id", source["sequence_field"]: "utr", source["endpoint_field"]: "mrl"}).to_csv(ROOT / "protocol" / "evaluator_labels.tsv", sep="\t", index=False)
    return audit, stopped


def kmer_features(sequence):
    values = np.zeros(259, dtype=np.float32)
    for index in range(len(sequence) - 3):
        code = 0
        for base in sequence[index:index + 4]:
            code = code * 4 + BASES[base]
        values[code] += 1
    values[:256] /= len(sequence) - 3
    values[256] = (sequence.count("G") + sequence.count("C")) / len(sequence)
    values[257] = float("ATG" in sequence)
    values[258] = max((len(part) for part in sequence.replace("A", " ").replace("C", " ").replace("G", " ").split()), default=0) / 50
    return values


def model_hash(model):
    digest = hashlib.sha256()
    digest.update(model.__class__.__name__.encode())
    state = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu()
        state.update(name.encode())
        state.update(str(value.dtype).encode())
        state.update(str(tuple(value.shape)).encode())
        state.update(value.numpy().tobytes())
    digest.update(state.hexdigest().encode())
    return digest.hexdigest()


def cuda_predict(protocol):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    import torch
    from torch import nn
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(protocol["cuda"]["seed"])
    torch.cuda.manual_seed_all(protocol["cuda"]["seed"])
    device = torch.device("cuda:0")
    rows = pd.read_csv(ROOT / "protocol" / "eligible_sequences.tsv", sep="\t")
    if set(rows.columns) != {"sequence_id", "utr"}:
        raise RuntimeError("Prediction input must not contain labels")
    features = np.stack([kmer_features(x) for x in rows.utr])
    member_predictions = []
    reports = []
    for spec in protocol["model"]["checkpoints"]:
        checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=False)
        if checkpoint["seed"] != spec["seed"]:
            raise ValueError("Checkpoint seed mismatch")
        class KmerRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(259, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x):
                return self.layers(x).squeeze(1)
        wrapped = KmerRegressor()
        wrapped.load_state_dict(checkpoint["model_state"])
        wrapped = wrapped.to(device=device, dtype=torch.float32).eval()
        if model_hash(wrapped) != spec["model_hash"]:
            raise ValueError("Frozen model hash mismatch")
        if any(not p.is_cuda for p in wrapped.parameters()):
            raise RuntimeError("CPU model fallback")
        x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
        if x.device != device:
            raise RuntimeError("CPU input fallback")
        with torch.inference_mode():
            normalized = wrapped(x)
            if normalized.device != device:
                raise RuntimeError("CPU output fallback")
            output = normalized * checkpoint["scaler"]["scale"] + checkpoint["scaler"]["mean"]
            if output.device != device:
                raise RuntimeError("CPU scaler fallback")
            member_predictions.append(output.detach().cpu().numpy())
        torch.cuda.synchronize(device)
        reports.append({"seed": spec["seed"], "model_hash": spec["model_hash"], "input_device": str(x.device), "output_device": str(output.device), "parameter_devices": sorted({str(p.device) for p in wrapped.parameters()}), "item_forwards": len(rows)})
    values = np.stack(member_predictions)
    output = rows.copy()
    output["prediction"] = values.mean(axis=0)
    output["member_std"] = values.std(axis=0, ddof=1)
    for index, spec in enumerate(protocol["model"]["checkpoints"]):
        output[f"member_{spec['seed']}"] = values[index]
    output.to_csv(ROOT / "predictions" / "predictions.tsv", sep="\t", index=False, float_format="%.17g")
    gpu = {
        "requested_device": "cuda:0",
        "actual_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "dtype": "torch.float32",
        "deterministic_algorithms": True,
        "tf32": False,
        "cpu_fallback": False,
        "labels_read": False,
        "members": reports,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
    }
    dump_json(gpu, ROOT / "predictions" / "cuda_inference.json")
    return gpu


def order_for_seed(n, seed):
    return np.random.default_rng(seed).permutation(n)


def choose(order, scores, ids, cap=None):
    candidates = order if cap is None else order[:cap]
    return sorted(candidates, key=lambda i: (-scores[i], ids[i], int(i)))[0]


def run_proposals(labels, scores, ids, seeds, seed_offset=None):
    n = len(labels)
    records = []
    for position, seed in enumerate(seeds):
        actual_seed = seed if seed_offset is None else [seed_offset[0], seed_offset[1], seed]
        order = order_for_seed(n, actual_seed)
        oracle = choose(order, labels, ids)
        random_index = int(order[0])
        bon_index = choose(order, scores, ids, 16)
        oracle_value = labels[oracle]
        random_regret = oracle_value - labels[random_index]
        bon_regret = oracle_value - labels[bon_index]
        records.append((random_regret, bon_regret, bon_regret - random_regret, float(labels[random_index] == oracle_value), float(labels[bon_index] == oracle_value)))
    return np.asarray(records)


def metric_values(labels, scores):
    return np.asarray([stats.spearmanr(scores, labels).statistic, stats.pearsonr(scores, labels).statistic, np.mean(np.abs(scores - labels))])


def summarize(protocol):
    predictions = pd.read_csv(ROOT / "predictions" / "predictions.tsv", sep="\t")
    labels_frame = pd.read_csv(ROOT / "protocol" / "evaluator_labels.tsv", sep="\t")
    merged = labels_frame.merge(predictions, on=["sequence_id", "utr"], validate="one_to_one")
    if len(merged) != len(labels_frame) or not np.isfinite(merged[["mrl", "prediction"]].to_numpy()).all():
        raise ValueError("Prediction-label join failed")
    labels = merged.mrl.to_numpy(dtype=float)
    scores = merged.prediction.to_numpy(dtype=float)
    ids = merged.sequence_id.astype(str).tolist()
    utrs = merged.utr.tolist()
    seeds = list(range(20261125, 20261325))
    base = run_proposals(labels, scores, ids, seeds)
    per_seed = pd.DataFrame({"algorithm_seed": seeds, "random_regret": base[:, 0], "bon_regret": base[:, 1], "delta": base[:, 2], "random_oracle_coverage": base[:, 3], "bon_oracle_coverage": base[:, 4]})
    per_seed.to_csv(ROOT / "results" / "per_seed.tsv", sep="\t", index=False, float_format="%.17g")
    clusters = sorted(set(utrs))
    cluster_rows = [np.asarray([i for i, value in enumerate(utrs) if value == cluster], dtype=int) for cluster in clusters]
    rng = np.random.default_rng(20261126)
    bootstrap = np.empty((10000, 8), dtype=float)
    for b in range(10000):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([cluster_rows[i] for i in sampled])
        b_labels = labels[indices]
        b_scores = scores[indices]
        b_ids = [f"{ids[i]}#{rank}" for rank, i in enumerate(indices)]
        proposals = run_proposals(b_labels, b_scores, b_ids, seeds, (20261127, b))
        bootstrap[b, 0] = proposals[:, 2].mean()
        bootstrap[b, 1:4] = metric_values(b_labels, b_scores)
        bootstrap[b, 4:] = proposals[:, [0, 1, 3, 4]].mean(axis=0)
    np.save(ROOT / "results" / "bootstrap_replicates.npy", bootstrap)
    def interval(column):
        return [float(x) for x in np.percentile(bootstrap[:, column], [2.5, 97.5])]
    delta = bootstrap[:, 0]
    p_value = min(1.0, 2 * min(((delta >= 0).sum() + 1) / 10001, ((delta <= 0).sum() + 1) / 10001))
    ci = interval(0)
    base_metrics = metric_values(labels, scores)
    summary = {
        "analysis_id": protocol["analysis_id"],
        "evidence_boundary": protocol["evidence_boundary"],
        "constructs": len(merged),
        "utr_clusters": len(clusters),
        "algorithm_seeds": len(seeds),
        "bootstrap_replicates": len(bootstrap),
        "primary": {
            "bon_minus_random_regret": float(base[:, 2].mean()),
            "ci95": ci,
            "p_two_sided_sign": float(p_value),
            "significant_benefit": bool(ci[1] < 0 and p_value < 0.05),
            "direction": "BoN favored" if base[:, 2].mean() < 0 else "Random favored" if base[:, 2].mean() > 0 else "Tie"
        },
        "secondary": {
            "spearman": {"value": float(base_metrics[0]), "ci95": interval(1)},
            "pearson": {"value": float(base_metrics[1]), "ci95": interval(2)},
            "mae": {"value": float(base_metrics[2]), "ci95": interval(3)},
            "random_top1_regret": {"value": float(base[:, 0].mean()), "ci95": interval(4)},
            "bon_top1_regret": {"value": float(base[:, 1].mean()), "ci95": interval(5)},
            "random_oracle_coverage": {"value": float(base[:, 3].mean()), "ci95": interval(6)},
            "bon_oracle_coverage": {"value": float(base[:, 4].mean()), "ci95": interval(7)}
        },
        "statistical_compute": "CPU only; CUDA was used exclusively for frozen model inference",
        "outer_tuning": False,
        "negative_results_retained": True
    }
    dump_json(summary, ROOT / "results" / "summary.json")
    return summary


def write_manifest():
    records = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            records[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    dump_json(records, ROOT / "artifact_manifest.json")
    return records
