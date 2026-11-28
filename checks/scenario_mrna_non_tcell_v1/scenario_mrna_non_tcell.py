#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASES = {"A": 0, "C": 1, "G": 2, "T": 3}
N50_PREFIX = "ACATCGTAGAGAGTCGTACTTA"
CAPS = [1, 2, 4, 8, 16]
PRIMARY_ENDPOINT = "primary"


def dump_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path):
    return json.loads(Path(path).read_text()), Path(path)


def verify_raw(protocol):
    records = {}
    for name in ("editing", "stability"):
        spec = protocol["raw"][name]
        path = Path(spec["path"])
        actual = sha256(path)
        if actual != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError(f"Raw input mismatch: {name}")
        records[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    for role, path in protocol["sample_exclusion"]["paths"].items():
        actual = sha256(path)
        if actual != protocol["sample_exclusion"]["sha256"][role]:
            raise ValueError(f"Sample reference mismatch: {role}")
        records[f"sample_{role}"] = {"path": path, "sha256": actual}
    for member in protocol["model"]["checkpoints"]:
        actual = sha256(member["path"])
        if actual != member["sha256"]:
            raise ValueError(f"Checkpoint mismatch: {member['seed']}")
        records[f"checkpoint_{member['seed']}"] = {"path": member["path"], "sha256": actual}
    return records


def read_panel(protocol):
    spec = protocol["raw"]["editing"]
    frame = pd.read_excel(spec["path"], sheet_name=spec["sheet_panel"], engine="openpyxl")
    if len(frame) != 48 or frame["Sequence ID"].nunique() != 48:
        raise ValueError("Expected 48 unique panel UTRs")
    frame = frame.rename(columns={"Sequence ID": "source_row_id", "5'UTR": "utr", "Measured MRL": "measured_mrl", "Predicted MRL": "published_predicted_mrl", "Description": "description"})
    frame["source_row_id"] = frame["source_row_id"].astype(int)
    frame["sequence_id"] = frame["source_row_id"].map(lambda x: f"seq_{x:02d}")
    frame["construct_id"] = frame["sequence_id"]
    frame["cluster_id"] = frame["sequence_id"]
    frame["utr"] = frame["utr"].astype(str).str.upper()
    if any(set(x) - set(BASES) or len(x) < 4 for x in frame["utr"]):
        raise ValueError("Invalid native UTR")
    if frame["measured_mrl"].notna().sum() != 12 or frame["published_predicted_mrl"].notna().sum() != 45:
        raise ValueError("Measured/predicted MRL coverage mismatch")
    return frame.sort_values("source_row_id").reset_index(drop=True)


def read_sample_references(protocol):
    result = {}
    for role, path in protocol["sample_exclusion"]["paths"].items():
        with gzip.open(path, "rt", newline="") as handle:
            result[role] = {row["utr50"] for row in csv.DictReader(handle, delimiter="\t")}
    return result


def conflict_audit(panel, references):
    conflicts = {}
    for row in panel.itertuples(index=False):
        versions = {"native_full_utr": row.utr}
        if row.description.startswith("N50") and len(row.utr) > 50:
            versions["n50_variable_region"] = row.utr[-50:]
        found = []
        for version, sequence in versions.items():
            for role, refs in references.items():
                if sequence in refs:
                    found.append({"version": version, "role": role, "type": "exact"})
                if len(sequence) == 50:
                    for ref in refs:
                        if sum(a != b for a, b in zip(sequence, ref)) == 1:
                            found.append({"version": version, "role": role, "type": "hamming1"})
                            break
        if found:
            conflicts[row.sequence_id] = found
    return conflicts


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


def predict_cpu(panel, protocol, root):
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    features = np.stack([kmer_features(x) for x in panel["utr"]])
    member_values = []
    reports = []
    for spec in protocol["model"]["checkpoints"]:
        checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=False)
        class KmerRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(259, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x):
                return self.layers(x).squeeze(1)
        model = KmerRegressor()
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device=device, dtype=torch.float32).eval()
        if model_hash(model) != spec["model_hash"]:
            raise ValueError("Frozen model hash mismatch")
        x = torch.from_numpy(features).to(device=device)
        with torch.inference_mode():
            normalized = model(x)
            output = normalized * checkpoint["scaler"]["scale"] + checkpoint["scaler"]["mean"]
        if output.device.type != "cpu":
            raise ValueError("Unexpected non-CPU inference")
        member_values.append(output.detach().numpy())
        reports.append({"seed": spec["seed"], "model_hash": model_hash(model), "input_device": str(x.device), "output_device": str(output.device), "item_forwards": len(panel)})
    values = np.stack(member_values)
    output = panel[["sequence_id", "utr"]].copy()
    output["kmer_mlp"] = values.mean(axis=0)
    output["member_std"] = values.std(axis=0, ddof=1)
    for index, spec in enumerate(protocol["model"]["checkpoints"]):
        output[f"member_{spec['seed']}"] = values[index]
    output.to_csv(root / "predictions" / "predictions.tsv", sep="\t", index=False, float_format="%.17g")
    report = {
        "requested_device": "cpu",
        "actual_device": "cpu",
        "cpu_fallback": "not_applicable_cpu_protocol",
        "cuda_used": False,
        "cnn_inference": False,
        "training": False,
        "torch_version": torch.__version__,
        "torch_threads": 1,
        "dtype": "torch.float32",
        "labels_read": False,
        "members": reports,
    }
    dump_json(report, root / "predictions" / "execution_device.json")
    return output


def normalized_indel(protocol, include_flagged=False):
    frames = []
    for cell in protocol["raw"]["editing"]["cells"]:
        raw = pd.read_excel(protocol["raw"]["editing"]["path"], sheet_name=cell, engine="openpyxl")
        raw["source_row_id"] = raw["Sequence ID"].astype(int)
        raw["sequence_id"] = raw["source_row_id"].map(lambda x: f"seq_{x:02d}")
        raw["cell"] = cell
        raw["included"] = True if include_flagged else raw["Flagged"].isna()
        retained = raw.loc[raw["included"]].copy()
        reference = retained.loc[retained["source_row_id"] == 48].groupby(["megaTAL", "mRNA pmol"])["Indel %"].mean()
        retained["normalized_indel"] = retained.apply(lambda r: r["Indel %"] / reference.loc[(r["megaTAL"], r["mRNA pmol"])], axis=1)
        raw["normalized_indel"] = np.nan
        raw.loc[retained.index, "normalized_indel"] = retained["normalized_indel"]
        raw["endpoint"] = "kozak_normalized_indel_all_rows" if include_flagged else "kozak_normalized_indel_author"
        frames.append(raw)
    tidy = pd.concat(frames, ignore_index=True)
    grouped = tidy.loc[tidy["included"]].groupby(["cell", "megaTAL", "sequence_id"], sort=True)["normalized_indel"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"mean": "measured_value", "std": "sample_sd", "count": "n_dose_replicate_ratios"})
    if len(grouped) != 192:
        raise ValueError("Expected 192 indel summaries")
    return tidy, grouped


def validate_author_summary(protocol, grouped):
    author = pd.read_csv(protocol["raw"]["author_summary"]["path"])
    if len(author) != 192:
        raise ValueError("Author summary row mismatch")
    merged = author.merge(grouped, left_on=["Dataset", "megaTAL", "Sequence ID"], right_on=["cell", "megaTAL", "sequence_id"], validate="one_to_one") if False else None
    check = grouped.copy()
    check["Sequence ID"] = check["sequence_id"].str[4:].astype(int)
    merged = author.merge(check, left_on=["Dataset", "megaTAL", "Sequence ID"], right_on=["cell", "megaTAL", "Sequence ID"], validate="one_to_one")
    mean_error = float(np.max(np.abs(merged["Mean normalized indel %"] - merged["measured_value"])))
    sd_error = float(np.max(np.abs(merged["Std normalized indel %"] - merged["sample_sd"])))
    if mean_error > 1e-12 or sd_error > 1e-12:
        raise ValueError(f"Author normalization mismatch: {mean_error}, {sd_error}")
    return {"rows": 192, "maximum_absolute_mean_error": mean_error, "maximum_absolute_sd_error": sd_error}


def absolute_2pmol(protocol):
    frames = []
    for cell in protocol["raw"]["editing"]["cells"]:
        raw = pd.read_excel(protocol["raw"]["editing"]["path"], sheet_name=cell, engine="openpyxl")
        raw = raw.loc[raw["mRNA pmol"] == 2.0].copy()
        raw["sequence_id"] = raw["Sequence ID"].astype(int).map(lambda x: f"seq_{x:02d}")
        raw["cell"] = cell
        frames.append(raw)
    tidy = pd.concat(frames, ignore_index=True)
    return tidy.groupby(["cell", "megaTAL", "sequence_id"], sort=True)["Indel %"].agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "measured_value", "std": "sample_sd", "count": "n_replicates"})


def compute_stability(protocol, panel):
    raw = pd.read_csv(protocol["raw"]["stability"]["path"])
    experimental = raw.loc[raw["seq_name"].str.startswith("seq_")].copy().set_index("seq_name")
    spikes = raw.loc[raw["seq_name"].str.startswith("spikein_")].copy().set_index("seq_name")
    if len(raw) != 35 or len(experimental) != 27 or len(spikes) != 8:
        raise ValueError("Stability 27+8 count mismatch")
    times = np.asarray(protocol["stability"]["times_hours"], dtype=float)
    columns = protocol["stability"]["time_columns"]
    concentrations = np.asarray([1.0, 0.1, 0.01])
    spike_ids = ["spikein_1", "spikein_2", "spikein_3"]
    normalized = pd.DataFrame(index=experimental.index, columns=columns, dtype=float)
    calibration = []
    for time_value, column in zip(times, columns):
        x = spikes.loc[spike_ids, column].to_numpy(dtype=float)
        fit = stats.linregress(np.log10(x), np.log10(concentrations))
        normalized[column] = 10 ** (fit.slope * np.log10(experimental[column].to_numpy(dtype=float)) + fit.intercept)
        calibration.append({"time_hours": float(time_value), "column": column, "spikeins": spike_ids, "slope": float(fit.slope), "intercept": float(fit.intercept), "r_squared": float(fit.rvalue ** 2)})
    normalized = normalized / (experimental["ivt"].to_numpy(dtype=float).reshape(-1, 1) / experimental["ivt"].mean())
    integrated = np.trapezoid(normalized[columns].to_numpy(dtype=float), x=times, axis=1)
    result = pd.DataFrame({
        "stability_id": experimental.index,
        "source_row_id": [int(x.split("_")[1]) for x in experimental.index],
        "stability_sequence_key": experimental["seq"],
        "integrated_abundance": integrated,
        "free_to_total": experimental["polprof_0"].to_numpy(dtype=float) / experimental["polprof_total"].to_numpy(dtype=float),
        "ivt": experimental["ivt"].to_numpy(dtype=float),
    })
    result["sequence_id"] = result["source_row_id"].map(lambda x: f"seq_{x:02d}")
    panel_keys = panel.set_index("sequence_id")["utr"]
    if result["sequence_id"].nunique() != 27 or not all(panel_keys[row.sequence_id].endswith(row.stability_sequence_key) for row in result.itertuples(index=False)):
        raise ValueError("Stability/panel mapping failed")
    for index, column in enumerate(columns):
        result[column] = normalized[column].to_numpy(dtype=float)
    long = result.melt(id_vars=["sequence_id"], value_vars=columns, var_name="time_column", value_name="normalized_abundance")
    long["time_hours"] = long["time_column"].map(dict(zip(columns, times)))
    return result.sort_values("source_row_id").reset_index(drop=True), long, calibration


def order_for_seed(pool_code, seed_index, bootstrap_index=None, entropy=20261127):
    spawn = (pool_code, seed_index) if bootstrap_index is None else (pool_code, bootstrap_index, seed_index)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=entropy, spawn_key=spawn)))


def random_n_position(pool_code, seed_index, cap_index, cap, bootstrap_index=None):
    spawn = (pool_code, seed_index, cap_index) if bootstrap_index is None else (pool_code, bootstrap_index, seed_index, cap_index)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=20261129, spawn_key=spawn)))
    return int(rng.integers(0, cap))


def best_index(values, ids, candidates=None):
    if candidates is None:
        candidates = range(len(values))
    return min(candidates, key=lambda i: (-float(values[i]), ids[i], int(i)))


def top_tail_metrics(scores, endpoint, ids):
    n = len(ids)
    k = max(1, math.ceil(0.10 * n))
    proxy_order = sorted(range(n), key=lambda i: (-float(scores[i]), ids[i], int(i)))
    endpoint_order = sorted(range(n), key=lambda i: (-float(endpoint[i]), ids[i], int(i)))
    top = proxy_order[:k]
    bottom = proxy_order[-k:]
    oracle = float(endpoint[endpoint_order[0]])
    return {
        "n": n,
        "k": k,
        "top_decile_recall": float(len(set(top) & set(endpoint_order[:k])) / k),
        "top_tail_regret": float(oracle - max(float(endpoint[i]) for i in top)),
        "top_tail_mean_endpoint": float(np.mean(endpoint[top])),
        "bottom_tail_mean_endpoint": float(np.mean(endpoint[bottom])),
    }


def run_original_pool(pool_name, pool_code, ids, scores, endpoints, seeds):
    n = len(ids)
    if n < 1:
        raise ValueError(f"Pool {pool_name} is empty")
    caps = [cap for cap in CAPS if cap <= n]
    rank_order = sorted(range(n), key=lambda i: (-float(scores[i]), ids[i], int(i)))
    ranks = np.empty(n, dtype=int)
    for rank, index in enumerate(rank_order):
        ranks[index] = rank
    oracle = {name: best_index(values, ids) for name, values in endpoints.items()}
    rows = []
    for seed_index, seed in enumerate(seeds):
        order = order_for_seed(pool_code, seed_index).permutation(n)
        for cap_index, cap in enumerate(caps):
            prefix = order[:cap]
            bon = int(min(prefix, key=lambda i: ranks[i]))
            random_n = int(prefix[random_n_position(pool_code, seed_index, cap_index, cap)])
            random1 = int(prefix[0])
            for endpoint_name, values in endpoints.items():
                best = float(values[oracle[endpoint_name]])
                bon_regret = best - float(values[bon])
                random_regret = best - float(values[random_n])
                random1_regret = best - float(values[random1])
                rows.append({"pool": pool_name, "context": endpoint_name, "cap": cap, "algorithm_seed": seed, "n": n, "bon_selected_id": ids[bon], "random_n_selected_id": ids[random_n], "random1_selected_id": ids[random1], "bon_regret": bon_regret, "random_n_regret": random_regret, "random1_regret": random1_regret, "bon_minus_random_n": bon_regret - random_regret, "bon_selected_endpoint": float(values[bon]), "random_n_selected_endpoint": float(values[random_n]), "oracle_endpoint": best})
    return rows


def summarize_original(records):
    frame = pd.DataFrame(records)
    group_columns = ["pool", "context", "cap", "n"]
    summary = frame.groupby(group_columns, sort=True).agg(
        algorithm_seeds=("algorithm_seed", "count"),
        bon_regret=("bon_regret", "mean"),
        random_n_regret=("random_n_regret", "mean"),
        random1_regret=("random1_regret", "mean"),
        bon_minus_random_n=("bon_minus_random_n", "mean"),
        bon_selected_endpoint=("bon_selected_endpoint", "mean"),
        random_n_selected_endpoint=("random_n_selected_endpoint", "mean"),
        bon_oracle_coverage=("bon_regret", lambda x: float(np.mean(x == 0))),
        random_n_oracle_coverage=("random_n_regret", lambda x: float(np.mean(x == 0))),
    ).reset_index()
    return summary


def bootstrap_pool(pool_name, pool_code, ids, scores, endpoints, seeds, replicates, protocol):
    n = len(ids)
    contexts = sorted(endpoints)
    deltas = {(context, cap): [] for context in contexts for cap in CAPS}
    spearman = {context: [] for context in contexts}
    tails = {(context, field): [] for context in contexts for field in ("top_decile_recall", "top_tail_regret", "top_tail_mean_endpoint", "bottom_tail_mean_endpoint")}
    rng = np.random.default_rng([protocol["uncertainty"]["cluster_seed"], pool_code])
    proposal_orders = [order_for_seed(pool_code, seed_index).permutation(n) for seed_index in range(len(seeds))]
    random_positions = [[random_n_position(pool_code, seed_index, cap_index, cap) for cap_index, cap in enumerate(CAPS)] for seed_index in range(len(seeds))]
    for bootstrap_index in range(replicates):
        sampled = rng.integers(0, n, size=n)
        b_ids = [f"{ids[index]}#{copy}" for copy, index in enumerate(sampled)]
        b_scores = np.asarray(scores, dtype=float)[sampled]
        b_endpoints = {name: np.asarray(values, dtype=float)[sampled] for name, values in endpoints.items()}
        rank_order = sorted(range(n), key=lambda i: (-b_scores[i], b_ids[i], i))
        ranks = np.empty(n, dtype=int)
        for rank, index in enumerate(rank_order):
            ranks[index] = rank
        oracle = {name: float(np.max(values)) for name, values in b_endpoints.items()}
        accum = {(context, cap): [] for context in contexts for cap in CAPS}
        for seed_index, _ in enumerate(seeds):
            order = proposal_orders[seed_index]
            for cap_index, cap in enumerate(CAPS):
                prefix = order[:cap]
                bon = int(min(prefix, key=lambda i: ranks[i]))
                random_n = int(prefix[random_positions[seed_index][cap_index]])
                for context in contexts:
                    values = b_endpoints[context]
                    accum[(context, cap)].append(float(values[random_n] - values[bon]))
        for key, values in accum.items():
            deltas[key].append(float(np.mean(values)))
        for context in contexts:
            values = b_endpoints[context]
            spearman[context].append(float(stats.spearmanr(b_scores, values).statistic))
            tail = top_tail_metrics(b_scores, values, b_ids)
            for field in tails:
                if field[0] == context:
                    tails[field].append(tail[field[1]])
    return deltas, spearman, tails


def interval_and_p(values):
    values = np.asarray(values, dtype=float)
    lower = (1 + int(np.count_nonzero(values <= 0))) / (len(values) + 1)
    upper = (1 + int(np.count_nonzero(values >= 0))) / (len(values) + 1)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])], float(min(1.0, 2.0 * min(lower, upper)))


def add_holm(rows, family_columns):
    for _, group in pd.DataFrame(rows).groupby(family_columns, sort=False):
        indices = list(group.index)
        order = sorted(indices, key=lambda i: (rows[i]["bootstrap_p"], i))
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(order) - rank) * rows[index]["bootstrap_p"]))
            rows[index]["holm_p"] = running


def make_pool_data(panel, predictions, primary, all_rows, absolute, stability, conflicts):
    frame = panel.merge(predictions[["sequence_id", "kmer_mlp"]], on="sequence_id", validate="one_to_one")
    for endpoint_type, source in (("primary", primary), ("all_flagged", all_rows), ("absolute_2pmol", absolute)):
        for row in source.itertuples(index=False):
            context = f"{row.cell}|{row.megaTAL}"
            frame.loc[frame["sequence_id"] == row.sequence_id, f"{endpoint_type}|{context}"] = row.measured_value
    frame = frame.merge(stability[["sequence_id", "integrated_abundance", "free_to_total"]], on="sequence_id", how="left", validate="one_to_one")
    frame["sample_conflict"] = frame["sequence_id"].isin(conflicts)
    return frame


def pool_specs(frame):
    specs = []
    indel_contexts = sorted({column.split("|", 1)[1] for column in frame.columns if column.startswith("primary|")})
    indel_endpoints = {}
    for endpoint_type in ("primary", "all_flagged", "absolute_2pmol"):
        for context in indel_contexts:
            indel_endpoints[f"{endpoint_type}|{context}"] = f"{endpoint_type}|{context}"
    for name, code_key, proxy, mask in (
        ("indel_kmer_48", "indel_kmer_48", "kmer_mlp", frame["kmer_mlp"].notna()),
        ("indel_published_45", "indel_published_45", "published_predicted_mrl", frame["published_predicted_mrl"].notna()),
        ("indel_measured_12", "indel_measured_12", "measured_mrl", frame["measured_mrl"].notna()),
        ("indel_strict_h1", "indel_strict_h1", "kmer_mlp", ~frame["sample_conflict"]),
    ):
        selected = frame.loc[mask].copy()
        endpoints = {label: selected[column].to_numpy(dtype=float) for label, column in indel_endpoints.items()}
        specs.append({"name": name, "code_key": code_key, "proxy": proxy, "ids": selected["sequence_id"].tolist(), "scores": selected[proxy].to_numpy(dtype=float), "endpoints": endpoints, "frame": selected})
    stable = frame.loc[frame["integrated_abundance"].notna()].copy()
    for name, code_key, proxy in (("stability_kmer_27", "stability_kmer_27", "kmer_mlp"), ("stability_published_27", "stability_published_27", "published_predicted_mrl")):
        endpoints = {"primary|integrated_abundance": stable["integrated_abundance"].to_numpy(dtype=float)}
        for context in indel_contexts:
            endpoints[f"cross_endpoint_indel|{context}"] = stable[f"primary|{context}"].to_numpy(dtype=float)
        specs.append({"name": name, "code_key": code_key, "proxy": proxy, "ids": stable["sequence_id"].tolist(), "scores": stable[proxy].to_numpy(dtype=float), "endpoints": endpoints, "frame": stable})
    return specs


def frozen_mrl_diagnostics(protocol, root):
    pool_root = Path(protocol["historical_inputs"]["strict_h1_pool_root"])
    source_root = Path(protocol["historical_inputs"]["strict_h1_root"])
    per_pool = []
    combined = {}
    for context in ("n50_hek293t_r1", "defined_hepg2_r1"):
        rows = []
        for pool_index in range(16):
            pool_id = f"castillo_{context}_{pool_index:03d}"
            with gzip.open(pool_root / context / "learner" / f"{pool_id}.jsonl.gz", "rt") as handle:
                learner = [json.loads(line) for line in handle if line.strip()]
            ids = [row["sequence_id"] for row in learner]
            with gzip.open(pool_root / context / "evaluator" / "labels.tsv.gz", "rt", newline="") as handle:
                labels = {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
            score_payload = json.loads((source_root / context / "cnn_scores" / f"{pool_id}.json").read_text())
            scores_map = {row["sequence_id"]: float(row["proxy_mean"]) for row in score_payload["records"]}
            scores = np.asarray([scores_map[x] for x in ids])
            endpoint = np.asarray([labels[x] for x in ids])
            row = {"context": context, "pool_id": pool_id, "mae": float(np.mean(np.abs(scores - endpoint))), "spearman": float(stats.spearmanr(scores, endpoint).statistic)}
            row.update(top_tail_metrics(scores, endpoint, ids))
            per_pool.append(row)
            rows.extend({"pool_id": pool_id, "sequence_id": ids[i], "score": scores[i], "endpoint": endpoint[i]} for i in range(len(ids)))
        combined[context] = pd.DataFrame(rows)
    diagnostics = []
    rng = np.random.default_rng(20261230)
    for context, frame in combined.items():
        scores = frame["score"].to_numpy(dtype=float)
        endpoint = frame["endpoint"].to_numpy(dtype=float)
        pool_ids = sorted(frame["pool_id"].unique())
        pool_rows = [frame.loc[frame["pool_id"] == pool_id] for pool_id in pool_ids]
        pool_diag = [row for row in per_pool if row["context"] == context]
        bootstrap = np.empty((10000, 6), dtype=float)
        for b in range(10000):
            sampled = rng.integers(0, 16, size=16)
            selected = pd.concat([pool_rows[i] for i in sampled], ignore_index=True)
            b_scores = selected["score"].to_numpy(dtype=float)
            b_endpoint = selected["endpoint"].to_numpy(dtype=float)
            bootstrap[b, 0] = np.mean(np.abs(b_scores - b_endpoint))
            bootstrap[b, 1] = stats.spearmanr(b_scores, b_endpoint).statistic
            for column, field in enumerate(("top_decile_recall", "top_tail_regret", "top_tail_mean_endpoint", "bottom_tail_mean_endpoint"), start=2):
                bootstrap[b, column] = np.mean([pool_diag[i][field] for i in sampled])
        tail = {field: float(np.mean([row[field] for row in pool_diag])) for field in ("top_decile_recall", "top_tail_regret", "top_tail_mean_endpoint", "bottom_tail_mean_endpoint")}
        diagnostics.append({
            "context": context,
            "pools": 16,
            "candidates": len(frame),
            "mae": float(np.mean(np.abs(scores - endpoint))),
            "mae_ci95": [float(x) for x in np.percentile(bootstrap[:, 0], [2.5, 97.5])],
            "spearman": float(stats.spearmanr(scores, endpoint).statistic),
            "spearman_ci95": [float(x) for x in np.percentile(bootstrap[:, 1], [2.5, 97.5])],
            **tail,
            **{f"{field}_ci95": [float(x) for x in np.percentile(bootstrap[:, i + 2], [2.5, 97.5])] for i, field in enumerate(("top_decile_recall", "top_tail_regret", "top_tail_mean_endpoint", "bottom_tail_mean_endpoint"))},
        })
    pd.DataFrame(per_pool).to_csv(root / "results" / "mrl_frozen_per_pool_diagnostics.tsv", sep="\t", index=False, float_format="%.17g")
    dump_json(diagnostics, root / "results" / "mrl_frozen_diagnostics.json")
    return diagnostics


def copy_historical(protocol, root):
    budget = json.loads(Path(protocol["historical_inputs"]["budget_results"]).read_text())
    rows = []
    for family, family_rows in budget["families"].items():
        for row in family_rows:
            if row["context"] in ("n50_hek293t_r1", "defined_hepg2_r1"):
                rows.append({"family": family, **row})
    pd.DataFrame(rows).to_csv(root / "results" / "mrl_frozen_regret.tsv", sep="\t", index=False, float_format="%.17g")
    leppek = json.loads(Path(protocol["historical_inputs"]["leppek_summary"]).read_text())
    schlusser = json.loads(Path(protocol["historical_inputs"]["schlusser_summary"]).read_text())
    execution = json.loads(Path(protocol["historical_inputs"]["schlusser_execution"]).read_text())
    if not execution["outer_metrics_computed"] or not execution["one_time_unblinding"]:
        raise ValueError("Schlusser historical execution status mismatch")
    dump_json(leppek, root / "results" / "leppek_historical_summary.json")
    dump_json({"source_sha256": sha256(protocol["historical_inputs"]["schlusser_summary"]), "historical_once_only": True, "new_unblinding": False, "summary": schlusser}, root / "results" / "schlusser_historical_summary.json")
    return budget, leppek, schlusser


def build_unified(frame, specs, decision_summary, bootstrap_rows, mrl_diagnostics, budget, leppek, schlusser, root):
    unified = []
    for context in ("n50_hek293t_r1", "defined_hepg2_r1"):
        diagnostic = next(x for x in mrl_diagnostics if x["context"] == context)
        for cap in CAPS:
            source = next(x for x in budget["families"]["bon_minus_random_n"] if x["context"] == context and x["cap"] == cap)
            unified.append({"scenario": context, "endpoint": "measured_MRL", "proxy": "frozen_CNN", "n_candidates": 1024, "n_clusters": 16, "cap": cap, "mae": diagnostic["mae"], "spearman": diagnostic["spearman"], "top_decile_recall": diagnostic["top_decile_recall"], "top_tail_regret": diagnostic["top_tail_regret"], "bon_regret": next(x["bon_regret"] for x in budget["context_summaries"] if x["context"] == context and x["cap"] == cap), "random_n_regret": next(x["random_n_regret"] for x in budget["context_summaries"] if x["context"] == context and x["cap"] == cap), "bon_minus_random_n": source["effect"], "ci95": source["nominal_95_ci"], "p": source["p"], "holm_p": source["holm_p"], "evidence": "historical retrospective frozen result"})
    boot_lookup = {(x["pool"], x["context"], x["cap"]): x for x in bootstrap_rows}
    for spec in specs:
        primary_contexts = [x for x in spec["endpoints"] if x.startswith("primary|")]
        for context in primary_contexts:
            values = spec["endpoints"][context]
            for cap in [cap for cap in CAPS if cap <= len(spec["ids"])]:
                row = decision_summary.loc[(decision_summary["pool"] == spec["name"]) & (decision_summary["context"] == context) & (decision_summary["cap"] == cap)].iloc[0]
                boot = boot_lookup.get((spec["name"], context, cap), {})
                tail = top_tail_metrics(spec["scores"], values, spec["ids"])
                unified.append({"scenario": spec["name"], "endpoint": context, "proxy": spec["proxy"], "n_candidates": len(spec["ids"]), "n_clusters": len(spec["ids"]), "cap": int(cap), "mae": None, "spearman": float(stats.spearmanr(spec["scores"], values).statistic), "top_decile_recall": tail["top_decile_recall"], "top_tail_regret": tail["top_tail_regret"], "bon_regret": float(row["bon_regret"]), "random_n_regret": float(row["random_n_regret"]), "bon_minus_random_n": float(row["bon_minus_random_n"]), "ci95": boot.get("ci95"), "p": boot.get("bootstrap_p"), "holm_p": boot.get("holm_p"), "evidence": "new retrospective fixed-pool analysis"})
    unified.append({"scenario": "leppek_hek293t_variable_mrl", "endpoint": "measured_MRL", "proxy": "frozen_4mer_MLP", "n_candidates": leppek["constructs"], "n_clusters": leppek["utr_clusters"], "cap": 16, "mae": leppek["secondary"]["mae"]["value"], "spearman": leppek["secondary"]["spearman"]["value"], "top_decile_recall": None, "top_tail_regret": leppek["secondary"]["bon_top1_regret"]["value"], "bon_regret": leppek["secondary"]["bon_top1_regret"]["value"], "random_n_regret": leppek["secondary"]["random_top1_regret"]["value"], "bon_minus_random_n": leppek["primary"]["bon_minus_random_regret"], "ci95": leppek["primary"]["ci95"], "p": leppek["primary"]["p_two_sided_sign"], "holm_p": None, "evidence": "historical retrospective; Random comparator is Random1 not new RandomN"})
    pair = schlusser["primary"]["paired_comparison"]
    unified.append({"scenario": "schlusser_hepg2_endogenous_te", "endpoint": "measured_TE", "proxy": "historical_frozen_CNN", "n_candidates": schlusser["candidate_rows"], "n_clusters": schlusser["unique_genes"], "cap": 16, "mae": None, "spearman": schlusser["endpoint_alignment_descriptive"]["spearman_cnn_MRL_vs_TE"], "top_decile_recall": None, "top_tail_regret": None, "bon_regret": schlusser["primary"]["configuration"]["mean_regret"], "random_n_regret": schlusser["primary"]["configuration"]["mean_regret"] - pair["mean_regret_difference_left_minus_right"], "bon_minus_random_n": pair["mean_regret_difference_left_minus_right"], "ci95": pair["gene_cluster_bootstrap_95_interval"], "p": None, "holm_p": None, "evidence": "historical one-time result only; Random1 comparator; no new unblinding"})
    pd.DataFrame(unified).to_json(root / "results" / "unified_scenario_metrics.json", orient="records", indent=2)
    return unified


def run(protocol_path):
    started = time.time()
    protocol_start_sha256 = sha256(protocol_path)
    protocol, protocol_path = load_protocol(protocol_path)
    root = Path(protocol["root"])
    inputs = verify_raw(protocol)
    panel = read_panel(protocol)
    references = read_sample_references(protocol)
    conflicts = conflict_audit(panel, references)
    dump_json({"native_and_n50_variable_region_audit": True, "conflicted_utrs": len(conflicts), "conflict_free_utrs": 48 - len(conflicts), "conflicts": conflicts}, root / "protocol" / "isolation_audit.json")
    learner = panel[["sequence_id", "utr"]].copy()
    learner.to_csv(root / "protocol" / "learner_sequences.tsv", sep="\t", index=False)
    panel[["sequence_id", "source_row_id", "construct_id", "cluster_id", "measured_mrl"]].to_csv(root / "protocol" / "evaluator_mrl_labels.tsv", sep="\t", index=False, float_format="%.17g")
    predictions = predict_cpu(panel, protocol, root)
    tidy_primary, primary = normalized_indel(protocol, include_flagged=False)
    author_check = validate_author_summary(protocol, primary)
    tidy_all, all_rows = normalized_indel(protocol, include_flagged=True)
    absolute = absolute_2pmol(protocol)
    tidy_primary.to_csv(root / "results" / "indel_normalized_rows_primary.tsv", sep="\t", index=False, float_format="%.17g")
    tidy_all.to_csv(root / "results" / "indel_normalized_rows_all_flagged.tsv", sep="\t", index=False, float_format="%.17g")
    primary.assign(endpoint_type="primary").to_csv(root / "results" / "indel_summary_primary.tsv", sep="\t", index=False, float_format="%.17g")
    all_rows.assign(endpoint_type="all_flagged").to_csv(root / "results" / "indel_summary_all_flagged.tsv", sep="\t", index=False, float_format="%.17g")
    absolute.to_csv(root / "results" / "indel_summary_absolute_2pmol.tsv", sep="\t", index=False, float_format="%.17g")
    stability, stability_long, calibration = compute_stability(protocol, panel)
    stability.to_csv(root / "results" / "stability_summary.tsv", sep="\t", index=False, float_format="%.17g")
    stability_long.to_csv(root / "results" / "stability_timecourse.tsv", sep="\t", index=False, float_format="%.17g")
    dump_json(calibration, root / "results" / "stability_calibration.json")
    frame = make_pool_data(panel, predictions, primary, all_rows, absolute, stability, conflicts)
    frame.to_csv(root / "results" / "candidate_master.tsv", sep="\t", index=False, float_format="%.17g")
    specs = pool_specs(frame)
    seeds = list(range(protocol["selection"]["algorithm_seeds"][0], protocol["selection"]["algorithm_seeds"][1] + 1))
    records = []
    for spec in specs:
        records.extend(run_original_pool(spec["name"], protocol["selection"]["pool_codes"][spec["code_key"]], spec["ids"], spec["scores"], spec["endpoints"], seeds))
    pd.DataFrame(records).to_csv(root / "results" / "decision_per_seed.tsv", sep="\t", index=False, float_format="%.17g")
    decision_summary = summarize_original(records)
    decision_summary.to_csv(root / "results" / "decision_summary.tsv", sep="\t", index=False, float_format="%.17g")
    bootstrap_rows = []
    diagnostic_rows = []
    for spec in specs:
        if spec["name"] == "indel_measured_12":
            continue
        primary_endpoints = {key: value for key, value in spec["endpoints"].items() if key.startswith("primary|")}
        replicates = protocol["uncertainty"]["primary_bootstrap_replicates"] if spec["name"] in ("indel_kmer_48", "stability_kmer_27") else protocol["uncertainty"]["secondary_bootstrap_replicates"]
        deltas, spearman_values, tail_values = bootstrap_pool(spec["name"], protocol["selection"]["pool_codes"][spec["code_key"]], spec["ids"], spec["scores"], primary_endpoints, seeds, replicates, protocol)
        for (context, cap), values in deltas.items():
            ci, p_value = interval_and_p(values)
            bootstrap_rows.append({"pool": spec["name"], "context": context, "cap": cap, "bootstrap_replicates": replicates, "ci95": ci, "bootstrap_p": p_value, "bootstrap_delta_mean": float(np.mean(values))})
        for context in primary_endpoints:
            ci, _ = interval_and_p(spearman_values[context])
            tail = top_tail_metrics(spec["scores"], primary_endpoints[context], spec["ids"])
            diagnostic_rows.append({"pool": spec["name"], "context": context, "n": len(spec["ids"]), "spearman": float(stats.spearmanr(spec["scores"], primary_endpoints[context]).statistic), "spearman_ci95": ci, **tail, **{f"{field}_ci95": [float(x) for x in np.percentile(tail_values[(context, field)], [2.5, 97.5])] for field in ("top_decile_recall", "top_tail_regret", "top_tail_mean_endpoint", "bottom_tail_mean_endpoint")}})
    add_holm(bootstrap_rows, ["pool"])
    pd.DataFrame(bootstrap_rows).to_json(root / "results" / "decision_bootstrap.json", orient="records", indent=2)
    pd.DataFrame(diagnostic_rows).to_json(root / "results" / "decision_diagnostics.json", orient="records", indent=2)
    baseline_rows = []
    for spec in specs[:4]:
        if "seq_48" not in spec["ids"]:
            continue
        index = spec["ids"].index("seq_48")
        for context, values in spec["endpoints"].items():
            if not context.startswith("primary|"):
                continue
            oracle = float(np.max(values))
            baseline_rows.append({"pool": spec["name"], "context": context, "selected_id": "seq_48", "selected_endpoint": float(values[index]), "regret": oracle - float(values[index])})
    pd.DataFrame(baseline_rows).to_csv(root / "results" / "strong_kozak_baseline.tsv", sep="\t", index=False, float_format="%.17g")
    mrl_diagnostics = frozen_mrl_diagnostics(protocol, root)
    budget, leppek, schlusser = copy_historical(protocol, root)
    build_unified(frame, specs, decision_summary, bootstrap_rows, mrl_diagnostics, budget, leppek, schlusser, root)
    execution = {
        "status": "PASS",
        "analysis_id": protocol["analysis_id"],
        "protocol_sha256": protocol_start_sha256,
        "input_files": inputs,
        "author_summary_validation": author_check,
        "panel_utrs": 48,
        "measured_mrl": 12,
        "published_predicted_mrl": 45,
        "indel_raw_rows": {"HepG2": 672, "K562": 768},
        "flagged_excluded_primary": int(pd.concat([pd.read_excel(protocol["raw"]["editing"]["path"], sheet_name="K562")["Flagged"] == 1]).sum()),
        "indel_construct_context_summaries": 192,
        "stability_experimental": 27,
        "stability_spikeins_calibration_only": 8,
        "decision_per_seed_rows": len(records),
        "algorithm_seeds": 200,
        "cpu_statistics": True,
        "cpu_mlp_inference": True,
        "cuda_used": False,
        "new_training": False,
        "new_blind_test": False,
        "schlusser_recomputed": False,
        "elapsed_seconds": time.time() - started,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "executable": os.sys.executable},
    }
    dump_json(execution, root / "verification" / "round1_execution.json")
    print(json.dumps({"status": "PASS", "elapsed_seconds": execution["elapsed_seconds"], "conflicted_utrs": len(conflicts)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    if args.command == "run":
        import fcntl
        protocol = json.loads(Path(args.protocol).read_text())
        lock_path = Path(protocol["root"]) / "protocol" / "execution.lock"
        lock_handle = open(lock_path, "w")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another scenario analysis process holds the execution lock")
        try:
            run(args.protocol)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


if __name__ == "__main__":
    main()
