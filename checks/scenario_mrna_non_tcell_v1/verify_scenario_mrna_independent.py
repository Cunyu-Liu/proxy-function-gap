#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/scenario_mrna_non_tcell_v1")
PROTOCOL = Path("/home/proxy-function-gap-scenario-mrna/configs/scenario_mrna_non_tcell_v1.json")
CAPS = [1, 2, 4, 8, 16]
SEEDS = list(range(20261127, 20261327))
POOL_CODES = {"indel_kmer_48": 1, "indel_published_45": 2, "indel_measured_12": 3, "stability_kmer_27": 4, "stability_published_27": 5, "indel_strict_h1": 6}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(left, right, tolerance=1e-10, label="value"):
    error = float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))
    if error > tolerance:
        raise AssertionError(f"{label}: maximum absolute error {error}")
    return error


def independent_features(sequences):
    alphabet = "ACGT"
    code = {base: index for index, base in enumerate(alphabet)}
    result = np.zeros((len(sequences), 259), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for start in range(len(sequence) - 3):
            index = 0
            for base in sequence[start:start + 4]:
                index = index * 4 + code[base]
            result[row, index] += np.float32(1.0 / (len(sequence) - 3))
        result[row, 256] = (sequence.count("G") + sequence.count("C")) / len(sequence)
        result[row, 257] = float("ATG" in sequence)
        longest = 0
        current = 0
        for base in sequence:
            current = current + 1 if base == "T" else 0
            longest = max(longest, current)
        result[row, 258] = longest / 50
    return result


def independent_indel(config, include_flagged):
    summaries = {}
    tidy_rows = {}
    for cell in ("HepG2", "K562"):
        raw = pd.read_excel(config["raw"]["editing"]["path"], sheet_name=cell, engine="openpyxl")
        if not include_flagged:
            raw = raw.loc[raw["Flagged"].isna()].copy()
        for target in ("TGFBR2", "PDCD1"):
            target_rows = raw.loc[raw["megaTAL"] == target]
            reference = target_rows.loc[target_rows["Sequence ID"] == 48].groupby("mRNA pmol")["Indel %"].mean().to_dict()
            values = {sequence_id: [] for sequence_id in range(1, 49)}
            for sequence_id, dose, indel in target_rows[["Sequence ID", "mRNA pmol", "Indel %"]].itertuples(index=False, name=None):
                sequence_id = int(sequence_id)
                dose = float(dose)
                indel = float(indel)
                values[sequence_id].append(indel / reference[dose])
            for sequence_id, sequence_values in values.items():
                if not sequence_values:
                    raise AssertionError("Missing indel summary")
                key = f"primary|{cell}|{target}"
                summaries.setdefault(key, {})[f"seq_{sequence_id:02d}"] = float(np.mean(sequence_values))
                tidy_rows[(cell, target, sequence_id)] = (float(np.mean(sequence_values)), float(np.std(sequence_values, ddof=1)), len(sequence_values))
    return summaries, tidy_rows


def independent_absolute(config):
    result = {}
    for cell in ("HepG2", "K562"):
        raw = pd.read_excel(config["raw"]["editing"]["path"], sheet_name=cell, engine="openpyxl")
        raw = raw.loc[raw["mRNA pmol"] == 2.0]
        for target in ("TGFBR2", "PDCD1"):
            for sequence_id in range(1, 49):
                values = raw.loc[(raw["megaTAL"] == target) & (raw["Sequence ID"] == sequence_id), "Indel %"].to_numpy(dtype=float)
                if len(values) != 2:
                    raise AssertionError("2 pmol replicate count mismatch")
                result.setdefault(f"absolute_2pmol|{cell}|{target}", {})[f"seq_{sequence_id:02d}"] = float(values.mean())
    return result


def independent_stability(config):
    raw = pd.read_csv(config["raw"]["stability"]["path"])
    spikes = raw.loc[raw["seq_name"].str.startswith("spikein")].set_index("seq_name")
    experimental = raw.loc[raw["seq_name"].str.startswith("seq_")].copy()
    columns = ["stab_1", "stab_2", "stab_3", "stab_4", "stab_5", "stab_6"]
    times = np.asarray([0.5, 1, 2, 4, 8, 24], dtype=float)
    abundances = np.zeros((27, 6), dtype=float)
    for column_index, column in enumerate(columns):
        x = np.log10(spikes.loc[["spikein_1", "spikein_2", "spikein_3"], column].to_numpy(dtype=float))
        y = np.log10(np.asarray([1.0, 0.1, 0.01]))
        slope, intercept = np.polyfit(x, y, 1)
        abundances[:, column_index] = 10 ** (slope * np.log10(experimental[column].to_numpy(dtype=float)) + intercept)
    abundances /= (experimental["ivt"].to_numpy(dtype=float) / experimental["ivt"].mean())[:, None]
    integrated = np.sum((abundances[:, 1:] + abundances[:, :-1]) * np.diff(times)[None, :] / 2.0, axis=1)
    result = {}
    free = {}
    for index, row in experimental.reset_index(drop=True).iterrows():
        sequence_id = f"seq_{int(row['seq_name'].split('_')[1]):02d}"
        result[sequence_id] = float(integrated[index])
        free[sequence_id] = float(row["polprof_0"] / row["polprof_total"])
    return result, free, abundances


def independent_predictions(config):
    import torch
    learner = pd.read_csv(ROOT / "protocol" / "learner_sequences.tsv", sep="\t")
    features = independent_features(learner["utr"].tolist())
    members = []
    for spec in config["model"]["checkpoints"]:
        checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=False)
        state = checkpoint["model_state"]
        hidden = np.maximum(0.0, features @ state["layers.0.weight"].numpy().T + state["layers.0.bias"].numpy())
        normalized = hidden @ state["layers.2.weight"].numpy().T + state["layers.2.bias"].numpy()
        values = normalized[:, 0] * checkpoint["scaler"]["scale"] + checkpoint["scaler"]["mean"]
        members.append(values)
    return learner["sequence_id"].tolist(), np.mean(members, axis=0), np.max(np.abs(members[0] - members[1]))


def proposal_permutation(pool_code, seed_index, n, bootstrap=None, bootstrap_mode=False):
    entropy = 20261227 if bootstrap_mode else 20261127
    spawn = (pool_code, seed_index) if bootstrap is None else (pool_code, bootstrap, seed_index)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=entropy, spawn_key=spawn))).permutation(n)


def random_position(pool_code, seed_index, cap_index, cap, bootstrap=None):
    spawn = (pool_code, seed_index, cap_index) if bootstrap is None else (pool_code, bootstrap, seed_index, cap_index)
    return int(np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=20261129, spawn_key=spawn))).integers(cap))


def independent_original(pool, ids, scores, endpoints):
    code = POOL_CODES[pool]
    n = len(ids)
    rank = {index: position for position, index in enumerate(sorted(range(n), key=lambda i: (-scores[i], ids[i], i)))}
    result = {}
    for seed_index, seed in enumerate(SEEDS):
        permutation = proposal_permutation(code, seed_index, n)
        for cap_index, cap in enumerate([cap for cap in CAPS if cap <= n]):
            prefix = permutation[:cap]
            bon = min(prefix, key=lambda i: rank[i])
            random_n = prefix[random_position(code, seed_index, cap_index, cap)]
            random1 = prefix[0]
            for context, endpoint_map in endpoints.items():
                values = np.asarray([endpoint_map[x] for x in ids])
                oracle = max(values)
                key = (context, cap, seed)
                result[key] = (ids[bon], ids[random_n], ids[random1], float(oracle - values[bon]), float(oracle - values[random_n]), float(oracle - values[random1]))
    return result


def independent_bootstrap(pool, ids, scores, endpoint_maps, replicates):
    code = POOL_CODES[pool]
    n = len(ids)
    scores = np.asarray(scores, dtype=float)
    original_endpoints = {context: np.asarray([mapping[x] for x in ids]) for context, mapping in endpoint_maps.items()}
    means = {(context, cap): [] for context in endpoint_maps for cap in CAPS}
    rng = np.random.default_rng([20261130, code])
    proposal_orders = [proposal_permutation(code, seed_index, n) for seed_index in range(len(SEEDS))]
    random_positions = [[random_position(code, seed_index, cap_index, cap) for cap_index, cap in enumerate(CAPS)] for seed_index in range(len(SEEDS))]
    for bootstrap in range(replicates):
        sampled = rng.integers(n, size=n)
        b_ids = [f"{ids[index]}#{copy}" for copy, index in enumerate(sampled)]
        b_scores = scores[sampled]
        b_endpoints = {context: values[sampled] for context, values in original_endpoints.items()}
        order = sorted(range(n), key=lambda i: (-b_scores[i], b_ids[i], i))
        rank = {index: position for position, index in enumerate(order)}
        cap_deltas = {(context, cap): [] for context in endpoint_maps for cap in CAPS}
        for seed_index in range(len(SEEDS)):
            permutation = proposal_orders[seed_index]
            for cap_index, cap in enumerate(CAPS):
                prefix = permutation[:cap]
                bon = min(prefix, key=lambda i: rank[i])
                random_n = prefix[random_positions[seed_index][cap_index]]
                for context, b_endpoint in b_endpoints.items():
                    cap_deltas[(context, cap)].append(float(b_endpoint[random_n] - b_endpoint[bon]))
        for key, deltas in cap_deltas.items():
            means[key].append(float(np.mean(deltas)))
    return means


def main():
    started = time.time()
    config = json.loads(PROTOCOL.read_text())
    checks = {}
    panel = pd.read_excel(config["raw"]["editing"]["path"], sheet_name="5'UTR sequences", engine="openpyxl")
    ids = [f"seq_{int(x):02d}" for x in panel["Sequence ID"]]
    measured = dict(zip(ids, pd.to_numeric(panel["Measured MRL"], errors="coerce")))
    published = dict(zip(ids, pd.to_numeric(panel["Predicted MRL"], errors="coerce")))
    if len(panel) != 48 or np.isfinite(list(measured.values())).sum() != 12 or np.isfinite(list(published.values())).sum() != 45:
        raise AssertionError("Panel n/coverage")
    checks["panel"] = {"utrs": 48, "measured_mrl": 12, "published_predicted_mrl": 45, "lengths": {str(int(k)): int(v) for k, v in panel["5'UTR"].str.len().value_counts().sort_index().items()}}
    raw_row_counts = {}
    for cell in ("HepG2", "K562"):
        raw = pd.read_excel(config["raw"]["editing"]["path"], sheet_name=cell, engine="openpyxl")
        raw_row_counts[cell] = {"rows": len(raw), "flagged": int((raw["Flagged"] == 1).sum())}
    if raw_row_counts != {"HepG2": {"rows": 672, "flagged": 0}, "K562": {"rows": 768, "flagged": 25}}:
        raise AssertionError("Raw editing rows/flags")
    checks["raw_editing_rows"] = raw_row_counts
    primary, primary_tidy = independent_indel(config, False)
    all_flagged, _ = independent_indel(config, True)
    absolute = independent_absolute(config)
    production_primary = pd.read_csv(ROOT / "results" / "indel_summary_primary.tsv", sep="\t")
    expected = []
    for row in production_primary.itertuples(index=False):
        expected.append(primary_tidy[(row.cell, row.megaTAL, int(row.sequence_id[4:]))][0])
    checks["indel_primary_max_error"] = assert_close(production_primary["measured_value"], expected, 1e-12, "indel primary")
    author = pd.read_csv(config["raw"]["author_summary"]["path"])
    author_mean = []
    author_sd = []
    for dataset, target, sequence_id in author[["Dataset", "megaTAL", "Sequence ID"]].itertuples(index=False, name=None):
        values = primary_tidy[(dataset, target, int(sequence_id))]
        author_mean.append(values[0])
        author_sd.append(values[1])
    checks["author_mean_max_error"] = assert_close(author["Mean normalized indel %"], author_mean, 1e-12, "author means")
    checks["author_sd_max_error"] = assert_close(author["Std normalized indel %"], author_sd, 1e-12, "author SDs")
    production_all = pd.read_csv(ROOT / "results" / "indel_summary_all_flagged.tsv", sep="\t")
    expected_all = [all_flagged[f"primary|{row.cell}|{row.megaTAL}"][row.sequence_id] for row in production_all.itertuples(index=False)]
    checks["indel_all_flagged_max_error"] = assert_close(production_all["measured_value"], expected_all, 1e-12, "all flagged")
    integrated, free, abundances = independent_stability(config)
    production_stability = pd.read_csv(ROOT / "results" / "stability_summary.tsv", sep="\t")
    checks["stability_integrated_max_error"] = assert_close(production_stability["integrated_abundance"], [integrated[x] for x in production_stability["sequence_id"]], 1e-10, "integrated abundance")
    checks["stability_free_max_error"] = assert_close(production_stability["free_to_total"], [free[x] for x in production_stability["sequence_id"]], 1e-12, "free to total")
    prediction_ids, prediction_values, member_spread = independent_predictions(config)
    production_predictions = pd.read_csv(ROOT / "predictions" / "predictions.tsv", sep="\t")
    checks["prediction_max_error"] = assert_close(production_predictions["kmer_mlp"], prediction_values, 2e-6, "independent numpy predictions")
    kmer = dict(zip(prediction_ids, prediction_values))
    isolation = json.loads((ROOT / "protocol" / "isolation_audit.json").read_text())
    conflict_ids = set(isolation["conflicts"])
    stability_ids = sorted(integrated)
    pool_definitions = {}
    for pool, proxy, selected_ids in (
        ("indel_kmer_48", kmer, ids),
        ("indel_published_45", published, [x for x in ids if np.isfinite(published[x])]),
        ("indel_measured_12", measured, [x for x in ids if np.isfinite(measured[x])]),
        ("indel_strict_h1", kmer, [x for x in ids if x not in conflict_ids]),
    ):
        endpoints = {}
        for source, prefix in ((primary, "primary"), (all_flagged, "all_flagged"), (absolute, "absolute_2pmol")):
            for context, mapping in source.items():
                label = context if prefix == "primary" else context.replace("primary", prefix, 1)
                endpoints[label] = mapping
        pool_definitions[pool] = (selected_ids, proxy, endpoints)
    for pool, proxy in (("stability_kmer_27", kmer), ("stability_published_27", published)):
        endpoints = {"primary|integrated_abundance": integrated}
        for context, mapping in primary.items():
            endpoints[context.replace("primary", "cross_endpoint_indel", 1)] = mapping
        pool_definitions[pool] = (stability_ids, proxy, endpoints)
    production_records = pd.read_csv(ROOT / "results" / "decision_per_seed.tsv", sep="\t")
    max_id_mismatches = 0
    max_regret_error = 0.0
    compared = 0
    for pool, (selected_ids, proxy, endpoints) in pool_definitions.items():
        independent = independent_original(pool, selected_ids, [proxy[x] for x in selected_ids], endpoints)
        rows = production_records.loc[production_records["pool"] == pool]
        for row in rows.itertuples(index=False):
            expected = independent[(row.context, row.cap, row.algorithm_seed)]
            max_id_mismatches += int((row.bon_selected_id, row.random_n_selected_id, row.random1_selected_id) != expected[:3])
            max_regret_error = max(max_regret_error, float(np.max(np.abs(np.asarray([row.bon_regret, row.random_n_regret, row.random1_regret]) - expected[3:]))))
            compared += 1
    if max_id_mismatches or max_regret_error > 1e-10 or compared != len(production_records):
        raise AssertionError("Original decision recomputation mismatch")
    checks["original_decisions"] = {"rows": compared, "selected_id_mismatches": max_id_mismatches, "maximum_regret_error": max_regret_error}
    production_boot = pd.read_json(ROOT / "results" / "decision_bootstrap.json")
    boot_max_error = 0.0
    boot_ci_error = 0.0
    boot_rows = 0
    for pool in ("indel_kmer_48", "indel_published_45", "indel_strict_h1", "stability_kmer_27", "stability_published_27"):
        selected_ids, proxy, endpoints = pool_definitions[pool]
        primary_endpoints = {context: mapping for context, mapping in endpoints.items() if context.startswith("primary|")}
        replicates = 10000 if pool in ("indel_kmer_48", "stability_kmer_27") else 2000
        distributions = independent_bootstrap(pool, selected_ids, [proxy[x] for x in selected_ids], primary_endpoints, replicates)
        for context in primary_endpoints:
            for cap in CAPS:
                target = production_boot.loc[(production_boot["pool"] == pool) & (production_boot["context"] == context) & (production_boot["cap"] == cap)].iloc[0]
                values = np.asarray(distributions[(context, cap)])
                boot_max_error = max(boot_max_error, abs(float(values.mean()) - float(target["bootstrap_delta_mean"])))
                ci = np.percentile(values, [2.5, 97.5])
                boot_ci_error = max(boot_ci_error, float(np.max(np.abs(ci - np.asarray(target["ci95"], dtype=float)))))
                boot_rows += 1
    if boot_max_error > 1e-10 or boot_ci_error > 1e-10:
        raise AssertionError("Bootstrap recomputation mismatch")
    checks["bootstrap"] = {"rows": boot_rows, "maximum_mean_error": boot_max_error, "maximum_ci_error": boot_ci_error}
    budget_path = Path(config["historical_inputs"]["budget_results"])
    copied = pd.read_csv(ROOT / "results" / "mrl_frozen_regret.tsv", sep="\t")
    budget = json.loads(budget_path.read_text())
    expected_rows = sum(1 for family in budget["families"].values() for row in family if row["context"] in ("n50_hek293t_r1", "defined_hepg2_r1"))
    if len(copied) != expected_rows or expected_rows != 30:
        raise AssertionError("Frozen MRL export incomplete")
    schlusser_path = Path(config["historical_inputs"]["schlusser_summary"])
    historical = json.loads((ROOT / "results" / "schlusser_historical_summary.json").read_text())
    if historical["source_sha256"] != sha256(schlusser_path) or historical["new_unblinding"] is not False:
        raise AssertionError("Schlusser historical-only boundary")
    checks["historical"] = {"mrl_regret_rows": expected_rows, "schlusser_source_sha256": sha256(schlusser_path), "schlusser_new_unblinding": False}
    output = {"status": "PASS", "method": "Standalone raw XLSX/CSV, NumPy neural-network, selection and full bootstrap recomputation; production module not imported", "checks": checks, "elapsed_seconds": time.time() - started}
    (ROOT / "verification" / "round2_independent.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
