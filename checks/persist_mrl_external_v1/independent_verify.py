import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.stats import rankdata

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/persist_mrl_external_v1")
CODE = Path("/home/proxy-function-gap-persist-mrl")
ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def features_independent(seq):
    out = np.zeros(259, dtype=np.float32)
    denominator = len(seq) - 3
    for start in range(denominator):
        word = seq[start:start + 4]
        index = ((ALPHABET[word[0]] * 4 + ALPHABET[word[1]]) * 4 + ALPHABET[word[2]]) * 4 + ALPHABET[word[3]]
        out[index] += 1.0 / denominator
    out[256] = sum(base in "GC" for base in seq) / len(seq)
    out[257] = "ATG" in seq
    longest = current = 0
    for base in seq:
        current = current + 1 if base == "T" else 0
        longest = max(longest, current)
    out[258] = longest / 50
    return out


def proposal_stats(y, p, names, entropy=None):
    random_regrets, bon_regrets, random_hits, bon_hits = [], [], [], []
    for seed in range(20261125, 20261325):
        rng = np.random.default_rng(seed if entropy is None else entropy + [seed])
        order = rng.permutation(len(y))
        oracle = sorted(range(len(y)), key=lambda i: (-y[i], names[i], i))[0]
        bon = sorted(order[:16], key=lambda i: (-p[i], names[i], int(i)))[0]
        random = order[0]
        random_regrets.append(y[oracle] - y[random])
        bon_regrets.append(y[oracle] - y[bon])
        random_hits.append(float(y[random] == y[oracle]))
        bon_hits.append(float(y[bon] == y[oracle]))
    random_regrets = np.asarray(random_regrets)
    bon_regrets = np.asarray(bon_regrets)
    return np.asarray([np.mean(bon_regrets - random_regrets), random_regrets.mean(), bon_regrets.mean(), np.mean(random_hits), np.mean(bon_hits)])


def main():
    protocol = json.loads((CODE / "frozen_protocol.json").read_text())
    assert file_hash(CODE / "frozen_protocol.json") == (ROOT / "protocol" / "frozen_protocol.sha256").read_text().strip()
    workbook = load_workbook(protocol["source"]["path"], read_only=True, data_only=True)
    sheet = workbook[protocol["source"]["sheet"]]
    iterator = sheet.iter_rows(values_only=True)
    header = list(next(iterator))
    source_rows = [dict(zip(header, row)) for row in iterator if any(value is not None for value in row)]
    group_rows = [r for r in source_rows if r["Group"] == "UTR (5') variants"]
    assert len(source_rows) == 233 and len(group_rows) == 59
    references = set()
    role_sizes = {}
    for role, path in protocol["sample_exclusion"]["paths"].items():
        with gzip.open(path, "rt") as f:
            role_sequences = {r["utr50"] for r in csv.DictReader(f, delimiter="\t")}
        role_sizes[role] = len(role_sequences)
        references |= role_sequences
    excluded = set()
    for seq in {r["Sequence 5 UTR"] for r in group_rows}:
        if seq in references or any(len(seq) == len(ref) and sum(a != b for a, b in zip(seq, ref)) == 1 for ref in references):
            excluded.add(seq)
    eligible = [r for r in group_rows if r["Sequence 5 UTR"] not in excluded]
    ids = [r["Sequence ID"] for r in eligible]
    utrs = [r["Sequence 5 UTR"] for r in eligible]
    y = np.asarray([float(r["Ribosome load"]) for r in eligible])
    saved = list(csv.DictReader(open(ROOT / "predictions" / "predictions.tsv"), delimiter="\t"))
    saved_by_id = {r["sequence_id"]: r for r in saved}
    assert set(saved_by_id) == set(ids)
    x = np.stack([features_independent(seq) for seq in utrs])

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    import torch
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_x = torch.tensor(x, dtype=torch.float32, device=device)
    max_member_error = 0.0
    member_values = []
    for spec in protocol["model"]["checkpoints"]:
        ckpt = torch.load(spec["path"], map_location="cpu", weights_only=False)
        state = ckpt["model_state"]
        w0 = state["layers.0.weight"].to(device=device)
        b0 = state["layers.0.bias"].to(device=device)
        w2 = state["layers.2.weight"].to(device=device)
        b2 = state["layers.2.bias"].to(device=device)
        assert all(t.is_cuda for t in (gpu_x, w0, b0, w2, b2))
        with torch.inference_mode():
            hidden = torch.relu(gpu_x @ w0.T + b0)
            normalized = (hidden @ w2.T + b2).squeeze(1)
            output = normalized * ckpt["scaler"]["scale"] + ckpt["scaler"]["mean"]
            assert output.is_cuda and hidden.is_cuda
            values = output.cpu().numpy()
        member_values.append(values)
        expected = np.asarray([float(saved_by_id[i][f"member_{spec['seed']}"]) for i in ids])
        max_member_error = max(max_member_error, float(np.max(np.abs(values - expected))))
    assert max_member_error < 2e-6
    p = np.mean(member_values, axis=0)
    expected_mean = np.asarray([float(saved_by_id[i]["prediction"]) for i in ids])
    max_mean_error = float(np.max(np.abs(p - expected_mean)))
    assert max_mean_error < 2e-6

    base = proposal_stats(y, p, ids)
    clusters = sorted(set(utrs))
    members = [[i for i, seq in enumerate(utrs) if seq == cluster] for cluster in clusters]
    rng = np.random.default_rng(20261126)
    boot = np.empty((10000, 8))
    for b in range(10000):
        draws = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([members[d] for d in draws])
        by, bp = y[indices], p[indices]
        bnames = [f"{ids[i]}#{rank}" for rank, i in enumerate(indices)]
        values = proposal_stats(by, bp, bnames, [20261127, b])
        ranked_p, ranked_y = rankdata(bp), rankdata(by)
        boot[b, 0] = values[0]
        boot[b, 1] = np.corrcoef(ranked_p, ranked_y)[0, 1]
        boot[b, 2] = np.corrcoef(bp, by)[0, 1]
        boot[b, 3] = np.mean(np.abs(bp - by))
        boot[b, 4:] = values[1:]
    expected_boot = np.load(ROOT / "results" / "bootstrap_replicates.npy")
    differences = np.abs(boot - expected_boot)
    max_boot_error = float(np.max(differences))
    print("BASE", base.tolist(), "EXPECTED_PRIMARY", json.loads((ROOT / "results" / "summary.json").read_text())["primary"]["bon_minus_random_regret"], flush=True)
    print("BOOT_COLUMN_MAX", differences.max(axis=0).tolist(), flush=True)
    print("FIRST_DIFF", np.unravel_index(np.argmax(differences), differences.shape), boot[np.unravel_index(np.argmax(differences), differences.shape)[0]].tolist(), expected_boot[np.unravel_index(np.argmax(differences), differences.shape)[0]].tolist(), flush=True)
    np.save(ROOT / "verification" / "round2_independent_bootstrap.npy", boot)
    assert differences[:, 0].max() == 0.0
    assert np.all(differences[:, [4, 5, 6, 7]] == 0.0)
    assert differences[:, [1, 2, 3]].max() < 1e-6
    expected_summary = json.loads((ROOT / "results" / "summary.json").read_text())
    assert abs(base[0] - expected_summary["primary"]["bon_minus_random_regret"]) < 1e-12
    assert np.allclose(np.percentile(boot[:, 0], [2.5, 97.5]), expected_summary["primary"]["ci95"], atol=1e-12)
    delta = boot[:, 0]
    pvalue = min(1.0, 2 * min(((delta >= 0).sum() + 1) / 10001, ((delta <= 0).sum() + 1) / 10001))
    assert pvalue == expected_summary["primary"]["p_two_sided_sign"]
    report = {
        "status": "PASS",
        "method": "Separate openpyxl reader, independent feature loops, manual CUDA matrix network, independent proposal/bootstrap implementation",
        "source_rows": len(source_rows),
        "eligible_constructs": len(eligible),
        "eligible_utr_clusters": len(clusters),
        "excluded_utr_clusters": len(excluded),
        "sample_role_unique_sequences": role_sizes,
        "max_member_prediction_absolute_difference": max_member_error,
        "max_mean_prediction_absolute_difference": max_mean_error,
        "max_bootstrap_absolute_difference": max_boot_error,
        "bootstrap_column_max_absolute_difference": differences.max(axis=0).tolist(),
        "comparison_rule": "Primary effect and all regret/coverage columns exact; secondary continuous columns <1e-6 for independent CUDA matrix paths",
        "primary_recomputed": float(base[0]),
        "primary_ci95_recomputed": np.percentile(delta, [2.5, 97.5]).tolist(),
        "p_value_recomputed": float(pvalue),
        "gpu_name": torch.cuda.get_device_name(device),
        "cpu_fallback": False,
        "bootstrap_replicates_recomputed": 10000
    }
    (ROOT / "verification" / "round2_independent.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
