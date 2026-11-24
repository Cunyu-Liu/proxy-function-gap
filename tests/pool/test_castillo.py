import csv
import gzip
import json

import numpy as np
import pytest

from proxy_function_gap.castillo import _write_jsonl, build_castillo_context, evaluate_castillo_pool, load_castillo_pool
from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash
from validate_castillo_pools import validate_castillo_pools


def test_castillo_pool_builder_and_evaluator(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    sequences = ["A" * 50, "C" * 50, "G" * 50, "T" * 50, "AC" * 25, "GT" * 25]
    with gzip.open(raw, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["UTR", "rl"])
        writer.writeheader()
        for index, sequence in enumerate(sequences):
            writer.writerow({"UTR": sequence, "rl": index + 1})
    sample = tmp_path / "sample.tsv.gz"
    with gzip.open(sample, "wt", newline="") as handle:
        handle.write("utr50\n")
    train = tmp_path / "train.npz"
    np.savez(train, utr50=np.asarray(["A" * 50]))
    output = tmp_path / "context"
    manifest = build_castillo_context(raw, "test", [sample], train, output, pool_count=1, pool_size=2)
    pool_id = next(iter(manifest["pool_hashes"]))
    pool = load_castillo_pool(output, pool_id, train)
    assert len(pool) == 2 and pool.measured is None
    assert all("measured_rl" not in row for row in [json.loads(line) for line in gzip.open(output / "learner" / f"{pool_id}.jsonl.gz", "rt")])
    record = SelectionRecord(pool_id=pool_id, strategy="bon", alpha=0, proposal_cap=1, seed=101, selected_index=0, selected_id=pool.sequence_ids[0], regret=None, proxy_item_forwards=1, ensemble_member_item_forwards=3, member_item_forwards={11: 1, 23: 1, 37: 1}, proposals=1, abstain=False)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_config_sha256": selection_config_hash([record]), "execution_metadata": {"outer_labels_read": False}, "records": [record.to_dict()]}))
    result = evaluate_castillo_pool(output, pool_id, selection, tmp_path / "evaluation.json")
    assert result["labels_returned_to_learner"] is False
    evaluated = json.loads((tmp_path / "evaluation.json").read_text())["records"][0]
    assert evaluated["regret"] >= 0


def test_right_half_hamming_one_is_blocked():
    from collections import defaultdict
    from proxy_function_gap.castillo import _near_existing, _add_half_index
    first = "CAAGAGGCAGGGCGACGGGCGGGATGAACCGGGAGGGCTGGGGATCAACA"
    second = "CAAGAGGCAGGGCGACGGGCGGCATGAACCGGGAGGGCTGGGGATCAACA"
    index = defaultdict(list)
    _add_half_index(first, index)
    assert _near_existing(second, index)


def _write_raw_rows(path, rows):
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["UTR", "rl"])
        writer.writeheader()
        writer.writerows(rows)


def _write_sample_sequences(path, sequences):
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utr50"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({"utr50": sequence} for sequence in sequences)


def test_exact_duplicates_are_aggregated_before_selection(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    repeated = "A" * 50
    _write_raw_rows(raw, [{"UTR": repeated, "rl": 1}, {"UTR": repeated, "rl": 3}, {"UTR": "C" * 50, "rl": 2}])
    sample = tmp_path / "sample.tsv.gz"
    _write_sample_sequences(sample, [])
    train = tmp_path / "train.npz"
    np.savez(train, utr50=np.asarray([repeated]))
    output = tmp_path / "context"
    manifest = build_castillo_context(raw, "test", [sample], train, output, pool_count=1, pool_size=2)
    pool_id = next(iter(manifest["pool_hashes"]))
    pool = load_castillo_pool(output, pool_id, train)
    assert len(pool.utr50) == len(set(pool.utr50)) == 2
    with gzip.open(output / "evaluator" / "labels.tsv.gz", "rt", newline="") as handle:
        labels = {row["utr50"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
    assert labels[repeated] == 2


def test_validator_rejects_exact_duplicates(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    _write_raw_rows(raw, [{"UTR": "A" * 50, "rl": 1}, {"UTR": "C" * 50, "rl": 2}])
    sample = tmp_path / "sample.tsv.gz"
    _write_sample_sequences(sample, [])
    train = tmp_path / "train.npz"
    np.savez(train, utr50=np.asarray(["A" * 50]))
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_castillo_context(raw, "test", [sample], train, first, pool_count=1, pool_size=2)
    build_castillo_context(raw, "test", [sample], train, second, pool_count=1, pool_size=2)
    pool_id = next(iter(manifest["pool_hashes"]))
    for directory in (first, second):
        path = directory / "learner" / f"{pool_id}.jsonl.gz"
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows[1] = rows[0]
        _write_jsonl(path, rows)
    report = validate_castillo_pools(first, second, "test", [sample], count=1, pool_size=2)
    assert not report["pass"]
    assert f"exact_duplicate:{pool_id}" in report["errors"]
    assert "global_exact_duplicate" in report["errors"]


def test_sample_hamming_one_candidate_is_rejected(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    sample_sequence = "A" * 50
    _write_raw_rows(raw, [{"UTR": "A" * 49 + "C", "rl": 1}])
    sample = tmp_path / "sample.tsv.gz"
    _write_sample_sequences(sample, [sample_sequence])
    train = tmp_path / "train.npz"
    np.savez(train, utr50=np.asarray([sample_sequence]))
    output = tmp_path / "context"
    with pytest.raises(ValueError, match="not enough independent"):
        build_castillo_context(raw, "test", [sample], train, output, pool_count=1, pool_size=1)
    assert not output.exists()


def test_validator_rejects_cross_study_hamming_one(tmp_path):
    raw = tmp_path / "raw.csv.gz"
    _write_raw_rows(raw, [{"UTR": "A" * 50, "rl": 1}, {"UTR": "C" * 50, "rl": 2}])
    sample = tmp_path / "sample.tsv.gz"
    _write_sample_sequences(sample, [])
    train = tmp_path / "train.npz"
    np.savez(train, utr50=np.asarray(["A" * 50]))
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_castillo_context(raw, "test", [sample], train, first, pool_count=1, pool_size=2)
    build_castillo_context(raw, "test", [sample], train, second, pool_count=1, pool_size=2)
    pool_id = next(iter(manifest["pool_hashes"]))
    pool = load_castillo_pool(first, pool_id, train)
    sequence = pool.utr50[0]
    conflict = sequence[:-1] + ("C" if sequence[-1] != "C" else "G")
    _write_sample_sequences(sample, [conflict])
    report = validate_castillo_pools(first, second, "test", [sample], count=1, pool_size=2)
    assert not report["pass"]
    assert "cross_study_hamming1" in report["errors"]
