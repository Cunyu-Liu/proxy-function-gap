import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from gate0_schlusser2024.acquire import acquire, download_ranged
from gate0_schlusser2024.build import build, hamming1
from gate0_schlusser2024.common import file_record, is_missing, normalize_sequence, terminal_50
from gate0_schlusser2024.verify import verify


def test_sequence_normalization_and_terminal_input():
    assert normalize_sequence(" acguN ") == "ACGTN"
    assert normalize_sequence("ACGX") is None
    assert is_missing("NaN") and is_missing("") and not is_missing("0")
    assert terminal_50("U" * 49) is None
    assert terminal_50("A" + "U" * 50) == "T" * 50
    assert terminal_50("N" + "A" * 49) is None
    assert hamming1("A" * 50, "A" * 49 + "C")
    assert not hamming1("A" * 50, "C" * 50)


def write_tsv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_config(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    sample = tmp_path / "train.tsv.gz"
    with gzip.open(sample, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utr50"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({"utr50": "A" * 50})
    sequence = "C" * 50
    write_tsv(raw / "HepG2_TE.tsv", ["transcript", "UTR", "TE", "model_prediction"], [
        {"transcript": "included", "UTR": sequence, "TE": "1.25", "model_prediction": "9.9"},
        {"transcript": "short", "UTR": "C" * 49, "TE": "2.5", "model_prediction": "8.8"},
        {"transcript": "exact", "UTR": "A" * 50, "TE": "3.5", "model_prediction": "7.7"},
        {"transcript": "hamming", "UTR": "A" * 49 + "C", "TE": "4.5", "model_prediction": "6.6"},
    ])
    definition = {"file": "HepG2_TE.tsv", "system": "endogenous_mRNA", "endpoint": "TE", "role": "frozen_confirmatory_study_level_OOD", "reason": "test"}
    return {
        "dataset": "synthetic-schlusser-test",
        "raw_root": str(raw),
        "derived_root": str(tmp_path / "derived"),
        "upstream_commit": "test",
        "article_doi": "test",
        "article_pmcid": "test",
        "licenses": {},
        "studies": {"hepg2_te": definition},
        "field_mapping": {"hepg2_te": {"sequence": "UTR", "measured": ["TE"], "predicted": ["model_prediction"], "identifiers": ["transcript"]}},
        "frozen_outer": {"study": "hepg2_te", "allocation_unit": "entire_study", "outer_metrics_computed": False, "labels_returned_to_learner": False},
        "sample_role_paths": [str(sample)],
        "zenodo_files": [],
    }


def test_build_and_three_verification_rounds_keep_labels_isolated(tmp_path):
    config = synthetic_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    summary = build(config_path)
    assert summary["outer_included_rows"] == 1
    derived = tmp_path / "derived"
    with gzip.open(derived / "learner" / "hepg2_te_sequences_isolated.tsv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["source_id"] == "included"
    assert "TE" not in rows[0] and "model_prediction" not in rows[0]
    with gzip.open(derived / "evaluator" / "hepg2_te_labels.tsv.gz", "rt", newline="") as handle:
        vault = list(csv.DictReader(handle, delimiter="\t"))
    assert len(vault) == 4 and vault[0]["TE"] == "1.25"
    for round_number in (1, 2, 3):
        assert verify(config_path, round_number)["pass"]


def test_ranged_download_uses_chunk_specific_resume_directory(tmp_path, monkeypatch):
    payload = b"abcdefghij"
    spec = {"name": "table.tsv", "bytes": len(payload), "md5": hashlib.md5(payload).hexdigest(), "url": "https://example.invalid"}
    calls = []

    def fake_request(url, byte_range=None, timeout=180, attempts=10):
        calls.append(byte_range)
        start, end = byte_range
        return payload[start:end + 1]

    monkeypatch.setattr("gate0_schlusser2024.acquire.request_bytes", fake_request)
    destination = tmp_path / "table.tsv"
    record = download_ranged(spec, destination, 4, 3)
    assert destination.read_bytes() == payload
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert calls == [(0, 3), (4, 7), (8, 9)]
    assert not (tmp_path / "table.tsv.parts_4").exists()


def test_acquisition_progress_preserves_verified_file_names(tmp_path, monkeypatch):
    payload = b"sequence-level-table\n"
    spec = {"name": "HepG2_TE.tsv", "study": "hepg2_te", "bytes": len(payload), "md5": "published-md5", "url": "https://example.invalid"}
    config = {"raw_root": str(tmp_path), "zenodo_files": [spec], "chunk_bytes": 4, "download_workers": 1, "upstream_commit": "test", "licenses": {}}
    config_path = tmp_path / "acquire.json"
    config_path.write_text(json.dumps(config))

    def fake_download(spec, destination, chunk_bytes, workers):
        destination.write_bytes(payload)
        return file_record(destination, {"md5": spec["md5"]})

    monkeypatch.setattr("gate0_schlusser2024.acquire.download_ranged", fake_download)
    monkeypatch.setattr("gate0_schlusser2024.acquire.md5_file", lambda path: "published-md5")
    manifest = acquire(config_path)
    assert manifest["all_zenodo_md5_verified"]
    progress = json.loads((tmp_path / "download_progress.json").read_text())
    assert progress["pending"] == []
    assert progress["complete"][0]["name"] == "HepG2_TE.tsv"


def test_verifier_rejects_changed_vault_without_revealing_labels(tmp_path):
    config = synthetic_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    build(config_path)
    vault = tmp_path / "derived" / "evaluator" / "hepg2_te_labels.tsv.gz"
    vault.write_bytes(b"corrupted")
    with pytest.raises(Exception):
        verify(config_path, 3)
    first = tmp_path / "derived" / "verification_round_3.json"
    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    with pytest.raises(Exception):
        verify(config_path, 3)
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_hash
    assert (tmp_path / "derived" / "verification_round_3_attempt_2.json").exists()
