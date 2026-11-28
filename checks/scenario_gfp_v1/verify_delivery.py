import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.is_file(), relative
        assert sha256_file(path) == expected["sha256"], relative
        assert path.stat().st_size == expected["bytes"], relative
    round1 = json.loads((root / "verification" / "round1_raw_gate.json").read_text())
    round2 = json.loads((root / "verification" / "round2_independent.json").read_text())
    assert round1["status"] == round2["status"] == "PASS"
    assert round1["selected_high_order_count"] == 16000
    assert round1["pool_count"] == 16
    assert round1["low_order_train_dev_hamming1_crossings"] == 0
    assert round2["low_order_train_dev_hamming1_crossings"] == 0
    assert round2["minimum_possible_low_test_hamming"] >= 3
    correction = json.loads((root / "verification" / "posthoc_hamming_correction.json").read_text())
    assert correction["status"] == "CORRECTION_PASS"
    assert correction["correct_low_order_train_dev_minimum_hamming"] == 2
    assert correction["low_order_to_permanent_test_minimum_hamming"] == 3
    assert round2["independent_pair_comparisons"] == 128
    freeze = json.loads((root / "predictions" / "PREDICTIONS_FROZEN.json").read_text())
    started = json.loads((root / "results" / "EVALUATION_STARTED.json").read_text())
    assert started["evaluation_count"] == 1
    assert freeze["all_cuda_runs_had_allocation"] is True
    assert freeze["cuda_training_runs"] == 20
    assert freeze["test_label_columns_in_learner_view"] == []
    results = json.loads((root / "results" / "results.json").read_text())
    primary = json.loads((root / "results" / "primary.json").read_text())
    pairs = json.loads((root / "results" / "paired_pool_results.json").read_text())
    assert "blind" in results["study_status"] and "not_new_blind_test" in results["study_status"]
    for method in results["methods"]:
        delivered = next(m for m in primary["methods"] if m["method"] == method["method"])
        assert delivered["point_metrics"] == method["point_metrics"]
        rng = np.random.default_rng(20261127)
        for budget_result in method["primary"]:
            budget = budget_result["budget"]
            rows = [r for r in pairs if r["method"] == method["method"] and r["budget"] == budget]
            assert len(rows) == 16
            delta = np.asarray([r["bon_minus_random_regret"] for r in rows])
            assert math.isclose(float(delta.mean()), budget_result["mean_bon_minus_random_regret"], abs_tol=1e-12)
            indices = rng.integers(0, 16, size=(10000, 16))
            ci = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
            assert np.allclose(ci, [budget_result["ci95_bon_minus_random_regret"]["low"], budget_result["ci95_bon_minus_random_regret"]["high"]], atol=1e-12)
    forbidden_suffixes = {".csv", ".npy", ".npz", ".pt"}
    unexpected = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.suffix in forbidden_suffixes]
    assert not unexpected, unexpected
    report = (root / "REPORT.md").read_text()
    assert "回顾性" in report and "不是新盲测" in report
    output = {"status": "PASS", "manifest_file_count": len(manifest["files"]), "round1_raw_gate": "PASS", "round2_independent_recalculation": "PASS", "round3_local_delivery": "PASS", "pair_rows_rechecked": 128, "primary_ci_rechecked": 8, "cuda_training_runs": freeze["cuda_training_runs"], "permanent_test_evaluation_count": 1, "sequence_level_files_present": unexpected, "retrospective_wording_present": True}
    (root / "verification" / "round3_delivery.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
