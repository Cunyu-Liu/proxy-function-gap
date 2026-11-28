import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "checks" / "scenario_mrna_non_tcell.py"
SPEC = importlib.util.spec_from_file_location("scenario_mrna_non_tcell", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_kmer_features_length_and_composition():
    features = MODULE.kmer_features("AAAATTTT")
    assert features.shape == (259,)
    assert features.dtype == np.float32
    assert np.isclose(features[:256].sum(), 1.0)
    assert features[0] == 0.2
    assert features[255] == 0.2
    assert features[256] == 0.0
    assert features[257] == 0.0
    assert features[258] == 0.08


def test_best_index_deterministic_tie():
    assert MODULE.best_index([1.0, 2.0, 2.0], ["seq_03", "seq_02", "seq_01"]) == 2
    assert MODULE.best_index([1.0, 2.0, 2.0], ["seq_03", "seq_02", "seq_01"], [0, 1]) == 1


def test_top_tail_perfect_and_reversed():
    ids = [f"seq_{i:02d}" for i in range(10)]
    perfect = MODULE.top_tail_metrics(np.arange(10), np.arange(10), ids)
    reversed_ = MODULE.top_tail_metrics(np.arange(10), np.arange(10)[::-1], ids)
    assert perfect["k"] == 1
    assert perfect["top_decile_recall"] == 1.0
    assert perfect["top_tail_regret"] == 0.0
    assert reversed_["top_decile_recall"] == 0.0
    assert reversed_["top_tail_regret"] == 9.0


def test_original_cap1_identity_and_equal_budget_sign():
    ids = [f"seq_{i:02d}" for i in range(16)]
    scores = np.arange(16, dtype=float)
    endpoints = {"primary|test": np.arange(16, dtype=float)}
    rows = MODULE.run_original_pool("test", 99, ids, scores, endpoints, [101, 202])
    assert len(rows) == 10
    for row in rows:
        if row["cap"] == 1:
            assert row["bon_selected_id"] == row["random_n_selected_id"] == row["random1_selected_id"]
            assert row["bon_minus_random_n"] == 0.0
        assert row["bon_regret"] >= 0.0
        assert row["random_n_regret"] >= 0.0
        assert np.isclose(row["bon_minus_random_n"], row["bon_regret"] - row["random_n_regret"])
    assert all(row["bon_regret"] == 0.0 for row in rows if row["cap"] == 16)


def test_random_n_reproducible_and_bounded():
    values = [MODULE.random_n_position(1, 7, 3, 8) for _ in range(5)]
    assert values == [values[0]] * 5
    assert 0 <= values[0] < 8
    assert 0 <= MODULE.random_n_position(1, 7, 3, 8, 2) < 8


def test_bootstrap_sign_interval_deterministic():
    ci, p = MODULE.interval_and_p(np.zeros(100))
    assert ci == [0.0, 0.0]
    assert 0.0 < p <= 1.0


def test_n50_73nt_terminal_variable_region_is_audited():
    variable = "A" * 49 + "C"
    panel = pd.DataFrame([{"sequence_id": "seq_01", "utr": "G" * 23 + variable, "description": "N50, DEN"}])
    conflicts = MODULE.conflict_audit(panel, {"train": {variable}, "dev": set(), "calibration": set()})
    assert conflicts["seq_01"][0]["version"] == "n50_variable_region"
    assert conflicts["seq_01"][0]["type"] == "exact"


def test_panel_missing_roles_not_imputed(tmp_path):
    protocol = {
        "raw": {"editing": {"path": str(tmp_path / "panel.xlsx"), "sheet_panel": "5'UTR sequences"}},
    }
    frame = pd.DataFrame({
        "Sequence ID": range(1, 49),
        "5'UTR": ["ACGTACGT"] * 48,
        "Measured MRL": [1.0] * 12 + [np.nan] * 36,
        "Predicted MRL": [1.0] * 45 + [np.nan] * 3,
        "Description": ["x"] * 48,
    })
    with pd.ExcelWriter(protocol["raw"]["editing"]["path"], engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="5'UTR sequences", index=False)
    result = MODULE.read_panel(protocol)
    assert result["measured_mrl"].notna().sum() == 12
    assert result["published_predicted_mrl"].notna().sum() == 45
    assert result["measured_mrl"].isna().sum() == 36
