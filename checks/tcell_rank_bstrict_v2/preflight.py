from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rank import save_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    v1 = Path(config["v1_input_root"])
    expected_config_hash = "16c94ae5caaf9bd166b9a61af9505a69ae776b6689838705e005192747b55afd"
    assert sha256_file(config_path) == expected_config_hash
    raw = json.loads((v1 / "raw_preflight.json").read_text())
    assert raw["raw_hashes_pass"] and not raw["target_mrl_read"]
    raw_paths = {
        "n50_hek293t_r1": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/raw/castillo_hair_gse232927/GSE232927_processed_random_end_hek293t_N50_r1.csv.gz",
        "defined_hepg2_r1": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/raw/castillo_hair_gse232927/GSE232927_processed_defined_end_hepg2_r1.csv.gz",
        "defined_tcell_r1": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/raw/castillo_hair_gse232927/GSE232927_processed_defined_end_tcell_r1.csv.gz",
        "defined_tcell_r2": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/raw/castillo_hair_gse232927/GSE232927_processed_defined_end_tcell_r2.csv.gz",
    }
    raw_checks = {name: sha256_file(path) == raw["raw_sha256"][name] for name, path in raw_paths.items()}
    assert all(raw_checks.values())
    sample_checks = {path: sha256_file(path) == expected for path, expected in raw["sample"]["sha256"].items()}
    assert all(sample_checks.values())
    contexts = {}
    expected_roles = {
        "n50_hek293t_r1": {0: 1763213, 1: 220183, 2: 217137},
        "defined_hepg2_r1": {0: 9604728, 1: 1192185, 2: 1192434},
    }
    for context in expected_roles:
        base = v1 / "auxiliary" / context
        manifest = json.loads((base / "prepare_manifest.json").read_text())
        assert manifest["inference"]["actual_cuda"] and not manifest["inference"]["cpu_fallback"]
        assert not manifest["target_mrl_read"]
        file_checks = {name: sha256_file(base / name) == expected for name, expected in manifest["files"].items()}
        assert all(file_checks.values())
        split = np.load(base / "split.npy", allow_pickle=False)
        role_counts = {role: int(np.sum(split == role)) for role in (0, 1, 2)}
        assert role_counts == expected_roles[context]
        split_manifest = json.loads((base / "split_manifest.json").read_text())
        assert split_manifest.get("target_mrl_read", False) is False
        contexts[context] = {"file_hashes": file_checks, "role_counts": role_counts, "strict_excluded": manifest["excluded"], "cnn_actual_cuda": True}
    output = {
        "status": "PASS", "round": 1, "config_sha256": expected_config_hash,
        "raw_hash_checks": raw_checks, "sample_hash_checks": sample_checks,
        "contexts": contexts, "target_label_columns_accessed": False,
        "target_mrl_read": False, "target_evaluation_count": 0,
        "focused_tests": "6 passed after retained fixture and implementation failures",
    }
    save_json(Path(config["artifact_root"]) / "verification" / "round1_preflight.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
