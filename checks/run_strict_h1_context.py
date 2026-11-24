import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from proxy_function_gap.mechanism import evaluate_mechanism
from proxy_function_gap.trainer.data import sha256_file


def _execute(command, log_path, environment):
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)


def _check_cuda_payload(path, key="metadata"):
    payload = json.loads(path.read_text())
    metadata = payload[key]
    reports = metadata.get("member_gpu", metadata.get("gpu", []))
    if not metadata.get("actual_cuda_inference") or not reports:
        raise ValueError(f"CUDA provenance missing from {path}")
    if any(report.get("actual_device") != "cuda:0" or report.get("cpu_fallback") is not False for report in reports):
        raise ValueError(f"non-CUDA execution reported in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--context", required=True, choices=["n50_hek293t_r1", "defined_hepg2_r1", "defined_tcell_r1", "defined_tcell_r2"])
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    context = args.context
    pool_dir = Path(config["pool_root"]) / context
    validation = json.loads((pool_dir / "validation.json").read_text())
    if not validation["pass"]:
        raise ValueError("strict H1 pools must pass before analysis")
    output = Path(config["analysis_root"]) / context
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = config["cuda"]["cublas_workspace_config"]
    checks = Path(__file__).parent
    common = ["--pool-dir", str(pool_dir), "--context", context, "--count", str(config["pool_count"])]
    selection = output / "selection"
    evaluation = output / "evaluation"
    scores = output / "cnn_scores"
    mechanism = output / "mechanism"
    cross = output / "cross_architecture"
    workers = str(config["cuda"]["pool_workers_per_context"])
    _execute([sys.executable, str(checks / "run_castillo_grids.py"), "--python", sys.executable, *common, "--train-reference", config["train_reference"], "--inference-config", config["inference_config"], "--output-dir", str(selection), "--workers", workers], logs / "cnn_bon_pressure.log", environment)
    _execute([sys.executable, str(checks / "evaluate_castillo_context.py"), *common, "--selection-dir", str(selection), "--output-dir", str(evaluation)], logs / "evaluate_pressure.log", environment)
    _execute([sys.executable, str(checks / "run_mechanism_scores.py"), *common, "--train-reference", config["train_reference"], "--inference-config", config["inference_config"], "--output-dir", str(scores), "--workers", workers], logs / "cnn_candidate_scores.log", environment)
    mechanism.mkdir()
    for index in range(config["pool_count"]):
        pool_id = f"castillo_{context}_{index:03d}"
        evaluate_mechanism(pool_dir, pool_id, scores / f"{pool_id}.json", selection / f"{pool_id}.json", mechanism / f"{pool_id}.json")
    (logs / "evaluate_score_matching.log").write_text(json.dumps({"pools": config["pool_count"], "pass": True}) + "\n")
    _execute([sys.executable, str(checks / "run_cross_arch.py"), *common, "--train-reference", config["train_reference"], "--checkpoints", *config["kmer_checkpoints"], "--cnn-scores-dir", str(scores), "--output-dir", str(cross)], logs / "kmer_common_error_cuda.log", environment)
    for round_index in (1, 2, 3):
        _execute([sys.executable, str(checks / "validate_castillo_evaluation.py"), *common, "--selection-dir", str(selection), "--evaluation-dir", str(evaluation), "--round", str(round_index)], logs / f"pressure_validation_round{round_index}.log", environment)
        _execute([sys.executable, str(checks / "validate_mechanism.py"), *common, "--scores-dir", str(scores), "--selection-dir", str(selection), "--evaluation-dir", str(mechanism), "--round", str(round_index)], logs / f"mechanism_validation_round{round_index}.log", environment)
        _execute([sys.executable, str(checks / "validate_cross_arch.py"), *common, "--cross-dir", str(cross), "--cnn-scores-dir", str(scores), "--round", str(round_index)], logs / f"common_error_validation_round{round_index}.log", environment)
    for index in range(config["pool_count"]):
        pool_id = f"castillo_{context}_{index:03d}"
        _check_cuda_payload(selection / f"{pool_id}.json", "execution_metadata")
        _check_cuda_payload(scores / f"{pool_id}.json")
        _check_cuda_payload(cross / f"{pool_id}_kmer.json")
    report = {
        "pass": True,
        "context": context,
        "pools": config["pool_count"],
        "validation_rounds": 3,
        "actual_cuda_inference": True,
        "cpu_fallback": False,
        "frozen_cnn_config_sha256": sha256_file(config["inference_config"]),
        "frozen_config_sha256": sha256_file(config_path),
        "pool_manifest_sha256": sha256_file(pool_dir / "pool_manifest.json"),
        "artifacts": {name: sha256_file(path / filename) for name, path, filename in (
            ("pressure", evaluation, "evaluation_manifest.json"),
            ("common_error", cross, "summary.json"),
        )},
        "evidence_boundary": config["evidence_boundary"],
        "new_blind_test": False,
    }
    (output / "context_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
