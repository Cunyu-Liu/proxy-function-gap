import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import h5py
import numpy as np
import torch

from .data import STEP_FIELDS, TARGET_FIELDS, attach_predictions, load_design_rows, validate_gate0_rows, write_rows
from .model import load_both_models, predict_both, require_cuda, sha256_file
from .provenance import DATA_SHA256, DEFAULT_MODEL_ROLE_MANIFEST, DESIGNED_DATA, GATE0_SEQUENCE_INDEX, OFFICIAL_LICENSE, OFFICIAL_REPOSITORY, RANDOM_DATA, SCALER, STAGING_ROOT
from .validation import independent_sample_check, validate_prediction_tables, validate_target_info2
from .views import generate_views, permission_negative_tests, validate_views


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _git(*args):
    return subprocess.check_output(["git", "-C", str(STAGING_ROOT), *args], text=True).strip()


def _gpu_driver():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip()
    uuid, name, driver = [part.strip() for part in output.splitlines()[0].split(",", 2)]
    return {"uuid": uuid, "name": name, "driver": driver}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--requested-device", default="cuda:0")
    parser.add_argument("--model-role-manifest", type=Path, default=DEFAULT_MODEL_ROLE_MANIFEST)
    args = parser.parse_args()
    if args.batch_size < 1 or args.samples_per_group < 2:
        raise ValueError("Batch size must be positive and at least two samples are required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    role_path = args.model_role_manifest.resolve()
    role_relative = role_path.relative_to(STAGING_ROOT)
    _git("ls-files", "--error-unmatch", str(role_relative))
    if _git("status", "--porcelain"):
        raise RuntimeError("Staging Git worktree must be clean before a provenance run")
    code_commit = _git("rev-parse", "HEAD")
    started = datetime.now(timezone.utc).isoformat()
    for path, expected in ((DESIGNED_DATA, DATA_SHA256["designed"]), (RANDOM_DATA, DATA_SHA256["random"])):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Data hash mismatch for {path}: {actual}")
    gate0_sha256 = sha256_file(GATE0_SEQUENCE_INDEX)
    device = require_cuda(args.requested_device)
    driver = _gpu_driver()
    models, weight_info = load_both_models(device)
    step_rows, target_rows = load_design_rows()
    gate0_assertions = validate_gate0_rows(step_rows, target_rows)
    step_predictions = predict_both(models, [row["utr"] for row in step_rows], device, args.batch_size)
    target_predictions = predict_both(models, [row["utr"] for row in target_rows], device, args.batch_size)
    step_rows = attach_predictions(step_rows, step_predictions)
    target_rows = attach_predictions(target_rows, target_predictions)
    validation = {
        "round_2_target_info2": validate_target_info2(target_rows),
        "round_2_independent_numpy": independent_sample_check(step_rows, target_rows, args.samples_per_group),
        "round_3_connections": validate_prediction_tables(step_rows, target_rows),
        "round_3_gate0_identity": gate0_assertions,
    }
    provenance = {
        "code_commit": code_commit,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_license": OFFICIAL_LICENSE,
        "weights": weight_info,
        "data": {
            "designed": {"path": str(DESIGNED_DATA), "sha256": DATA_SHA256["designed"]},
            "random": {"path": str(RANDOM_DATA), "sha256": DATA_SHA256["random"]},
            "gate0_sequence_index": {"path": str(GATE0_SEQUENCE_INDEX), "sha256": gate0_sha256},
        },
        "model_role_manifest": {"path": str(role_relative), "sha256": sha256_file(role_path)},
        "scaler": SCALER,
        "requested_device": args.requested_device,
        "actual_device": str(device),
        "input_device": str(device),
        "output_device": str(device),
        "gpu": driver,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "h5py_version": h5py.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "dtype": "float32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "batch_size": args.batch_size,
        "training": False,
        "cpu_fallback": False,
        "counts": {"step": len(step_rows), "target": len(target_rows)},
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".sample-runtime-", dir=args.output_dir.parent))
    try:
        write_rows(temporary / "step_predictions.csv", step_rows, STEP_FIELDS)
        write_rows(temporary / "target_predictions.csv", target_rows, TARGET_FIELDS)
        generate_views(temporary, step_rows, target_rows, role_path)
        validation["round_3_view_permissions"] = validate_views(temporary, step_rows, target_rows, role_path)
        validation["round_3_permission_negative_tests"] = permission_negative_tests()
        _json_write(temporary / "provenance.json", provenance)
        _json_write(temporary / "validation.json", validation)
        names = sorted(
            str(path.relative_to(temporary)) for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"artifact_hashes.json", "run_metadata.json"}
        )
        hashes = {name: sha256_file(temporary / name) for name in names}
        _json_write(temporary / "artifact_hashes.json", hashes)
        _json_write(temporary / "run_metadata.json", {
            "code_commit": code_commit,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "output_dir": str(args.output_dir),
            "requested_device": args.requested_device,
            "actual_device": str(device),
            "artifact_hashes": hashes,
        })
        temporary.rename(args.output_dir)
    except Exception:
        shutil.rmtree(temporary)
        raise
    print(json.dumps({"output_dir": str(args.output_dir), "code_commit": code_commit, "requested_device": args.requested_device, "actual_device": str(device), "gpu": driver, "artifact_hashes": hashes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
