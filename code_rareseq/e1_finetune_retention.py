#!/usr/bin/env python3
"""Retain auditable checkpoint proofs and clean only accepted non-representative weights."""
import argparse
import json
import os
import time

import pandas as pd

from e1_finetune import atomic_json, atomic_marker, load_checkpoint_compressed
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file

APPROVED_ROOT = "/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rnafm_finetune_strict3train_p20"


def representative_tasks(run, validation):
    if os.path.basename(run).startswith("pilot_cuda_"):
        return set(zip(validation.experiment, validation.track)), "pilot_permanent"
    config = json.load(open(os.path.join(run, "finetune_config.json")))
    args = config["args"]
    if args["folds"] == "0" and int(args["seed"]) == 0:
        experiments = []
        for tier in ("head", "mid", "tail"):
            subset = validation[validation.tier == tier]
            if subset.empty:
                raise ValueError(f"Missing tier for representative selection: {tier}")
            experiments.append(min(subset.experiment.unique()))
        selected = validation[validation.experiment.isin(experiments)]
        return set(zip(selected.experiment, selected.track)), "fold0_seed0_tier_lexicographic_permanent"
    return set(), "matrix_proof_only"


def proof_hashes(run, directory):
    test_dir = os.path.join(run, "test_tasks", os.path.basename(directory))
    paths = {"history": os.path.join(directory, "history.jsonl"),
             "validation_predictions": os.path.join(directory, "validation_predictions.npz"),
             "test_predictions": os.path.join(test_dir, "test_predictions.npz"),
             "test_metadata": os.path.join(test_dir, "metadata.json"),
             "finetune_config": os.path.join(run, "finetune_config.json"),
             "finetune_eval_config": os.path.join(run, "finetune_eval_config.json"),
             "pre_retention_verification": os.path.join(run, "pre_retention_verification.json")}
    return {name: {"path": os.path.relpath(path, run), "sha256": sha256_file(path),
                   "bytes": os.path.getsize(path)} for name, path in paths.items()}


def audited_checkpoint_path(run, directory, manifest):
    run_real = os.path.realpath(run)
    approved_real = os.path.realpath(APPROVED_ROOT)
    if os.path.commonpath([run_real, approved_real]) != approved_real:
        raise ValueError("Run is outside the approved new-artifact root")
    if os.path.islink(directory) or os.path.realpath(os.path.dirname(directory)) != os.path.join(run_real, "tasks"):
        raise ValueError("Task directory failed direct-parent/non-symlink audit")
    if os.path.commonpath([os.path.realpath(directory), approved_real]) != approved_real:
        raise ValueError("Task realpath escaped the approved artifact root")
    if manifest["checkpoint"] != "best.pt.zlib":
        raise ValueError("Refusing to clean a nonstandard checkpoint name")
    checkpoint = os.path.join(directory, "best.pt.zlib")
    if os.path.islink(checkpoint) or not os.path.isfile(checkpoint):
        raise ValueError("Checkpoint must be a regular non-symlink file")
    if os.path.realpath(checkpoint) != os.path.join(os.path.realpath(directory), "best.pt.zlib"):
        raise ValueError("Checkpoint realpath audit failed")
    if sha256_file(checkpoint) != manifest["compressed_sha256"]:
        raise ValueError("Checkpoint SHA differs before cleanup")
    return checkpoint


def retain_run(run, verification, selection=None, representative_file=None):
    report = json.load(open(verification))
    if report.get("status") != "passed" or not report.get("require_eval"):
        raise ValueError("Independent pre-retention evaluation verification is required")
    if len(report.get("runs", [])) != 1 or not report["runs"][0].get("eval_passed"):
        raise ValueError("Retention requires one independently passed run")
    if os.path.realpath(report["runs"][0]["run"]) != os.path.realpath(run):
        raise ValueError("Verification report run mismatch")
    if not os.path.isfile(os.path.join(run, ".train_complete")) or not os.path.isfile(os.path.join(run, ".eval_complete")):
        raise ValueError("Incomplete train/eval markers")
    config = json.load(open(os.path.join(run, "finetune_config.json")))
    args = config["args"]
    if args["mode"] == "lora":
        if not selection or int(json.load(open(selection))["selected_rank"]) != int(args["lora_r"]):
            raise ValueError("LoRA retention requires the matching frozen rank selection")
    validation = pd.read_csv(os.path.join(run, "finetune_validation_results.tsv"), sep="\t")
    if representative_file:
        representative_doc = json.load(open(representative_file))
        representatives = {(row["experiment"], row["track"]) for row in representative_doc["tasks"]}
        representative_policy = representative_doc["policy"]
    else:
        representatives, representative_policy = representative_tasks(run, validation)
    counts = {"retained_full_checkpoint": 0, "retained_proof_only": 0}
    bytes_cleaned = 0
    for directory in sorted(os.path.join(run, "tasks", name) for name in os.listdir(os.path.join(run, "tasks"))):
        metadata = json.load(open(os.path.join(directory, "metadata.json")))
        manifest = json.load(open(os.path.join(directory, "checkpoint_manifest.json")))
        retention_path = os.path.join(directory, "retention.json")
        if os.path.isfile(retention_path):
            previous = json.load(open(retention_path))
            if previous["status"] == "retained_proof_only":
                if os.path.exists(os.path.join(directory, "best.pt.zlib")):
                    raise ValueError("Proof-only task still has encoder weights")
                counts["retained_proof_only"] += 1
                continue
            if previous["status"] == "retained_full_checkpoint":
                if not os.path.isfile(os.path.join(directory, "best.pt.zlib")):
                    raise ValueError("Permanent checkpoint is missing")
                counts["retained_full_checkpoint"] += 1
                continue
        test_metadata = json.load(open(os.path.join(run, "test_tasks", os.path.basename(directory), "metadata.json")))
        if not test_metadata.get("checkpoint_validation_reproduced") or not test_metadata.get("checkpoint_used_for_test"):
            raise ValueError("Checkpoint validation/test proof is incomplete")
        key = (metadata["experiment"], metadata["track"])
        permanent = key in representatives
        checkpoint = os.path.join(directory, "best.pt.zlib")
        if not os.path.isfile(checkpoint):
            plan_path = os.path.join(directory, "retention_plan.json")
            if permanent or not os.path.isfile(plan_path):
                raise ValueError("Missing checkpoint without an approved cleanup plan")
            plan = json.load(open(plan_path))
            if plan["compressed_sha256"] != manifest["compressed_sha256"]:
                raise ValueError("Cleanup plan SHA mismatch")
        else:
            audited_checkpoint_path(run, directory, manifest)
        recovered = load_checkpoint_compressed(directory) if os.path.isfile(checkpoint) else None
        head_path = os.path.join(directory, "head.pt")
        if recovered is not None:
            import torch
            tmp = head_path + ".tmp"
            torch.save({"head_state": recovered["head_state"],
                        "task": {key: metadata[key] for key in ("experiment", "fold", "track")},
                        "source_checkpoint_raw_sha256": manifest["raw_sha256"],
                        "scaler": "none_mean_masked_pooling"}, tmp)
            os.replace(tmp, head_path)
            del recovered
        if not os.path.isfile(head_path):
            raise ValueError("Small head retention failed")
        proof = {"status": "retained_full_checkpoint" if permanent else "retained_proof_only",
                 "retention_policy": representative_policy,
                 "task": {key: metadata[key] for key in ("experiment", "rbp", "tier", "fold", "track")},
                 "mode": args["mode"], "seed": args["seed"], "lora_r": args["lora_r"],
                 "checkpoint_format": manifest["format"], "raw_sha256": manifest["raw_sha256"],
                 "compressed_sha256": manifest["compressed_sha256"],
                 "raw_bytes": manifest["raw_bytes"], "compressed_bytes": manifest["compressed_bytes"],
                 "compression_roundtrip_verified": manifest["roundtrip_sha256_verified"],
                 "bitwise_recovery_verified_at_save": True,
                 "checkpoint_validation_reproduced": True, "checkpoint_used_for_test": True,
                 "head": {"path": "head.pt", "sha256": sha256_file(head_path),
                          "bytes": os.path.getsize(head_path)},
                 "scaler": "none_mean_masked_pooling", "artifact_hashes": proof_hashes(run, directory)}
        if permanent:
            atomic_json(retention_path, proof)
            counts["retained_full_checkpoint"] += 1
            continue
        if not os.path.isfile(checkpoint):
            proof["cleaned_at"] = plan["planned_at"]
            proof["cleaned_bytes"] = plan["compressed_bytes"]
            proof["cleaned_file"] = "best.pt.zlib"
            atomic_json(retention_path, proof)
            counts["retained_proof_only"] += 1
            continue
        plan = {**proof, "status": "approved_for_single_file_cleanup",
                "cleanup_path": checkpoint, "cleanup_file": "best.pt.zlib",
                "cleanup_sha256": manifest["compressed_sha256"], "planned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        atomic_json(os.path.join(directory, "retention_plan.json"), plan)
        audited_checkpoint_path(run, directory, manifest)
        size = os.path.getsize(checkpoint)
        os.remove(checkpoint)
        proof["cleaned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        proof["cleaned_bytes"] = size
        proof["cleaned_file"] = "best.pt.zlib"
        atomic_json(retention_path, proof)
        counts["retained_proof_only"] += 1
        bytes_cleaned += size
    atomic_marker(os.path.join(run, ".retention_complete"))
    return {**counts, "bytes_cleaned": bytes_cleaned, "representative_policy": representative_policy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--representative-file")
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    lg = RunLogger(args.log, stage="E1", kind="finetune-artifact-retention")
    lg.start(config=vars(args), data_version="e0-v2-id90-strict3train-main",
             inputs=[args.verification], code_version=sha256_file(os.path.abspath(__file__)))
    try:
        result = retain_run(os.path.abspath(args.run), os.path.abspath(args.verification), args.selection,
                            args.representative_file)
        lg.finish(metrics=result, outputs=[args.run])
        print("E1_FINETUNE_RETENTION_DONE " + json.dumps(result))
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.run])
        raise


if __name__ == "__main__":
    main()
