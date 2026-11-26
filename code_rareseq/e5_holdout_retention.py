#!/usr/bin/env python3
"""Plan, apply, and independently verify E5 non-representative checkpoint retention."""
import argparse
import contextlib
import gzip
import hashlib
import json
import os
import shutil
import stat
import time

HOLDOUTS = "HNRNPC,PTBP1,U2AF2,SRSF1,RBFOX2,QKI,IGF2BP1,LIN28B,TAF15,UPF1,CSTF2T,YBX3".split(",")
TRACKS = ("neg1c", "neg2")
SEEDS = (0, 1, 2)
DEFAULT_RUN_ROOT = "/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5/holdout_convergence"
DEFAULT_SUMMARY_DIR = "/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5/holdout_convergence_summary"
POLICY = "one_lexicographically_first_holdout_per_seed_track_retained_full__others_gzip9_bitwise_recovery"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value, path):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def file_record(path):
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"Not a regular non-symlink file: {path}")
    return {"path": os.path.abspath(path), "bytes": info.st_size, "sha256": sha256_file(path),
            "device": info.st_dev, "inode": info.st_ino}


def require_direct_file(path, parent):
    path = os.path.abspath(path)
    parent = os.path.realpath(parent)
    if os.path.islink(path) or os.path.realpath(os.path.dirname(path)) != parent:
        raise ValueError(f"Direct-parent/non-symlink audit failed: {path}")
    if os.path.realpath(path) != os.path.join(parent, os.path.basename(path)):
        raise ValueError(f"Realpath audit failed: {path}")
    return path


def require_within(path, root):
    if os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) != os.path.realpath(root):
        raise ValueError(f"Path escapes run root: {path}")


def checkpoint_paths(run_root, seed, track, holdout):
    run_dir = os.path.join(run_root, f"holdout_convergence_s{seed}", track)
    prefix = f"{holdout}_{track}"
    return {
        "checkpoint": require_direct_file(os.path.join(run_dir, "checkpoints", prefix + ".pt"),
                                          os.path.join(run_dir, "checkpoints")),
        "manifest": require_direct_file(os.path.join(run_dir, "manifests", prefix + ".json"),
                                        os.path.join(run_dir, "manifests")),
        "history": require_direct_file(os.path.join(run_dir, "history", prefix + ".tsv"),
                                       os.path.join(run_dir, "history")),
        "pair": require_direct_file(os.path.join(run_dir, "pairs", prefix + ".tsv"),
                                    os.path.join(run_dir, "pairs")),
        "scores": require_direct_file(os.path.join(run_dir, "scores", prefix + ".npz"),
                                      os.path.join(run_dir, "scores")),
    }


def verify_torch_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint.get("state_dict"):
        raise ValueError(f"Invalid checkpoint state: {path}")
    for key, value in checkpoint["state_dict"].items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint state {key}: {path}")
    return True


def decompress_exact(compressed, recovered, expected_sha256):
    require_direct_file(compressed, os.path.dirname(compressed))
    with gzip.open(compressed, "rb") as source, open(recovered, "xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if sha256_file(recovered) != expected_sha256:
        raise ValueError(f"Bitwise recovery SHA mismatch: {compressed}")


@contextlib.contextmanager
def materialize_checkpoint(checkpoint_path, expected_sha256=None):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        actual = sha256_file(checkpoint_path)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(f"Checkpoint SHA mismatch: {checkpoint_path}")
        yield checkpoint_path
        return
    proof_path = checkpoint_path[:-3] + ".retention.json"
    with open(proof_path) as handle:
        proof = json.load(handle)
    compressed = checkpoint_path + ".gz"
    if proof.get("status") != "compressed_original_removed" or proof.get("original", {}).get("path") != checkpoint_path:
        raise ValueError(f"Missing valid retention proof: {checkpoint_path}")
    if proof.get("compressed", {}).get("path") != compressed or not proof.get("bitwise_recovery_verified"):
        raise ValueError(f"Invalid compressed retention proof: {checkpoint_path}")
    if sha256_file(compressed) != proof["compressed"]["sha256"]:
        raise ValueError(f"Compressed checkpoint SHA mismatch: {compressed}")
    raw_sha256 = proof["original"]["sha256"]
    if expected_sha256 is not None and raw_sha256 != expected_sha256:
        raise ValueError(f"Retained raw SHA differs from manifest: {checkpoint_path}")
    recovered = checkpoint_path + f".recovery.{os.getpid()}.tmp"
    try:
        decompress_exact(compressed, recovered, raw_sha256)
        yield recovered
    finally:
        if os.path.exists(recovered):
            os.remove(recovered)


def build_plan(run_root, summary_dir, audit_dir):
    run_root = os.path.abspath(run_root)
    summary_dir = os.path.abspath(summary_dir)
    audit_dir = os.path.abspath(audit_dir)
    require_within(audit_dir, run_root)
    evidence = {}
    for name in ("verification.json", "independent_verification.json"):
        path = require_direct_file(os.path.join(summary_dir, name), summary_dir)
        with open(path) as handle:
            report = json.load(handle)
        if report.get("status") != "complete" or report.get("n_pairs_completed") != 72 or report.get("n_pairs_expected") != 72:
            raise ValueError(f"Final 72/72 acceptance is required: {path}")
        evidence[name] = file_record(path)
    files = []
    for seed in SEEDS:
        for track in TRACKS:
            representative = min(HOLDOUTS)
            for holdout in HOLDOUTS:
                paths = checkpoint_paths(run_root, seed, track, holdout)
                for path in paths.values():
                    require_within(path, run_root)
                with open(paths["manifest"]) as handle:
                    manifest = json.load(handle)
                if manifest["heldout"] != holdout or manifest["track"] != track or int(manifest["seed"]) != seed:
                    raise ValueError(f"Manifest identity mismatch: {paths['manifest']}")
                records = {name: file_record(path) for name, path in paths.items()}
                for name, key in (("pair", "pair_sha256"), ("history", "history_sha256"),
                                  ("checkpoint", "checkpoint_sha256"), ("scores", "scores_sha256")):
                    if records[name]["sha256"] != manifest["artifacts"][key]:
                        raise ValueError(f"Manifest artifact SHA mismatch: {paths[name]}")
                verify_torch_checkpoint(paths["checkpoint"])
                action = "retain_full" if holdout == representative else "gzip9_verify_then_remove_original"
                files.append({"seed": seed, "track": track, "holdout": holdout, "action": action,
                              "representative": action == "retain_full", "artifacts": records})
    if len(files) != 72 or sum(row["representative"] for row in files) != 6:
        raise ValueError("Retention plan must contain exactly 72 tasks and six representatives")
    plan = {"schema_version": 1, "status": "planned", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_root": run_root, "summary_dir": summary_dir, "policy": POLICY,
            "final_acceptance_evidence": evidence, "task_count": 72, "representative_count": 6,
            "nonrepresentative_count": 66, "files": files}
    os.makedirs(audit_dir, exist_ok=True)
    plan_path = os.path.join(audit_dir, "plan.json")
    if os.path.exists(plan_path):
        raise FileExistsError(plan_path)
    atomic_json(plan, plan_path)
    return plan_path


def load_plan(plan_path, run_root):
    with open(plan_path) as handle:
        plan = json.load(handle)
    if plan.get("status") != "planned" or plan.get("policy") != POLICY:
        raise ValueError("Unrecognized retention plan")
    if os.path.realpath(plan["run_root"]) != os.path.realpath(run_root):
        raise ValueError("Plan/run-root mismatch")
    if len(plan.get("files", [])) != 72:
        raise ValueError("Plan does not contain 72 tasks")
    for name, record in plan["final_acceptance_evidence"].items():
        if file_record(record["path"])["sha256"] != record["sha256"]:
            raise ValueError(f"Final acceptance evidence changed: {name}")
    for row in plan["files"]:
        expected_paths = checkpoint_paths(run_root, row["seed"], row["track"], row["holdout"])
        for name, path in expected_paths.items():
            record = row["artifacts"][name]
            if record["path"] != path:
                raise ValueError(f"Plan path mismatch: {path}")
            require_within(path, run_root)
            if name != "checkpoint" or row["action"] == "retain_full":
                if file_record(path)["sha256"] != record["sha256"]:
                    raise ValueError(f"Plan artifact SHA mismatch: {path}")
    return plan


def apply_plan(plan_path, run_root, audit_dir):
    plan = load_plan(plan_path, run_root)
    plan_sha256 = sha256_file(plan_path)
    for row in plan["files"]:
        checkpoint = row["artifacts"]["checkpoint"]["path"]
        if row["action"] == "retain_full":
            if sha256_file(checkpoint) != row["artifacts"]["checkpoint"]["sha256"]:
                raise ValueError(f"Representative checkpoint changed: {checkpoint}")
            proof = {"status": "retained_full", "policy": POLICY, "plan_sha256": plan_sha256,
                     "task": {key: row[key] for key in ("seed", "track", "holdout")},
                     "original": row["artifacts"]["checkpoint"], "torch_load_verified": True,
                     "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            atomic_json(proof, checkpoint[:-3] + ".retention.json")
            continue
        if row["action"] != "gzip9_verify_then_remove_original":
            raise ValueError(f"Unknown action: {row['action']}")
        original_record = row["artifacts"]["checkpoint"]
        current = file_record(checkpoint)
        for key in ("sha256", "bytes", "device", "inode"):
            if current[key] != original_record[key]:
                raise ValueError(f"Checkpoint identity changed before compression: {checkpoint}")
        compressed = checkpoint + ".gz"
        proof_path = checkpoint[:-3] + ".retention.json"
        if os.path.exists(compressed) or os.path.exists(proof_path):
            raise FileExistsError(compressed)
        temporary = compressed + f".{os.getpid()}.tmp"
        recovered = checkpoint + f".recovery.{os.getpid()}.tmp"
        try:
            with open(checkpoint, "rb") as source, open(temporary, "xb") as raw_target:
                with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw_target, mtime=0) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            os.replace(temporary, compressed)
            compressed_record = file_record(compressed)
            decompress_exact(compressed, recovered, original_record["sha256"])
            verify_torch_checkpoint(recovered)
            os.remove(recovered)
            task = {key: row[key] for key in ("seed", "track", "holdout")}
            proof = {"status": "approved_for_single_file_cleanup", "policy": POLICY,
                     "plan_sha256": plan_sha256, "task": task, "original": original_record,
                     "compressed": compressed_record, "bitwise_recovery_verified": True,
                     "torch_load_verified": True, "artifacts": row["artifacts"],
                     "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            atomic_json(proof, proof_path)
            current = file_record(checkpoint)
            for key in ("sha256", "bytes", "device", "inode"):
                if current[key] != original_record[key]:
                    raise ValueError(f"Checkpoint identity changed before removal: {checkpoint}")
            os.remove(checkpoint)
            proof.update({"status": "compressed_original_removed", "original_removed": True,
                          "cleaned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            atomic_json(proof, proof_path)
            with materialize_checkpoint(checkpoint, original_record["sha256"]) as restored:
                verify_torch_checkpoint(restored)
        finally:
            for path in (temporary, recovered):
                if os.path.exists(path):
                    os.remove(path)
    summary = {"status": "applied", "plan_sha256": plan_sha256,
               "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "representatives_retained": 6, "originals_removed": 66}
    atomic_json(summary, os.path.join(audit_dir, "apply_summary.json"))
    return summary


def verify_plan(plan_path, run_root, audit_dir, pass_index):
    if pass_index not in (1, 2, 3):
        raise ValueError("Pass index must be 1, 2, or 3")
    plan = load_plan(plan_path, run_root)
    original_removed_bytes = 0
    compressed_retained_bytes = 0
    for row in plan["files"]:
        checkpoint = row["artifacts"]["checkpoint"]["path"]
        expected = row["artifacts"]["checkpoint"]["sha256"]
        if row["action"] == "retain_full":
            if not os.path.isfile(checkpoint) or sha256_file(checkpoint) != expected:
                raise ValueError(f"Representative checkpoint missing/changed: {checkpoint}")
            with open(checkpoint[:-3] + ".retention.json") as handle:
                proof = json.load(handle)
            if proof.get("status") != "retained_full":
                raise ValueError(f"Representative proof missing: {checkpoint}")
        else:
            if os.path.exists(checkpoint):
                raise ValueError(f"Non-representative original still exists: {checkpoint}")
            with open(checkpoint[:-3] + ".retention.json") as handle:
                proof = json.load(handle)
            if proof.get("plan_sha256") != sha256_file(plan_path) or not proof.get("original_removed"):
                raise ValueError(f"Cleanup proof mismatch: {checkpoint}")
            original_removed_bytes += row["artifacts"]["checkpoint"]["bytes"]
            compressed_retained_bytes += os.path.getsize(checkpoint + ".gz")
        with materialize_checkpoint(checkpoint, expected) as restored:
            verify_torch_checkpoint(restored)
    report = {"status": "passed", "pass_index": pass_index, "plan_sha256": sha256_file(plan_path),
              "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tasks_verified": 72,
              "representatives_verified": 6, "compressed_checkpoints_verified": 66,
              "artifact_sha256_verified": True, "bitwise_recovery_verified": True,
              "torch_load_verified": True, "original_removed_bytes": original_removed_bytes,
              "compressed_retained_bytes": compressed_retained_bytes,
              "net_file_bytes_released": original_removed_bytes - compressed_retained_bytes}
    atomic_json(report, os.path.join(audit_dir, f"verification_pass{pass_index}.json"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "apply", "verify"))
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--summary-dir", default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--pass-index", type=int)
    args = parser.parse_args()
    plan_path = os.path.join(args.audit_dir, "plan.json")
    if args.command == "plan":
        result = {"plan_path": build_plan(args.run_root, args.summary_dir, args.audit_dir)}
    elif args.command == "apply":
        result = apply_plan(plan_path, args.run_root, args.audit_dir)
    else:
        if args.pass_index is None:
            parser.error("verify requires --pass-index")
        result = verify_plan(plan_path, args.run_root, args.audit_dir, args.pass_index)
    print("E5_RETENTION " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
