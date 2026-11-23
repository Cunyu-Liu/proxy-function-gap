import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E0_SOURCE = ROOT / "src" / "proxy_function_gap"
TRAINER_SOURCE = E0_SOURCE / "trainer"
GATE0_SOURCE = ROOT / "gate0" / "gate0_sample2019"
RUNTIME_SOURCE = ROOT / "runtime" / "src" / "sample_runtime"
E0_IMPORTS = {"argparse", "dataclasses", "enum", "hashlib", "json", "math", "pathlib", "random", "typing"}
GATE0_IMPORTS = {
    "argparse", "ast", "csv", "datetime", "fcntl", "gzip", "hashlib", "io", "itertools", "json",
    "collections", "math", "numpy", "os", "pathlib", "platform", "re", "struct", "subprocess", "sys", "zipfile",
}
RUNTIME_IMPORTS = {
    "argparse", "csv", "datetime", "gzip", "hashlib", "h5py", "json", "math", "numpy",
    "os", "pathlib", "platform", "shutil", "subprocess", "sys", "tempfile", "torch",
}
TRAINER_IMPORTS = {
    "argparse", "concurrent", "copy", "csv", "dataclasses", "gzip", "hashlib", "json", "math",
    "multiprocessing", "numpy", "os", "pathlib", "queue", "subprocess", "tempfile", "time", "torch", "traceback", "typing",
}
SOURCE_COMMITS = {
    "gate0": "25d6fd4a8133aadde85f6cc4718503c0efe21249",
    "split_v2": "05eb96b5e730c9dd6cebc96486220bd6266c30d9",
    "runtime": "3c8d047816a4a216a3954c66112c1e858c7dc2c7",
}
RAW_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".hdf5", ".h5")


def inspect_sources(source, recursive=False):
    hashes = {}
    imports = set()
    paths = source.rglob("*.py") if recursive else source.glob("*.py")
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    return hashes, imports


def main():
    e0_hashes, e0_imports = inspect_sources(E0_SOURCE)
    gate0_hashes, gate0_imports = inspect_sources(GATE0_SOURCE)
    runtime_hashes, runtime_imports = inspect_sources(RUNTIME_SOURCE)
    trainer_hashes, trainer_imports = inspect_sources(TRAINER_SOURCE, recursive=True)
    assert e0_imports <= E0_IMPORTS, sorted(e0_imports - E0_IMPORTS)
    assert gate0_imports <= GATE0_IMPORTS, sorted(gate0_imports - GATE0_IMPORTS)
    assert runtime_imports <= RUNTIME_IMPORTS, sorted(runtime_imports - RUNTIME_IMPORTS)
    assert trainer_imports <= TRAINER_IMPORTS, sorted(trainer_imports - TRAINER_IMPORTS)

    selection = (E0_SOURCE / "selection.py").read_text(encoding="utf-8")
    pool = (E0_SOURCE / "pool.py").read_text(encoding="utf-8")
    budget = (E0_SOURCE / "budget.py").read_text(encoding="utf-8")
    noise = (E0_SOURCE / "noise.py").read_text(encoding="utf-8")
    for forbidden in ("synthetic_truth", "true_values", "oracle", "regret("):
        assert forbidden not in selection, forbidden
        assert forbidden not in pool, forbidden
    for required in (
        'FP_TILT = "fp-tilt"', 'FP_REJ = "fp-rej"', 'ABSTAIN = "abstain"',
        "zero_weight_policy: ZeroWeightPolicy = ZeroWeightPolicy.ABSTAIN",
    ):
        assert required in selection, required
    for required in (
        "proxy_item_forwards", "verifier_item_forwards", "qhat_item_forwards",
        "combined = verifier * qhat", "return combined ** alpha",
    ):
        assert required in budget, required
    assert "qhat" not in pool
    assert "proxy_calls" not in selection
    assert "MRL" not in noise

    gate0_runtime = "\n".join(path.read_text(encoding="utf-8") for path in GATE0_SOURCE.glob("*.py"))
    for forbidden in ("torch", "tensorflow", ".cuda(", "train("):
        assert forbidden not in gate0_runtime, forbidden
    assert gate0_runtime.count("subprocess.check_output") == gate0_runtime.count('subprocess.check_output(["git"')
    runtime_source = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_SOURCE.glob("*.py"))
    assert "CPU inference is forbidden" in runtime_source
    assert "shell=True" not in runtime_source

    trainer_runtime = "\n".join(path.read_text(encoding="utf-8") for path in TRAINER_SOURCE.rglob("*.py"))
    for required in (
        "CPU training fallback is forbidden",
        "set_per_process_memory_fraction",
        'get_context("spawn")',
        "class DualPlateau",
        "while True",
        "os.replace",
        "train_eval_loss",
        "dev_eval_loss",
        "canonical_row_order_hash",
        "completed checkpoint cannot be resumed",
        "model_definition_sha256",
        "initial_model_hash",
    ):
        assert required in trainer_runtime, required
    for forbidden in ("max_epochs", "max_steps", 'torch.device("cpu")', "shell=True"):
        assert forbidden not in trainer_runtime, forbidden
    assert "ROLES = (\"train\", \"dev\", \"calibration\")" in trainer_runtime

    for name, commit in SOURCE_COMMITS.items():
        subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"])
    for document in (ROOT / "gate0" / "PROVENANCE.md", ROOT / "runtime" / "PROVENANCE.md"):
        assert document.is_file()

    tracked = [item for item in subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"], text=True).split("\0") if item]
    candidates = [
        item for item in subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"], text=True
        ).split("\0") if item
    ]
    raw_tracked = [item for item in candidates if item.endswith(RAW_SUFFIXES)]
    large_tracked = [item for item in candidates if (ROOT / item).stat().st_size > 1024 * 1024]
    assert not raw_tracked, raw_tracked
    assert not large_tracked, large_tracked
    max_tracked_bytes = max((ROOT / item).stat().st_size for item in candidates)

    print(json.dumps({
        "status": "PASS",
        "e0_runtime_files": len(e0_hashes),
        "gate0_split_runtime_files": len(gate0_hashes),
        "sample_runtime_files": len(runtime_hashes),
        "trainer_runtime_files": len(trainer_hashes),
        "source_commits": SOURCE_COMMITS,
        "tracked_files": len(tracked),
        "audited_candidate_files": len(candidates),
        "max_tracked_bytes": max_tracked_bytes,
        "raw_or_large_tracked_files": [],
        "official_gpl_code_or_weights_in_git": False,
        "formal_training_started_by_audit": False,
        "sha256": {**e0_hashes, **gate0_hashes, **runtime_hashes, **trainer_hashes},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
