import ast
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E0_SOURCE = ROOT / "src" / "proxy_function_gap"
GATE0_SOURCE = ROOT / "gate0" / "gate0_sample2019"
E0_IMPORTS = {
    "argparse",
    "dataclasses",
    "enum",
    "hashlib",
    "json",
    "math",
    "pathlib",
    "random",
    "typing",
}
GATE0_IMPORTS = {
    "argparse",
    "csv",
    "gzip",
    "hashlib",
    "itertools",
    "json",
    "collections",
    "math",
    "pathlib",
    "platform",
    "subprocess",
    "sys",
}
SOURCE_COMMIT = "25d6fd4a8133aadde85f6cc4718503c0efe21249"
RAW_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".npy", ".npz", ".pt", ".pth", ".ckpt")


def inspect_sources(source):
    hashes = {}
    imports = set()
    for path in sorted(source.glob("*.py")):
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
    assert e0_imports <= E0_IMPORTS, sorted(e0_imports - E0_IMPORTS)
    assert gate0_imports <= GATE0_IMPORTS, sorted(gate0_imports - GATE0_IMPORTS)

    selection = (E0_SOURCE / "selection.py").read_text(encoding="utf-8")
    pool = (E0_SOURCE / "pool.py").read_text(encoding="utf-8")
    budget = (E0_SOURCE / "budget.py").read_text(encoding="utf-8")
    noise = (E0_SOURCE / "noise.py").read_text(encoding="utf-8")
    for forbidden in ("synthetic_truth", "true_values", "oracle", "regret("):
        assert forbidden not in selection, forbidden
        assert forbidden not in pool, forbidden
    for required in (
        'FP_TILT = "fp-tilt"',
        'FP_REJ = "fp-rej"',
        'ABSTAIN = "abstain"',
        "zero_weight_policy: ZeroWeightPolicy = ZeroWeightPolicy.ABSTAIN",
    ):
        assert required in selection, required
    for required in (
        "proxy_item_forwards",
        "verifier_item_forwards",
        "qhat_item_forwards",
        "combined = verifier * qhat",
        "return combined ** alpha",
    ):
        assert required in budget, required
    assert "qhat" not in pool
    assert "proxy_calls" not in selection
    assert "MRL" not in noise

    gate0_runtime = "\n".join(path.read_text(encoding="utf-8") for path in GATE0_SOURCE.glob("*.py"))
    for forbidden in ("torch", "tensorflow", ".cuda(", "train(", "vault"):
        assert forbidden not in gate0_runtime, forbidden
    assert gate0_runtime.count("subprocess.check_output") == 4
    assert gate0_runtime.count('subprocess.check_output(["git"') == 4
    provenance = (ROOT / "gate0" / "PROVENANCE.md").read_text(encoding="utf-8")
    assert SOURCE_COMMIT in provenance
    for excluded in (
        "gate0/configs/split_v1.json",
        "gate0/gate0_sample2019/freeze_split.py",
        "gate0/gate0_sample2019/protocol.py",
    ):
        assert not (ROOT / excluded).exists(), excluded
    subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"])

    tracked = [
        item
        for item in subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"], text=True).split("\0")
        if item
    ]
    candidates = [
        item
        for item in subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            text=True,
        ).split("\0")
        if item
    ]
    raw_tracked = [item for item in candidates if item.endswith(RAW_SUFFIXES)]
    large_tracked = [item for item in candidates if (ROOT / item).stat().st_size > 1024 * 1024]
    assert not raw_tracked, raw_tracked
    assert not large_tracked, large_tracked
    max_tracked_bytes = max((ROOT / item).stat().st_size for item in candidates)

    print(
        json.dumps(
            {
                "status": "PASS",
                "e0_runtime_files": len(e0_hashes),
                "gate0_runtime_files": len(gate0_hashes),
                "e0_runtime_imports": sorted(e0_imports),
                "gate0_runtime_imports": sorted(gate0_imports),
                "gate0_source_commit": SOURCE_COMMIT,
                "tracked_files": len(tracked),
                "audited_candidate_files": len(candidates),
                "max_tracked_bytes": max_tracked_bytes,
                "raw_or_large_tracked_files": [],
                "forbidden_paths": ["GPU", "training", "model inference", "vault", "raw data in git"],
                "sha256": {**e0_hashes, **gate0_hashes},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
