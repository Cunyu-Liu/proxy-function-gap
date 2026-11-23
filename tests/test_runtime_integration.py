import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_COMMIT = "3c8d047816a4a216a3954c66112c1e858c7dc2c7"
SPLIT_COMMIT = "05eb96b5e730c9dd6cebc96486220bd6266c30d9"


def test_runtime_source_commit_and_initial_tree_are_preserved():
    subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", RUNTIME_COMMIT, "HEAD"])
    source_tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", f"{RUNTIME_COMMIT}^{{tree}}"], text=True
    ).strip()
    integrated_tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "981a3de:runtime"], text=True
    ).strip()
    assert integrated_tree == source_tree
    assert RUNTIME_COMMIT in (ROOT / "runtime" / "PROVENANCE.md").read_text()
    assert not list(ROOT.rglob("*.hdf5"))


def test_split_v2_commit_and_manifest_are_integrated():
    subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", SPLIT_COMMIT, "HEAD"])
    subprocess.check_call(["git", "-C", str(ROOT), "cat-file", "-e", "56d3e3b:gate0/configs/split_v2.json"])
    assert (ROOT / "gate0" / "configs" / "split_v2.json").is_file()
    assert SPLIT_COMMIT in (ROOT / "gate0" / "PROVENANCE.md").read_text()
