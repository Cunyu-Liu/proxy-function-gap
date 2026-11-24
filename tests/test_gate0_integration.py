import json
import subprocess
from pathlib import Path

import gate0_sample2019


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "25d6fd4a8133aadde85f6cc4718503c0efe21249"


def test_gate0_package_and_pyproject_paths_are_integrated():
    assert Path(gate0_sample2019.__file__).parent == ROOT / "gate0" / "gate0_sample2019"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pytest_ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert 'where = ["src", "gate0", "runtime/src"]' in pyproject
    assert 'include = ["proxy_function_gap*", "gate0_sample2019*", "gate0_leppek2022*", "sample_runtime*"]' in pyproject
    assert "namespaces = false" in pyproject
    assert 'pythonpath = ["src", "gate0", "runtime/src"]' in pyproject
    assert 'testpaths = ["tests", "gate0/tests", "runtime/tests"]' in pyproject
    assert "pythonpath = src gate0 runtime/src" in pytest_ini
    assert "testpaths = tests gate0/tests runtime/tests" in pytest_ini


def test_gate0_source_commit_is_ancestor_and_initial_tree_matches():
    subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"])
    integrated_tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "cc8ec11:gate0"], text=True
    ).strip()
    source_tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"], text=True
    ).strip()
    assert integrated_tree == source_tree == "dfff55f48300b5e32f3ccc8e3bb41fa29e4e1024"
    provenance = (ROOT / "gate0" / "PROVENANCE.md").read_text(encoding="utf-8")
    assert SOURCE_COMMIT in provenance


def test_gate0_config_keeps_data_outside_git():
    config = json.loads((ROOT / "gate0" / "configs" / "gate0_v1.json").read_text())
    for key in ("input_manifest", "output_dir"):
        assert config[key].startswith("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/")
        assert not config[key].startswith(str(ROOT))
    assert "split_ready" in config["output_dir"] or "gate0_sample2019" in config["output_dir"]
