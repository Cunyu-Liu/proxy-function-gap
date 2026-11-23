import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "proxy_function_gap"
ALLOWED_IMPORTS = {
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


def main():
    hashes = {}
    imports = set()
    for path in sorted(SOURCE.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])
    assert imports <= ALLOWED_IMPORTS, sorted(imports - ALLOWED_IMPORTS)

    selection = (SOURCE / "selection.py").read_text(encoding="utf-8")
    pool = (SOURCE / "pool.py").read_text(encoding="utf-8")
    for forbidden in ("synthetic_truth", "true_values", "oracle", "regret("):
        assert forbidden not in selection, forbidden
        assert forbidden not in pool, forbidden
    all_runtime = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE.glob("*.py"))
    for forbidden in ("torch", "tensorflow", ".cuda(", "prefix", "train("):
        assert forbidden not in all_runtime, forbidden

    print(
        json.dumps(
            {
                "status": "PASS",
                "runtime_files": len(hashes),
                "runtime_imports": sorted(imports),
                "forbidden_paths": ["GPU", "training", "prefix generation", "truth in selector/pool"],
                "sha256": hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
