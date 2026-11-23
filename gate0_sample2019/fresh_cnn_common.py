import hashlib
import json
from pathlib import Path


def canonical_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_list_hash(rows):
    return hashlib.sha256("\n".join(sorted(rows)) + ("\n" if rows else "")).hexdigest()


def semantic_manifest(path):
    value = json.loads(Path(path).read_text())
    value.pop("created_utc", None)
    value.pop("build_dir", None)
    return value


def role_intersections(role_values):
    import itertools
    result = []
    for left, right in itertools.combinations(sorted(role_values), 2):
        result.append({"left": left, "right": right, "count": len(role_values[left] & role_values[right])})
    return result
