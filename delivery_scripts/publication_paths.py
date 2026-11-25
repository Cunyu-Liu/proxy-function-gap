from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def configured_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


EVIDENCE_ROOT = configured_path("PFG_EVIDENCE_ROOT", REPO_ROOT)
DELIVERABLES = EVIDENCE_ROOT / "deliverables"
OUTPUT_ROOT = configured_path("PFG_OUTPUT_ROOT", REPO_ROOT / "build" / "publication")


def evidence_path(relative: str) -> Path:
    return EVIDENCE_ROOT / relative


def display_path(path: Path) -> str:
    path = path.resolve()
    for root in (REPO_ROOT, EVIDENCE_ROOT):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return str(path)
