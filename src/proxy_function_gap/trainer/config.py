import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class TrainerConfig:
    data_manifest: Path
    output_root: Path
    seeds: Tuple[int, ...] = (11, 23, 37)
    memory_fractions: Tuple[float, ...] = (0.20, 0.20, 0.20)
    gpu_index: int = 0
    dtype: str = "float32"
    batch_size: int = 64
    learning_rate: float = 1e-3
    min_delta: float = 1e-4
    patience: int = 5
    resume_from: Tuple[Tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least three unique seeds are required")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must be integers")
        if len(self.memory_fractions) != len(self.seeds):
            raise ValueError("one memory fraction is required per seed/process")
        if any(not 0 < fraction <= 1 for fraction in self.memory_fractions):
            raise ValueError("memory fractions must be in (0, 1]")
        if sum(self.memory_fractions) > 1:
            raise ValueError("sum of per-process GPU memory fractions must not exceed 1")
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        if self.min_delta < 0 or self.patience <= 0:
            raise ValueError("min_delta must be non-negative and patience positive")
        resume_seeds = {seed for seed, _ in self.resume_from}
        if resume_seeds and resume_seeds != set(self.seeds):
            raise ValueError("resume checkpoints must be provided for every seed or none")

    @classmethod
    def from_dict(cls, value: Dict) -> "TrainerConfig":
        allowed = {
            "data_manifest",
            "output_root",
            "seeds",
            "memory_fractions",
            "gpu_index",
            "dtype",
            "batch_size",
            "learning_rate",
            "min_delta",
            "patience",
            "resume_from",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown trainer config fields: {sorted(unknown)}")
        resume = value.get("resume_from") or {}
        return cls(
            data_manifest=Path(value["data_manifest"]),
            output_root=Path(value["output_root"]),
            seeds=tuple(value.get("seeds", (11, 23, 37))),
            memory_fractions=tuple(value.get("memory_fractions", (0.20, 0.20, 0.20))),
            gpu_index=value.get("gpu_index", 0),
            dtype=value.get("dtype", "float32"),
            batch_size=value.get("batch_size", 64),
            learning_rate=value.get("learning_rate", 1e-3),
            min_delta=value.get("min_delta", 1e-4),
            patience=value.get("patience", 5),
            resume_from=tuple(sorted((int(seed), str(path)) for seed, path in resume.items())),
        )

    @classmethod
    def from_json(cls, path: Path) -> "TrainerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def resume_for(self, seed: int) -> Optional[Path]:
        return next((Path(path) for candidate, path in self.resume_from if candidate == seed), None)
