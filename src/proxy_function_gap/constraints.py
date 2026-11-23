from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class UTRConstraints:
    min_length: int = 1
    max_length: Optional[int] = None
    gc_min: float = 0.0
    gc_max: float = 1.0
    forbid_uaug: bool = False
    max_poly_u: Optional[int] = None

    def __post_init__(self) -> None:
        if self.min_length < 0:
            raise ValueError("min_length must be non-negative")
        if self.max_length is not None and self.max_length < self.min_length:
            raise ValueError("max_length must be at least min_length")
        if not 0 <= self.gc_min <= self.gc_max <= 1:
            raise ValueError("GC bounds must satisfy 0 <= min <= max <= 1")
        if self.max_poly_u is not None and self.max_poly_u < 0:
            raise ValueError("max_poly_u must be non-negative")


def utr_constraint_mask(
    sequences: Sequence[str], constraints: UTRConstraints = UTRConstraints()
) -> Tuple[bool, ...]:
    mask = []
    for sequence in sequences:
        if any(base not in "ACGU" for base in sequence):
            raise ValueError("UTR sequences must contain only uppercase A, C, G, U")
        length = len(sequence)
        length_ok = length >= constraints.min_length and (
            constraints.max_length is None or length <= constraints.max_length
        )
        gc = (sequence.count("G") + sequence.count("C")) / length if length else 0.0
        gc_ok = length > 0 and constraints.gc_min <= gc <= constraints.gc_max
        uaug_ok = not constraints.forbid_uaug or "AUG" not in sequence
        poly_u_ok = constraints.max_poly_u is None or "U" * (constraints.max_poly_u + 1) not in sequence
        mask.append(length_ok and gc_ok and uaug_ok and poly_u_ok)
    return tuple(mask)
