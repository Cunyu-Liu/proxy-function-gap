import math
from typing import Optional, Sequence


def regret(
    synthetic_truth: Sequence[float], selected_index: int, feasible: Optional[Sequence[bool]] = None
) -> float:
    truth = tuple(float(value) for value in synthetic_truth)
    if not truth or any(not math.isfinite(value) for value in truth):
        raise ValueError("synthetic truth must be non-empty and finite")
    if feasible is None:
        feasible_indices = range(len(truth))
    else:
        if len(feasible) != len(truth):
            raise ValueError("feasible mask length must match truth")
        feasible_indices = [index for index, allowed in enumerate(feasible) if allowed]
    if selected_index not in feasible_indices:
        raise ValueError("selected candidate must be feasible")
    oracle = max(truth[index] for index in feasible_indices)
    return oracle - truth[selected_index]
