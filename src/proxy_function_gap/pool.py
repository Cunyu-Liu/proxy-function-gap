import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    index: int
    sequence_id: str
    sequence: str


class CandidatePool:
    def __init__(
        self,
        sequence_ids: Sequence[str],
        sequences: Sequence[str],
        proposal_probs: Sequence[float],
        feasible: Optional[Sequence[bool]] = None,
    ) -> None:
        self.sequence_ids = tuple(sequence_ids)
        self.sequences = tuple(sequences)
        self.proposal_probs = tuple(float(p) for p in proposal_probs)
        self.feasible = tuple(True for _ in self.sequences) if feasible is None else tuple(feasible)
        lengths = {
            len(self.sequence_ids),
            len(self.sequences),
            len(self.proposal_probs),
            len(self.feasible),
        }
        if len(lengths) != 1 or not self.sequences:
            raise ValueError("pool fields must have the same non-zero length")
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("sequence IDs must be unique")
        if any(not sequence_id for sequence_id in self.sequence_ids):
            raise ValueError("sequence IDs must be non-empty")
        if any(not isinstance(sequence, str) for sequence in self.sequences):
            raise ValueError("sequences must be strings")
        if any(not math.isfinite(p) or p < 0 for p in self.proposal_probs):
            raise ValueError("proposal probabilities must be finite and non-negative")
        if not math.isclose(math.fsum(self.proposal_probs), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("proposal probabilities must sum to 1")
        feasible_mass = math.fsum(
            p for p, is_feasible in zip(self.proposal_probs, self.feasible) if is_feasible
        )
        if feasible_mass <= 0:
            raise ValueError("at least one feasible candidate must have positive proposal mass")
        self._feasible_mass = feasible_mass

    def __len__(self) -> int:
        return len(self.sequences)

    def candidate(self, index: int) -> Candidate:
        return Candidate(index, self.sequence_ids[index], self.sequences[index])

    def masked_proposal_probs(self) -> Tuple[float, ...]:
        return tuple(
            p / self._feasible_mass if is_feasible else 0.0
            for p, is_feasible in zip(self.proposal_probs, self.feasible)
        )

    def exact_tilted_distribution(
        self,
        verifier_scores: Sequence[float],
        alpha: float,
        qhat: Optional[float] = None,
    ) -> Tuple[float, ...]:
        scores = tuple(float(value) for value in verifier_scores)
        if len(scores) != len(self):
            raise ValueError("verifier score length must match pool size")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("verifier scores must be in [0, 1]")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if qhat is not None and (not math.isfinite(qhat) or not 0 <= qhat <= 1):
            raise ValueError("qhat must be in [0, 1]")
        base = self.masked_proposal_probs()
        if alpha == 0:
            return base
        joint_factor = 1.0 if qhat is None else qhat ** alpha
        weights = tuple(p * (score ** alpha) * joint_factor for p, score in zip(base, scores))
        normalizer = math.fsum(weights)
        if normalizer <= 0:
            raise ValueError("tilted distribution has zero normalizer")
        return tuple(weight / normalizer for weight in weights)
