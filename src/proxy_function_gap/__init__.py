"""VGSC-inspired fixed-pool E0 selection and synthetic validation."""

from .budget import Aggregation, BatchScorer, Budget, CallBudget, ScoreCache, guidance_weight
from .pool import Candidate, CandidatePool
from .selection import (
    MultiTrajectoryResult,
    SelectionResult,
    Strategy,
    ZeroWeightPolicy,
    select,
    select_trajectories,
)

__all__ = [
    "Aggregation",
    "BatchScorer",
    "Budget",
    "CallBudget",
    "Candidate",
    "CandidatePool",
    "MultiTrajectoryResult",
    "ScoreCache",
    "SelectionResult",
    "Strategy",
    "ZeroWeightPolicy",
    "guidance_weight",
    "select",
    "select_trajectories",
]

__version__ = "0.4.0"
