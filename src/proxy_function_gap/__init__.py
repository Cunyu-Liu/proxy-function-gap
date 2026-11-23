"""Fixed-pool E0 selection and synthetic validation."""

from .pool import Candidate, CandidatePool
from .selection import Budget, CallBudget, SelectionResult, Strategy, select

__all__ = [
    "Budget",
    "CallBudget",
    "Candidate",
    "CandidatePool",
    "SelectionResult",
    "Strategy",
    "select",
]

__version__ = "0.1.0"
