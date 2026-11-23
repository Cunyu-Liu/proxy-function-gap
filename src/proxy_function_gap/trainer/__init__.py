"""Fresh CUDA-only Sample-like proxy training."""

from .config import TrainerConfig
from .convergence import DualPlateau
from .data import PreflightReport, TargetScaler, preflight
from .model import FreshSampleCNN

__all__ = ["DualPlateau", "FreshSampleCNN", "PreflightReport", "TargetScaler", "TrainerConfig", "preflight"]
