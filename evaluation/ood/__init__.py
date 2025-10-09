"""OOD evaluation infrastructure."""

from .evaluation import evaluate_model_ood, OODMetrics

__all__ = [
    "evaluate_model_ood",
    "OODMetrics",
]
