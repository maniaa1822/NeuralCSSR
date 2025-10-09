"""Core infrastructure for unified evaluation framework."""

from .models import ModelHandle, load_model, parse_model_spec, compute_state_distances
from .data import generate_iid_data, generate_ood_data, OODGenerator
from .activations import collect_layer_activations
from .metrics import compute_lm_metrics, compute_classification_metrics

__all__ = [
    "ModelHandle",
    "load_model",
    "parse_model_spec",
    "compute_state_distances",
    "generate_iid_data",
    "generate_ood_data",
    "OODGenerator",
    "collect_layer_activations",
    "compute_lm_metrics",
    "compute_classification_metrics",
]
