"""
Minimal Neural CSSR utilities extracted from src/ for the nanoGPT-CSSR pipeline.

This file contains only the essential components needed by run_neural_cssr.py:
- NeuralCSSRProbabilityProvider: Adapts neural models to CSSR interface
- Placeholder classes for ClassicalCSSR and TransCSSRWrapper (to be simplified later)
"""

import torch
from typing import Dict


class NeuralCSSRProbabilityProvider:
    """
    Wraps a trained neural model to provide next-symbol distributions
    for arbitrary histories, as required by CSSR sufficiency tests.
    """
    
    def __init__(self, model, device='auto', context_window: int = 32):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.context_window = context_window
        # Binary alphabet mapping
        self.id_to_token = {0: '0', 1: '1'}
        self.token_to_id = {'0': 0, '1': 1}

    def predict_next_distribution(self, history: str) -> Dict[str, float]:
        """Predict next symbol distribution given history string."""
        with torch.no_grad():
            ids = [self.token_to_id[c] for c in history if c in self.token_to_id]
            ids = ids[-self.context_window:]
            if len(ids) == 0:
                return {'0': 0.5, '1': 0.5}
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)[:, -1]  # [1, 2]
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            return {'0': float(probs[0]), '1': float(probs[1])}

    def get_empirical_distribution(self, history: str) -> Dict[str, int]:
        """Provide pseudo-counts to fit CSSR API expecting counts."""
        dist = self.predict_next_distribution(history)
        scale = 100
        return {k: int(round(v * scale)) for k, v in dist.items()}


# TODO: These are placeholders - need to either simplify or use transCSSR directly
class ClassicalCSSR:
    """Placeholder - this needs to be simplified or replaced with transCSSR usage."""
    def __init__(self, **kwargs):
        raise NotImplementedError("ClassicalCSSR needs to be simplified - use transCSSR directly")


class TransCSSRWrapper:
    """Placeholder - this should just call transCSSR directly."""
    def __init__(self, **kwargs):
        raise NotImplementedError("TransCSSRWrapper needs to be simplified - use transCSSR directly")