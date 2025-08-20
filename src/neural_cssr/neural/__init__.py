"""Neural components for Neural CSSR."""

from .transformer import SimpleTransformer

__all__ = ['SimpleTransformer']


class NeuralCSSRProbabilityProvider:
    """
    Wraps a trained causal decoder (cssr_nlp) to provide next-symbol distributions
    for arbitrary histories, as required by CSSR sufficiency tests.
    """
    def __init__(self, model, device='auto', context_window: int = 32):
        import torch
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.context_window = context_window
        # Binary alphabet mapping
        self.id_to_token = {0: '0', 1: '1'}
        self.token_to_id = {'0': 0, '1': 1}

    def predict_next_distribution(self, history: str):
        import torch
        with torch.no_grad():
            ids = [self.token_to_id[c] for c in history if c in self.token_to_id]
            ids = ids[-self.context_window:]
            if len(ids) == 0:
                return {'0': 0.5, '1': 0.5}
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)[:, -1]  # [1, 2]
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            return {'0': float(probs[0]), '1': float(probs[1])}

    def get_empirical_distribution(self, history: str):
        # Provide pseudo-counts to fit CSSR API expecting counts
        dist = self.predict_next_distribution(history)
        scale = 100
        return {k: int(round(v * scale)) for k, v in dist.items()}
