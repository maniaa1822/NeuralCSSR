"""
Minimal Neural CSSR utilities extracted from src/ for the nanoGPT-CSSR pipeline.

This file contains only the essential components needed by run_neural_cssr.py:
- NeuralCSSRProbabilityProvider: Adapts neural models to CSSR interface
- Placeholder classes for ClassicalCSSR and TransCSSRWrapper (to be simplified later)
"""

import torch
from typing import Dict
from collections import defaultdict


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
    
    def get_probabilities(self, context: list) -> list:
        """Get probabilities as list [P(0), P(1)] for transCSSR integration."""
        history = ''.join(str(c) for c in context)
        dist = self.predict_next_distribution(history)
        return [dist['0'], dist['1']]


# TODO: These are placeholders - need to either simplify or use transCSSR directly
class ClassicalCSSR:
    """Placeholder - this needs to be simplified or replaced with transCSSR usage."""
    def __init__(self, **kwargs):
        raise NotImplementedError("ClassicalCSSR needs to be simplified - use transCSSR directly")


class TransCSSRWrapper:
    """Wrapper for transCSSR that uses neural probabilities instead of empirical counts."""
    
    def __init__(self, significance_level: float = 0.05, test_type: str = 'chi2', mix_empirical: float = 0.0):
        self.significance_level = significance_level
        self.test_type = test_type
        self.mix_empirical = max(0.0, min(1.0, mix_empirical))
    
    def run_cssr_with_neural_provider(self, data_tokens: list, provider: 'NeuralCSSRProbabilityProvider', L_max: int = 6):
        """Run transCSSR using neural probabilities instead of empirical counts."""
        import sys
        import os
        sys.path.append('transCSSR')
        
        try:
            from transCSSR import run_transCSSR, estimate_predictive_distributions
            import itertools
        except ImportError as e:
            raise ImportError(f"Failed to import transCSSR: {e}. Ensure transCSSR directory is available.")
        
        # Convert data to strings as expected by transCSSR
        stringX = '0' * len(data_tokens)  # Null input process
        stringY = ''.join(map(str, data_tokens))
        
        # Set up alphabets
        axs = ['0']  # Input alphabet (null process)
        ays = ['0', '1']  # Output alphabet
        e_symbols = list(itertools.product(axs, ays))
        
        # Build empirical word lookup dictionaries first to get correct structure
        print("Building empirical lookup structure...")
        word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(stringX, stringY, L_max)
        
        # Replace future probabilities with neural ones, but keep structure
        print("Replacing with neural probabilities...")
        self._replace_with_neural_probabilities(word_lookup_marg, word_lookup_fut, provider, ays)
        
        # Run transCSSR
        epsilon, invepsilon, morph_by_state = run_transCSSR(
            word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols,
            '', 'neural_data', alpha=self.significance_level, test_type=self.test_type
        )
        
        return epsilon, invepsilon, morph_by_state

    def run_cssr_empirical(self, data_tokens: list, L_max: int = 6):
        """Run transCSSR purely empirically (no neural replacement)."""
        import sys
        sys.path.append('transCSSR')
        from transCSSR import run_transCSSR, estimate_predictive_distributions
        import itertools

        stringX = '0' * len(data_tokens)
        stringY = ''.join(map(str, data_tokens))
        axs = ['0']
        ays = ['0','1']
        e_symbols = list(itertools.product(axs, ays))
        word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(stringX, stringY, L_max)
        epsilon, invepsilon, morph_by_state = run_transCSSR(
            word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols,
            '', 'empirical_data', alpha=self.significance_level, test_type=self.test_type
        )
        return epsilon, invepsilon, morph_by_state
    
    def _replace_with_neural_probabilities(self, word_lookup_marg, word_lookup_fut, provider, ays):
        """Replace empirical future counts with neural probability-based counts, preserving marginals.

        For each (history_x, history_y), we compute neural probs p, the empirical future
        distribution q (from word_lookup_fut), and set new future counts to:
            counts(y_next) = marg_count * [ mix_empirical * q + (1 - mix_empirical) * p ]
        ensuring sum_y counts(y) == marg_count.
        """
        # Collect unique Y-histories and aggregate empirical future counts per marginal
        histories = set()
        fut_by_marg = defaultdict(lambda: [0.0 for _ in ays])  # (hx, hy) -> counts per y_next index
        for (history_x, future_y), cnt in list(word_lookup_fut.items()):
            if len(future_y) > 0:
                history_y = future_y[:-1]
                y_next = future_y[-1]
                y_idx = ays.index(y_next)
                histories.add(history_y)
                fut_by_marg[(history_x, history_y)][y_idx] += float(cnt)

        print(f"Replacing probabilities for {len(histories)} unique histories...")

        # For each marginal, compute mixed distribution and assign counts proportionally to marg_count
        for (history_x, history_y), emp_counts in fut_by_marg.items():
            context = [int(c) for c in history_y] if history_y else []
            p = provider.get_probabilities(context)
            # Empirical distribution q
            emp_total = sum(emp_counts)
            if emp_total > 0:
                q = [c / emp_total for c in emp_counts]
            else:
                q = [1.0 / len(ays) for _ in ays]

            mix = self.mix_empirical
            d = [mix * qi + (1.0 - mix) * pi for qi, pi in zip(q, p)]
            # Normalize d
            s = sum(d) if sum(d) > 0 else 1.0
            d = [di / s for di in d]

            marg_key = (history_x, history_y)
            marg_count = float(word_lookup_marg.get(marg_key, emp_total if emp_total > 0 else 1.0))
            # Assign futures so they sum to marg_count
            for y_idx, y_next in enumerate(ays):
                fut_key = (history_x, history_y + y_next)
                word_lookup_fut[fut_key] = d[y_idx] * marg_count
    
    def _build_neural_word_lookups(self, stringX, stringY, provider, L_max, axs, ays):
        """Build word lookup dictionaries using neural probabilities - EFFICIENTLY!"""
        word_lookup_marg = {}
        word_lookup_fut = {}
        
        # Pseudo-count scale for converting probabilities to counts
        pseudo_count_scale = 1000
        T = len(stringY)
        
        # STEP 1: Collect all unique histories and build proper marginal/future structure
        unique_histories = {}  # history_y -> count
        all_marginals = {}     # (history_x, history_y) -> count
        all_futures = {}       # (history_x, history_y + y_next) -> (history_y, count)
        
        print(f"Collecting histories from {T} positions...")
        
        # Build all length combinations from 0 to L_max
        for t in range(T):
            for L in range(0, L_max + 1):  # Include L=0 for base cases
                if t >= L:
                    hist_start = max(0, t - L)
                    history_y = stringY[hist_start:t]
                    history_x = stringX[hist_start:t]
                    
                    # Marginal entry: history without next symbol
                    marg_key = (history_x, history_y)
                    if marg_key not in all_marginals:
                        all_marginals[marg_key] = 0
                    all_marginals[marg_key] += 1
                    
                    # Track unique Y-histories for neural evaluation
                    if history_y not in unique_histories:
                        unique_histories[history_y] = 0
                    unique_histories[history_y] += 1
                    
                    # Future entries: history with next symbol (if within bounds)
                    if t < T:
                        y_next = stringY[t]
                        future_y = history_y + y_next
                        future_x = history_x + '0'  # null input process
                        fut_key = (future_x, future_y)
                        
                        if fut_key not in all_futures:
                            all_futures[fut_key] = []
                        all_futures[fut_key].append((history_y, 1))
        
        print(f"Found {len(unique_histories)} unique histories")
        print(f"Found {len(all_marginals)} marginal combinations")
        
        # STEP 2: Batch evaluate neural probabilities for unique histories only
        print("Computing neural probabilities for unique histories...")
        neural_prob_cache = {}
        
        for history_y in unique_histories:
            context = [int(c) for c in history_y] if history_y else []
            probs = provider.get_probabilities(context)
            neural_prob_cache[history_y] = probs
        
        print(f"Cached {len(neural_prob_cache)} neural probability evaluations")
        
        # STEP 3: Build marginals directly
        word_lookup_marg = all_marginals.copy()
        
        # STEP 4: Build futures using neural probabilities
        print("Building future lookups with neural probabilities...")
        
        # Convert empirical futures to neural-based futures
        for fut_key in all_futures:
            history_x, future_y = fut_key
            if len(future_y) > 0:
                history_y = future_y[:-1]  # Remove the future symbol
                y_next = future_y[-1]     # The future symbol
                
                if history_y in neural_prob_cache:
                    probs = neural_prob_cache[history_y]
                    y_idx = ays.index(y_next)
                    
                    # Get empirical count for this specific future
                    emp_count = len(all_futures[fut_key])
                    
                    # Replace with neural probability weighted by occurrence
                    neural_count = probs[y_idx] * pseudo_count_scale * all_marginals.get((history_x, history_y), emp_count)
                    word_lookup_fut[fut_key] = neural_count
                else:
                    # Fallback to empirical count
                    word_lookup_fut[fut_key] = len(all_futures[fut_key])
        
        print(f"Built lookups: {len(word_lookup_marg)} marginal, {len(word_lookup_fut)} future")
        return word_lookup_marg, word_lookup_fut