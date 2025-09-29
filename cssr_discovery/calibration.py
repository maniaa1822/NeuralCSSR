"""
Calibration and threshold analysis functions.

This module provides functions for:
- Platt scaling calibration of model probabilities
- JS threshold computation and stability analysis
- Optimal L detection based on threshold plateaus
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from typing import Dict, Tuple, Optional

from js_metrics import extract_subsequences, js_divergence, get_next_token_distribution


def fit_platt_params(data: np.ndarray, model, L_max: int = 4, min_count: int = 5) -> Optional[Dict[str, float]]:
    """Fit Platt calibration parameters a,b using empirical next-token targets.

    We collect histories up to length L_max, compute model p1 and empirical p1,
    and solve weighted logistic regression with LBFGS.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device('cpu')

    counts = {}
    ones = {}
    N = len(data)
    for L in range(1, L_max + 1):
        for t in range(L, N):
            h = tuple(int(x) for x in data[t-L:t])
            counts[h] = counts.get(h, 0) + 1
            if int(data[t]) == 1:
                ones[h] = ones.get(h, 0) + 1

    histories = [h for h, c in counts.items() if c >= min_count]
    if not histories:
        return None

    margins = []
    targets = []
    weights = []
    ctx = getattr(getattr(model, 'config', None), 'block_size', None)
    for h in histories:
        ids = list(h)
        if ctx is not None and len(ids) > int(ctx):
            ids = ids[-int(ctx):]
        p = get_next_token_distribution(model, np.array(ids, dtype=np.int64), platt_params=None)
        p1 = float(p[1])
        p1c = max(1e-12, min(1.0 - 1e-12, p1))
        margins.append(np.log(p1c / (1.0 - p1c)))
        targets.append(ones.get(h, 0) / float(counts[h]))
        weights.append(float(counts[h]))

    if len(margins) == 0:
        return None

    margins_t = torch.tensor(margins, dtype=torch.float64, device=device)
    targets_t = torch.tensor(targets, dtype=torch.float64, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float64, device=device)
    a = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.25, max_iter=200)

    def closure():
        opt.zero_grad()
        s = a * margins_t + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, targets_t, weight=weights_t)
        loss.backward()
        return loss

    opt.step(closure)
    return {'a': float(a.detach().cpu().item()), 'b': float(b.detach().cpu().item())}


def compute_js_thresholds(data: np.ndarray, model, L_max: int = 10, n_samples: int = 5000,
                         platt_params: Optional[dict] = None) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute JS thresholds for different L values using random pairs."""
    thresholds = {}
    separations = {}

    for L in range(1, L_max + 1):
        print(f"Computing threshold for L={L}")

        # Extract subsequences
        subsequences = extract_subsequences(data, L)
        if len(subsequences) < n_samples:
            n_samples_L = len(subsequences) // 2
        else:
            n_samples_L = n_samples

        # Sample random pairs
        idx = np.random.choice(len(subsequences), size=(n_samples_L, 2))

        # Compute JS divergences
        js_values = []
        for i, j in idx:
            p = get_next_token_distribution(model, subsequences[i], platt_params=platt_params)
            q = get_next_token_distribution(model, subsequences[j], platt_params=platt_params)
            js_values.append(js_divergence(p, q))

        js_values = np.array(js_values)

        # Fit GMM and get threshold
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))

            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))
            weights = gmm.weights_

            lower_idx = np.argmin(means)
            threshold = means[lower_idx] + 2 * stds[lower_idx]
        except:
            threshold = np.percentile(js_values, 75)
            means = [np.mean(js_values)] * 2
            stds = [np.std(js_values)] * 2
            weights = [0.5, 0.5]

        thresholds[L] = threshold
        separations[L] = abs(means[1] - means[0]) / (stds[0] + stds[1]) if (stds[0] + stds[1]) > 1e-12 else 0.0

    return thresholds, separations


def find_optimal_L_thresholds(thresholds: Dict[int, float], stability_threshold: float = 0.1) -> int:
    """Find minimal L where threshold values stabilize (plateau)."""
    L_values = sorted(thresholds.keys())

    if len(L_values) < 3:
        return max(L_values)  # Need at least 3 points for stability check

    # Look for consecutive L values with relative change < threshold
    for i in range(2, len(L_values)):
        L_curr, L_prev1, L_prev2 = L_values[i], L_values[i-1], L_values[i-2]

        change1 = abs(thresholds[L_curr] - thresholds[L_prev1]) / max(thresholds[L_prev1], 1e-12)
        change2 = abs(thresholds[L_prev1] - thresholds[L_prev2]) / max(thresholds[L_prev2], 1e-12)

        if change1 < stability_threshold and change2 < stability_threshold:
            return L_prev1  # Return the L where stability starts

    return max(L_values)  # Return max L if no stability found


def find_optimal_L_separation(separations: Dict[int, float], stability_threshold: float = 0.1) -> int:
    """Find minimal L where separation metric stabilizes."""
    L_values = sorted(separations.keys())

    if len(L_values) < 3:
        return max(L_values)

    # Look for consecutive L values with separation change < threshold
    for i in range(2, len(L_values)):
        L_curr, L_prev1, L_prev2 = L_values[i], L_values[i-1], L_values[i-2]

        change1 = abs(separations[L_curr] - separations[L_prev1]) / max(separations[L_prev1], 1e-12)
        change2 = abs(separations[L_prev1] - separations[L_prev2]) / max(separations[L_prev2], 1e-12)

        if change1 < stability_threshold and change2 < stability_threshold:
            return L_prev1

    return max(L_values)