"""
Jensen-Shannon divergence metrics and distribution computation functions.

This module provides the core functions for computing JS divergences between
probability distributions, extracting subsequences, and getting model predictions.
"""

import numpy as np
import torch
from scipy.stats import entropy
from typing import List, Optional


def extract_subsequences(data: np.ndarray, L: int) -> List[np.ndarray]:
    """Extract all length-L subsequences from dataset."""
    subsequences = []
    for i in range(len(data) - L + 1):
        subsequences.append(data[i:i+L])
    return subsequences


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def get_next_token_distribution(model, history: np.ndarray, platt_params: Optional[dict] = None) -> np.ndarray:
    """Get P(x|history) from your trained transformer."""
    # Convert history to model input format
    with torch.no_grad():
        # Truncate to model context window if available
        ctx = getattr(getattr(model, 'config', None), 'block_size', None)
        if ctx is not None and len(history) > int(ctx):
            ids = history[-int(ctx):]
        else:
            ids = history
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        # Move to model device if possible
        try:
            device = next(model.parameters()).device
            x = x.to(device)
        except Exception:
            pass
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        if probs.shape[-1] < 2:
            return np.array([0.5, 0.5], dtype=np.float64)
        p0, p1 = float(probs[0]), float(probs[1])
        # Optional Platt calibration on p1
        if platt_params is not None and 'a' in platt_params and 'b' in platt_params:
            a = float(platt_params['a']); b = float(platt_params['b'])
            eps = 1e-12
            p1c = max(eps, min(1.0 - eps, p1))
            logit = np.log(p1c / (1.0 - p1c))
            s = a * logit + b
            p1 = float(1.0 / (1.0 + np.exp(-s)))
            p1 = max(1e-6, min(1.0 - 1e-6, p1))
            p0 = 1.0 - p1
        s = p0 + p1
        if s <= 0:
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.array([p0 / s, p1 / s], dtype=np.float64)


def get_kstep_distribution(model, history: np.ndarray, k: int, platt_params: Optional[dict] = None) -> np.ndarray:
    """
    Compute exact k-step distribution over binary sequences by chaining next-token probabilities.

    Returns a vector of length 2^k in lexicographic order (0..0, 0..1, ..., 1..1).
    """
    if k <= 0:
        return np.array([1.0], dtype=np.float64)

    num_paths = 1 << k
    probs = np.zeros(num_paths, dtype=np.float64)

    # stack entries: (history_tuple, depth, log_prob, index_prefix)
    initial_hist_tuple = tuple(int(x) for x in history.tolist())
    stack = [(initial_hist_tuple, 0, 0.0, 0)]

    while stack:
        hist_tuple, depth, logp, idx_prefix = stack.pop()
        if depth == k:
            probs[idx_prefix] = np.exp(logp)
            continue

        p = get_next_token_distribution(model, np.array(hist_tuple, dtype=np.int64), platt_params=platt_params)
        p0, p1 = float(p[0]), float(p[1])

        # Append 0 (left child)
        stack.append((hist_tuple + (0,), depth + 1, logp + np.log(max(1e-12, p0)), (idx_prefix << 1) | 0))
        # Append 1 (right child)
        stack.append((hist_tuple + (1,), depth + 1, logp + np.log(max(1e-12, p1)), (idx_prefix << 1) | 1))

    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(num_paths, 1.0 / num_paths, dtype=np.float64)
    return probs / s


def js_divergence_k(model, h1: np.ndarray, h2: np.ndarray, k: int, platt_params: Optional[dict] = None) -> float:
    """JS divergence between exact k-step rollout distributions for two histories."""
    p = get_kstep_distribution(model, h1, k, platt_params)
    q = get_kstep_distribution(model, h2, k, platt_params)
    return js_divergence(p, q)


def js_divergence_conditional_k(model, h1: np.ndarray, h2: np.ndarray, k: int, platt_params: Optional[dict] = None) -> float:
    """
    Conditional JS divergence between two histories using k-step rollouts.

    This computes a weighted average of JS divergences after conditioning on the first symbol,
    which removes mixture effects and focuses on successor kernel differences.

    Args:
        model: The neural model
        h1, h2: History arrays to compare
        k: Steps to roll out (must be >= 2)
        platt_params: Optional Platt calibration parameters

    Returns:
        Conditional JS divergence: w̄₀ * JS(p|₀, q|₀) + w̄₁ * JS(p|₁, q|₁)
    """
    if k < 2:
        raise ValueError("Conditional JS requires k >= 2")

    # Get k-step distributions for both histories
    p = get_kstep_distribution(model, h1, k, platt_params)  # length 2^k
    q = get_kstep_distribution(model, h2, k, platt_params)  # length 2^k

    # Split by first bit: first half (0...) and second half (1...)
    m = 1 << (k - 1)  # 2^(k-1)

    p0, p1 = p[:m], p[m:]  # p(k)(0·|h1), p(k)(1·|h1)
    q0, q1 = q[:m], q[m:]  # p(k)(0·|h2), p(k)(1·|h2)

    # Compute marginal probabilities for first symbol
    w0p, w1p = float(p0.sum()), float(p1.sum())  # Pr(0|h1), Pr(1|h1)
    w0q, w1q = float(q0.sum()), float(q1.sum())  # Pr(0|h2), Pr(1|h2)

    # Normalize conditional distributions: p|₀^(k-1)(·|h) = p^(k)(0·|h) / w₀(h)
    p0 = p0 / max(w0p, 1e-12)
    p1 = p1 / max(w1p, 1e-12)
    q0 = q0 / max(w0q, 1e-12)
    q1 = q1 / max(w1q, 1e-12)

    # Compute average weights: w̄ₐ = ½(wₐ(h1) + wₐ(h2))
    w0_bar = 0.5 * (w0p + w0q)
    w1_bar = 1.0 - w0_bar

    # Conditional JS: weighted average of JS divergences after branching
    js_cond = (w0_bar * js_divergence(p0, q0) +
               w1_bar * js_divergence(p1, q1))

    return float(js_cond)