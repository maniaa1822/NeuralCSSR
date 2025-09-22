#!/usr/bin/env python3
"""
Estimate an optimal JS epsilon (threshold) for CSSR tests by comparing
neural model predictions against ground-truth next-token probabilities,
without running CSSR.

For each history h up to L_max observed in the data, we compute:
  - p_neural(h): model probability of next=1 given history h
  - p_gt(h): ground-truth probability of next=1 given history h
  - JS(Bernoulli(p_neural) || Bernoulli(p_gt))

We then summarize the weighted distribution of JS divergences over histories
(weights are marginal counts n(h)) and report recommended epsilon values:
  - weighted mean
  - weighted quantiles (configurable, e.g., 0.9, 0.95, 0.99)

Usage example:
  uv run --with torch python -u experiments/tools/estimate_js_epsilon.py \
    --data experiments/datasets/hierarchical_4_state_10k/hierarchical_4_state/hierarchical_4_state.dat \
    --model_ckpt nanoGPT/out-hierarchical-4-state-char/ckpt.pt \
    --L_max 6 --quantiles 0.9 0.95 0.99 --min_count 5 \
    --output_json experiments/results/provider_eval_hier4_nanogpt/js_epsilon_estimates_L6.json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


# Local imports from repository
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_cssr_state_averaging import ModelProvider, load_model_from_ckpt  # type: ignore
from transcssr_neural_runner import (
    load_binary_string,  # type: ignore
    _p1_gt_from_history_hier4,  # type: ignore
    _p1_gt_from_history_seven_state,  # type: ignore
)


def js_divergence_bernoulli(p1: float, q1: float, clip: float = 1e-12) -> float:
    """JS divergence between Bernoulli(p1) and Bernoulli(q1)."""
    p1c = max(clip, min(1.0 - clip, float(p1)))
    q1c = max(clip, min(1.0 - clip, float(q1)))
    m1 = 0.5 * (p1c + q1c)
    # KL terms
    kl_p_m = p1c * math.log(p1c / m1) + (1.0 - p1c) * math.log((1.0 - p1c) / (1.0 - m1))
    kl_q_m = q1c * math.log(q1c / m1) + (1.0 - q1c) * math.log((1.0 - q1c) / (1.0 - m1))
    return 0.5 * (kl_p_m + kl_q_m)


def weighted_quantile(pairs: Sequence[Tuple[float, float]], q: float) -> float:
    """Weighted quantile of value-weight pairs."""
    if not pairs:
        return 0.0
    pairs_sorted = sorted(pairs, key=lambda t: t[0])
    total_w = sum(w for _, w in pairs_sorted)
    if total_w <= 0:
        return pairs_sorted[-1][0]
    thresh = max(0.0, min(1.0, float(q))) * total_w
    acc = 0.0
    for v, w in pairs_sorted:
        acc += w
        if acc >= thresh:
            return float(v)
    return float(pairs_sorted[-1][0])


@dataclass
class EpsilonSummary:
    weighted_mean: float
    weighted_quantiles: Dict[float, float]
    wmse_neural_vs_gt: float
    num_histories: int


def estimate_js_epsilon(
    stringY: str,
    provider: ModelProvider,
    L_max: int,
    device: torch.device,
    min_count: int = 5,
    context_window: int = 32,
    prob_clip: float = 1e-6,
    gt_machine: str = 'hierarchical_4_state',
    quantiles: Sequence[float] = (0.9, 0.95, 0.99),
    platt_fit: bool = True,
) -> EpsilonSummary:
    # Count marginals n(h) for histories up to L_max
    counts: Dict[str, int] = {}
    ones: Dict[str, int] = {}
    for L in range(1, L_max + 1):
        for t in range(L, len(stringY)):
            h = stringY[t - L : t]
            counts[h] = counts.get(h, 0) + 1
            if stringY[t] == '1':
                ones[h] = ones.get(h, 0) + 1

    # Optional Platt fit using empirical targets for calibration of provider
    if platt_fit:
        margins: List[float] = []
        targets: List[float] = []
        weights: List[float] = []
        provider.platt_params = None
        for h, c in counts.items():
            if c <= 0:
                continue
            ids = [int(cch) for cch in h][-context_window:]
            p0, p1 = provider.next_probs(ids)
            p1c = max(1e-12, min(1.0 - 1e-12, float(p1)))
            margins.append(math.log(p1c / (1.0 - p1c)))
            targets.append(ones.get(h, 0) / float(c))
            weights.append(float(c))
        if margins:
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
            provider.platt_params = {'a': float(a.detach().cpu().item()), 'b': float(b.detach().cpu().item())}

    # Choose GT function
    if gt_machine == 'seven_state_human':
        p1_gt_fn = _p1_gt_from_history_seven_state
    elif gt_machine == 'hierarchical_4_state':
        p1_gt_fn = _p1_gt_from_history_hier4
    else:
        raise SystemExit(f"Unsupported gt_machine: {gt_machine}")

    # Compute divergences
    js_pairs: List[Tuple[float, float]] = []
    wmse_num = 0.0
    wmse_den = 0.0
    for h, c in counts.items():
        if c < min_count:
            continue
        ids = [int(ch) for ch in h][-context_window:]
        p0, p1 = provider.next_probs(ids)
        p1_gt = float(p1_gt_fn(h))
        js_val = js_divergence_bernoulli(float(p1), p1_gt, clip=prob_clip)
        js_pairs.append((js_val, float(c)))
        wmse_num += float(c) * (p1 - p1_gt) * (p1 - p1_gt)
        wmse_den += float(c)

    total_w = sum(w for _, w in js_pairs)
    wmean = (sum(v * w for v, w in js_pairs) / total_w) if total_w > 0 else 0.0
    qvals: Dict[float, float] = {}
    for q in quantiles:
        qvals[float(q)] = weighted_quantile(js_pairs, float(q))

    wmse = float(wmse_num / wmse_den) if wmse_den > 0 else 0.0
    return EpsilonSummary(weighted_mean=float(wmean), weighted_quantiles=qvals, wmse_neural_vs_gt=wmse, num_histories=len(js_pairs))


def main() -> None:
    p = argparse.ArgumentParser(description='Estimate JS epsilon from neural vs GT without running CSSR')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model_ckpt', type=Path, required=True)
    p.add_argument('--L_max', type=int, default=6)
    p.add_argument('--context_window', type=int, default=32)
    p.add_argument('--min_count', type=int, default=5)
    p.add_argument('--prob_clip', type=float, default=1e-6)
    p.add_argument('--quantiles', type=float, nargs='+', default=[0.9, 0.95, 0.99])
    p.add_argument('--gt_machine', type=str, choices=['seven_state_human', 'hierarchical_4_state'], default=None,
                   help='If not set, attempt to infer from data path')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--platt_fit', action='store_true', default=True)
    p.add_argument('--output_json', type=Path, required=False)
    args = p.parse_args()

    # Resolve device
    dev_str = str(args.device)
    if dev_str.lower() == 'auto':
        dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_str)

    # Load data
    stringY = load_binary_string(args.data)

    # Infer GT machine if not provided
    gt_machine = args.gt_machine
    if gt_machine is None:
        dstr = str(args.data)
        if 'seven_state_human' in dstr:
            gt_machine = 'seven_state_human'
        elif 'hierarchical_4_state' in dstr:
            gt_machine = 'hierarchical_4_state'
        else:
            raise SystemExit('--gt_machine must be specified for unknown dataset types')

    # Load model/provider
    try:
        model, ctx_win_loaded, model_type = load_model_from_ckpt(args.model_ckpt, device)
        model = model.to(device).eval()
        context_window = int(args.context_window or ctx_win_loaded)
        used_model = model
    except Exception as e:
        # Fallback nanoGPT loader from runner to keep scripts decoupled
        from transcssr_neural_runner import _load_nano_gpt_model  # type: ignore
        print(f"Falling back to nanoGPT loader for {args.model_ckpt}: {e}")
        used_model, block_size = _load_nano_gpt_model(args.model_ckpt, device)
        context_window = int(args.context_window or block_size)

    provider = ModelProvider(used_model, device=device, context_window=context_window,
                             temperature=1.0, prob_clip=args.prob_clip)

    # Estimate
    summary = estimate_js_epsilon(
        stringY=stringY,
        provider=provider,
        L_max=int(args.L_max),
        device=device,
        min_count=int(args.min_count),
        context_window=int(context_window),
        prob_clip=float(args.prob_clip),
        gt_machine=str(gt_machine),
        quantiles=[float(q) for q in args.quantiles],
        platt_fit=bool(args.platt_fit),
    )

    # Print results
    print(f"Histories used: {summary.num_histories}")
    print(f"Weighted MSE (neural vs GT): {summary.wmse_neural_vs_gt:.6f}")
    print(f"Recommended epsilon (weighted mean): {summary.weighted_mean:.6f}")
    for q, v in sorted(summary.weighted_quantiles.items()):
        print(f"Recommended epsilon (weighted q={q:.3f}): {v:.6f}")

    # Save JSON if requested
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            'data': str(args.data),
            'model_ckpt': str(args.model_ckpt),
            'L_max': int(args.L_max),
            'context_window_used': int(context_window),
            'min_count': int(args.min_count),
            'prob_clip': float(args.prob_clip),
            'gt_machine': str(gt_machine),
            'device': str(dev_str),
            'wmse_neural_vs_gt': float(summary.wmse_neural_vs_gt),
            'weighted_mean_epsilon': float(summary.weighted_mean),
            'weighted_quantiles': {str(k): float(v) for k, v in summary.weighted_quantiles.items()},
            'num_histories': int(summary.num_histories),
            'platt_params': getattr(provider, 'platt_params', None),
        }
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"Saved epsilon estimates to {args.output_json}")


if __name__ == '__main__':
    main()


