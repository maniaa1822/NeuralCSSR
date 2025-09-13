#!/usr/bin/env python3
"""
Inject neural probabilities into the reference transCSSR algorithm.

Workflow (unchanged transCSSR mechanics):
- Build word lookups with estimate_predictive_distributions
- Replace future counts with neural-probability-based pseudo-counts
- Run run_transCSSR with chosen test (chi2/G) and alpha

Probabilities are computed via ModelProvider from neural_cssr_state_averaging.py
to adhere to the same proposal path used in our Neural CSSR implementation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_binary_string(dat_path: Path) -> str:
    s = dat_path.read_text().strip()
    s = ''.join(c for c in s if c in '01')
    if len(s) == 0:
        raise ValueError(f"No binary tokens in {dat_path}")
    return s


def main():
    p = argparse.ArgumentParser(description='Run transCSSR using neural probabilities for morphs')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model_ckpt', type=Path)
    p.add_argument('--L_max', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.001)
    p.add_argument('--test_method', type=str, choices=['chi2', 'G'], default='chi2')
    p.add_argument('--backend', type=str, choices=['neural', 'empirical'], default='neural', help='Use neural probabilities or empirical counts')
    p.add_argument('--pseudo_count_scale', type=float, default=50.0, help='Scale for converting probabilities to counts (neural only)')
    p.add_argument('--cap_marginal_counts', action='store_true', default=True, help='Cap per-history marginal counts to pseudo_count_scale when forming neural pseudo-counts')
    p.add_argument('--context_window', type=int, default=32)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--prob_clip', type=float, default=1e-6)
    p.add_argument('--mix_empirical', type=float, default=0.3, help='Linear mix with empirical p (0..1) applied to neural p')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--platt_fit', action='store_false', default=False, help='Apply Platt calibration to neural probabilities')
    p.add_argument('--output_json', type=Path, required=True)
    # Optional diff output (empirical vs neural probability per history)
    p.add_argument('--dump_prob_diff', type=Path, help='If set, dump CSV of empirical vs neural probabilities over histories')
    p.add_argument('--diff_min_count', type=int, default=5, help='Min marginal count for a history to include in the diff CSV and MSE')
    args = p.parse_args()

    # Import our ModelProvider and loader
    from neural_cssr_state_averaging import ModelProvider, load_model_from_ckpt

    # Import transCSSR
    repo_root = Path(__file__).resolve().parents[0]
    transcssr_dir = repo_root / 'transCSSR'
    sys.path.insert(0, str(transcssr_dir))
    from transCSSR import estimate_predictive_distributions, run_transCSSR, get_transitions
    import itertools

    device = torch.device(args.device)

    # Load data as strings for transCSSR
    stringY = load_binary_string(args.data)
    stringX = '0' * len(stringY)  # null input process

    provider = None
    context_window = args.context_window
    if args.backend == 'neural':
        if args.model_ckpt is None:
            raise SystemExit('--model_ckpt is required when --backend neural')
        # Load model
        model, ctx_win_loaded, model_type = load_model_from_ckpt(args.model_ckpt, device)
        model = model.to(device).eval()
        context_window = args.context_window if args.context_window else ctx_win_loaded
        provider = ModelProvider(model, device=device, context_window=context_window,
                                 temperature=args.temperature, prob_clip=args.prob_clip)

    # Optional Platt fit using histories up to L_max
    if args.platt_fit and args.backend == 'neural':
        counts = {}
        ones = {}
        for L in range(1, args.L_max + 1):
            for t in range(L, len(stringY)):
                h = stringY[t-L:t]
                counts[h] = counts.get(h, 0) + 1
                if stringY[t] == '1':
                    ones[h] = ones.get(h, 0) + 1
        # Weighted logistic regression on margins
        margins = []
        targets = []
        weights = []
        for h, c in counts.items():
            if c <= 0:
                continue
            ids = [int(cch) for cch in h][-context_window:]
            p0, p1 = provider.next_probs(ids)
            p1c = max(1e-12, min(1.0 - 1e-12, p1))
            import math
            margins.append(math.log(p1c / (1.0 - p1c)))
            targets.append(ones.get(h, 0) / float(c))
            weights.append(c)
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
            platt_params = {'a': float(a.detach().cpu().item()), 'b': float(b.detach().cpu().item())}
            provider.platt_params = platt_params
            print(f'Fitted Platt params: {platt_params}')

    # Get empirical lookups (structure)
    word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(stringX, stringY, args.L_max)

    # Replace futures with neural pseudo-counts scaled by marginal counts
    axs = ['0']
    ays = ['0', '1']
    e_symbols = list(itertools.product(axs, ays))

    if args.backend == 'neural':
        # Build a map of marginals for speed
        marg_counts: Dict[Tuple[str, str], float] = {}
        for (hist_x, hist_y), c in word_lookup_marg.items():
            marg_counts[(hist_x, hist_y)] = float(c)

        # Empirical next-probability per history (from original lookups)
        emp_p1_by_hist_y: Dict[str, float] = {}
        from collections import defaultdict
        fut_sum = defaultdict(float)
        fut1 = defaultdict(float)
        for (hist_x, fut_y), c in list(word_lookup_fut.items()):
            if len(fut_y) == 0:
                continue
            hist_y = fut_y[:-1]
            fut_sum[hist_y] += float(c)
            if fut_y[-1] == '1':
                fut1[hist_y] += float(c)
        for hy in fut_sum:
            s = fut_sum[hy]
            emp_p1_by_hist_y[hy] = (fut1.get(hy, 0.0) / s) if s > 0 else 0.5

        # Compute per-history neural probabilities (with calibration/mix) and weighted MSE
        scale = float(args.pseudo_count_scale)
        wmse_num = 0.0; wmse_den = 0.0
        rows = []
        for (hist_x, hist_y), m_raw in marg_counts.items():
            # Skip histories below threshold in diff/MSE
            if m_raw < args.diff_min_count:
                continue
            ids = [int(c) for c in hist_y][-context_window:]
            p0, p1 = provider.next_probs(ids)
            p_emp1 = emp_p1_by_hist_y.get(hist_y, 0.5)
            if args.mix_empirical > 0.0:
                mix = max(0.0, min(1.0, float(args.mix_empirical)))
                p1 = (1.0 - mix) * p1 + mix * p_emp1
                p1 = max(args.prob_clip, min(1.0 - args.prob_clip, p1))
            wmse_num += float(m_raw) * (p_emp1 - p1) * (p_emp1 - p1)
            wmse_den += float(m_raw)
            if args.dump_prob_diff is not None:
                rows.append([hist_y, int(m_raw), p_emp1, p1])

        if args.dump_prob_diff is not None and rows:
            import csv
            args.dump_prob_diff.parent.mkdir(parents=True, exist_ok=True)
            with args.dump_prob_diff.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['history', 'marg_count', 'p_emp', 'p_neural'])
                for r in sorted(rows, key=lambda x: (-x[1], x[0])):
                    w.writerow(r)
            print(f"Wrote probability comparison to {args.dump_prob_diff}")
        if wmse_den > 0:
            print(f"Weighted MSE (neural vs empirical): {wmse_num / wmse_den:.6f}")

        # Overwrite word_lookup_fut with neural counts that preserve marginals (optionally capped)
        for (hist_x, fut_y) in list(word_lookup_fut.keys()):
            if len(fut_y) == 0:
                continue
            hist_y = fut_y[:-1]
            # Model probability for next=1 given hist_y, with optional mixing
            ids = [int(c) for c in hist_y][-context_window:]
            p0, p1 = provider.next_probs(ids)
            if args.mix_empirical > 0.0:
                p_emp1 = emp_p1_by_hist_y.get(hist_y, 0.5)
                mix = max(0.0, min(1.0, float(args.mix_empirical)))
                p1 = (1.0 - mix) * p1 + mix * p_emp1
                p1 = max(args.prob_clip, min(1.0 - args.prob_clip, p1))
            # Marginal count for this history (transCSSR expects counts consistent with marginals)
            m = marg_counts.get((hist_x, hist_y), 0.0)
            if args.cap_marginal_counts and m > 0:
                m = min(m, scale)
            if m <= 0:
                # Fallback to a fixed scale
                m = scale
            # Assign counts coherently for both futures of this marginal
            word_lookup_fut[(hist_x, hist_y + '0')] = (1.0 - p1) * m
            word_lookup_fut[(hist_x, hist_y + '1')] = p1 * m

    # Run transCSSR
    test_type = args.test_method  # 'chi2' or 'G'
    epsilon, invepsilon, morph_by_state = run_transCSSR(
        word_lookup_marg, word_lookup_fut, args.L_max, axs, ays, e_symbols,
        '', 'neural_data', alpha=args.alpha, test_type=test_type
    )

    # Derive transitions
    trans_dict = get_transitions(epsilon, invepsilon, e_symbols, args.L_max)

    # Convert to JSON similar to our Neural CSSR output
    def state_distribution_vector(state_id: int):
        counts = morph_by_state[state_id]
        import numpy as np
        arr = torch.tensor(counts, dtype=torch.float32)
        # slice for x='0'
        p = arr[:len(ays)]
        s = float(p.sum().item())
        if s <= 0:
            return [0.5, 0.5]
        p = (p / s).tolist()
        return [float(1.0 - p[1]), float(p[1])]  # [p0, p1]

    # Build states list
    states_out = []
    # invepsilon is dict: state -> { (xhist, yhist): True }
    for sid in sorted(invepsilon.keys()):
        y_histories = sorted([y for (x, y) in invepsilon[sid].keys()])
        dv = state_distribution_vector(sid)
        # Derive transitions for y=0 and y=1 where available (input always '0')
        tmap = {}
        for y in ays:
            to = trans_dict.get((sid, ('0', y)), None)
            if to is not None:
                tmap[y] = to
        states_out.append({
            'id': int(sid),
            'histories': y_histories,
            'distribution_vector': dv,
            'weight': int(sum(len(h) for h in y_histories)),  # proxy
            'transitions': tmap
        })

    out = {
        'num_states': len(states_out),
        'states': states_out,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"Saved result to {args.output_json}")


if __name__ == '__main__':
    main()


