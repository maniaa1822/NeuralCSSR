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


def _load_nano_gpt_model(model_ckpt: Path, device: torch.device):
    """Load a nanoGPT checkpoint (ckpt.pt) and return (model, block_size)."""
    ckpt_path = Path(model_ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / 'ckpt.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"nanoGPT checkpoint not found: {ckpt_path}")
    # import nanoGPT model
    repo_root = Path(__file__).resolve().parents[0]
    nano_dir = repo_root / 'nanoGPT'
    if str(nano_dir) not in sys.path:
        sys.path.insert(0, str(nano_dir))
    from model import GPTConfig, GPT  # type: ignore
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model_args = ckpt.get('model_args', {})
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    block_size = getattr(model.config, 'block_size', 32)
    return model, int(block_size)


def _p1_gt_from_history_seven_state(hist_y: str) -> float:
    """GT P(1|hist_y) via stationary posterior mixture over Seven-State Human machine states."""
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _machine_spec():
        states = ['BB', 'AAA', 'AAAB', 'BA', 'BAB', 'BAAB', 'BAA']
        idx = {s: i for i, s in enumerate(states)}
        p1 = {
            'BB': 1.0/16.0,
            'AAA': 13.0/16.0,
            'AAAB': 9.0/16.0,
            'BA': 9.0/16.0,
            'BAB': 8.0/16.0,
            'BAAB': 9.0/16.0,
            'BAA': 13.0/16.0,
        }
        t0 = {
            'BB': 'BA', 'AAA': 'AAA', 'AAAB': 'BA', 'BA': 'BAA', 'BAB': 'BA', 'BAAB': 'BA', 'BAA': 'AAA'
        }
        t1 = {
            'BB': 'BB', 'AAA': 'AAAB', 'AAAB': 'BB', 'BA': 'BAB', 'BAB': 'BB', 'BAAB': 'BB', 'BAA': 'BAAB'
        }
        p1_arr = [float(p1[s]) for s in states]
        t0_idx = [idx[t0[s]] for s in states]
        t1_idx = [idx[t1[s]] for s in states]
        n = len(states)
        pi = [1.0 / n] * n
        for _ in range(10000):
            new = [0.0] * n
            for i in range(n):
                p1_i = p1_arr[i]
                p0_i = 1.0 - p1_i
                new[t0_idx[i]] += pi[i] * p0_i
                new[t1_idx[i]] += pi[i] * p1_i
            diff = sum(abs(new[i] - pi[i]) for i in range(n))
            pi = new
            if diff < 1e-15:
                break
        s = sum(pi)
        if s <= 0:
            pi = [1.0 / n] * n
        else:
            pi = [x / s for x in pi]
        return states, idx, t0_idx, t1_idx, p1_arr, pi

    states, idx, t0_idx, t1_idx, p1_arr, pi = _machine_spec()
    n = len(states)
    weights = [0.0] * n
    for i in range(n):
        cur = i
        prob = 1.0
        for ch in hist_y:
            if ch == '0':
                prob *= (1.0 - p1_arr[cur])
                cur = t0_idx[cur]
            elif ch == '1':
                prob *= p1_arr[cur]
                cur = t1_idx[cur]
            else:
                continue
        weights[cur] += pi[i] * prob
    denom = sum(weights)
    if denom <= 0.0:
        return float(sum(pi[k] * p1_arr[k] for k in range(n)))
    return float(sum(weights[k] * p1_arr[k] for k in range(n)) / denom)


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
    # Early-layer head integration
    p.add_argument('--probe_layer', type=str, default='final', help="'final' to use model's final head; otherwise an integer layer index for early-layer head")
    p.add_argument('--head_ckpt', type=Path, help='Path to trained early-layer head checkpoint (.pt)')
    p.add_argument('--platt_json', type=Path, help='Optional Platt calibration JSON with keys {a,b} for provider')
    # Enable on-the-fly Platt calibration using empirical targets derived from data
    p.add_argument('--platt_fit', action='store_true', default=False, help='Fit Platt calibration on-the-fly against empirical targets and apply to neural probabilities')
    p.add_argument('--output_json', type=Path, required=True)
    # Optional diff output (empirical vs neural probability per history)
    p.add_argument('--dump_prob_diff', type=Path, help='If set, dump CSV of empirical vs neural probabilities over histories')
    p.add_argument('--diff_min_count', type=int, default=5, help='Min marginal count for a history to include in the diff CSV and MSE')
    # Optional: compute ground-truth WMSE vs known machine
    p.add_argument('--gt_machine', type=str, choices=['seven_state_human'], help='If set, also compute WMSE vs ground-truth machine and include p_gt in dumps')
    args = p.parse_args()

    # Import our ModelProvider and loader
    from neural_cssr_state_averaging import ModelProvider, load_model_from_ckpt

    # Import transCSSR
    repo_root = Path(__file__).resolve().parents[0]
    transcssr_dir = repo_root / 'transCSSR'
    sys.path.insert(0, str(transcssr_dir))
    from transCSSR import estimate_predictive_distributions, run_transCSSR, get_transitions
    import itertools

    # Resolve device string; accept 'auto' for convenience
    dev_str = str(args.device)
    if dev_str.lower() == 'auto':
        dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_str)

    # Load data as strings for transCSSR
    stringY = load_binary_string(args.data)
    stringX = '0' * len(stringY)  # null input process

    # Default GT machine detection (enable GT metrics by default for seven_state dataset)
    if args.gt_machine is None:
        try:
            if 'seven_state_human' in str(args.data):
                args.gt_machine = 'seven_state_human'
        except Exception:
            pass

    provider = None
    context_window = args.context_window
    wmse_value = None
    if args.backend == 'neural':
        if args.model_ckpt is None:
            raise SystemExit('--model_ckpt is required when --backend neural')
        # Load base model (EBM/AR) with fallback to nanoGPT
        try:
            model, ctx_win_loaded, model_type = load_model_from_ckpt(args.model_ckpt, device)
            model = model.to(device).eval()
            context_window = args.context_window if args.context_window else ctx_win_loaded
            used_model = model
            # Optionally wrap with early-layer non-linear head (only for supported models)
            if str(args.probe_layer).lower() != 'final':
                try:
                    layer_index = int(args.probe_layer)
                except Exception:
                    raise SystemExit(f"--probe_layer must be 'final' or an integer index; got {args.probe_layer}")
                if args.head_ckpt is None:
                    raise SystemExit('--head_ckpt is required when using an early-layer head')
                # Lazy import to avoid hard dependency
                repo_root = Path(__file__).resolve().parents[0]
                ebm_dir = repo_root / 'experiments' / 'ebm'
                if str(ebm_dir) not in sys.path:
                    sys.path.insert(0, str(ebm_dir))
                from layer0_head import NonlinearEarlyLayerHead  # type: ignore
                d_model = getattr(model, 'd_model', None)
                if d_model is None:
                    raise SystemExit('Base model missing d_model attribute; cannot attach early-layer head')
                wrapper = NonlinearEarlyLayerHead(model, layer_index=layer_index, d_model=int(d_model))
                wrapper.load_head_state(args.head_ckpt)
                used_model = wrapper.to(device).eval()
            provider = ModelProvider(used_model, device=device, context_window=context_window,
                                     temperature=args.temperature, prob_clip=args.prob_clip)
        except Exception as e:
            # Fallback: load nanoGPT checkpoint
            print(f"Falling back to nanoGPT loader for {args.model_ckpt}: {e}")
            ngpt_model, block_size = _load_nano_gpt_model(args.model_ckpt, device)
            context_window = args.context_window if args.context_window else block_size
            used_model = ngpt_model
            provider = ModelProvider(used_model, device=device, context_window=int(context_window),
                                     temperature=args.temperature, prob_clip=args.prob_clip)

    # Optional: load Platt calibration from JSON
    if args.platt_json is not None and args.backend == 'neural':
        try:
            pl = json.loads(args.platt_json.read_text())
            if isinstance(pl, dict) and 'a' in pl and 'b' in pl:
                provider.platt_params = {'a': float(pl['a']), 'b': float(pl['b'])}
                print(f"Loaded Platt params from {args.platt_json}: {provider.platt_params}")
        except Exception as e:
            print(f"Warning: failed to load Platt JSON {args.platt_json}: {e}")

    # Optional Platt fit using histories up to L_max
    if args.platt_fit and args.backend == 'neural':
        counts = {}
        ones = {}
        # Ensure we fit using RAW model probabilities (no previous calibration)
        provider.platt_params = None
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
        print(f'Platt fitting: collected {len(margins)} history-probability pairs')
        if margins:
            print(f'Sample margins: {margins[:5]} ...')
            print(f'Sample targets: {targets[:5]} ...')
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
            # Done fitting; previous calibration is intentionally overridden by new fit
        else:
            print('No margins collected for Platt fitting - skipping calibration')

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

        # Compute per-history neural probabilities (with calibration/mix) and weighted MSE vs empirical
        scale = float(args.pseudo_count_scale)
        wmse_num = 0.0; wmse_den = 0.0
        wmse_gt_num = 0.0; wmse_gt_den = 0.0
        wmse_emp_gt_num = 0.0; wmse_emp_gt_den = 0.0
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
            # Optional GT WMSE
            if args.gt_machine == 'seven_state_human':
                p1_gt = _p1_gt_from_history_seven_state(hist_y)
                wmse_gt_num += float(m_raw) * (p1_gt - p1) * (p1_gt - p1)
                wmse_gt_den += float(m_raw)
                wmse_emp_gt_num += float(m_raw) * (p1_gt - p_emp1) * (p1_gt - p_emp1)
                wmse_emp_gt_den += float(m_raw)
            if args.dump_prob_diff is not None:
                if args.gt_machine == 'seven_state_human':
                    rows.append([hist_y, int(m_raw), p_emp1, p1, p1_gt])
                else:
                    rows.append([hist_y, int(m_raw), p_emp1, p1])

        if args.dump_prob_diff is not None and rows:
            import csv
            args.dump_prob_diff.parent.mkdir(parents=True, exist_ok=True)
            with args.dump_prob_diff.open('w', newline='') as f:
                w = csv.writer(f)
                if args.gt_machine == 'seven_state_human':
                    w.writerow(['history', 'marg_count', 'p_emp', 'p_neural', 'p_gt'])
                else:
                    w.writerow(['history', 'marg_count', 'p_emp', 'p_neural'])
                for r in sorted(rows, key=lambda x: (-x[1], x[0])):
                    w.writerow(r)
            print(f"Wrote probability comparison to {args.dump_prob_diff}")
        if wmse_den > 0:
            wmse_value = float(wmse_num / wmse_den)
            print(f"Weighted MSE (neural vs empirical): {wmse_value:.6f}")
        wmse_gt_value = None
        if wmse_gt_den > 0:
            wmse_gt_value = float(wmse_gt_num / wmse_gt_den)
            print(f"Weighted MSE (neural vs GT): {wmse_gt_value:.6f}")
        wmse_emp_gt_value = None
        if wmse_emp_gt_den > 0:
            wmse_emp_gt_value = float(wmse_emp_gt_num / wmse_emp_gt_den)
            print(f"Weighted MSE (empirical vs GT): {wmse_emp_gt_value:.6f}")

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

    # Build configuration summary
    cfg = {
        'data': str(args.data),
        'backend': str(args.backend),
        'L_max': int(args.L_max),
        'alpha': float(args.alpha),
        'test_method': str(args.test_method),
        'device': str(dev_str),
        'pseudo_count_scale': float(args.pseudo_count_scale),
        'cap_marginal_counts': bool(args.cap_marginal_counts),
        'context_window_used': int(context_window),
        'temperature': float(args.temperature),
        'prob_clip': float(args.prob_clip),
        'mix_empirical': float(args.mix_empirical),
        'probe_layer': str(args.probe_layer),
        'head_ckpt': (str(args.head_ckpt) if args.head_ckpt is not None else None),
        'model_ckpt': (str(args.model_ckpt) if args.model_ckpt is not None else None),
        'platt_json': (str(args.platt_json) if args.platt_json is not None else None),
    }
    # Attach platt params if available
    try:
        if provider is not None and getattr(provider, 'platt_params', None) is not None:
            cfg['platt_params'] = {
                'a': float(provider.platt_params.get('a', 1.0)),
                'b': float(provider.platt_params.get('b', 0.0)),
            }
    except Exception:
        pass

    out = {
        'num_states': len(states_out),
        'states': states_out,
        'config': cfg,
    }
    if wmse_value is not None:
        out['metrics'] = {'wmse_neural_vs_empirical': float(wmse_value)}
        try:
            if wmse_gt_value is not None:
                out['metrics']['wmse_neural_vs_gt'] = float(wmse_gt_value)
            if wmse_emp_gt_value is not None:
                out['metrics']['wmse_empirical_vs_gt'] = float(wmse_emp_gt_value)
        except Exception:
            pass
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"Saved result to {args.output_json}")


if __name__ == '__main__':
    main()


