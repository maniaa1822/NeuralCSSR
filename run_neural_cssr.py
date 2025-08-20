#!/usr/bin/env python3
"""
Run Neural CSSR on a .dat sequence using a causal decoder (cssr_nlp) as the
probability provider for sufficiency tests. This avoids sliding-window models
and uses standard LM block packing for training the transformer.

Example:
  uv run -- python -u run_neural_cssr.py \
    --data domain_machines/golden_mean/golden_mean/golden_mean.dat \
    --L_max 8 --alpha 0.05 \
    --context_window 32 --epochs 2 --d_model 32 --layers 2 --heads 2
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random

import torch
import torch.nn.functional as F

from src.neural_cssr.neural import NeuralCSSRProbabilityProvider
from src.neural_cssr.classical.cssr import ClassicalCSSR
from src.neural_cssr.classical.transcssr_wrapper import TransCSSRWrapper

# Optional: nanoGPT integration
try:
    from nanoGPT.model import GPTConfig as NanoGPTConfig, GPT as NanoGPT
    _HAS_NANOGPT = True
except Exception:
    _HAS_NANOGPT = False


class NanoGPTAdapter(torch.nn.Module):
    """Adapter to expose nanoGPT's (logits, loss) forward as logits-only for CSSR provider."""
    def __init__(self, gpt: 'NanoGPT'):
        super().__init__()
        self.gpt = gpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.gpt(x)
        return logits


def load_binary_tokens(dat_path: Path) -> List[int]:
    content = dat_path.read_text().strip()
    tokens = [int(c) for c in content if c in '01']
    if not tokens:
        raise ValueError(f"No binary tokens found in {dat_path}")
    return tokens


def make_fixed_blocks(tokens: List[int], block_size: int, offset: int) -> List[List[int]]:
    blocks: List[List[int]] = []
    start = offset % block_size
    for i in range(start, len(tokens) - block_size + 1, block_size):
        blocks.append(tokens[i:i + block_size])
    return blocks


def train_cssr_nlp_on_dat(data_path: Path, model, context_window: int, epochs: int, lr: float, device: torch.device,
                          random_offset: bool = True, quiet: bool = False) -> None:
    tokens = load_binary_tokens(data_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        offset = random.randrange(context_window) if random_offset else 0
        blocks = make_fixed_blocks(tokens, context_window, offset)
        total_loss = 0.0
        for seq in blocks:
            inp = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            tgt = torch.tensor(seq[1:], dtype=torch.long, device=device)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(blocks))
        if not quiet:
            print(f"Train epoch {epoch} | offset {offset} | blocks {len(blocks)} | loss {avg_loss:.4f}")


@torch.no_grad()
def model_next_probs(model, history_ids: List[int], device: torch.device, temperature: float = 1.0,
                     prob_clip: float = 1e-3) -> Tuple[float, float]:
    ids = history_ids
    if not ids:
        p0 = p1 = 0.5
    else:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[:, -1]
        if temperature and temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        p0 = float(probs[0].item()); p1 = float(probs[1].item())
    # clip and renormalize
    p0 = max(prob_clip, min(1.0 - prob_clip, p0))
    p1 = max(prob_clip, min(1.0 - prob_clip, p1))
    s = p0 + p1
    return p0 / s, p1 / s


@torch.no_grad()
def future_distribution(model, history_ids: List[int], device: torch.device, horizon: int, max_len: int,
                        temperature: float = 1.0, prob_clip: float = 1e-3) -> List[float]:
    """Return probability over all binary sequences of length horizon in lexicographic order."""
    # BFS over futures
    seqs = [([], 1.0, list(history_ids))]  # (future, prob, hist_ids)
    for _ in range(horizon):
        new = []
        for fut, pr, hids in seqs:
            # keep last max_len tokens
            hcut = hids[-max_len:]
            p0, p1 = model_next_probs(model, hcut, device, temperature, prob_clip)
            new.append((fut + [0], pr * p0, hids + [0]))
            new.append((fut + [1], pr * p1, hids + [1]))
        seqs = new
    # order futures lexicographically by bitstring
    seqs.sort(key=lambda x: ''.join(str(b) for b in x[0]))
    probs = [pr for _, pr, _ in seqs]
    s = sum(probs)
    if s > 0:
        probs = [p / s for p in probs]
    return probs


def js_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    import math
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, eps); bi = max(bi, eps)
            s += ai * math.log(ai / bi)
        return s
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def observed_histories(tokens: List[int], L: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in range(L, len(tokens)):
        hist = ''.join(str(x) for x in tokens[t - L:t])
        counts[hist] = counts.get(hist, 0) + 1
    return counts


def run_neural_cssr_js(data_path: Path, model, L_max: int, context_window: int, device: torch.device,
                        metric: str = 'js', horizon: int = 3, threshold: float = 0.02,
                        temperature: float = 1.0, prob_clip: float = 1e-3,
                        min_count: int = 5) -> Dict:
    tokens = load_binary_tokens(data_path)
    states: List[Dict] = []  # each: {'histories': set, 'vec': avg_dist, 'weight': total_count}
    # Iterate lengths
    for L in range(1, L_max + 1):
        hist_counts = observed_histories(tokens, L)
        # process histories sorted by count desc for stability
        for hist, cnt in sorted(hist_counts.items(), key=lambda x: -x[1]):
            if cnt < min_count:
                continue
            ids = [int(c) for c in hist][-context_window:]
            if metric == 'js':
                p0, p1 = model_next_probs(model, ids, device, temperature, prob_clip)
                vec = [p0, p1]
            else:  # 'jsh'
                vec = future_distribution(model, ids, device, horizon, context_window, temperature, prob_clip)
            # find best state
            best_idx = -1; best_d = float('inf')
            for i, st in enumerate(states):
                d = js_divergence(vec, st['vec'])
                if d < best_d:
                    best_d = d; best_idx = i
            if best_idx == -1 or best_d >= threshold:
                # create new state
                states.append({'histories': {hist}, 'vec': vec[:], 'weight': cnt})
            else:
                st = states[best_idx]
                st['histories'].add(hist)
                # weighted average update
                w_old = st['weight']; w_new = w_old + cnt
                st['vec'] = [ (w_old * a + cnt * b) / w_new for a, b in zip(st['vec'], vec) ]
                st['weight'] = w_new

    # Build epsilon and invepsilon mappings
    epsilon = {}
    invepsilon = {}
    for sid, st in enumerate(states):
        invepsilon[sid] = list(st['histories'])
        for h in st['histories']:
            # record as (Xpast, Ypast) with X all zeros for compatibility
            epsilon[( '0' * len(h), h)] = sid

    # Build simple morphs from state vec
    morph_by_state = {}
    for sid, st in enumerate(states):
        if metric == 'js':
            p0, p1 = st['vec']
            morph_by_state[sid] = [p0, p1]
        else:
            # Not a morph; keep last-step distribution approximated from futures (marginalize first bit)
            # Compute from one step next probs averaged from member histories
            accum0 = 0.0; accum1 = 0.0; w=0.0
            for h in st['histories']:
                ids = [int(c) for c in h][-context_window:]
                a0, a1 = model_next_probs(model, ids, device, temperature, prob_clip)
                c = 1.0
                accum0 += c * a0; accum1 += c * a1; w += c
            p0 = accum0 / max(w, 1e-9); p1 = accum1 / max(w, 1e-9)
            morph_by_state[sid] = [p0, p1]

    result = {
        'discovered_structure': {
            'num_states': len(states),
            'states': {str(i): {
                'id': i,
                'histories': invepsilon[i],
                'distribution_vector': states[i]['vec'],
                'weight': states[i]['weight']
            } for i in range(len(states))},
            'epsilon_mapping': {f"({k[0]}, {k[1]})": v for k, v in epsilon.items()},
        },
        'execution_info': {
            'converged': True,
            'algorithm': f'neural-js-{metric}',
            'parameters': {
                'L_max': L_max,
                'threshold': threshold,
                'horizon': horizon,
                'prob_clip': prob_clip,
                'temperature': temperature,
                'min_count': min_count,
            }
        }
    }
    return result


def main():
    p = argparse.ArgumentParser(description="Neural CSSR with transformer probability provider")
    p.add_argument('--data', type=Path, required=True, help='Path to .dat file')
    p.add_argument('--L_max', type=int, default=8, help='Maximum history length for CSSR')
    p.add_argument('--alpha', type=float, default=0.05, help='Significance level for sufficiency tests')
    p.add_argument('--context_window', type=int, default=32, help='Transformer context window (block size)')
    p.add_argument('--epochs', type=int, default=2, help='Training epochs for transformer')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate for transformer')
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--test_type', type=str, choices=['chi_square','kl_divergence','permutation'], default='kl_divergence')
    p.add_argument('--backend', type=str, choices=['internal','transcssr','neural_js'], default='internal', help='Which CSSR backend to use')
    p.add_argument('--pseudo_count_scale', type=int, default=100, help='Pseudo-count mass for transCSSR backend')
    p.add_argument('--json_only', action='store_true', help='Emit only final JSON to stdout (suppress logs)')
    p.add_argument('--mix_empirical', type=float, default=0.0, help='Mix factor in [0,1] to blend empirical counts with neural probs (transcssr backend)')
    p.add_argument('--temp_counts', type=float, default=1.0, help='Temperature to soften probabilities before forming counts (transcssr backend)')
    # JS/JS-H settings
    p.add_argument('--state_metric', type=str, choices=['js','jsh'], default='js')
    p.add_argument('--js_threshold', type=float, default=0.02)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--prob_clip', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--min_count', type=int, default=5)
    # Model I/O
    p.add_argument('--model_out', type=Path, help='Path to save trained model .pt')
    p.add_argument('--model_in', type=Path, help='Path to load model .pt (skip training if provided)')
    # nanoGPT integration
    p.add_argument('--nanogpt_out_dir', type=Path, help='Path to nanoGPT out_dir containing ckpt.pt (use nanoGPT model)')
    # DOT export
    p.add_argument('--dot_out', type=Path, help='Optional path to save DOT for neural_js backend')
    args = p.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu'))

    # Create / load model
    if args.nanogpt_out_dir:
        if not _HAS_NANOGPT:
            raise RuntimeError("nanoGPT not available for import. Ensure nanoGPT package directory is present.")
        ckpt_path = args.nanogpt_out_dir / 'ckpt.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(f"nanoGPT checkpoint not found at {ckpt_path}")
        ckpt_ng = torch.load(ckpt_path, map_location=device)
        gptconf = NanoGPTConfig(**ckpt_ng['model_args'])
        gpt = NanoGPT(gptconf).to(device)
        state_dict = ckpt_ng['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        gpt.load_state_dict(state_dict)
        model = NanoGPTAdapter(gpt)
        if not args.json_only:
            print(f"Loaded nanoGPT model from {ckpt_path} (vocab_size={gpt.config.vocab_size}, block_size={gpt.config.block_size})")
    elif args.model_in and args.model_in.exists():
        ckpt = torch.load(args.model_in, map_location=device)
        cfg = ckpt.get('config', {})
        # Create model matching checkpoint config if available
        d_model = cfg.get('d_model', args.d_model)
        n_layers = cfg.get('n_layers', args.layers)
        n_heads = cfg.get('n_heads', args.heads)
        dropout = cfg.get('dropout', args.dropout)
        model = create_model('cssr_nlp', d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        model.load_state_dict(ckpt['state_dict'])
        if not args.json_only:
            print(f"Loaded model from {args.model_in}")
    else:
        model = create_model('cssr_nlp', d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout)
        if not args.json_only:
            print(f"Created cssr_nlp model with d_model={args.d_model}, layers={args.layers}, heads={args.heads}")
        train_cssr_nlp_on_dat(args.data, model, args.context_window, args.epochs, args.lr, device, quiet=args.json_only)
        # Save if requested
        if args.model_out:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'config': {'d_model': args.d_model, 'n_layers': args.layers, 'n_heads': args.heads, 'dropout': args.dropout}}, args.model_out)
            if not args.json_only:
                print(f"Saved model to {args.model_out}")

    # Build neural probability provider
    provider = NeuralCSSRProbabilityProvider(model, device=device.type, context_window=args.context_window)

    if args.backend == 'internal':
        # Run CSSR with our internal neural provider backend
        cssr = ClassicalCSSR(significance_level=args.alpha, test_type=args.test_type, neural_probability_provider=provider)
        converged = cssr.run_cssr(max_history_length=args.L_max)
        if not args.json_only:
            print(f"CSSR converged: {converged}")
        summary = cssr.get_results_summary()
    elif args.backend == 'transcssr':
        # Run CSSR via transCSSR using neural-weighted observed counts
        string_y = args.data.read_text().strip()
        string_x = '0' * len(string_y)
        # Map our CLI test types to transCSSR's expected identifiers
        test_type_map = {
            'chi_square': 'chi2',
            'kl_divergence': 'G',
            'permutation': 'chi2',  # not supported in transCSSR; fallback
        }
        wrapper = TransCSSRWrapper(significance_level=args.alpha, test_type=test_type_map.get(args.test_type, 'chi2'))
        result = wrapper.run_cssr_neural(
            provider, string_x, string_y, max_length=args.L_max,
            pseudo_count_scale=args.pseudo_count_scale,
            prob_clip=args.prob_clip if hasattr(args, 'prob_clip') else 1e-3,
            temperature=args.temp_counts,
            mix_empirical=args.mix_empirical,
        )
        summary = result
    else:
        # JS/JS-H backend
        summary = run_neural_cssr_js(
            args.data, model, args.L_max, args.context_window, device,
            metric=args.state_metric, horizon=args.horizon, threshold=args.js_threshold,
            temperature=args.temperature, prob_clip=args.prob_clip, min_count=args.min_count
        )
        # Optional DOT export for neural_js
        if args.dot_out:
            try:
                def next_morph_from_vec(vec: List[float], metric: str, horizon: int) -> Tuple[float,float]:
                    if metric == 'js':
                        p0, p1 = vec
                        return p0, p1
                    # jsh: sum futures where first bit is 0 vs 1
                    half = len(vec)//2
                    p1 = sum(vec[half:])
                    p0 = sum(vec[:half])
                    s = p0 + p1 if (p0+p1)>0 else 1.0
                    return p0/s, p1/s

                states = summary['discovered_structure']['states']
                epsilon_map = summary['discovered_structure']['epsilon_mapping']
                # Build transitions by majority vote of extensions
                # Create reverse map: history -> state id
                hist_to_state: Dict[str,int] = {}
                for sid, sd in states.items():
                    for h in sd['histories']:
                        hist_to_state[h] = int(sid)

                transitions: Dict[int, Dict[int,int]] = {}
                for sid, sd in states.items():
                    transitions[int(sid)] = {}
                    votes0: Dict[int,int] = {}
                    votes1: Dict[int,int] = {}
                    for h in sd['histories']:
                        ext0 = (h + '0')[-args.L_max:]
                        ext1 = (h + '1')[-args.L_max:]
                        # Assign to nearest state using current centroids
                        ids0 = [int(c) for c in ext0][-args.context_window:]
                        ids1 = [int(c) for c in ext1][-args.context_window:]
                        if args.state_metric == 'js':
                            v0 = list(model_next_probs(model, ids0, device, args.temperature, args.prob_clip))
                            v1 = list(model_next_probs(model, ids1, device, args.temperature, args.prob_clip))
                        else:
                            v0 = future_distribution(model, ids0, device, args.horizon, args.context_window, args.temperature, args.prob_clip)
                            v1 = future_distribution(model, ids1, device, args.horizon, args.context_window, args.temperature, args.prob_clip)
                        # nearest centroid
                        best0, bestd0 = None, 1e9
                        best1, bestd1 = None, 1e9
                        for sid2, sd2 in states.items():
                            d0 = js_divergence(v0, sd2['distribution_vector'])
                            if d0 < bestd0:
                                bestd0 = d0; best0 = int(sid2)
                            d1 = js_divergence(v1, sd2['distribution_vector'])
                            if d1 < bestd1:
                                bestd1 = d1; best1 = int(sid2)
                        votes0[best0] = votes0.get(best0, 0) + 1
                        votes1[best1] = votes1.get(best1, 0) + 1
                    # pick majority
                    if votes0:
                        transitions[int(sid)][0] = max(votes0.items(), key=lambda x: x[1])[0]
                    if votes1:
                        transitions[int(sid)][1] = max(votes1.items(), key=lambda x: x[1])[0]

                # Write DOT
                args.dot_out.parent.mkdir(parents=True, exist_ok=True)
                with open(args.dot_out, 'w') as f:
                    f.write('digraph  {\n')
                    f.write('size = "6,8.5";\nratio = "fill";\nnode\n[shape = circle];\n')
                    f.write('node [fontsize = 24];\nnode [penwidth = 5];\nedge [fontsize = 24];\n')
                    f.write('node [fontname = "CMU Serif Roman"];\ngraph [fontname = "CMU Serif Roman"];\nedge [fontname = "CMU Serif Roman"];\n')
                    # Node names A, B, C, ...
                    sid_to_name = {i: chr(ord('A') + i) for i in range(len(states))}
                    # Edges with labels y|0:prob
                    for sid, sd in states.items():
                        sid_int = int(sid)
                        p0, p1 = next_morph_from_vec(sd['distribution_vector'], args.state_metric, args.horizon)
                        # 1 edge
                        if 1 in transitions[sid_int]:
                            dst = transitions[sid_int][1]
                            f.write(f"{sid_to_name[sid_int]} -> {sid_to_name[dst]} [label = \"1|0:{p1:.3f}\\l\"];\n")
                        # 0 edge
                        if 0 in transitions[sid_int]:
                            dst = transitions[sid_int][0]
                            f.write(f"{sid_to_name[sid_int]} -> {sid_to_name[dst]} [label = \"0|0:{p0:.3f}\\l\"];\n")
                    f.write('}\n')
                if not args.json_only:
                    print(f"Saved DOT to {args.dot_out}")
            except Exception as e:
                print(f"Warning: failed to write DOT: {e}")

    import json
    if args.json_only:
        # Emit compact JSON suitable for piping
        print(json.dumps(summary))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()


