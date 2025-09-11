#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import math
from typing import List

import torch
import torch.nn.functional as F

# Local import path for models
try:
    from .models import EnergyBasedBinaryLM
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from models import EnergyBasedBinaryLM


def load_tokens(path: Path) -> List[int]:
    s = path.read_text().strip()
    return [int(c) for c in s if c in '01']


@torch.no_grad()
def eval_loss(model: EnergyBasedBinaryLM, tokens: List[int], device: torch.device, context_window: int, steps: int = 500, batch_size: int = 64):
    model.eval()
    losses = []
    for _ in range(steps):
        starts = torch.randint(low=0, high=max(1, len(tokens) - context_window - 1), size=(batch_size,))
        X, Y = [], []
        for s in starts.tolist():
            seq = tokens[s:s + context_window + 1]
            X.append(seq[:-1]); Y.append(seq[-1])
        x = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(Y, dtype=torch.long, device=device)
        attn = torch.ones((batch_size, context_window), dtype=torch.long, device=device)
        scores = model(x, attn)
        logits = scores[:, -1]
        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
    nats = sum(losses) / max(1, len(losses))
    return {
        'nats': nats,
        'bits': nats / math.log(2),
        'ppl': math.exp(nats),
    }


@torch.no_grad()
def estimate_p00_p01(model: EnergyBasedBinaryLM, tokens: List[int], device: torch.device, context_window: int, max_positions: int = 5000):
    n = len(tokens)
    indices = list(range(1, n))
    if len(indices) > max_positions:
        stride = max(1, len(indices) // max_positions)
        indices = indices[::stride][:max_positions]
    sum_p0_after0 = 0.0; cnt0 = 0
    sum_p0_after1 = 0.0; cnt1 = 0
    for i in indices:
        hist = tokens[max(0, i - context_window):i]
        x = torch.tensor([hist], dtype=torch.long, device=device)
        probs = model.generate_probabilities(x)[0]
        p0 = float(probs[0].item())
        if tokens[i - 1] == 0:
            sum_p0_after0 += p0; cnt0 += 1
        else:
            sum_p0_after1 += p0; cnt1 += 1
    p00 = sum_p0_after0 / max(1, cnt0)
    p01 = sum_p0_after1 / max(1, cnt1)
    return p00, p01


@torch.no_grad()
def generate_completion(model: EnergyBasedBinaryLM, prefix: List[int], num_steps: int, device: torch.device, context_window: int, temperature: float = 1.0):
    hist = list(prefix)
    for _ in range(num_steps):
        x = torch.tensor([hist[-context_window:]], dtype=torch.long, device=device)
        probs = model.generate_probabilities(x)[0]
        if temperature and temperature != 1.0:
            logits = torch.log(probs + 1e-9) / temperature
            probs = torch.softmax(logits, dim=-1)
        next_tok = int(torch.multinomial(probs, num_samples=1).item())
        hist.append(next_tok)
    return hist


def golden_mean_violation_rate(seq: List[int]) -> float:
    # Golden Mean forbids 00; measure fraction of transitions with 00
    if len(seq) < 2:
        return 0.0
    vio = 0; total = 0
    for i in range(1, len(seq)):
        if seq[i - 1] == 0:
            total += 1
            if seq[i] == 0:
                vio += 1
    return (vio / total) if total > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description='Evaluate EBM checkpoint on arbitrary dataset and GM violations')
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--context_window', type=int, default=128)
    ap.add_argument('--gen_prefix_len', type=int, default=64)
    ap.add_argument('--gen_steps', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=1.0)
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    model = EnergyBasedBinaryLM(
        vocab_size=3,
        output_vocab_size=2,
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('heads', 8),
        num_layers=cfg.get('layers', 4),
        max_len=cfg.get('context_window', args.context_window),
        dropout=cfg.get('dropout', 0.0),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    tokens = load_tokens(args.data)

    # Loss/perplexity
    loss_stats = eval_loss(model, tokens, device, args.context_window)
    p00, p01 = estimate_p00_p01(model, tokens, device, args.context_window)

    # Generation & GM violation
    prefix = tokens[:args.gen_prefix_len]
    completion = generate_completion(model, prefix, args.gen_steps, device, args.context_window, temperature=args.temperature)
    vio_rate = golden_mean_violation_rate(completion)

    out = {
        'loss': loss_stats,
        'cond_probs': {'p0_after0': p00, 'p0_after1': p01},
        'gm_violation_rate_generated': vio_rate,
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()


