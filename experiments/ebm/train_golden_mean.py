#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import math
import csv
from typing import List

import torch
import torch.nn.functional as F

try:
    from .models import EnergyBasedBinaryLM  # when executed as a module
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from models import EnergyBasedBinaryLM  # when executed as a script


def load_binary_tokens(dat_path: Path) -> List[int]:
    content = dat_path.read_text().strip()
    tokens = [int(c) for c in content if c in '01']
    if not tokens:
        raise ValueError(f"No binary tokens found in {dat_path}")
    return tokens


def make_batches(tokens: List[int], block_size: int, batch_size: int, device: torch.device):
    starts = torch.randint(low=0, high=max(1, len(tokens) - block_size - 1), size=(batch_size,))
    X = []
    Y = []
    for s in starts.tolist():
        seq = tokens[s:s + block_size + 1]
        X.append(seq[:-1])
        Y.append(seq[-1])
    x = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(Y, dtype=torch.long, device=device)
    attn = torch.ones((batch_size, block_size), dtype=torch.long, device=device)
    return x, attn, y


@torch.no_grad()
def estimate_conditional_probs(model: EnergyBasedBinaryLM, tokens: List[int], device: torch.device, context_window: int):
    def p_next(last_bit: int) -> float:
        counts = 0
        acc = 0.0
        for i in range(1, len(tokens)):
            if tokens[i - 1] == last_bit:
                hist = tokens[max(0, i - context_window):i]
                x = torch.tensor([hist], dtype=torch.long, device=device)
                probs = model.generate_probabilities(x)
                acc += float(probs[0, 0].item())
                counts += 1
        return acc / max(1, counts)
    return p_next(0), p_next(1)


@torch.no_grad()
def zero_violation_rate(model: EnergyBasedBinaryLM, tokens: List[int], device: torch.device, context_window: int) -> float:
    total = 0
    violations = 0
    for i in range(1, len(tokens)):
        if tokens[i - 1] == 0:
            hist = tokens[max(0, i - context_window):i]
            x = torch.tensor([hist], dtype=torch.long, device=device)
            probs = model.generate_probabilities(x)[0]
            pred = int(torch.argmax(probs).item())
            if pred == 0:
                violations += 1
            total += 1
    return (violations / total) if total > 0 else 0.0


@torch.no_grad()
def evaluate_validation(model: EnergyBasedBinaryLM, tokens: List[int], device: torch.device, context_window: int, steps: int = 200, batch_size: int = 64) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        x, attn, y = make_batches(tokens, context_window, batch_size, device)
        scores = model(x, attn)
        logits = scores[:, -1]
        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))


def compute_parity_labels(tokens: List[int]) -> List[int]:
    """Compute parity state (0=E, 1=O) before each position i based on full history.
    E at start; toggle on 1; reset to E on 0.
    Returns list of length len(tokens) with state before emitting tokens[i]."""
    parity_before: List[int] = []
    s = 0  # 0=E, 1=O
    for t in range(len(tokens)):
        parity_before.append(s)
        if tokens[t] == 1:
            s = 1 - s
        else:
            s = 0
    return parity_before


@torch.no_grad()
def parity_metrics(model: EnergyBasedBinaryLM, tokens: List[int], device: torch.device, context_window: int, max_positions: int = 5000):
    """Compute parity-aware metrics for Even Process.
    - mean_p0_E: average P(0|hist) over E contexts
    - mean_p0_O: average P(0|hist) over O contexts (should be ~0)
    - frac_O: fraction of O contexts
    - argmax_zero_rate_O: fraction of O contexts where argmax predicts 0 (should be ~0)
    """
    states = compute_parity_labels(tokens)
    n = len(tokens)
    indices = list(range(1, n))
    if len(indices) > max_positions:
        # sample roughly uniformly
        stride = max(1, len(indices) // max_positions)
        indices = indices[::stride][:max_positions]
    sum_p0_E = 0.0; cnt_E = 0
    sum_p0_O = 0.0; cnt_O = 0
    argmax_zero_O = 0
    for i in indices:
        parity = states[i]
        hist = tokens[max(0, i - context_window):i]
        x = torch.tensor([hist], dtype=torch.long, device=device)
        probs = model.generate_probabilities(x)[0]
        p0 = float(probs[0].item())
        pred = int(torch.argmax(probs).item())
        if parity == 0:
            sum_p0_E += p0; cnt_E += 1
        else:
            sum_p0_O += p0; cnt_O += 1
            if pred == 0:
                argmax_zero_O += 1
    mean_p0_E = (sum_p0_E / cnt_E) if cnt_E > 0 else 0.0
    mean_p0_O = (sum_p0_O / cnt_O) if cnt_O > 0 else 0.0
    frac_O = cnt_O / max(1, (cnt_E + cnt_O))
    argmax_zero_rate_O = (argmax_zero_O / cnt_O) if cnt_O > 0 else 0.0
    return mean_p0_E, mean_p0_O, frac_O, argmax_zero_rate_O


def main():
    ap = argparse.ArgumentParser(description="Train EBM on binary datasets (Golden Mean, Even Process)")
    ap.add_argument('--preset', type=str, choices=['golden_mean', 'even_process', 'custom'], default='golden_mean')
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--context_window', type=int, default=64)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--steps_per_epoch', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_out', type=Path, default=None)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--val_steps', type=int, default=200)
    ap.add_argument('--log_csv', type=Path, default=None)
    # GPU / precision
    ap.add_argument('--amp', action='store_true', help='Enable CUDA AMP mixed precision')
    ap.add_argument('--tf32', action='store_true', help='Enable TF32 matmul on Ampere+ GPUs')
    # Subset control
    ap.add_argument('--max_tokens', type=int, default=0, help='If >0, train/val on the first N tokens')
    args = ap.parse_args()

    # Resolve preset defaults
    if args.preset == 'golden_mean':
        default_data = Path('/home/matteo/NeuralCSSR/experiments/datasets/golden_mean/golden_mean.dat')
        default_ckpt = Path('/home/matteo/NeuralCSSR/experiments/ebm/checkpoints/golden_mean_ebm.pt')
        default_csv = Path('/home/matteo/NeuralCSSR/experiments/ebm/metrics/golden_mean_metrics.csv')
    elif args.preset == 'even_process':
        default_data = Path('/home/matteo/NeuralCSSR/experiments/datasets/even_process/even_process.dat')
        default_ckpt = Path('/home/matteo/NeuralCSSR/experiments/ebm/checkpoints/even_process_ebm.pt')
        default_csv = Path('/home/matteo/NeuralCSSR/experiments/ebm/metrics/even_process_metrics.csv')
    else:
        default_data = None; default_ckpt = None; default_csv = None

    args.data = args.data or default_data
    args.ckpt_out = args.ckpt_out or default_ckpt or Path('ebm_ckpt.pt')
    args.log_csv = args.log_csv or default_csv or Path('ebm_metrics.csv')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if device.type == 'cuda':
        # Optional performance knobs
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data is None or not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset file not found. Provide --data or --preset with available default. Got: {args.data}")
    tokens = load_binary_tokens(args.data)
    if args.max_tokens and args.max_tokens > 0:
        tokens = tokens[:args.max_tokens]
        print(f"Using subset of dataset: first {len(tokens)} tokens")
    if len(tokens) <= args.context_window + 1:
        raise ValueError(f"Not enough tokens ({len(tokens)}) for context_window {args.context_window}. Increase --max_tokens or reduce --context_window.")
    split_idx = max(args.context_window, int(len(tokens) * (1.0 - args.val_frac)))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx - args.context_window:]

    model = EnergyBasedBinaryLM(
        vocab_size=3,
        output_vocab_size=2,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        max_len=args.context_window,
        dropout=args.dropout,
    ).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and args.amp))

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step in range(1, args.steps_per_epoch + 1):
            x, attn, y = make_batches(train_tokens, args.context_window, args.batch_size, device)
            opt.zero_grad(set_to_none=True)
            if device.type == 'cuda' and args.amp:
                with torch.amp.autocast('cuda', enabled=True):
                    scores = model(x, attn)
                    logits = scores[:, -1]
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                scores = model(x, attn)
                logits = scores[:, -1]
                loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {running/50:.4f}")
                running = 0.0

        model.eval()
        # Validation metrics
        val_loss_nats = evaluate_validation(model, val_tokens, device, args.context_window, steps=args.val_steps, batch_size=args.batch_size)
        val_loss_bits = val_loss_nats / math.log(2)
        val_ppl = math.exp(val_loss_nats)
        p00, p01 = estimate_conditional_probs(model, val_tokens, device, args.context_window)
        zvr = zero_violation_rate(model, val_tokens, device, args.context_window)
        # Parity-aware metrics for even_process preset
        parity_msg = ""
        if args.preset == 'even_process':
            mean_p0_E, mean_p0_O, frac_O, argmax_zero_O = parity_metrics(model, val_tokens, device, args.context_window)
            parity_msg = f", parity: p0|E≈{mean_p0_E:.3f}, p0|O≈{mean_p0_O:.3f}, frac_O≈{frac_O:.3f}, argmax0|O≈{argmax_zero_O:.3f}"
        print(f"After epoch {epoch}: val_loss={val_loss_nats:.4f} nats ({val_loss_bits:.4f} bits), ppl={val_ppl:.3f}, p(0|0)≈{p00:.3f}, p(0|1)≈{p01:.3f}, zero_violation≈{zvr:.4f}{parity_msg}")
        # CSV logging
        args.log_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.log_csv.exists()
        with open(args.log_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                header = ['epoch', 'val_loss_nats', 'val_loss_bits', 'val_ppl', 'p00', 'p01', 'zero_violation']
                if args.preset == 'even_process':
                    header += ['p0_given_E', 'p0_given_O', 'frac_O', 'argmax0_given_O']
                w.writerow(header)
            row = [epoch, f"{val_loss_nats:.6f}", f"{val_loss_bits:.6f}", f"{val_ppl:.6f}", f"{p00:.6f}", f"{p01:.6f}", f"{zvr:.6f}"]
            if args.preset == 'even_process':
                row += [f"{mean_p0_E:.6f}", f"{mean_p0_O:.6f}", f"{frac_O:.6f}", f"{argmax_zero_O:.6f}"]
            w.writerow(row)
        model.train()

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': {
        'd_model': args.d_model,
        'layers': args.layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'context_window': args.context_window,
        'model_type': 'ebm_binary'
    }}, args.ckpt_out)
    print(f"Saved checkpoint to {args.ckpt_out}")


if __name__ == '__main__':
    main()


