import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

import sys
import csv

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_binary_tokens(dat_path: Path) -> List[int]:
    content = dat_path.read_text().strip()
    tokens = [int(c) for c in content if c in '01']
    if not tokens:
        raise ValueError(f"No binary tokens found in {dat_path}")
    return tokens


def make_context_label_pairs(tokens: List[int], context_window: int, stride: int = 1, offset: int = 0) -> Tuple[List[List[int]], List[int]]:
    contexts: List[List[int]] = []
    labels: List[int] = []
    start = max(offset, context_window)
    for t in range(start, len(tokens), stride):
        ctx = tokens[t - context_window:t]
        if t >= len(tokens):
            break
        y = tokens[t]
        contexts.append(ctx)
        labels.append(y)
    return contexts, labels


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    sys.path.append(str(Path(__file__).parent))
    from models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM  # type: ignore

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg: Dict = ckpt.get('config', {})
    model_type = cfg.get('model_type', 'ebm_binary')
    context_window = int(cfg.get('context_window', 32))

    if model_type == 'ar_binary':
        model = AutoRegressiveBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(cfg.get('d_model', 128)),
            nhead=int(cfg.get('heads', 8)),
            num_layers=int(cfg.get('layers', 4)),
            max_len=context_window,
            dropout=float(cfg.get('dropout', 0.0)),
        ).to(device)
    else:
        model = EnergyBasedBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(cfg.get('d_model', 128)),
            nhead=int(cfg.get('heads', 8)),
            num_layers=int(cfg.get('layers', 4)),
            max_len=context_window,
            dropout=float(cfg.get('dropout', 0.0)),
        ).to(device)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, context_window, model_type


@torch.no_grad()
def get_last_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    # Support [B, T, 2] or [B, 2]
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    return logits


def apply_temperature_and_bias(logits: torch.Tensor, temperature: float = 1.0, bias: Tuple[float, float] | None = None) -> torch.Tensor:
    z = logits
    if bias is not None:
        b = torch.tensor([bias[0], bias[1]], dtype=z.dtype, device=z.device)
        z = z + b
    if temperature and temperature != 1.0:
        z = z / float(temperature)
    probs = torch.softmax(z, dim=-1)
    return probs


def compute_calibration_bins(prob_pos: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> Dict:
    # prob_pos, labels are 1D tensors on CPU
    assert prob_pos.dim() == 1 and labels.dim() == 1
    n = prob_pos.shape[0]
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    counts: List[int] = []
    avg_pred: List[float] = []
    emp_acc: List[float] = []

    ece = 0.0
    mce = 0.0
    for lo, hi in zip(bin_lowers, bin_uppers):
        # Include right edge only on last bin
        if hi.item() == 1.0:
            mask = (prob_pos >= lo) & (prob_pos <= hi)
        else:
            mask = (prob_pos >= lo) & (prob_pos < hi)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            counts.append(0)
            avg_pred.append(float((lo + hi) / 2.0))
            emp_acc.append(0.0)
            continue
        p_bin = prob_pos[idx]
        y_bin = labels[idx]
        conf = float(p_bin.mean().item())
        acc = float(y_bin.float().mean().item())
        counts.append(int(idx.numel()))
        avg_pred.append(conf)
        emp_acc.append(acc)
        gap = abs(acc - conf)
        ece += (idx.numel() / n) * gap
        mce = max(mce, gap)

    return {
        'bin_lowers': [float(x.item()) for x in bin_lowers],
        'bin_uppers': [float(x.item()) for x in bin_uppers],
        'counts': counts,
        'avg_pred': avg_pred,
        'emp_acc': emp_acc,
        'ECE': float(ece),
        'MCE': float(mce),
    }


def compute_metrics(prob_pos: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> Dict:
    # Ensure CPU tensors
    prob_pos = prob_pos.detach().cpu()
    labels = labels.detach().cpu()
    eps = 1e-12
    # NLL (log loss)
    p = torch.clamp(prob_pos, min=eps, max=1 - eps)
    nll = -torch.where(labels == 1, torch.log(p), torch.log(1 - p)).mean().item()
    # Brier score (binary)
    brier = torch.mean((prob_pos - labels.float()) ** 2).item()
    # Calibration bins
    bins = compute_calibration_bins(prob_pos, labels, num_bins=num_bins)
    return {
        'NLL': float(nll),
        'Brier': float(brier),
        'ECE': float(bins['ECE']),
        'MCE': float(bins['MCE']),
        'bins': bins,
    }


def plot_reliability(bins: Dict, out_path: Path | None = None, title: str | None = None) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot")
        return
    x = bins['avg_pred']
    y = bins['emp_acc']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(x, y, marker='o', label='Model')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability (class 1)')
    ax.set_ylabel('Empirical frequency (class 1)')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved reliability diagram to {out_path}")
    else:
        plt.show()


def export_bins_csv(bins: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_lower', 'bin_upper', 'count', 'avg_pred', 'emp_acc'])
        for lo, hi, c, ap, ea in zip(
            bins['bin_lowers'], bins['bin_uppers'], bins['counts'], bins['avg_pred'], bins['emp_acc']
        ):
            writer.writerow([lo, hi, c, ap, ea])
    print(f"Saved calibration bins CSV to {out_csv}")


def fit_temperature_on_logits(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 500, lr: float = 0.05, init_temperature: float = 1.0) -> float:
    # Optimize a single scalar temperature T > 0 to minimize NLL
    device = logits.device
    log_T = torch.tensor(math.log(max(init_temperature, 1e-6)), dtype=logits.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    labels_long = labels.long().to(device)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        z = logits / T
        loss = F.cross_entropy(z, labels_long)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_star = float(torch.exp(log_T).item())
    return max(T_star, 1e-6)


def main():
    p = argparse.ArgumentParser(description='Evaluate and visualize calibration for AR/EBM binary models')
    p.add_argument('--ebm_ar_ckpt', type=Path, required=True, help='Path to EBM/AR model checkpoint (.pt)')
    p.add_argument('--data', type=Path, required=True, help='Path to .dat file containing 0/1 sequence')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--context_window', type=int, default=0, help='Override context window (0 to use model cfg)')
    p.add_argument('--stride', type=int, default=1, help='Stride when sampling contexts')
    p.add_argument('--offset', type=int, default=0, help='Offset for context start index')
    p.add_argument('--max_samples', type=int, default=0, help='Limit number of samples (0 = use all)')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--bins', type=int, default=15)
    p.add_argument('--temperature', type=float, default=1.0, help='Temperature applied to logits before softmax')
    p.add_argument('--fit_temperature', action='store_true', help='Fit optimal temperature on the evaluation set')
    p.add_argument('--out_plot', type=Path, help='Path to save reliability diagram (png)')
    p.add_argument('--out_csv', type=Path, help='Path to save calibration bins as CSV')
    args = p.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu'))

    model, cfg_ctx, model_type = load_model_from_ckpt(args.ebm_ar_ckpt, device)
    ctx = cfg_ctx if args.context_window <= 0 else args.context_window

    tokens = load_binary_tokens(args.data)
    contexts, labels = make_context_label_pairs(tokens, context_window=ctx, stride=args.stride, offset=args.offset)

    if args.max_samples and args.max_samples > 0 and len(contexts) > args.max_samples:
        # Uniformly subsample without replacement
        step = max(1, len(contexts) // args.max_samples)
        contexts = contexts[::step][:args.max_samples]
        labels = labels[::step][:args.max_samples]

    # Batched logits computation
    probs_list: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    batch_size = max(1, args.batch_size)
    for i in range(0, len(contexts), batch_size):
        batch_ctx = contexts[i:i + batch_size]
        x = torch.tensor(batch_ctx, dtype=torch.long, device=device)
        z = get_last_logits(model, x)
        logits_list.append(z.detach())

    logits_full = torch.cat(logits_list, dim=0)

    # Optional temperature fitting
    temperature = float(args.temperature)
    if args.fit_temperature:
        temperature = fit_temperature_on_logits(logits_full, labels_tensor, max_iter=200, lr=0.25, init_temperature=max(temperature, 1.0))
        print(f"Fitted temperature: {temperature:.4f}")

    # Calibrated probabilities
    prob_full = apply_temperature_and_bias(logits_full, temperature=temperature, bias=None)
    prob_pos = prob_full[:, 1]

    # Metrics
    metrics = compute_metrics(prob_pos, labels_tensor, num_bins=args.bins)
    print({k: v for k, v in metrics.items() if k != 'bins'})

    # Plot
    title = f"Calibration ({model_type}, T={temperature:.2f})"
    if args.out_plot is not None:
        plot_reliability(metrics['bins'], args.out_plot, title=title)
    else:
        plot_reliability(metrics['bins'], None, title=title)

    # CSV export
    if args.out_csv is not None:
        export_bins_csv(metrics['bins'], args.out_csv)


if __name__ == '__main__':
    main()



