import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

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


def build_empirical_morphs(tokens: List[int], L_max: int, min_count: int = 1) -> Dict[str, Dict]:
    """Return mapping history -> {count, count1, p_emp, L} for all lengths up to L_max."""
    hist_stats: Dict[str, Dict] = {}
    for L in range(1, L_max + 1):
        for t in range(L, len(tokens)):
            hist_bits = tokens[t - L:t]
            next_bit = tokens[t]
            h = ''.join(str(b) for b in hist_bits)
            if h not in hist_stats:
                hist_stats[h] = {'count': 0, 'count1': 0, 'L': L}
            d = hist_stats[h]
            d['count'] += 1
            d['count1'] += int(next_bit == 1)
            # Keep the smallest L encountered for this history representation
            if L < d['L']:
                d['L'] = L
    # finalize probabilities and filter by min_count
    out: Dict[str, Dict] = {}
    for h, d in hist_stats.items():
        c = d['count']
        if c >= min_count:
            p_emp = d['count1'] / float(c)
            out[h] = {'count': c, 'count1': d['count1'], 'p_emp': p_emp, 'L': d['L']}
    return out


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    import sys
    # Ensure repository root is on path for 'experiments.ebm.models'
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from experiments.ebm.models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM  # type: ignore
    except Exception:
        # Fallback: import directly from experiments/ebm if package import fails
        ebm_dir = repo_root / 'experiments' / 'ebm'
        if str(ebm_dir) not in sys.path:
            sys.path.insert(0, str(ebm_dir))
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
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    return logits


def histories_to_tensor(histories: List[str], device: torch.device) -> torch.Tensor:
    if not histories:
        return torch.empty(0, 0, dtype=torch.long, device=device)
    max_len = max(len(h) for h in histories)
    # Right align histories and left pad with zeros
    batch = []
    for h in histories:
        ids = [int(c) for c in h]
        pad = [0] * (max_len - len(ids))
        batch.append(pad + ids)
    return torch.tensor(batch, dtype=torch.long, device=device)


def compute_model_probs_and_margins(model: torch.nn.Module, histories: List[str], device: torch.device, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    probs_list: List[torch.Tensor] = []
    margins_list: List[torch.Tensor] = []
    for i in range(0, len(histories), batch_size):
        batch_h = histories[i:i + batch_size]
        x = histories_to_tensor(batch_h, device)
        if x.numel() == 0:
            continue
        logits = get_last_logits(model, x)
        probs = torch.softmax(logits, dim=-1)
        margin = logits[:, 1] - logits[:, 0]
        probs_list.append(probs[:, 1].detach())
        margins_list.append(margin.detach())
    if probs_list:
        p = torch.cat(probs_list, dim=0)
        s = torch.cat(margins_list, dim=0)
        return p, s
    return torch.empty(0, device=device), torch.empty(0, device=device)


def fit_temperature_on_margins(margins: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, max_iter: int = 500, lr: float = 0.1, init_T: float = 1.0) -> float:
    log_T = torch.tensor(math.log(max(init_T, 1e-6)), dtype=margins.dtype, device=margins.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        s_prime = margins / T
        loss = F.binary_cross_entropy_with_logits(s_prime, targets, weight=weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_star = float(torch.exp(log_T).item())
    return max(T_star, 1e-6)


def fit_platt_on_margins(margins: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, max_iter: int = 1000, lr: float = 0.1) -> Tuple[float, float]:
    a = torch.tensor(1.0, dtype=margins.dtype, device=margins.device, requires_grad=True)
    b = torch.tensor(0.0, dtype=margins.dtype, device=margins.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        s_prime = a * margins + b
        loss = F.binary_cross_entropy_with_logits(s_prime, targets, weight=weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(a.item()), float(b.item())


def fit_beta_on_probs(probs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, max_iter: int = 2000, lr: float = 0.05) -> Tuple[float, float, float]:
    # p' = sigmoid(alpha*log p + beta*log(1-p) + gamma)
    eps = 1e-8
    alpha = torch.tensor(1.0, dtype=probs.dtype, device=probs.device, requires_grad=True)
    beta = torch.tensor(1.0, dtype=probs.dtype, device=probs.device, requires_grad=True)
    gamma = torch.tensor(0.0, dtype=probs.dtype, device=probs.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([alpha, beta, gamma], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        lp = torch.log(torch.clamp(probs, eps, 1 - eps))
        lq = torch.log(torch.clamp(1 - probs, eps, 1 - eps))
        s = alpha * lp + beta * lq + gamma
        loss = F.binary_cross_entropy_with_logits(s, targets, weight=weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(alpha.item()), float(beta.item()), float(gamma.item())


def apply_calibration(margins: torch.Tensor, probs: torch.Tensor, method: str, params: Dict) -> torch.Tensor:
    if method == 'none':
        return probs
    if method == 'temperature':
        T = float(params.get('T', 1.0))
        s_prime = margins / max(T, 1e-6)
        return torch.sigmoid(s_prime)
    if method == 'platt':
        a = float(params.get('a', 1.0)); b = float(params.get('b', 0.0))
        s_prime = a * margins + b
        return torch.sigmoid(s_prime)
    if method == 'beta':
        alpha = float(params.get('alpha', 1.0)); beta = float(params.get('beta', 1.0)); gamma = float(params.get('gamma', 0.0))
        eps = 1e-8
        lp = torch.log(torch.clamp(probs, eps, 1 - eps))
        lq = torch.log(torch.clamp(1 - probs, eps, 1 - eps))
        s = alpha * lp + beta * lq + gamma
        return torch.sigmoid(s)
    raise ValueError(f"Unknown calibration method: {method}")


def weighted_mse(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor) -> float:
    return float(torch.sum(w * (p - q) ** 2) / torch.sum(w))


def plot_scatter(emp: torch.Tensor, model: torch.Tensor, calib: torch.Tensor | None, out_path: Path | None, title: str) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='y=x')
    ax.scatter(emp.cpu().numpy(), model.cpu().numpy(), s=8, alpha=0.4, label='model')
    if calib is not None:
        ax.scatter(emp.cpu().numpy(), calib.cpu().numpy(), s=8, alpha=0.4, label='calibrated')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Empirical P(1|h)')
    ax.set_ylabel('Model P(1|h)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved scatter to {out_path}")
    else:
        plt.show()


def export_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['history', 'L', 'count', 'p_emp', 'p_model', 'p_calibrated']
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved comparison CSV to {out_csv}")


def main():
    p = argparse.ArgumentParser(description='Compare model probabilities vs empirical morphs over histories')
    p.add_argument('--ebm_ar_ckpt', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--L_max', type=int, default=8)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--min_count', type=int, default=5)
    p.add_argument('--calibration', type=str, choices=['none', 'temperature', 'platt', 'beta'], default='none')
    p.add_argument('--temperature', type=float, default=1.0, help='Temperature if calibration=temperature and not fitting')
    p.add_argument('--fit', action='store_true', help='Fit calibration parameters (if supported)')
    p.add_argument('--out_plot', type=Path)
    p.add_argument('--out_csv', type=Path)
    args = p.parse_args()

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu'))

    model, ctx_cfg, model_type = load_model_from_ckpt(args.ebm_ar_ckpt, device)

    tokens = load_binary_tokens(args.data)
    stats = build_empirical_morphs(tokens, L_max=args.L_max, min_count=args.min_count)
    histories = sorted(stats.keys())
    counts = torch.tensor([stats[h]['count'] for h in histories], dtype=torch.float32, device=device)
    emp = torch.tensor([stats[h]['p_emp'] for h in histories], dtype=torch.float32, device=device)

    p_model, margins = compute_model_probs_and_margins(model, histories, device, batch_size=args.batch_size)
    assert p_model.numel() == emp.numel() == counts.numel(), "Mismatch in number of histories"

    calib_method = args.calibration
    params: Dict = {}
    p_calib = None

    if calib_method == 'temperature':
        if args.fit:
            T_star = fit_temperature_on_margins(margins, emp, counts)
            params['T'] = T_star
            print(f"Fitted T={T_star:.4f}")
        else:
            params['T'] = float(args.temperature)
        p_calib = apply_calibration(margins, p_model, 'temperature', params)
    elif calib_method == 'platt':
        if args.fit:
            a, b = fit_platt_on_margins(margins, emp, counts)
            params['a'] = a; params['b'] = b
            print(f"Fitted Platt a={a:.4f}, b={b:.4f}")
        else:
            params['a'] = 1.0; params['b'] = 0.0
        p_calib = apply_calibration(margins, p_model, 'platt', params)
    elif calib_method == 'beta':
        if args.fit:
            alpha, beta, gamma = fit_beta_on_probs(p_model, emp, counts)
            params['alpha'] = alpha; params['beta'] = beta; params['gamma'] = gamma
            print(f"Fitted Beta alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")
        else:
            params['alpha'] = 1.0; params['beta'] = 1.0; params['gamma'] = 0.0
        p_calib = apply_calibration(margins, p_model, 'beta', params)
    else:
        p_calib = None

    # Metrics
    wmse_model = weighted_mse(p_model, emp, counts)
    print(f"Weighted MSE (model vs emp): {wmse_model:.6f}")
    if p_calib is not None:
        wmse_calib = weighted_mse(p_calib, emp, counts)
        print(f"Weighted MSE (calibrated vs emp): {wmse_calib:.6f}")

    # Plot
    title = f"Model vs Empirical ({model_type}, calib={calib_method})"
    plot_scatter(emp, p_model, p_calib, args.out_plot if args.out_plot else None, title)

    # CSV
    if args.out_csv is not None:
        rows: List[Dict] = []
        for i, h in enumerate(histories):
            rows.append({
                'history': h,
                'L': stats[h]['L'],
                'count': stats[h]['count'],
                'p_emp': float(emp[i].item()),
                'p_model': float(p_model[i].item()),
                'p_calibrated': float(p_calib[i].item()) if p_calib is not None else ''
            })
        export_csv(rows, args.out_csv)


if __name__ == '__main__':
    main()


