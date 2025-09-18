#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Import both EBM and AR models
try:
    from ..ebm.models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM  # when executed as a module
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'ebm'))
    from models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM  # type: ignore


def load_binary_tokens(dat_path: Path) -> List[int]:
    s = dat_path.read_text().strip()
    tokens = [int(c) for c in s if c in '01']
    if not tokens:
        raise ValueError(f"No binary tokens found in {dat_path}")
    return tokens


def load_custom_states(states_path: Path) -> List[int]:
    """Load precomputed causal states from a .states(.dat) file as integer IDs.

    The file is expected to be a single line/string of state symbols (e.g., 'ABCD...').
    We map unique symbols to consecutive integers in first-appearance order.
    """
    raw = states_path.read_text().strip()
    # Keep non-whitespace characters as symbols
    symbols = [c for c in raw if not c.isspace()]
    mapping: dict[str, int] = {}
    states: List[int] = []
    for c in symbols:
        if c not in mapping:
            mapping[c] = len(mapping)
        states.append(mapping[c])
    if not states:
        raise ValueError(f"No states found in {states_path}")
    return states


def compute_states_golden_mean(tokens: List[int]) -> List[int]:
    """Golden Mean causal states before emitting tokens[i]:
    - state 0 (A): start or last token was 1 → both 0/1 allowed
    - state 1 (B): last token was 0 → only 1 allowed
    """
    states: List[int] = []
    st = 0
    for t in range(len(tokens)):
        states.append(st)
        st = 1 if tokens[t] == 0 else 0
    return states


def compute_states_even_process(tokens: List[int]) -> List[int]:
    """Even Process parity states before each token: 0=E, 1=O. Toggle on 1; reset to E on 0."""
    states: List[int] = []
    s = 0
    for t in range(len(tokens)):
        states.append(s)
        if tokens[t] == 1:
            s = 1 - s
        else:
            s = 0
    return states


def set_tune_scope(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, scope: str) -> None:
    """scope: 'full' or 'head' (only energy_head/lm_head trainable)."""
    if scope == 'full':
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        # Use the appropriate head based on model type
        if hasattr(model, 'energy_head'):
            for p in model.energy_head.parameters():
                p.requires_grad = True
        elif hasattr(model, 'lm_head'):
            for p in model.lm_head.parameters():
                p.requires_grad = True


def build_model_from_ckpt(ckpt_path: Path, device: torch.device, context_window: int | None = None) -> Tuple[EnergyBasedBinaryLM | AutoRegressiveBinaryLM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('config', {})
    cw = context_window or int(cfg.get('context_window', 128))
    model_type = cfg.get('model_type', 'ebm_binary')
    
    if model_type == 'ar_binary':
        model = AutoRegressiveBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(cfg.get('d_model', 128)),
            nhead=int(cfg.get('heads', 8)),
            num_layers=int(cfg.get('layers', 4)),
            max_len=cw,
            dropout=float(cfg.get('dropout', 0.0)),
        ).to(device)
    else:
        model = EnergyBasedBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(cfg.get('d_model', 128)),
            nhead=int(cfg.get('heads', 8)),
            num_layers=int(cfg.get('layers', 4)),
            max_len=cw,
            dropout=float(cfg.get('dropout', 0.0)),
        ).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


def sample_surjective_mapping(num_states: int, rng: np.random.Generator) -> np.ndarray:
    """Return array M[s] ∈ {0,1}, ensuring both labels appear when possible."""
    if num_states == 1:
        return np.array([0], dtype=np.int64)
    if num_states == 2:
        return np.array([0, 1], dtype=np.int64) if rng.integers(0, 2) == 0 else np.array([1, 0], dtype=np.int64)
    # num_states > 2
    while True:
        M = rng.integers(0, 2, size=num_states, dtype=np.int64)
        if 0 in M and 1 in M:
            return M


def batch_from_indices(tokens: np.ndarray, indices: np.ndarray, context_window: int, pad_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad to max length; provide attention mask. Returns (input_ids, attn)"""
    lengths = np.minimum(context_window, indices).astype(np.int64)
    Lmax = int(lengths.max()) if lengths.size else 1
    B = len(indices)
    x = np.full((B, Lmax), pad_id, dtype=np.int64)
    attn = np.zeros((B, Lmax), dtype=np.int64)
    for i, idx in enumerate(indices):
        L = int(lengths[i])
        if L <= 0:
            continue
        start = int(idx - L)
        hist = tokens[start:idx]
        x[i, Lmax - L:Lmax] = hist
        attn[i, Lmax - L:Lmax] = 1
    return torch.from_numpy(x).to(device), torch.from_numpy(attn).to(device)


@torch.no_grad()
def predict_on_positions(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, tokens: np.ndarray, positions: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> np.ndarray:
    preds: List[int] = []
    model.eval()
    for s in range(0, len(positions), batch_size):
        idx = positions[s:s + batch_size]
        x, attn = batch_from_indices(tokens, idx, context_window, pad_id, device)
        scores = model(x, attn)  # (B,T,2)
        last_idx = attn.sum(dim=1) - 1
        b_idx = torch.arange(scores.size(0), device=device)
        logits = scores[b_idx, last_idx]  # (B,2)
        p = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        preds.extend(p.tolist())
    return np.array(preds, dtype=np.int64)


@torch.no_grad()
def logits_on_positions(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, tokens: np.ndarray, positions: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    model.eval()
    for s in range(0, len(positions), batch_size):
        idx = positions[s:s + batch_size]
        x, attn = batch_from_indices(tokens, idx, context_window, pad_id, device)
        scores = model(x, attn)
        last_idx = attn.sum(dim=1) - 1
        b_idx = torch.arange(scores.size(0), device=device)
        logits = scores[b_idx, last_idx]
        outs.append(logits.detach())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def eval_next_token_metrics(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, preset: str, tokens: np.ndarray, positions: np.ndarray, states_before: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> dict:
    logits = logits_on_positions(model, tokens, positions, context_window, pad_id, device, batch_size)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    targets = tokens[positions].astype(np.int64)
    ce = float(F.cross_entropy(logits.cpu(), torch.from_numpy(targets).long()).item())
    pred = probs.argmax(axis=1).astype(np.int64)
    acc = float((pred == targets).mean())

    metrics = {'val_ce': ce, 'val_acc': acc}

    if preset == 'even_process':
        s = states_before[positions]
        mask_E = (s == 0)
        mask_O = (s == 1)
        p0 = probs[:, 0]
        mean_p0_E = float(p0[mask_E].mean()) if mask_E.any() else 0.0
        mean_p0_O = float(p0[mask_O].mean()) if mask_O.any() else 0.0
        frac_O = float(mask_O.mean())
        argmax_zero_O = float((pred[mask_O] == 0).mean()) if mask_O.any() else 0.0
        metrics.update({'mean_p0_E': mean_p0_E, 'mean_p0_O': mean_p0_O, 'frac_O': frac_O, 'argmax0_O': argmax_zero_O})
    else:
        # golden_mean
        last_bit = tokens[positions - 1]
        p0 = probs[:, 0]
        mask0 = (last_bit == 0)
        mask1 = (last_bit == 1)
        p00 = float(p0[mask0].mean()) if mask0.any() else 0.0
        p01 = float(p0[mask1].mean()) if mask1.any() else 0.0
        zvr = float((pred[mask0] == 0).mean()) if mask0.any() else 0.0
        metrics.update({'p0_given_0': p00, 'p0_given_1': p01, 'zero_violation_rate': zvr})

    return metrics


@torch.no_grad()
def last_hidden_on_positions(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, tokens: np.ndarray, positions: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    model.eval()
    d_model = model.d_model
    for s in range(0, len(positions), batch_size):
        idx = positions[s:s + batch_size]
        x_ids, attn = batch_from_indices(tokens, idx, context_window, pad_id, device)
        # Embed + PE (mirror forward)
        x = model.token_embedding(x_ids) * math.sqrt(d_model)  # (B,L,C)
        x = x.transpose(0, 1)  # (L,B,C)
        x = model.positional_encoding(x)
        x = x.transpose(0, 1)  # (B,L,C)
        x = model.dropout(x)
        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (attn == 0)
        x = model.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)  # (B,L,C)
        last_idx = attn.sum(dim=1) - 1
        b_idx = torch.arange(x.size(0), device=device)
        h = x[b_idx, last_idx]  # (B,C)
        feats.append(h.detach())
    return torch.cat(feats, dim=0)


def train_probe_head(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, tokens: np.ndarray, positions: np.ndarray, labels: np.ndarray, context_window: int, pad_id: int, device: torch.device, steps: int, lr: float, batch_size: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = torch.nn.Linear(model.d_model, 2).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    N = len(positions)
    for t in range(steps):
        sel = rng.choice(N, size=min(batch_size, N), replace=(N < batch_size))
        idx = positions[sel]
        yb = torch.from_numpy(labels[sel].astype(np.int64)).to(device)
        H = last_hidden_on_positions(model, tokens, idx, context_window, pad_id, device, batch_size=len(idx))  # (B,C)
        logits = head(H)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Light logging cadence handled by caller if needed
    head.eval()
    return head


@torch.no_grad()
def predict_with_probe_head(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, head: torch.nn.Module, tokens: np.ndarray, positions: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> np.ndarray:
    preds: List[int] = []
    model.eval(); head.eval()
    for s in range(0, len(positions), batch_size):
        idx = positions[s:s + batch_size]
        H = last_hidden_on_positions(model, tokens, idx, context_window, pad_id, device, batch_size=len(idx))
        logits = head(H)
        p = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
        preds.extend(p.tolist())
    return np.array(preds, dtype=np.int64)


# -------- Layer-specific feature extraction (via forward hooks) --------
@torch.no_grad()
def layer_last_hidden_on_positions(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, layer_spec: str, tokens: np.ndarray, positions: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    """Capture last-position hidden states from an intermediate encoder layer.
    layer_spec: 'final' or integer index into model.encoder.layers
    Returns (N,C) features corresponding to each position in positions.
    """
    if layer_spec == 'final':
        return last_hidden_on_positions(model, tokens, positions, context_window, pad_id, device, batch_size)

    try:
        layer_index = int(layer_spec)
        hook_module = model.encoder.layers[layer_index]
    except Exception as e:
        raise ValueError(f"Invalid --probe_layer '{layer_spec}': {e}")

    feats: list[torch.Tensor] = []
    model.eval()
    d_model = model.d_model

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inp, out):
        captured['h'] = out.detach()

    handle = hook_module.register_forward_hook(hook_fn)
    try:
        for s in range(0, len(positions), batch_size):
            idx = positions[s:s + batch_size]
            x_ids, attn = batch_from_indices(tokens, idx, context_window, pad_id, device)
            x = model.token_embedding(x_ids) * math.sqrt(d_model)
            x = x.transpose(0, 1)
            x = model.positional_encoding(x)
            x = x.transpose(0, 1)
            x = model.dropout(x)
            L = x.size(1)
            causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
            padding_mask = (attn == 0)
            _ = model.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)  # trigger hook
            if 'h' not in captured:
                raise RuntimeError('Layer hook did not capture output')
            h = captured['h']  # (B,L,C)
            last_idx = attn.sum(dim=1) - 1
            b_idx = torch.arange(h.size(0), device=device)
            hl = h[b_idx, last_idx]
            feats.append(hl.cpu())
    finally:
        handle.remove()

    return torch.cat(feats, dim=0)


def train_probe_head_from_layer(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, layer_spec: str, tokens: np.ndarray, positions: np.ndarray, labels: np.ndarray, context_window: int, pad_id: int, device: torch.device, steps: int, lr: float, batch_size: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    head = torch.nn.Linear(model.d_model, 2).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    N = len(positions)
    for t in range(steps):
        sel = rng.choice(N, size=min(batch_size, N), replace=(N < batch_size))
        idx = positions[sel]
        yb = torch.from_numpy(labels[sel].astype(np.int64)).to(device)
        H = layer_last_hidden_on_positions(model, layer_spec, tokens, idx, context_window, pad_id, device, batch_size=len(idx))
        logits = head(H.to(device))
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict_with_probe_head_from_layer(model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM, head: torch.nn.Module, layer_spec: str, tokens: np.ndarray, positions: np.ndarray, context_window: int, pad_id: int, device: torch.device, batch_size: int = 256) -> np.ndarray:
    preds: List[int] = []
    model.eval(); head.eval()
    for s in range(0, len(positions), batch_size):
        idx = positions[s:s + batch_size]
        H = layer_last_hidden_on_positions(model, layer_spec, tokens, idx, context_window, pad_id, device, batch_size=len(idx))
        logits = head(H.to(device))
        p = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
        preds.extend(p.tolist())
    return np.array(preds, dtype=np.int64)


def compute_task_labels(
    task: str,
    preset: str,
    tokens: np.ndarray,
    positions: np.ndarray,
    states_all: np.ndarray,
    base_model: EnergyBasedBinaryLM | AutoRegressiveBinaryLM,
    context_window: int,
    pad_id: int,
    device: torch.device,
) -> np.ndarray:
    task = task.strip().lower()
    if task in ('random_state_mapping', 'state_random'):
        raise RuntimeError('compute_task_labels should not be used for random_state_mapping')

    if task in ('even_state->1', 'even_state_1', 'even_state_one'):
        # 1 for Even parity, 0 for Odd
        if preset != 'even_process':
            print('[warn] even_state task on non-even preset; computing parity from tokens directly')
            # recompute parity on tokens
            states_all = np.array(compute_states_even_process(tokens.tolist()), dtype=np.int64)
        s = states_all[positions]
        return (s == 0).astype(np.int64)

    if task in ('even_state->0', 'even_state_0', 'even_state_zero'):
        if preset != 'even_process':
            print('[warn] even_state task on non-even preset; computing parity from tokens directly')
            states_all = np.array(compute_states_even_process(tokens.tolist()), dtype=np.int64)
        s = states_all[positions]
        return (s == 1).astype(np.int64)

    if task in ('p1_gt_0.5', 'p1>0.5', 'next_prob_gt_half'):
        logits = logits_on_positions(base_model, tokens, positions, context_window, pad_id, device, batch_size=512)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return (probs[:, 1] > 0.5).astype(np.int64)

    if task in ('length_even', 'len_even'):
        # positions are lengths of prefix
        return ((positions % 2) == 0).astype(np.int64)

    if task in ('last3_sum_odd', 'last_three_sum_odd'):
        y = []
        for t in positions.tolist():
            start = max(0, int(t) - 3)
            s = int(np.sum(tokens[start:int(t)]))
            y.append(1 if (s % 2) == 1 else 0)
        return np.array(y, dtype=np.int64)

    raise ValueError(f'Unknown task: {task}')


def finetune_on_task(base_state: dict, ckpt_cfg: dict, device: torch.device, scope: str, tokens: np.ndarray, train_positions: np.ndarray, train_labels: np.ndarray, context_window: int, pad_id: int, steps: int, lr: float, batch_size: int, seed: int) -> EnergyBasedBinaryLM | AutoRegressiveBinaryLM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Rebuild model fresh from base_state each task
    model_type = ckpt_cfg.get('model_type', 'ebm_binary')
    
    if model_type == 'ar_binary':
        model = AutoRegressiveBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(ckpt_cfg.get('d_model', 128)),
            nhead=int(ckpt_cfg.get('heads', 8)),
            num_layers=int(ckpt_cfg.get('layers', 4)),
            max_len=context_window,
            dropout=float(ckpt_cfg.get('dropout', 0.0)),
        ).to(device)
    else:
        model = EnergyBasedBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(ckpt_cfg.get('d_model', 128)),
            nhead=int(ckpt_cfg.get('heads', 8)),
            num_layers=int(ckpt_cfg.get('layers', 4)),
            max_len=context_window,
            dropout=float(ckpt_cfg.get('dropout', 0.0)),
        ).to(device)
    model.load_state_dict(base_state)
    set_tune_scope(model, scope)
    model.train()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    N = len(train_positions)
    print_every = max(1, steps // 5)
    for t in range(steps):
        sel = rng.choice(N, size=min(batch_size, N), replace=(N < batch_size))
        idx = train_positions[sel]
        yb = torch.from_numpy(train_labels[sel].astype(np.int64)).to(device)
        x, attn = batch_from_indices(tokens, idx, context_window, pad_id, device)
        scores = model(x, attn)
        last_idx = attn.sum(dim=1) - 1
        b_idx = torch.arange(scores.size(0), device=device)
        logits = scores[b_idx, last_idx]  # (B,2)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ((t + 1) % print_every == 0) or (t + 1 == steps):
            print(f"  [FT] step {t + 1}/{steps} loss {loss.item():.4f}")

    model.eval()
    return model


def compute_rib_dib(states: np.ndarray, preds: np.ndarray) -> Tuple[float, float]:
    """R-IB: same-state pairs with same pred; D-IB: 1 - (diff-state pairs with same pred)."""
    states = states.astype(np.int64)
    preds = preds.astype(np.int64)
    S = int(states.max()) + 1 if states.size else 0

    n_s = np.array([np.sum(states == s) for s in range(S)], dtype=np.int64)
    n_s_c = np.zeros((S, 2), dtype=np.int64)
    for s in range(S):
        m = (states == s)
        if m.any():
            n_s_c[s, 0] = int(np.sum(preds[m] == 0))
            n_s_c[s, 1] = int(np.sum(preds[m] == 1))

    # R-IB
    rib_num = 0.0
    rib_den = 0.0
    for s in range(S):
        ns = int(n_s[s])
        if ns >= 2:
            c0 = int(n_s_c[s, 0]); c1 = int(n_s_c[s, 1])
            rib_num += c0 * (c0 - 1) / 2.0 + c1 * (c1 - 1) / 2.0
            rib_den += ns * (ns - 1) / 2.0
    rib = (rib_num / rib_den) if rib_den > 0 else 1.0

    # D-IB
    total_pairs = 0.0
    same_pred_pairs = 0.0
    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            n1 = int(n_s[s1]); n2 = int(n_s[s2])
            if n1 == 0 or n2 == 0:
                continue
            total_pairs += n1 * n2
            for c in (0, 1):
                same_pred_pairs += int(n_s_c[s1, c]) * int(n_s_c[s2, c])
    frac_same = (same_pred_pairs / total_pairs) if total_pairs > 0 else 0.0
    dib = 1.0 - frac_same
    return float(rib), float(dib)


def main():
    ap = argparse.ArgumentParser(description='Inductive Bias Probe for EBM (GM / Even) with fine-tuning per task')
    ap.add_argument('--ckpt', type=Path, required=True, help='EBM checkpoint path')
    ap.add_argument('--preset', type=str, choices=['golden_mean', 'even_process', 'custom'], default='golden_mean')
    ap.add_argument('--data', type=Path, default=None, help='Path to .dat file (required if preset=custom)')
    ap.add_argument('--val_data', type=Path, default=None, help='Optional separate validation .dat')
    ap.add_argument('--states_data', type=Path, default=None, help='Optional .states or .states.dat file with precomputed train states for preset=custom')
    ap.add_argument('--val_states_data', type=Path, default=None, help='Optional .states or .states.dat file with precomputed val states for preset=custom')
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--context_window', type=int, default=None)
    ap.add_argument('--pad_id', type=int, default=2)

    # Task sampling / dataset sizes
    ap.add_argument('--num_probe_datasets', type=int, default=20)
    ap.add_argument('--train_examples', type=int, default=100)
    ap.add_argument('--val_examples', type=int, default=2000)
    ap.add_argument('--surjective_only', action='store_true', default=True)
    ap.add_argument('--balance_by_label', action='store_true', default=True)
    ap.add_argument('--seed', type=int, default=0)

    # Fine-tuning hyperparams
    ap.add_argument('--tune_scope', type=str, choices=['full', 'head'], default='full')
    ap.add_argument('--ft_steps', type=int, default=200)
    ap.add_argument('--ft_lr', type=float, default=1e-3)
    ap.add_argument('--ft_batch_size', type=int, default=64)
    ap.add_argument('--use_probe_head', action='store_true', help='Train a separate linear probe on last hidden states; preserve LM head')
    ap.add_argument('--probe_layer', type=str, default='final', help="Layer to probe from: 'final' or integer index like '0'")
    ap.add_argument('--tasks', type=str, default='', help='Comma-separated custom tasks: even_state->1, even_state->0, p1_gt_0.5, length_even, last3_sum_odd')

    # Eval batch
    ap.add_argument('--eval_batch_size', type=int, default=256)

    # Optional logging of next-token metrics pre/post fine-tune
    ap.add_argument('--log_next_token_metrics', action='store_true')

    ap.add_argument('--output', type=Path, default=Path('ebm_ibp_results.json'))
    args = ap.parse_args()

    if args.preset == 'golden_mean' and args.data is None:
        args.data = Path('/home/matteo/NeuralCSSR/experiments/datasets/golden_mean/golden_mean.dat')
    elif args.preset == 'even_process' and args.data is None:
        args.data = Path('/home/matteo/NeuralCSSR/experiments/datasets/even_process/even_process.dat')
    elif args.preset == 'custom' and (args.data is None or not args.data.exists()):
        raise FileNotFoundError('Provide --data for preset=custom')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Build base model + cfg to clone per task
    base_model, ckpt = build_model_from_ckpt(args.ckpt, device, args.context_window)
    ckpt_cfg = ckpt.get('config', {})
    context_window = args.context_window or int(ckpt_cfg.get('context_window', 128))
    base_state = base_model.state_dict()
    print(f"[Init] preset={args.preset} device={device.type} cw={context_window} tune_scope={args.tune_scope}")
    print(f"[Init] tasks={args.num_probe_datasets} train_examples={args.train_examples} val_examples={args.val_examples} steps={args.ft_steps} lr={args.ft_lr}")

    # Load train tokens
    if args.data is None or not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    tokens_tr = np.array(load_binary_tokens(args.data), dtype=np.int64)
    print(f"[Data] train tokens: {len(tokens_tr)} from {args.data}")

    # Load validation tokens (separate or same)
    if args.val_data is not None:
        tokens_val = np.array(load_binary_tokens(args.val_data), dtype=np.int64)
    else:
        tokens_val = tokens_tr
    print(f"[Data] val tokens: {len(tokens_val)} from {args.val_data or args.data}")

    # Compute or load states before each token
    if args.preset == 'even_process':
        states_tr_all = np.array(compute_states_even_process(tokens_tr.tolist()), dtype=np.int64)
        states_val_all = np.array(compute_states_even_process(tokens_val.tolist()), dtype=np.int64)
    elif args.preset == 'golden_mean':
        states_tr_all = np.array(compute_states_golden_mean(tokens_tr.tolist()), dtype=np.int64)
        states_val_all = np.array(compute_states_golden_mean(tokens_val.tolist()), dtype=np.int64)
    else:
        # custom: prefer provided .states file; fallback to GM heuristic
        if args.states_data is not None and args.states_data.exists():
            states_tr_all = np.array(load_custom_states(args.states_data), dtype=np.int64)
            if len(states_tr_all) != len(tokens_tr):
                raise ValueError(f"Length mismatch: states ({len(states_tr_all)}) vs tokens ({len(tokens_tr)}) in train")
        else:
            print('[warn] No --states_data provided for custom preset; defaulting to golden-mean heuristic states')
            states_tr_all = np.array(compute_states_golden_mean(tokens_tr.tolist()), dtype=np.int64)
        if args.val_states_data is not None and args.val_states_data.exists():
            states_val_all = np.array(load_custom_states(args.val_states_data), dtype=np.int64)
            if len(states_val_all) != len(tokens_val):
                raise ValueError(f"Length mismatch: val states ({len(states_val_all)}) vs tokens ({len(tokens_val)})")
        else:
            states_val_all = np.array(compute_states_golden_mean(tokens_val.tolist()), dtype=np.int64)

    # Positions available (skip position 0 to avoid empty history)
    pos_tr_all = np.arange(1, len(tokens_tr), dtype=np.int64)
    pos_val_all = np.arange(1, len(tokens_val), dtype=np.int64)

    # Fix the validation subset once (across tasks)
    rng = np.random.default_rng(args.seed)
    if args.val_examples > 0 and len(pos_val_all) > args.val_examples:
        pos_val = rng.choice(pos_val_all, args.val_examples, replace=False)
    else:
        pos_val = pos_val_all.copy()
    s_val = states_val_all[pos_val]

    # Ensure validation has at least two states
    uniq_val = np.unique(s_val)
    if uniq_val.size < 2:
        raise ValueError(f"Validation set lacks state diversity: states={uniq_val.tolist()}")
    state_counts_val = np.bincount(s_val, minlength=int(uniq_val.max(initial=0)) + 1)
    print(f"[Data] val positions: {len(pos_val)} state_counts={state_counts_val.tolist()}")

    # Probe across tasks
    rib_list: List[float] = []
    dib_list: List[float] = []
    task_info: List[dict] = []

    num_states = int(max(states_tr_all.max(initial=0), states_val_all.max(initial=0))) + 1
    custom_tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    has_custom = len(custom_tasks) > 0
    total_runs = len(custom_tasks) if has_custom else args.num_probe_datasets

    for k in range(total_runs):
        rng_k = np.random.default_rng(args.seed * 1000 + k)
        if has_custom:
            task_name = custom_tasks[k]
            print(f"[Task {k + 1}/{total_runs}] custom task: {task_name}")
            labels_tr_all = compute_task_labels(
                task_name, args.preset, tokens_tr, pos_tr_all, states_tr_all, base_model, context_window, args.pad_id, device
            )
        else:
            # State->label mapping (surjective)
            if args.surjective_only:
                M = sample_surjective_mapping(num_states, rng_k)
            else:
                M = rng_k.integers(0, 2, size=num_states, dtype=np.int64)
            m_counts = [int(np.sum(M == 0)), int(np.sum(M == 1))]
            print(f"[Task {k + 1}/{args.num_probe_datasets}] mapping counts (0,1)={m_counts}")
            # Build labels for train pool
            labels_tr_all = M[states_tr_all[pos_tr_all]]
        # Balanced sampling by label for training
        idx0 = pos_tr_all[labels_tr_all == 0]
        idx1 = pos_tr_all[labels_tr_all == 1]
        if len(idx0) == 0 or len(idx1) == 0:
            # Degenerate mapping given train pool (skip or downweight)
            # Fall back to random sample to avoid crash; mark as degenerate
            sel_tr = rng_k.choice(pos_tr_all, min(args.train_examples, len(pos_tr_all)), replace=False)
            if has_custom:
                y_tr = compute_task_labels(task_name, args.preset, tokens_tr, sel_tr, states_tr_all, base_model, context_window, args.pad_id, device)
            else:
                y_tr = M[states_tr_all[sel_tr]]
            degenerate = True
        else:
            half = max(1, args.train_examples // 2)
            n0 = min(half, len(idx0))
            n1 = min(half, len(idx1))
            sel0 = rng_k.choice(idx0, n0, replace=False)
            sel1 = rng_k.choice(idx1, n1, replace=False)
            sel_tr = np.concatenate([sel0, sel1])
            if sel_tr.size < args.train_examples and len(pos_tr_all) > sel_tr.size:
                # Top-up randomly (won't be perfectly balanced, acceptable)
                extra_needed = args.train_examples - sel_tr.size
                pool = np.setdiff1d(pos_tr_all, sel_tr, assume_unique=False)
                if pool.size > 0:
                    extra = rng_k.choice(pool, min(extra_needed, pool.size), replace=False)
                    sel_tr = np.concatenate([sel_tr, extra])
            if has_custom:
                y_tr = compute_task_labels(task_name, args.preset, tokens_tr, sel_tr, states_tr_all, base_model, context_window, args.pad_id, device)
            else:
                y_tr = M[states_tr_all[sel_tr]]
            degenerate = False
        y_counts = [int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))]
        print(f"  train set size={len(sel_tr)} label_counts (0,1)={y_counts} degenerate={degenerate}")

        # Optional pre fine-tune next-token metrics
        nt_pre = None
        if args.log_next_token_metrics:
            model_pre, _ = build_model_from_ckpt(args.ckpt, device, context_window)
            nt_pre = eval_next_token_metrics(
                model_pre, args.preset, tokens_val, pos_val, states_val_all, context_window, args.pad_id, device, batch_size=args.eval_batch_size
            )
            print(f"  nt_pre: ce={nt_pre['val_ce']:.4f} acc={nt_pre['val_acc']:.4f}")

        if args.use_probe_head:
            # Train separate probe head on selected layer's hidden states
            model_k, _ = build_model_from_ckpt(args.ckpt, device, context_window)
            if args.probe_layer == 'final':
                probe_head = train_probe_head(
                    model_k, tokens_tr, sel_tr, y_tr, context_window, args.pad_id, device,
                    steps=args.ft_steps, lr=args.ft_lr, batch_size=args.ft_batch_size, seed=args.seed * 1000 + k
                )
            else:
                probe_head = train_probe_head_from_layer(
                    model_k, args.probe_layer, tokens_tr, sel_tr, y_tr, context_window, args.pad_id, device,
                    steps=args.ft_steps, lr=args.ft_lr, batch_size=args.ft_batch_size, seed=args.seed * 1000 + k
                )
        else:
            # Fine-tune LM (head/full)
            model_k = finetune_on_task(
                base_state=base_state,
                ckpt_cfg=ckpt_cfg,
                device=device,
                scope=args.tune_scope,
                tokens=tokens_tr,
                train_positions=sel_tr,
                train_labels=y_tr,
                context_window=context_window,
                pad_id=args.pad_id,
                steps=args.ft_steps,
                lr=args.ft_lr,
                batch_size=args.ft_batch_size,
                seed=args.seed * 1000 + k,
            )

        # Predict on fixed validation set
        if args.use_probe_head:
            if args.probe_layer == 'final':
                preds_val = predict_with_probe_head(
                    model_k, probe_head, tokens_val, pos_val, context_window, args.pad_id, device, batch_size=args.eval_batch_size
                )
            else:
                preds_val = predict_with_probe_head_from_layer(
                    model_k, probe_head, args.probe_layer, tokens_val, pos_val, context_window, args.pad_id, device, batch_size=args.eval_batch_size
                )
        else:
            preds_val = predict_on_positions(
                model_k, tokens_val, pos_val, context_window, args.pad_id, device, batch_size=args.eval_batch_size
            )
        rib, dib = compute_rib_dib(s_val, preds_val)
        rib_list.append(rib)
        dib_list.append(dib)
        nt_post = None
        if args.log_next_token_metrics:
            nt_post = eval_next_token_metrics(
                model_k, args.preset, tokens_val, pos_val, states_val_all, context_window, args.pad_id, device, batch_size=args.eval_batch_size
            )
            print(f"  nt_post: ce={nt_post['val_ce']:.4f} acc={nt_post['val_acc']:.4f}")
        entry = {'degenerate': degenerate, 'rib': float(rib), 'dib': float(dib)}
        if has_custom:
            entry['task'] = task_name
        if nt_pre is not None:
            entry['nt_pre'] = nt_pre
        if nt_post is not None:
            entry['nt_post'] = nt_post
        task_info.append(entry)
        print(f"  metrics: R-IB={rib:.3f} D-IB={dib:.3f}")

        # Free CUDA memory between tasks
        del model_k
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    results = {
        'preset': args.preset,
        'ckpt': str(Path(args.ckpt).resolve()),
        'data': str(Path(args.data).resolve()),
        'val_data': (str(Path(args.val_data).resolve()) if args.val_data is not None else None),
        'num_states': int(num_states),
        'val_examples': int(len(pos_val)),
        'train_examples': int(args.train_examples),
        'num_probe_datasets': int(args.num_probe_datasets),
        'tune_scope': args.tune_scope,
        'ft_steps': int(args.ft_steps),
        'ft_lr': float(args.ft_lr),
        'ft_batch_size': int(args.ft_batch_size),
        'surjective_only': bool(args.surjective_only),
        'balance_by_label': bool(args.balance_by_label),
        'rib_mean': float(np.mean(rib_list)),
        'rib_std': float(np.std(rib_list)),
        'dib_mean': float(np.mean(dib_list)),
        'dib_std': float(np.std(dib_list)),
        'per_task': task_info,
    }

    print(json.dumps(results, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved IBP results to {args.output}")


if __name__ == '__main__':
    main()