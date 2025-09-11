#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Local imports for the EBM model
try:
    from .models import EnergyBasedBinaryLM
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from models import EnergyBasedBinaryLM

# Reuse visualization utilities from nanoGPT/rep_viz.py
_ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.append(str(_ROOT / 'nanoGPT'))
import rep_viz as rv  # type: ignore


def load_binary_tokens(dat_path: Path) -> List[int]:
    s = dat_path.read_text().strip()
    tokens = [int(c) for c in s if c in '01']
    if not tokens:
        raise ValueError(f"No binary tokens found in {dat_path}")
    return tokens


def compute_states_golden_mean(tokens: List[int]) -> List[int]:
    """Two causal states for Golden Mean:
    - state 0 (A): start or last token was 1 → both 0/1 allowed
    - state 1 (B): last token was 0 → only 1 allowed
    Returns states before emitting tokens[i].
    """
    states: List[int] = []
    st = 0
    for t in range(len(tokens)):
        states.append(st)
        st = 1 if tokens[t] == 0 else 0
    return states


def compute_states_even_process(tokens: List[int]) -> List[int]:
    """Even Process parity states before each token: 0=E, 1=O.
    Toggle on 1; reset to E on 0.
    """
    states: List[int] = []
    s = 0
    for t in range(len(tokens)):
        states.append(s)
        if tokens[t] == 1:
            s = 1 - s
        else:
            s = 0
    return states


def build_true_transition(states: np.ndarray, S: int) -> np.ndarray:
    T = np.zeros((S, S), dtype=np.float64)
    for a, b in zip(states[:-1], states[1:]):
        T[int(a), int(b)] += 1.0
    T = T + 1e-8
    T = T / T.sum(axis=1, keepdims=True)
    return T


@torch.no_grad()
def _encoder_hidden_and_p1_for_chunk(model: EnergyBasedBinaryLM,
                                     chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute hidden features (post-encoder) and per-position P(token=1) for a chunk.
    chunk: LongTensor of shape (1, T)
    Returns: (H: (T, d_model), p1: (T,))
    """
    device = chunk.device
    d_model = model.d_model
    x = model.token_embedding(chunk) * math.sqrt(d_model)  # (1,T,C)
    x = x.transpose(0, 1)  # (T,1,C)
    x = model.positional_encoding(x)  # (T,1,C)
    x = x.transpose(0, 1)  # (1,T,C)
    x = model.dropout(x)
    seq_len = chunk.size(1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    x = model.encoder(x, mask=causal_mask, src_key_padding_mask=None)  # (1,T,C)
    energies = model.energy_head(x)  # (1,T,2)
    scores = -energies
    probs = F.softmax(scores, dim=-1).squeeze(0)  # (T,2)
    return x.squeeze(0), probs[:, 1]


@torch.no_grad()
def extract_prefix_features_and_nextp(model: EnergyBasedBinaryLM,
                                      tokens: np.ndarray,
                                      device: torch.device,
                                      context_window: int,
                                      max_positions: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract prefix-only features aligned to s_t (positions 1..).
    Returns H: (N,d), p1: (N,)
    """
    model.eval()
    feat_list: List[torch.Tensor] = []
    prob_list: List[torch.Tensor] = []
    prev_last_h: Optional[torch.Tensor] = None
    prev_last_p: Optional[torch.Tensor] = None

    limit = len(tokens) - (len(tokens) % context_window)
    for i in range(0, limit, context_window):
        chunk_np = tokens[i:i + context_window].astype(np.int64)
        chunk = torch.from_numpy(chunk_np).unsqueeze(0).to(device)
        h, p1 = _encoder_hidden_and_p1_for_chunk(model, chunk)
        # continuity across chunk boundaries: carry over last position of previous chunk
        if prev_last_h is not None and prev_last_p is not None:
            feat_list.append(prev_last_h.unsqueeze(0))
            prob_list.append(prev_last_p.view(1,))
        # take all but last positions from current chunk
        if h.size(0) > 1:
            feat_list.append(h[:-1, :])
            prob_list.append(p1[:-1])
        prev_last_h = h[-1, :]
        prev_last_p = p1[-1]

        if max_positions is not None:
            cur_n = int(sum(x.size(0) for x in feat_list))
            if cur_n >= max_positions:
                break

    # append the final last state
    if prev_last_h is not None and prev_last_p is not None:
        feat_list.append(prev_last_h.unsqueeze(0))
        prob_list.append(prev_last_p.view(1,))

    if not feat_list:
        return np.empty((0, int(model.d_model)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    H = torch.cat(feat_list, dim=0).cpu().numpy().astype(np.float32)
    p1 = torch.cat(prob_list, dim=0).cpu().numpy().astype(np.float32)
    return H, p1


@torch.no_grad()
def extract_hidden_states_prefix_only_with_hook(model: EnergyBasedBinaryLM,
                                                hook_module: torch.nn.Module,
                                                tokens: np.ndarray,
                                                device: torch.device,
                                                context_window: int,
                                                max_positions: Optional[int] = None) -> np.ndarray:
    """Capture hidden features from an intermediate module inside the encoder.
    Returns X: (N,d)
    """
    model.eval()
    captured = {}

    def hook_fn(_module, _inp, out):
        captured['h'] = out.detach()

    handle = hook_module.register_forward_hook(hook_fn)
    feat_list: List[torch.Tensor] = []
    prev_last_h: Optional[torch.Tensor] = None
    try:
        limit = len(tokens) - (len(tokens) % context_window)
        for i in range(0, limit, context_window):
            chunk_np = tokens[i:i + context_window].astype(np.int64)
            chunk = torch.from_numpy(chunk_np).unsqueeze(0).to(device)
            # run through embedding + encoder to trigger hook
            d_model = model.d_model
            x = model.token_embedding(chunk) * math.sqrt(d_model)
            x = x.transpose(0, 1)
            x = model.positional_encoding(x)
            x = x.transpose(0, 1)
            x = model.dropout(x)
            seq_len = chunk.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            _ = model.encoder(x, mask=causal_mask, src_key_padding_mask=None)
            h = captured['h'].squeeze(0)  # (T,C)

            if prev_last_h is not None:
                feat_list.append(prev_last_h.unsqueeze(0))
            if h.size(0) > 1:
                feat_list.append(h[:-1, :])
            prev_last_h = h[-1, :]

            if max_positions is not None:
                cur_n = int(sum(x.size(0) for x in feat_list))
                if cur_n >= max_positions:
                    break
        if prev_last_h is not None:
            feat_list.append(prev_last_h.unsqueeze(0))
    finally:
        handle.remove()

    if not feat_list:
        return np.empty((0, int(model.d_model)), dtype=np.float32)
    X = torch.cat(feat_list, dim=0).cpu().numpy().astype(np.float32)
    return X


def main():
    ap = argparse.ArgumentParser(description='Generate EBM representation visualizations and caches')
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--preset', type=str, choices=['golden_mean', 'even_process', 'custom'], default='golden_mean')
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--context_window', type=int, default=None)
    ap.add_argument('--embed', type=str, default='pca', choices=['pca', 'umap', 'tsne'])
    ap.add_argument('--sample', type=int, default=5000)
    ap.add_argument('--traj_len', type=int, default=200)
    ap.add_argument('--layers', type=str, default='final', help="Comma-separated: 'final' or encoder layer indices like '0,1,3'")
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--max_tokens', type=int, default=0)
    args = ap.parse_args()

    # Resolve defaults
    if args.preset == 'golden_mean' and args.data is None:
        args.data = Path('/home/matteo/NeuralCSSR/notebook_experiments/golden_mean/data/golden_mean/golden_mean.dat')
    elif args.preset == 'even_process' and args.data is None:
        args.data = Path('/home/matteo/NeuralCSSR/notebook_experiments/even_process/even_process.dat')

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    context_window = args.context_window or int(cfg.get('context_window', 128))
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

    if args.data is None or not args.data.exists():
        raise FileNotFoundError(f"Dataset not found. Provide --data or a supported --preset. Got: {args.data}")
    tokens = np.array(load_binary_tokens(args.data), dtype=np.int64)
    if args.max_tokens and args.max_tokens > 0:
        tokens = tokens[:args.max_tokens]

    # Trim to multiple of context window for chunked extraction
    L = (len(tokens) // context_window) * context_window
    tokens = tokens[:L]
    if len(tokens) < context_window:
        raise ValueError(f"Not enough tokens ({len(tokens)}) for context_window {context_window}")

    # Causal states before each token
    if args.preset == 'even_process':
        states_all = np.array(compute_states_even_process(tokens.tolist()), dtype=np.int64)
    else:
        states_all = np.array(compute_states_golden_mean(tokens.tolist()), dtype=np.int64)

    # Extract features and predicted next-token probabilities
    H, p1 = extract_prefix_features_and_nextp(model, tokens, device, context_window)
    # Align to positions 1.. to avoid the initial ambiguous state
    if H.shape[0] == len(tokens):
        H = H[1:, :]
        p1 = p1[1:]
    s = states_all[1:1 + H.shape[0]]

    # Ensure output dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter with P(1) overlay
    rv.plot_state_scatter(H, s, next_probs=p1, title='EBM hidden state scatter with P(1)', embed_method=args.embed, sample=args.sample)
    import matplotlib.pyplot as plt
    plt.savefig(str(args.out_dir / 'viz_scatter.png'), dpi=150)
    plt.close()

    # Centroid graph using empirical transitions of states sequence
    S = int(s.max()) + 1
    T_true = build_true_transition(s, S)
    rv.plot_centroid_graph(H, s, T_true, embed_method='pca')
    plt.savefig(str(args.out_dir / 'viz_centroids.png'), dpi=150)
    plt.close()

    # Trajectory plot for first traj_len
    H_seq = H[: args.traj_len]
    s_seq = s[: args.traj_len]
    tok_seq = tokens[1:1 + len(H_seq)]
    rv.plot_hidden_trajectory(H_seq, s_seq, tokens_seq=tok_seq, embed_method='pca')
    plt.savefig(str(args.out_dir / 'viz_trajectory.png'), dpi=150)
    plt.close()

    # Save cache (final layer)
    np.savez_compressed(str(args.out_dir / 'viz_cache.npz'), H=H, states=s, p1=p1)

    # Optional per-layer caches
    layers = [ls.strip() for ls in args.layers.split(',') if ls.strip()]

    def get_hook_module(layer_spec: str):
        if layer_spec == 'final':
            return model.encoder.layers[-1]
        else:
            li = int(layer_spec)
            return model.encoder.layers[li]

    for ls in layers:
        try:
            hook_module = get_hook_module(ls)
        except Exception:
            continue
        X = extract_hidden_states_prefix_only_with_hook(model, hook_module, tokens, device, context_window)
        if X.shape[0] == len(tokens):
            X = X[1:, :]
        s_l = states_all[1:1 + len(X)]
        np.savez_compressed(str(args.out_dir / f'viz_cache_layer_{ls}.npz'), H=X.astype(np.float32), states=s_l)

    print(json.dumps({
        'H_shape': list(H.shape),
        'states_shape': list(s.shape),
        'p1_shape': list(p1.shape),
        'scatter_png': str((args.out_dir / 'viz_scatter.png').resolve()),
        'centroids_png': str((args.out_dir / 'viz_centroids.png').resolve()),
        'trajectory_png': str((args.out_dir / 'viz_trajectory.png').resolve()),
        'cache_npz': str((args.out_dir / 'viz_cache.npz').resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()


