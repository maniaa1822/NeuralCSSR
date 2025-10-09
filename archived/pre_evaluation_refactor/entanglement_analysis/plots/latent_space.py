"""
Latent space projection plots for baseline vs multitask checkpoints.

Generates 2D projections (PCA or UMAP) of nanoGPT activations at a chosen
layer, colored by ε-state labels (and optionally by a primitive factor).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root on path when invoked directly
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from machines import get_machine, generate_sequence_with_states
from entanglement_analysis.analysis_pipeline import EntanglementAnalysisPipeline


def generate_sequences(machine, n_sequences: int, length: int, seed: int) -> List[torch.Tensor]:
    sequences: List[torch.Tensor] = []
    rng = np.random.default_rng(seed)
    base = int(rng.integers(0, 1_000_000))
    for i in range(n_sequences):
        seq_str, _ = generate_sequence_with_states(machine, length=length, seed=base + i)
        sequences.append(torch.tensor([int(c) for c in seq_str], dtype=torch.long))
    return sequences


def collect_layer_activations(
    pipeline: EntanglementAnalysisPipeline,
    sequences: List[torch.Tensor],
    n_samples: int,
    min_history_length: int,
    layer: str,
) -> Tuple[np.ndarray, np.ndarray]:
    histories, epsilon_states, _ = pipeline.data_extractor.get_balanced_sample(
        [s.cpu().numpy() for s in sequences],
        n_per_state=max(1, n_samples // len(pipeline.machine.states)),
        min_length=min_history_length,
    )
    acts = pipeline.feature_extractor.extract_batch(torch.tensor(histories, dtype=torch.long))
    if layer not in acts:
        raise ValueError(f"Requested layer '{layer}' not in captured activations: {list(acts.keys())}")
    return acts[layer], epsilon_states


def make_projection(X: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    if method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=random_state)
            return reducer.fit_transform(X)
        except Exception:
            pass  # Fall back to PCA
    # PCA fallback
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :2] * S[:2])


def plot_scatter(ax, Z: np.ndarray, y: np.ndarray, title: str, state_names: List[str]):
    classes = sorted(np.unique(y))
    # consistent palette
    cmap = plt.cm.get_cmap('tab10', len(classes))
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.7, label=state_names[int(cls)], color=cmap(i))
    ax.set_title(title)
    ax.set_xlabel('comp-1')
    ax.set_ylabel('comp-2')
    ax.legend(fontsize=8, markerscale=2, loc='best', frameon=False)
    ax.grid(True, alpha=0.2, linestyle=':')


def main():
    parser = argparse.ArgumentParser(description="Plot latent projections for baseline vs multitask checkpoints")
    parser.add_argument('--baseline_path', type=str, default='', help='Path to baseline ckpt (optional)')
    parser.add_argument('--mtl_path', type=str, default='', help='Path to multitask ckpt (optional)')
    parser.add_argument('--model_path', type=str, default='', help='Single model path (if comparing one)')
    parser.add_argument('--baseline_label', type=str, default='', help='Override label for baseline panel')
    parser.add_argument('--mtl_label', type=str, default='', help='Override label for multitask panel')
    parser.add_argument('--machine', type=str, default='seven_state_human')
    parser.add_argument('--layer', type=str, default='lm_head_input')
    parser.add_argument('--n_sequences', type=int, default=120)
    parser.add_argument('--sequence_length', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--min_history_length', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'umap'])
    parser.add_argument('--out', type=str, default='entanglement_plots/latent_space_comparison.png')

    args = parser.parse_args()

    machine = get_machine(args.machine)

    # Determine which models to plot
    def label_from_ckpt(default_label: str, ckpt_path: str, override: str) -> str:
        if override:
            return override
        # try to append step count if available
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            it = ckpt.get('iter_num', None)
            if it is not None:
                return f"{default_label} (iter {it})"
        except Exception:
            pass
        return default_label

    models: List[Tuple[str, str]] = []
    if args.baseline_path:
        models.append((label_from_ckpt("Baseline", args.baseline_path, args.baseline_label), args.baseline_path))
    if args.mtl_path:
        models.append((label_from_ckpt("Multitask", args.mtl_path, args.mtl_label), args.mtl_path))
    if args.model_path and not models:
        models.append((label_from_ckpt(Path(args.model_path).stem, args.model_path, ''), args.model_path))
    if not models:
        raise ValueError("Provide --baseline_path/--mtl_path or a single --model_path")

    sequences = generate_sequences(machine, args.n_sequences, args.sequence_length, args.seed)

    n_plots = len(models)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)

    # Keep projections comparable by using same PCA basis across models if desired.
    # For simplicity we compute independently per model here.
    for ax, (label, ckpt_path) in zip(axes[0], models):
        pipeline = EntanglementAnalysisPipeline(machine=machine, model_path=ckpt_path, output_dir='entanglement_results')
        X, y = collect_layer_activations(pipeline, sequences, args.n_samples, args.min_history_length, args.layer)
        Z = make_projection(X, method=args.method, random_state=args.seed)
        plot_scatter(ax, Z, y, f"{label} — {args.layer}", machine.states)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    main()
