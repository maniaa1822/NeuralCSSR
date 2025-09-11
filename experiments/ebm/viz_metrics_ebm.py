#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse visualization utilities from nanoGPT
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
_sys.path.append(str(_ROOT / 'nanoGPT'))
import rep_viz as rv  # type: ignore


def load_cache(path: Path):
    z = np.load(path)
    H = z['H']
    s = z['states']
    p1 = z['p1'] if 'p1' in z.files else None
    return H, s, p1


def plot_redundancy(cache_path: Path, out_png: Path, subsample: int = 4000):
    H, s, _ = load_cache(cache_path)
    n = H.shape[0]
    m = min(subsample, n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, m, replace=False)
    Hs, ss = H[idx], s[idx]
    fr = np.linspace(0.1, 1.0, 12)
    fr, acc, auc = rv.redundancy_auc(Hs, ss, fractions=fr, n_trials=5, C=1.0, seed=0)
    rv.plot_redundancy_curve(fr, acc, auc, title='EBM redundancy (subsampled)')
    plt.savefig(str(out_png), dpi=150)
    plt.close()


def plot_kernel_pred(cache_path: Path, out_txt: Path):
    H, s, p1 = load_cache(cache_path)
    if p1 is None:
        out_txt.write_text('p1 missing in cache')
        return
    S = int(s.max()) + 1
    pred_avg = []
    for k in range(S):
        mask = s == k
        pred_avg.append(float(p1[mask].mean()) if mask.any() else float('nan'))
    out_txt.write_text(str({'pred_mean_p1_per_state': pred_avg}))


def cka_from_layer_caches(cache_paths: List[Path], labels: List[str], out_png: Path, subsample: int = 4000):
    rng = np.random.default_rng(0)
    H_list = []
    for path in cache_paths:
        z = np.load(path)
        H = z['H']
        n = H.shape[0]
        m = min(subsample, n)
        idx = rng.choice(n, m, replace=False)
        H_list.append(H[idx])
    rv.plot_cka_matrix(H_list, labels)
    plt.savefig(str(out_png), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Compute EBM viz metrics from caches')
    ap.add_argument('--out_dir', type=Path, required=True, help='Directory with viz_cache.npz and optional layer caches')
    ap.add_argument('--layers', type=str, default='0,1,final')
    args = ap.parse_args()

    cache = args.out_dir / 'viz_cache.npz'
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_redundancy(cache, args.out_dir / 'viz_redundancy.png')
    plot_kernel_pred(cache, args.out_dir / 'viz_kernel_pred.txt')

    layers = [ls.strip() for ls in args.layers.split(',') if ls.strip()]
    layer_paths: List[Path] = []
    labels: List[str] = []
    for ls in layers:
        p = args.out_dir / f'viz_cache_layer_{ls}.npz'
        if p.exists():
            layer_paths.append(p)
            labels.append(ls)
    if len(layer_paths) >= 2:
        cka_from_layer_caches(layer_paths, labels, args.out_dir / 'viz_cka.png')
    else:
        (args.out_dir / 'viz_cka.txt').write_text('Not enough layer caches for CKA')


if __name__ == '__main__':
    main()


