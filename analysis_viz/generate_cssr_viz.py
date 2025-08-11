#!/usr/bin/env python3
"""
Generate visualizations and analysis for CSSR-enhanced extraction results.

Inputs:
  - Extracted results JSON (from cssr_enhanced_extractor.py)
  - Ground-truth seven-state machine (domain_specific.py)

Outputs (saved to output directory):
  - emissions_scatter.png: Extracted vs GT emission probabilities
  - alignment_bars.png: L1 error per matched GT state
  - state_sizes.png: Counts per extracted state (descending)
  - distance_heatmap.png: L1 distances (extracted ↔ GT emissions)
  - transition_heatmap_0.png and transition_heatmap_1.png: Extracted transitions (rows=current, cols=next), reordered by GT alignment when possible
  - summary.json: Metrics (avg L1, coverage, mapping)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_extracted(results_path: Path) -> Tuple[List[str], Dict[str, dict], Dict[str, dict]]:
    with open(results_path, 'r') as f:
        data = json.load(f)
    eps = data['epsilon_machine']
    state_ids: List[str] = list(eps['causal_state_info'].keys())
    state_info: Dict[str, dict] = eps['causal_state_info']
    transitions: Dict[str, dict] = eps.get('transitions', {})
    return state_ids, state_info, transitions


def compute_gt_emissions() -> Tuple[List[str], np.ndarray]:
    import sys
    sys.path.insert(0, 'src')
    from neural_cssr.machines.domain_specific import SevenStateHumanSequenceMachine
    m = SevenStateHumanSequenceMachine().create_machine()
    names: List[str] = list(m.states)
    em = []
    for s in names:
        counts = {'0': 0.0, '1': 0.0}
        for (src, sym), outs in m.transitions.items():
            if src == s:
                for to, p in outs:
                    counts[sym] = counts.get(sym, 0.0) + p
        em.append([counts.get('0', 0.0), counts.get('1', 0.0)])
    return names, np.array(em)  # [7,2]


def build_extracted_emissions(state_ids: List[str], state_info: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    emissions = []
    counts = []
    for sid in state_ids:
        probs = state_info[sid]['future_probabilities']
        emissions.append([probs.get('0', 0.0), probs.get('1', 0.0)])
        counts.append(state_info[sid]['count'])
    return np.array(emissions), np.array(counts), state_ids


def align_gt_to_extracted(gt_names: List[str], gt_em: np.ndarray, ex_ids: List[str], ex_em: np.ndarray) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Assign each GT state to a unique extracted state by Hungarian (rows=GT, cols=EX)."""
    from scipy.optimize import linear_sum_assignment
    # Cost matrix is L1 emission distance
    cost = np.abs(gt_em[:, None, 0] - ex_em[None, :, 0]) + np.abs(gt_em[:, None, 1] - ex_em[None, :, 1])
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[str, str] = {}
    errors: Dict[str, float] = {}
    for r, c in zip(row_ind, col_ind):
        mapping[gt_names[r]] = ex_ids[c]
        errors[gt_names[r]] = float(cost[r, c])
    return mapping, errors


def plot_emissions_scatter(output_dir: Path, gt_names: List[str], gt_em: np.ndarray, ex_ids: List[str], ex_em: np.ndarray, mapping: Dict[str, str]) -> None:
    plt.figure(figsize=(7, 6))
    # Extracted
    plt.scatter(ex_em[:, 0], ex_em[:, 1], c='#1f77b4', alpha=0.7, label='Extracted states')
    # GT
    plt.scatter(gt_em[:, 0], gt_em[:, 1], c='#d62728', marker='X', s=100, label='GT states')
    # Connect mapped pairs
    inv_map = {v: k for k, v in mapping.items()}
    for i, sid in enumerate(ex_ids):
        if sid in inv_map:
            gname = inv_map[sid]
            gi = gt_names.index(gname)
            plt.plot([ex_em[i, 0], gt_em[gi, 0]], [ex_em[i, 1], gt_em[gi, 1]], c='gray', alpha=0.4, linewidth=1)
    plt.xlabel('P(0)')
    plt.ylabel('P(1)')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.title('Emission probabilities: Extracted vs Ground Truth')
    plt.tight_layout()
    plt.savefig(output_dir / 'emissions_scatter.png', dpi=200)
    plt.close()


def plot_alignment_bars(output_dir: Path, errors: Dict[str, float]) -> None:
    keys = list(errors.keys())
    vals = [errors[k] for k in keys]
    plt.figure(figsize=(7, 4))
    sns.barplot(x=keys, y=vals, color='#1f77b4')
    plt.ylabel('L1 emission error')
    plt.xlabel('GT state')
    plt.title('Per-GT-state emission error (aligned)')
    plt.tight_layout()
    plt.savefig(output_dir / 'alignment_bars.png', dpi=200)
    plt.close()


def plot_state_sizes(output_dir: Path, ex_ids: List[str], counts: np.ndarray) -> None:
    order = np.argsort(-counts)
    labels = [ex_ids[i] for i in order]
    values = counts[order]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=values, color='#2ca02c')
    plt.ylabel('Count')
    plt.xlabel('Extracted state')
    plt.title('Extracted state sizes (descending)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'state_sizes.png', dpi=200)
    plt.close()


def plot_distance_heatmap(output_dir: Path, gt_names: List[str], gt_em: np.ndarray, ex_ids: List[str], ex_em: np.ndarray) -> None:
    dist = np.abs(gt_em[:, None, 0] - ex_em[None, :, 0]) + np.abs(gt_em[:, None, 1] - ex_em[None, :, 1])
    plt.figure(figsize=(0.5 + 0.5 * dist.shape[1] + 2, 5))
    sns.heatmap(dist, annot=True, fmt='.3f', cmap='viridis', xticklabels=ex_ids, yticklabels=gt_names)
    plt.xlabel('Extracted states')
    plt.ylabel('GT states')
    plt.title('L1 emission distance (GT ↔ Extracted)')
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_heatmap.png', dpi=200)
    plt.close()


def build_transition_matrix(ex_ids: List[str], transitions: Dict[str, dict], symbol: str, order: List[str] | None = None) -> np.ndarray:
    ids = order if order is not None else ex_ids
    idx = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    mat = np.zeros((n, n))
    for s in ex_ids:
        if s not in transitions:
            continue
        if symbol not in transitions[s]:
            continue
        for t, p in transitions[s][symbol].items():
            if s in idx and t in idx:
                mat[idx[s], idx[t]] = p
    return mat


def plot_transition_heatmaps(output_dir: Path, ex_ids: List[str], transitions: Dict[str, dict], mapping: Dict[str, str], gt_names: List[str]) -> None:
    # Reorder extracted states: mapped ones in GT order, then the rest
    mapped_ex = [mapping[g] for g in gt_names if g in mapping and mapping[g] in ex_ids]
    rest = [sid for sid in ex_ids if sid not in mapped_ex]
    order = mapped_ex + rest
    for sym in ['0', '1']:
        mat = build_transition_matrix(ex_ids, transitions, sym, order)
        plt.figure(figsize=(max(6, 0.5 * len(order)), max(5, 0.5 * len(order))))
        sns.heatmap(mat, annot=True, fmt='.2f', cmap='Blues', xticklabels=order, yticklabels=order)
        plt.xlabel('Next state')
        plt.ylabel('Current state')
        plt.title(f'Extracted transition matrix for token "{sym}" (GT-aligned order where possible)')
        plt.tight_layout()
        plt.savefig(output_dir / f'transition_heatmap_{sym}.png', dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True, help='Path to cssr_enhanced_results.json')
    parser.add_argument('--output', type=Path, required=True, help='Output directory for figures')
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    ex_ids, ex_info, ex_trans = load_extracted(args.results)
    ex_em, ex_counts, ex_ids = build_extracted_emissions(ex_ids, ex_info)
    gt_names, gt_em = compute_gt_emissions()

    mapping, errors = align_gt_to_extracted(gt_names, gt_em, ex_ids, ex_em)

    # Metrics
    avg_l1 = float(np.mean([errors[g] for g in errors])) if errors else float('nan')
    covered_ids = set(mapping.values())
    covered = int(np.sum([ex_info[sid]['count'] for sid in covered_ids]))
    total_count = int(np.sum([ex_info[sid]['count'] for sid in ex_ids]))
    coverage = covered / total_count if total_count > 0 else 0.0

    # Plots
    plot_emissions_scatter(out_dir, gt_names, gt_em, ex_ids, ex_em, mapping)
    plot_alignment_bars(out_dir, errors)
    plot_state_sizes(out_dir, ex_ids, ex_counts)
    plot_distance_heatmap(out_dir, gt_names, gt_em, ex_ids, ex_em)
    plot_transition_heatmaps(out_dir, ex_ids, ex_trans, mapping, gt_names)

    # Summary
    summary = {
        'results_path': str(args.results),
        'num_extracted_states': len(ex_ids),
        'avg_l1_emission_error': avg_l1,
        'coverage_fraction': coverage,
        'coverage_counts': {
            'covered': covered,
            'total': total_count
        },
        'alignment_mapping': mapping,
        'per_gt_errors': errors,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved figures to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()




