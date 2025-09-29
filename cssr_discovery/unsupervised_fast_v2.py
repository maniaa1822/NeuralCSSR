#!/usr/bin/env python3
"""Unsupervised epsilon machine discovery backed by unified machine specs.

Example commands (run from repo root):
    uv run python cssr_discovery/unsupervised_fast_v2.py --machine seven_state_human --variant char_large --history-length 5 --n-samples 100 --backward-stability --enable-remerging --output-json results/seven_state_human_v2.json
    uv run python cssr_discovery/unsupervised_fast_v2.py --machine even_process --variant char_large --history-length 6 --n-samples 150 --backward-stability --enable-remerging --output-json results/even_process_v2.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from machines import get_machine, list_machines
from cssr_discovery.calibration import fit_platt_params
from cssr_discovery.unsupervised_fast_original import (
    FastClusteringResult,
    emission_stratified_sampling,
    efficient_two_stage_clustering,
    random_history_sampling,
    compute_epsilon_machine_loss,
)
from transcssr_baseline.transcssr_neural_runner import _load_nano_gpt_model, load_binary_string


def evaluate_against_machine(result: FastClusteringResult, machine) -> Dict:
    clusters = result.clusters
    cluster_stats = []

    for idx, cluster in enumerate(clusters):
        state_counts = Counter()
        samples = []
        for hist in cluster:
            gt_state = machine.get_gt_state(hist)
            if gt_state is not None:
                state_counts[gt_state] += 1
            samples.append(''.join(map(str, hist)))

        total_labeled = sum(state_counts.values())
        purity = max(state_counts.values()) / total_labeled if total_labeled else 0.0
        dominant = max(state_counts, key=state_counts.get) if state_counts else None

        cluster_stats.append({
            'cluster_id': idx,
            'size': len(cluster),
            'gt_state_counts': dict(state_counts),
            'purity': purity,
            'dominant_gt_state': dominant,
            'emission_signature': result.cluster_emission_sigs[idx] if idx < len(result.cluster_emission_sigs) else None,
            'sample_histories': samples[:5],
        })

    total_histories = sum(len(cluster) for cluster in clusters)
    weighted_purity = (
        sum(entry['purity'] * entry['size'] for entry in cluster_stats) / total_histories
        if total_histories else 0.0
    )

    discovered_states = set()
    for entry in cluster_stats:
        discovered_states.update(entry['gt_state_counts'].keys())

    expected_states = set(machine.states)
    coverage = len(discovered_states & expected_states) / len(expected_states) if expected_states else 0.0

    return {
        'num_clusters_discovered': len(clusters),
        'total_histories': total_histories,
        'weighted_purity': weighted_purity,
        'gt_state_coverage': coverage,
        'discovered_gt_states': sorted(discovered_states),
        'expected_gt_states': sorted(expected_states),
        'cluster_analysis': cluster_stats,
        'cache_hits': result.cache_hits,
        'cache_misses': result.cache_misses,
    }


def parse_args() -> argparse.Namespace:
    machines = list_machines()
    parser = argparse.ArgumentParser(description="Fast epsilon machine discovery (v2)")
    parser.add_argument('--machine', choices=machines, default='seven_state_human')
    parser.add_argument('--variant', default='char_large', help="Model variant suffix (default: char_large)")
    parser.add_argument('--model-root', type=Path, default=repo_root / 'nanoGPT')
    parser.add_argument('--data-root', type=Path, default=repo_root / 'experiments' / 'datasets')
    parser.add_argument('--model-ckpt', type=Path, help='Override model checkpoint path')
    parser.add_argument('--data', type=Path, help='Override dataset .dat path')
    parser.add_argument('--history-length', '--L', type=int, default=5)
    parser.add_argument('--k-refine', type=int, default=4)
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--sampling-strategy', choices=['random', 'emission_stratified'], default='emission_stratified')
    parser.add_argument('--stage-a-threshold', type=float, default=0.001)
    parser.add_argument('--stage-b-threshold', type=float, default=0.001)
    parser.add_argument('--emission-precision', type=int, default=2)
    parser.add_argument('--max-representatives', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backward-stability', action='store_true')
    parser.add_argument('--tolerance-bits', type=float, default=1e-3)
    parser.add_argument('--min-suffix-len', type=int, default=2)
    parser.add_argument('--enable-remerging', action='store_true', default=True)
    parser.add_argument('--disable-remerging', action='store_true')
    parser.add_argument('--remerge-emission-threshold', type=float, default=1e-4)
    parser.add_argument('--remerge-rollout-threshold', type=float, default=5e-4)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--list', action='store_true', help='List available machines and exit')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for name in list_machines():
            m = get_machine(name)
            memory = 'infinite' if m.memory_length is None else f'finite (L={m.memory_length})'
            print(f"{name:25} → {m.num_states} states, memory={memory}")
        return

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    machine = get_machine(args.machine)

    model_path = args.model_ckpt or machine.get_model_path(args.model_root, args.variant)
    if not model_path.exists():
        alt_dir = model_path.parent.parent / model_path.parent.name.replace('_', '-')
        alt_path = alt_dir / model_path.name
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"nanoGPT checkpoint not found: {model_path}")
    data_path = args.data or machine.get_dataset_path(args.data_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = _load_nano_gpt_model(model_path, device)
    sequence = load_binary_string(data_path)
    data = np.fromiter((int(c) for c in sequence), dtype=np.int64)

    print("Fast Unsupervised Epsilon Machine Discovery (v2)")
    print(f"Machine: {machine.display_name} ({machine.name})")
    print(f"Model checkpoint: {model_path}")
    print(f"Dataset: {data_path} (length={len(data)})")
    print(f"Sampling strategy: {args.sampling_strategy}, n_samples={args.n_samples}, L={args.history_length}")

    print("\nFitting Platt calibration...")
    platt = fit_platt_params(data, model, L_max=args.history_length, min_count=5)
    if platt:
        print(f"Platt params: {platt}")

    if args.sampling_strategy == 'random':
        sampled_histories = random_history_sampling(data, args.history_length, args.n_samples, args.seed)
    else:
        sampled_histories = emission_stratified_sampling(
            data,
            args.history_length,
            model,
            args.n_samples,
            platt,
            n_strata=5,
            seed=args.seed,
        )
    print(f"Sampled {len(sampled_histories)} histories")

    enable_remerging = args.enable_remerging and not args.disable_remerging

    result = efficient_two_stage_clustering(
        sampled_histories,
        model,
        args.k_refine,
        platt,
        args.stage_a_threshold,
        args.stage_b_threshold,
        args.emission_precision,
        args.max_representatives,
        args.backward_stability,
        args.tolerance_bits,
        args.min_suffix_len,
        enable_remerging,
        args.remerge_emission_threshold,
        args.remerge_rollout_threshold,
    )

    print("\n====================== EVALUATION (GT VIA MACHINE) ======================")
    evaluation = evaluate_against_machine(result, machine)
    print(f"Discovered {evaluation['num_clusters_discovered']} states")
    print(f"Weighted purity: {evaluation['weighted_purity']:.3f}")
    print(f"Ground truth coverage: {evaluation['gt_state_coverage']:.3f}")
    print(f"Expected states: {evaluation['expected_gt_states']}")
    print(f"Discovered GT states: {evaluation['discovered_gt_states']}")

    for entry in evaluation['cluster_analysis']:
        print(f"\nCluster {entry['cluster_id']}: size={entry['size']} purity={entry['purity']:.3f}")
        print(f" Dominant GT: {entry['dominant_gt_state']} emission={entry['emission_signature']}")
        print(f" GT counts: {entry['gt_state_counts']}")
        print(f" Sample: {entry['sample_histories']}")

    print("\n====================== EPSILON MACHINE LOSS ======================")
    loss_results = compute_epsilon_machine_loss(result, model, data, args.history_length, platt)
    print(f"States: {loss_results['num_states']}")
    print(f"Average loss: {loss_results['avg_loss_per_symbol_bits']:.4f} bits")
    print(f"Predictions: {loss_results['num_predictions']}")

    if args.output_json:
        parameters = vars(args).copy()
        for key, value in list(parameters.items()):
            if isinstance(value, Path):
                parameters[key] = str(value)

        payload = {
            'parameters': parameters,
            'machine': {
                'name': machine.name,
                'display_name': machine.display_name,
                'num_states': machine.num_states,
                'memory_type': machine.memory_type,
                'memory_length': machine.memory_length,
            },
            'platt_params': platt,
            'clustering_result': {
                'num_clusters': len(result.clusters),
                'clusters': [
                    {
                        'cluster_id': idx,
                        'size': len(cluster),
                        'histories': [hist.tolist() for hist in cluster],
                        'emission_signature': result.cluster_emission_sigs[idx],
                    }
                    for idx, cluster in enumerate(result.clusters)
                ],
                'cache_hits': result.cache_hits,
                'cache_misses': result.cache_misses,
            },
            'evaluation': evaluation,
            'loss_evaluation': loss_results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"Results saved to {args.output_json}")


if __name__ == '__main__':
    main()
