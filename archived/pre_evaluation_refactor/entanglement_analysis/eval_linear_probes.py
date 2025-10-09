"""Quick utility to evaluate linear probe accuracy on nanoGPT activations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Ensure project root is on the import path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch

from machines import get_machine, generate_sequence_with_states
from entanglement_analysis.analysis_pipeline import EntanglementAnalysisPipeline


def generate_sequences(machine, n_sequences: int, length: int, seed: int) -> list:
    sequences = []
    rng = np.random.default_rng(seed)
    start_seed = rng.integers(0, 10_000_000)
    for i in range(n_sequences):
        seq_str, _ = generate_sequence_with_states(
            machine, length=length, seed=int(start_seed + i)
        )
        seq_tensor = torch.tensor([int(c) for c in seq_str], dtype=torch.long)
        sequences.append(seq_tensor)
    return sequences


def eval_linear_probes(args: argparse.Namespace) -> Dict:
    machine = get_machine(args.machine)
    pipeline = EntanglementAnalysisPipeline(
        machine=machine,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    sequences = generate_sequences(
        machine,
        n_sequences=args.n_sequences,
        length=args.sequence_length,
        seed=args.data_seed,
    )

    histories, epsilon_states, primitive_labels = pipeline.data_extractor.get_balanced_sample(
        [seq.cpu().numpy() for seq in sequences],
        n_per_state=max(1, args.n_samples // len(machine.states)),
        min_length=args.min_history_length,
    )

    labels = {"epsilon_state": epsilon_states}
    labels.update(primitive_labels)

    batch_input = torch.tensor(histories, dtype=torch.long)
    activations = pipeline.feature_extractor.extract_batch(batch_input)

    linear_results = pipeline.linear_probes.fit_probes(activations, labels)

    epsilon_summary = {
        layer_name: metrics.get("epsilon_state", {}).get("accuracy", float("nan"))
        for layer_name, metrics in linear_results.items()
    }

    result = {
        "model_path": args.model_path,
        "machine": machine.name,
        "n_samples": len(histories),
        "layer_accuracies": epsilon_summary,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate linear probe accuracy on nanoGPT activations")
    parser.add_argument("--model_path", required=True, type=str, help="Path to checkpoint")
    parser.add_argument("--machine", default="seven_state_human", type=str, help="Machine name")
    parser.add_argument("--output_dir", default="entanglement_results/tmp_linear_eval", type=str, help="Dir for temp artifacts")
    parser.add_argument("--n_samples", default=500, type=int, help="Number of histories for balanced sample")
    parser.add_argument("--n_sequences", default=50, type=int, help="Number of sequences to generate")
    parser.add_argument("--sequence_length", default=64, type=int, help="Length of generated sequences")
    parser.add_argument("--min_history_length", default=2, type=int, help="Minimum history length for sampling")
    parser.add_argument("--data_seed", default=42, type=int, help="Seed for sequence generation")
    parser.add_argument("--output_json", default="", type=str, help="Optional path to save JSON summary")

    args = parser.parse_args()
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    summary = eval_linear_probes(args)

    print("Model:", summary["model_path"])
    print("Samples:", summary["n_samples"])
    print("Layer accuracies (ε-state linear probe):")
    for layer, acc in summary["layer_accuracies"].items():
        print(f"  {layer}: {acc:.3f}")


if __name__ == "__main__":
    main()
