#!/usr/bin/env python3
"""Run probe experiments on baseline and MTL models.

Replaces nanoGPT/probes/state_probing/probe_state.py with unified infrastructure.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machines import get_machine
from evaluation.core import (
    load_model,
    parse_model_spec,
    generate_iid_data,
    collect_layer_activations,
    compute_state_distances,
)
from evaluation.probes import LinearProbeSuite, MLPProbeSuite, ProbeConfig, ProbeEvaluator


def load_meta(dataset: str) -> tuple[dict[str, int], int]:
    """Load dataset metadata."""
    meta_path = REPO_ROOT / "nanoGPT" / "data" / dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], int(meta["vocab_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model spec: name=path/to/ckpt.pt")
    parser.add_argument("--dataset", default="seven_state_human_mtl100k", help="Dataset for vocab metadata")
    parser.add_argument("--machine", default="seven_state_human", help="Machine name for state labels")
    parser.add_argument("--tokens", type=int, default=65536, help="Number of tokens to sample")
    parser.add_argument("--block-size", type=int, default=64, help="Sequence length per window")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for probes")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per probe")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mlp-hidden", type=int, default=256, help="MLP hidden dimension")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    stoi, vocab_size = load_meta(args.dataset)
    machine = get_machine(args.machine)
    distance_info = compute_state_distances(machine)

    model_name, model_path = parse_model_spec(args.model)
    ckpt_path = model_path if model_path.is_absolute() else (REPO_ROOT / model_path)

    print(f"Loading model: {ckpt_path}")
    handle = load_model(model_name, ckpt_path, device, vocab_size, distance_info)
    print(f"Model type: {handle.kind}")

    block_size = handle.block_size or args.block_size
    if block_size != args.block_size:
        print(f"Using block size {block_size} from checkpoint (overriding --block-size {args.block_size})")

    print(f"Generating {args.tokens} IID tokens...")
    x, _, eps, _ = generate_iid_data(
        machine, stoi, block_size, args.tokens, distance_info, args.seed
    )

    print("Extracting layer activations...")
    token_dataset = TensorDataset(x)
    token_loader = DataLoader(token_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    activations = collect_layer_activations(handle.model, token_loader, device)
    labels_eps = eps.reshape(-1).to(torch.long)

    print(f"Training probes on {labels_eps.numel()} tokens across {len(activations)} layers...")
    probe_config = ProbeConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    labels = {"epsilon_state": labels_eps}

    linear_suite = LinearProbeSuite(probe_config)
    mlp_suite = MLPProbeSuite(probe_config, hidden_dim=args.mlp_hidden)

    linear_results = linear_suite.fit_probes(activations, labels)
    mlp_results = mlp_suite.fit_probes(activations, labels)

    # Compute NLI
    evaluator = ProbeEvaluator()
    nli_results = evaluator.compute_nonlinearity_index(linear_results, mlp_results)

    # Print results
    print("\nResults:")
    print("layer_idx\tlayer_name\tprobe_type\thidden_dim\tloss\t\taccuracy\tnum_tokens")
    records = []
    for layer_idx, (layer_name, layer_linear) in enumerate(linear_results.items()):
        eps_linear = layer_linear["epsilon_state"]
        eps_mlp = mlp_results[layer_name]["epsilon_state"]
        nli = nli_results[layer_name]["epsilon_state"]

        # Linear record
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "probe_type": "linear",
            "hidden_dim": "",
            "loss": f"{eps_linear.loss:.6f}",
            "accuracy": f"{eps_linear.accuracy:.6f}",
            "num_tokens": eps_linear.samples,
            "nli": "",
        })
        print(f"{layer_idx}\t{layer_name}\tlinear\t\t{eps_linear.loss:.6f}\t{eps_linear.accuracy:.6f}\t{eps_linear.samples}")

        # MLP record
        records.append({
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "probe_type": "mlp",
            "hidden_dim": args.mlp_hidden,
            "loss": f"{eps_mlp.loss:.6f}",
            "accuracy": f"{eps_mlp.accuracy:.6f}",
            "num_tokens": eps_mlp.samples,
            "nli": f"{nli:.6f}",
        })
        print(f"{layer_idx}\t{layer_name}\tmlp\t{args.mlp_hidden}\t{eps_mlp.loss:.6f}\t{eps_mlp.accuracy:.6f}\t{eps_mlp.samples}")

    # Write CSV if requested
    if args.output:
        output_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            fieldnames = ["layer_idx", "layer_name", "probe_type", "hidden_dim", "loss", "accuracy", "num_tokens", "nli"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
