"""
In-Context Learning Evaluation for NanoGPT models.

Tests whether a model trained on one machine (e.g., 7-state human) can adapt
to another machine's behavior (e.g., golden mean) after seeing its history.

Usage:
    uv run python icl/icl_eval.py \
        --model_path nanoGPT/out-seven-state-human-char-icl/ckpt.pt \
        --target_machine golden_mean \
        --prompt_lengths 16,32,64,128 \
        --n_samples 100 \
        --output icl/results/icl_golden_mean.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from contextlib import nullcontext

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from machines import get_machine, generate_sequence_with_states
from nanoGPT.model import GPT, GPTConfig


def load_model(ckpt_path: str, device: str = 'cuda') -> Tuple[GPT, Dict]:
    """Load a trained nanoGPT model."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Handle compiled model prefix
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model, checkpoint


def generate_prompt(machine_name: str, length: int, seed: int = None) -> Tuple[np.ndarray, List[str]]:
    """Generate a prompt sequence from a target machine."""
    machine = get_machine(machine_name)
    sequence_str, states = generate_sequence_with_states(machine, length, seed=seed)

    # Convert string sequence to numpy array of integers
    sequence = np.array([int(c) for c in sequence_str], dtype=np.int64)
    return sequence, states


def compute_next_token_probs(model: GPT, context: torch.Tensor, device: str) -> np.ndarray:
    """Get model's probability distribution over next token."""
    with torch.no_grad():
        logits, _ = model(context)
        # Take last position
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()[0]


def evaluate_adaptation(
    model: GPT,
    target_machine_name: str,
    prompt_length: int,
    n_samples: int,
    device: str = 'cuda'
) -> Dict:
    """Evaluate how well model adapts to target machine after seeing prompt.

    Args:
        model: Trained nanoGPT model
        target_machine_name: Name of target machine to test on
        prompt_length: Length of conditioning prompt
        n_samples: Number of test samples
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    machine = get_machine(target_machine_name)

    # Metrics to track
    log_likelihoods = []
    cross_entropies = []
    top1_accuracies = []

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=dtype)

    for sample_idx in range(n_samples):
        # Generate prompt + test token from target machine
        test_length = prompt_length + 1
        sequence, states = generate_prompt(target_machine_name, test_length, seed=sample_idx)

        prompt = sequence[:prompt_length]
        true_next = sequence[prompt_length]

        # Convert to tensor
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)[None, ...]

        # Get model predictions
        with ctx:
            probs = compute_next_token_probs(model, prompt_tensor, device)

        # Compute metrics
        true_prob = probs[true_next]
        log_likelihoods.append(np.log(true_prob + 1e-10))
        cross_entropies.append(-np.log2(true_prob + 1e-10))

        predicted_token = np.argmax(probs)
        top1_accuracies.append(1.0 if predicted_token == true_next else 0.0)

    # Aggregate results
    results = {
        'prompt_length': prompt_length,
        'n_samples': n_samples,
        'mean_log_likelihood': float(np.mean(log_likelihoods)),
        'mean_cross_entropy': float(np.mean(cross_entropies)),
        'top1_accuracy': float(np.mean(top1_accuracies)),
        'perplexity': float(2 ** np.mean(cross_entropies)),
    }

    return results


def evaluate_baseline_no_prompt(
    model: GPT,
    target_machine_name: str,
    n_samples: int,
    device: str = 'cuda'
) -> Dict:
    """Evaluate model with minimal context (just 1 token) - no adaptation."""
    machine = get_machine(target_machine_name)

    log_likelihoods = []
    cross_entropies = []
    top1_accuracies = []

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=dtype)

    for sample_idx in range(n_samples):
        # Generate 2 tokens: 1 for context, 1 for test
        sequence, _ = generate_prompt(target_machine_name, 2, seed=sample_idx)

        context = torch.tensor([sequence[0]], dtype=torch.long, device=device)[None, ...]
        true_next = sequence[1]

        with ctx:
            probs = compute_next_token_probs(model, context, device)

        true_prob = probs[true_next]
        log_likelihoods.append(np.log(true_prob + 1e-10))
        cross_entropies.append(-np.log2(true_prob + 1e-10))

        predicted_token = np.argmax(probs)
        top1_accuracies.append(1.0 if predicted_token == true_next else 0.0)

    return {
        'prompt_length': 1,
        'n_samples': n_samples,
        'mean_log_likelihood': float(np.mean(log_likelihoods)),
        'mean_cross_entropy': float(np.mean(cross_entropies)),
        'top1_accuracy': float(np.mean(top1_accuracies)),
        'perplexity': float(2 ** np.mean(cross_entropies)),
    }


def main():
    parser = argparse.ArgumentParser(description='In-context learning evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--target_machine', type=str, required=True,
                        help='Target machine to test on (e.g., golden_mean)')
    parser.add_argument('--prompt_lengths', type=str, default='16,32,64,128',
                        help='Comma-separated prompt lengths to test')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples per prompt length')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--output', type=str, default='results/icl_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    # Parse prompt lengths
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(',')]

    print(f"Loading model from {args.model_path}")
    model, checkpoint = load_model(args.model_path, args.device)

    trained_on = checkpoint.get('config', {}).get('dataset', 'unknown')
    print(f"Model trained on: {trained_on}")
    print(f"Testing on: {args.target_machine}")
    print(f"Prompt lengths: {prompt_lengths}")

    # Run evaluations
    results = {
        'model_path': args.model_path,
        'trained_on': trained_on,
        'target_machine': args.target_machine,
        'evaluations': []
    }

    # Baseline: no prompt (minimal context)
    print("\nEvaluating baseline (no prompt)...")
    baseline = evaluate_baseline_no_prompt(model, args.target_machine, args.n_samples, args.device)
    results['baseline'] = baseline
    print(f"  Cross-entropy: {baseline['mean_cross_entropy']:.4f} bits")
    print(f"  Accuracy: {baseline['top1_accuracy']:.2%}")

    # Test different prompt lengths
    for prompt_len in prompt_lengths:
        print(f"\nEvaluating with prompt length {prompt_len}...")
        eval_result = evaluate_adaptation(
            model, args.target_machine, prompt_len, args.n_samples, args.device
        )
        results['evaluations'].append(eval_result)

        print(f"  Cross-entropy: {eval_result['mean_cross_entropy']:.4f} bits")
        print(f"  Accuracy: {eval_result['top1_accuracy']:.2%}")
        print(f"  Perplexity: {eval_result['perplexity']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Baseline (1 token): {baseline['top1_accuracy']:.2%} accuracy")
    for eval_result in results['evaluations']:
        improvement = eval_result['top1_accuracy'] - baseline['top1_accuracy']
        print(f"Prompt length {eval_result['prompt_length']:3d}: "
              f"{eval_result['top1_accuracy']:.2%} accuracy "
              f"(+{improvement:+.2%})")


if __name__ == '__main__':
    main()
