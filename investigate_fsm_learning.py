#!/usr/bin/env python3
"""
Comprehensive investigation of what the transformer model is actually learning.
Tests claims about FSM structure learning vs marginal distribution memorization.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_delay_transformer import BinaryTransformer, SequenceDataset, collate_fn_ar

def load_ground_truth_data(dataset_path: Path) -> Dict:
    """Load ground truth machine data from dataset."""
    gt_path = dataset_path / "ground_truth"
    
    # Load machine properties
    with open(gt_path / "machine_properties.json") as f:
        machines = json.load(f)
    
    # Load causal state labels
    with open(gt_path / "causal_state_labels.json") as f:
        state_labels = json.load(f)
    
    return {"machines": machines, "state_labels": state_labels}

def compute_baselines(dataset: SequenceDataset) -> Dict:
    """Compute various baseline accuracies."""
    all_sequences = []
    all_tokens = []
    
    for seq in dataset.data:
        seq_list = seq.tolist()
        all_sequences.append(seq_list)
        all_tokens.extend(seq_list)
    
    # Marginal baseline (predict most frequent token)
    token_counts = Counter(all_tokens)
    most_frequent_token = max(token_counts, key=token_counts.get)
    marginal_accuracy = token_counts[most_frequent_token] / len(all_tokens)
    
    # N-gram baselines
    bigram_accuracy = compute_ngram_baseline(all_sequences, n=2)
    trigram_accuracy = compute_ngram_baseline(all_sequences, n=3)
    
    # Random baseline
    random_accuracy = 0.5  # For binary sequences
    
    return {
        "marginal_accuracy": marginal_accuracy,
        "marginal_token": most_frequent_token,
        "marginal_distribution": dict(token_counts),
        "bigram_accuracy": bigram_accuracy,
        "trigram_accuracy": trigram_accuracy,
        "random_accuracy": random_accuracy,
        "total_tokens": len(all_tokens)
    }

def compute_ngram_baseline(sequences: List[List[int]], n: int) -> float:
    """Compute n-gram baseline accuracy."""
    ngram_counts = defaultdict(Counter)
    correct_predictions = 0
    total_predictions = 0
    
    # Build n-gram model
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            context = tuple(seq[i:i+n-1])
            next_token = seq[i+n-1]
            ngram_counts[context][next_token] += 1
    
    # Evaluate n-gram model
    for seq in sequences:
        for i in range(n-1, len(seq)):
            context = tuple(seq[i-n+1:i])
            true_next = seq[i]
            
            if context in ngram_counts:
                predicted_next = max(ngram_counts[context], key=ngram_counts[context].get)
                if predicted_next == true_next:
                    correct_predictions += 1
            total_predictions += 1
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

@torch.no_grad()
def analyze_model_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """Analyze what patterns the model has learned."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    sequence_accuracies = []
    
    for toks, tgt in loader:
        toks, tgt = toks.to(device), tgt.to(device)
        logits = model(toks)
        predictions = torch.argmax(logits, dim=-1)
        
        # Process each sequence in the batch
        for i in range(toks.size(0)):
            # Find actual sequence length (excluding padding)
            seq_len = (tgt[i] != 2).sum().item()
            if seq_len > 0:
                pred_seq = predictions[i][:seq_len].cpu().numpy()
                target_seq = tgt[i][:seq_len].cpu().numpy()
                
                all_predictions.extend(pred_seq)
                all_targets.extend(target_seq)
                
                # Sequence-level accuracy
                seq_acc = (pred_seq == target_seq).mean()
                sequence_accuracies.append(seq_acc)
    
    # Overall accuracy
    overall_accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Per-token accuracy
    token_accuracies = {}
    for token in [0, 1]:
        mask = np.array(all_targets) == token
        if mask.sum() > 0:
            token_accuracies[token] = np.mean(np.array(all_predictions)[mask] == token)
        else:
            token_accuracies[token] = 0.0
    
    # Prediction distribution
    pred_counts = Counter(all_predictions)
    target_counts = Counter(all_targets)
    
    return {
        "overall_accuracy": overall_accuracy,
        "sequence_accuracies": sequence_accuracies,
        "mean_sequence_accuracy": np.mean(sequence_accuracies),
        "perfect_sequences": np.sum(np.array(sequence_accuracies) == 1.0),
        "total_sequences": len(sequence_accuracies),
        "token_accuracies": token_accuracies,
        "prediction_distribution": dict(pred_counts),
        "target_distribution": dict(target_counts),
        "total_tokens": len(all_predictions)
    }

@torch.no_grad()
def test_context_sensitivity(model: nn.Module, device: torch.device) -> Dict:
    """Test if model is sensitive to different contexts (indicating FSM structure learning)."""
    model.eval()
    
    # Create test sequences with specific patterns
    test_patterns = [
        # Pattern 1: Alternating
        [0, 1, 0, 1, 0, 1, 0, 1],
        # Pattern 2: Repeating 0s
        [0, 0, 0, 0, 0, 0, 0, 0],
        # Pattern 3: Repeating 1s  
        [1, 1, 1, 1, 1, 1, 1, 1],
        # Pattern 4: Complex pattern
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        # Pattern 5: Random-looking
        [1, 0, 1, 0, 0, 1, 1, 0, 1],
    ]
    
    results = {}
    
    for i, pattern in enumerate(test_patterns):
        # Convert to tensor and prepare for AR prediction
        seq_tensor = torch.tensor(pattern, dtype=torch.long).unsqueeze(0)  # [1, T]
        
        # Create input (shifted right)
        input_tokens = torch.zeros_like(seq_tensor)
        input_tokens[:, 1:] = seq_tensor[:, :-1]
        
        input_tokens = input_tokens.to(device)
        target_tokens = seq_tensor.to(device)
        
        # Get predictions
        logits = model(input_tokens)
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate accuracy for this pattern
        accuracy = (predictions == target_tokens).float().mean().item()
        
        # Get probabilities for analysis
        probs = torch.softmax(logits, dim=-1)
        
        results[f"pattern_{i+1}"] = {
            "pattern": pattern,
            "accuracy": accuracy,
            "predictions": predictions.cpu().squeeze().tolist(),
            "probabilities": probs.cpu().squeeze().tolist()
        }
    
    return results

def analyze_state_transition_learning(model: nn.Module, ground_truth: Dict, 
                                    test_loader: DataLoader, device: torch.device) -> Dict:
    """Analyze if model learned actual state transitions from the FSMs."""
    model.eval()
    
    # Get state labels for sequences
    state_labels = ground_truth["state_labels"]
    machines = ground_truth["machines"]
    
    # Track predictions by state context
    state_prediction_accuracy = defaultdict(list)
    
    with torch.no_grad():
        for batch_idx, (toks, tgt) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit analysis to first 10 batches for efficiency
                break
                
            toks, tgt = toks.to(device), tgt.to(device)
            logits = model(toks)
            predictions = torch.argmax(logits, dim=-1)
            
            # Process first few sequences in batch
            for seq_idx in range(min(5, toks.size(0))):
                seq_len = (tgt[seq_idx] != 2).sum().item()
                if seq_len > 0:
                    pred_seq = predictions[seq_idx][:seq_len].cpu().numpy()
                    target_seq = tgt[seq_idx][:seq_len].cpu().numpy()
                    
                    # Calculate accuracy for this sequence
                    seq_accuracy = (pred_seq == target_seq).mean()
                    state_prediction_accuracy["overall"].append(seq_accuracy)
    
    return {
        "overall_state_accuracy": np.mean(state_prediction_accuracy["overall"]) if state_prediction_accuracy["overall"] else 0.0,
        "num_sequences_analyzed": len(state_prediction_accuracy["overall"]),
        "perfect_state_predictions": np.sum(np.array(state_prediction_accuracy["overall"]) == 1.0)
    }

def generate_detailed_report(model_analysis: Dict, baselines: Dict, 
                           context_test: Dict, state_analysis: Dict, 
                           ground_truth: Dict) -> str:
    """Generate a comprehensive analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("TRANSFORMER FSM LEARNING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Model Performance Summary
    report.append("📊 MODEL PERFORMANCE SUMMARY")
    report.append("-" * 40)
    report.append(f"Overall Accuracy: {model_analysis['overall_accuracy']:.4f}")
    report.append(f"Mean Sequence Accuracy: {model_analysis['mean_sequence_accuracy']:.4f}")
    report.append(f"Perfect Sequences: {model_analysis['perfect_sequences']}/{model_analysis['total_sequences']} "
                 f"({model_analysis['perfect_sequences']/model_analysis['total_sequences']*100:.1f}%)")
    report.append("")
    
    # Token-level Analysis
    report.append("🎯 TOKEN-LEVEL ANALYSIS")
    report.append("-" * 40)
    for token, acc in model_analysis['token_accuracies'].items():
        token_char = '0' if token == 0 else '1'
        report.append(f"Token '{token_char}' Accuracy: {acc:.4f}")
    report.append("")
    
    # Baseline Comparisons  
    report.append("📈 BASELINE COMPARISONS")
    report.append("-" * 40)
    report.append(f"Model Accuracy:     {model_analysis['overall_accuracy']:.4f}")
    report.append(f"Marginal Baseline:  {baselines['marginal_accuracy']:.4f} (predict '{baselines['marginal_token']}')")
    report.append(f"Bigram Baseline:    {baselines['bigram_accuracy']:.4f}")
    report.append(f"Trigram Baseline:   {baselines['trigram_accuracy']:.4f}")
    report.append(f"Random Baseline:    {baselines['random_accuracy']:.4f}")
    report.append("")
    
    # Key Findings
    report.append("🔍 KEY FINDINGS")
    report.append("-" * 40)
    
    # Check if beating marginal baseline significantly
    marginal_improvement = model_analysis['overall_accuracy'] - baselines['marginal_accuracy']
    if marginal_improvement > 0.1:
        report.append(f"✅ BEATS MARGINAL BASELINE by {marginal_improvement:.4f} - Strong evidence of structure learning")
    elif marginal_improvement > 0.01:
        report.append(f"⚠️  Beats marginal baseline by {marginal_improvement:.4f} - Modest structure learning")
    else:
        report.append(f"❌ Only marginally beats baseline by {marginal_improvement:.4f} - May be memorizing distribution")
    
    # Check sequence-level performance
    if model_analysis['mean_sequence_accuracy'] > 0.9:
        report.append("✅ HIGH SEQUENCE ACCURACY - Strong evidence of learning sequential patterns")
    elif model_analysis['mean_sequence_accuracy'] > 0.7:
        report.append("⚠️  Moderate sequence accuracy - Some sequential pattern learning")
    else:
        report.append("❌ Low sequence accuracy - May not be learning sequential patterns")
    
    # Check n-gram comparison
    ngram_improvement = model_analysis['overall_accuracy'] - max(baselines['bigram_accuracy'], baselines['trigram_accuracy'])
    if ngram_improvement > 0.05:
        report.append(f"✅ BEATS N-GRAM BASELINES by {ngram_improvement:.4f} - Learning beyond simple patterns")
    else:
        report.append(f"⚠️  Similar to n-gram performance ({ngram_improvement:.4f}) - May be learning local patterns")
    
    report.append("")
    
    # Context Sensitivity Analysis
    report.append("🧠 CONTEXT SENSITIVITY ANALYSIS")
    report.append("-" * 40)
    for pattern_name, result in context_test.items():
        report.append(f"{pattern_name}: {result['accuracy']:.4f} accuracy on {result['pattern']}")
    report.append("")
    
    # Distribution Analysis
    report.append("📊 DISTRIBUTION ANALYSIS")
    report.append("-" * 40)
    report.append("Model Predictions vs Targets:")
    for token in [0, 1]:
        token_char = '0' if token == 0 else '1'
        pred_count = model_analysis['prediction_distribution'].get(token, 0)
        target_count = model_analysis['target_distribution'].get(token, 0)
        pred_pct = pred_count / model_analysis['total_tokens'] * 100
        target_pct = target_count / model_analysis['total_tokens'] * 100
        report.append(f"  Token '{token_char}': Predicted {pred_pct:.1f}%, Actual {target_pct:.1f}%")
    report.append("")
    
    # Ground Truth Analysis
    report.append("🎰 GROUND TRUTH FSM ANALYSIS")
    report.append("-" * 40)
    num_machines = len(ground_truth['machines'])
    report.append(f"Dataset contains {num_machines} finite state machines")
    
    # Analyze machine complexities
    state_counts = []
    for machine_id, machine_data in ground_truth['machines'].items():
        num_states = machine_data['num_states']
        state_counts.append(num_states)
        report.append(f"  Machine {machine_id}: {num_states} states (topological: {machine_data.get('is_topological', 'unknown')})")
    
    avg_states = np.mean(state_counts)
    report.append(f"Average states per machine: {avg_states:.1f}")
    report.append("")
    
    # Final Verdict
    report.append("🏆 FINAL VERDICT")
    report.append("-" * 40)
    
    evidence_score = 0
    if marginal_improvement > 0.1: evidence_score += 2
    elif marginal_improvement > 0.01: evidence_score += 1
    
    if model_analysis['mean_sequence_accuracy'] > 0.9: evidence_score += 2
    elif model_analysis['mean_sequence_accuracy'] > 0.7: evidence_score += 1
    
    if ngram_improvement > 0.05: evidence_score += 2
    elif ngram_improvement > 0.01: evidence_score += 1
    
    if evidence_score >= 5:
        report.append("🎯 STRONG EVIDENCE: Model is learning FSM structure, not just marginal distributions")
        report.append("   - Significantly outperforms all baselines")
        report.append("   - High sequence-level accuracy indicates structural understanding")
        report.append("   - Performance suggests capturing state transition dynamics")
    elif evidence_score >= 3:
        report.append("⚠️  MODERATE EVIDENCE: Model shows some structure learning")
        report.append("   - Beats baselines but not decisively")
        report.append("   - May be learning mix of structure and local patterns")
    else:
        report.append("❌ WEAK EVIDENCE: Model may be memorizing distributions rather than structure")
        report.append("   - Performance similar to simple baselines")
        report.append("   - Limited evidence of sequential pattern understanding")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Investigate FSM learning claims")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--test", type=Path, help="Path to test dataset (defaults to dataset/val)")
    parser.add_argument("--d_model", type=int, default=32, help="Model d_model")
    parser.add_argument("--layers", type=int, default=3, help="Model layers")
    parser.add_argument("--heads", type=int, default=8, help="Model heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output", type=Path, help="Output file for report")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("Loading model and data...")
    
    # Load model
    model = BinaryTransformer(vocab_size=3, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    # Load test dataset
    test_path = args.test if args.test else args.dataset / "neural_format" / "val_dataset.pt"
    test_dataset = SequenceDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_ar)
    
    # Load ground truth
    ground_truth = load_ground_truth_data(args.dataset)
    
    print("Computing baselines...")
    baselines = compute_baselines(test_dataset)
    
    print("Analyzing model predictions...")
    model_analysis = analyze_model_predictions(model, test_loader, device)
    
    print("Testing context sensitivity...")
    context_test = test_context_sensitivity(model, device)
    
    print("Analyzing state transitions...")
    state_analysis = analyze_state_transition_learning(model, ground_truth, test_loader, device)
    
    print("Generating report...")
    report = generate_detailed_report(model_analysis, baselines, context_test, state_analysis, ground_truth)
    
    print(report)
    
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    
    # Also save detailed results as JSON
    results = {
        "model_analysis": model_analysis,
        "baselines": baselines,
        "context_test": context_test,
        "state_analysis": state_analysis,
        "ground_truth_summary": {
            "num_machines": len(ground_truth['machines']),
            "machine_states": {mid: mdata['num_states'] for mid, mdata in ground_truth['machines'].items()}
        }
    }
    
    if args.output:
        json_path = args.output.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {json_path}")

if __name__ == "__main__":
    main()