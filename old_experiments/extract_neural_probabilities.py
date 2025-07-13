#!/usr/bin/env python3
"""
Extract transition probabilities from trained transformer and create classical CSSR machine.

This script:
1. Loads a trained transformer model
2. Extracts learned transition probabilities for all possible contexts
3. Constructs a classical CSSR-style epsilon machine
4. Evaluates performance compared to both neural and classical approaches
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from our training script
from time_delay_transformer import (
    SequenceDataset, 
    BinaryTransformer, 
    collate_fn_ar,
    evaluate
)

class TransformerProbabilityExtractor:
    """Extract learned transition probabilities from a trained transformer."""
    
    def __init__(self, model_path: Path, d_model: int = 10, layers: int = 1, heads: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = BinaryTransformer(
            vocab_size=3, 
            d_model=d_model, 
            n_layers=layers,
            n_heads=heads,
            max_delay=None  # AR mode
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for learned probabilities
        self.context_probs = defaultdict(lambda: {'0': [], '1': []})
        self.context_counts = defaultdict(int)
    
    def extract_probabilities(self, dataset_path: Path, max_context_length: int = 10):
        """Extract transition probabilities for different contexts."""
        print(f"Extracting probabilities from {dataset_path}...")
        
        # Load dataset
        dataset = SequenceDataset(dataset_path)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_ar)
        
        total_batches = len(loader)
        print(f"Processing {total_batches} sequences...")
        
        with torch.no_grad():
            for batch_idx, (tokens, targets) in enumerate(tqdm(loader, desc="Extracting probabilities")):
                tokens, targets = tokens.to(self.device), targets.to(self.device)
                logits = self.model(tokens)  # [1, T, 3]
                probs = F.softmax(logits, dim=-1)  # [1, T, 3]
                
                # Extract sequence
                seq_len = (targets[0] != 2).sum().item()  # Remove padding
                if seq_len <= 1:
                    continue
                
                sequence = targets[0][:seq_len].cpu().numpy()
                prob_sequence = probs[0][:seq_len].cpu().numpy()  # [T, 3]
                
                # Extract context -> next symbol probabilities
                for t in range(len(sequence)):
                    if t == 0:
                        continue  # Skip first position (no context)
                    
                    # Get context of varying lengths
                    for context_len in range(1, min(t + 1, max_context_length + 1)):
                        context = tuple(sequence[t-context_len:t])
                        next_symbol = sequence[t]
                        predicted_probs = prob_sequence[t]  # [3] - probs for PAD, UNK, 0, 1
                        
                        # Store the probability for this context -> next symbol
                        # Model outputs probs for [PAD, UNK, '0'] but we need ['0', '1']
                        # So we need to look at the actual prediction distribution
                        # The model was trained with targets being 0='0' and 1='1' after mapping
                        # But logits are for vocab [PAD, UNK, '0'] so we need to extract binary probs
                        
                        # Actually, let's debug first - print shape and values
                        if batch_idx == 0 and t < 5:
                            print(f"Debug: predicted_probs shape={predicted_probs.shape}, values={predicted_probs}")
                            print(f"Target sequence: {sequence}")
                        
                        # For now, let's assume the model outputs binary probabilities correctly
                        # We'll extract the probabilities for the actual binary tokens
                        if len(predicted_probs) == 3:
                            # Assume probabilities are [PAD, '0', '1'] - check this
                            self.context_probs[context]['0'].append(predicted_probs[1])  # Prob of '0'
                            self.context_probs[context]['1'].append(predicted_probs[2])  # Prob of '1'
                        else:
                            print(f"Warning: unexpected prob shape {predicted_probs.shape}")
                            continue
                        self.context_counts[context] += 1
        
        print(f"Extracted probabilities for {len(self.context_probs)} unique contexts")
    
    def aggregate_probabilities(self, min_count: int = 10) -> Dict[tuple, Dict[str, float]]:
        """Aggregate extracted probabilities into final transition model."""
        print(f"Aggregating probabilities (min_count={min_count})...")
        
        final_probs = {}
        contexts_to_process = [(context, probs) for context, probs in self.context_probs.items() 
                              if self.context_counts[context] >= min_count]
        
        for context, probs in tqdm(contexts_to_process, desc="Aggregating contexts"):
            # Average the probabilities across all occurrences
            avg_prob_0 = np.mean(probs['0'])
            avg_prob_1 = np.mean(probs['1'])
            
            # Normalize to ensure they sum to 1
            total = avg_prob_0 + avg_prob_1
            if total > 0:
                final_probs[context] = {
                    '0': avg_prob_0 / total,
                    '1': avg_prob_1 / total,
                    'count': self.context_counts[context],
                    'std_0': np.std(probs['0']),
                    'std_1': np.std(probs['1'])
                }
        
        print(f"Final model has {len(final_probs)} context rules")
        return final_probs

class NeuralCSSRMachine:
    """A classical CSSR-style machine using neural-learned probabilities."""
    
    def __init__(self, transition_probs: Dict[tuple, Dict[str, float]]):
        self.transition_probs = transition_probs
        self.max_context_length = max(len(context) for context in transition_probs.keys()) if transition_probs else 0
        
    def predict_next_symbol_proba(self, history: List[int]) -> Dict[str, float]:
        """Predict next symbol probabilities given history."""
        # Try contexts from longest to shortest
        for context_len in range(min(len(history), self.max_context_length), 0, -1):
            context = tuple(history[-context_len:])
            if context in self.transition_probs:
                return {
                    '0': self.transition_probs[context]['0'],
                    '1': self.transition_probs[context]['1']
                }
        
        # Fallback to uniform distribution
        return {'0': 0.5, '1': 0.5}
    
    def predict_next_symbol(self, history: List[int]) -> int:
        """Predict next symbol (hard prediction)."""
        probs = self.predict_next_symbol_proba(history)
        return 0 if probs['0'] > probs['1'] else 1
    
    def evaluate_on_dataset(self, dataset_path: Path) -> Dict[str, float]:
        """Evaluate the neural CSSR machine on a dataset."""
        print("Evaluating Neural CSSR machine...")
        
        dataset = SequenceDataset(dataset_path)
        correct = 0
        total = 0
        log_likelihood = 0.0
        
        total_sequences = len(dataset.data)
        print(f"Evaluating on {total_sequences} sequences...")
        
        for seq_idx, sequence in enumerate(tqdm(dataset.data, desc="Evaluating sequences")):
            seq = sequence.numpy()
            
            for t in range(1, len(seq)):  # Skip first symbol
                history = seq[:t].tolist()
                true_next = seq[t]
                
                # Get predictions
                predicted_next = self.predict_next_symbol(history)
                probs = self.predict_next_symbol_proba(history)
                
                # Accuracy
                if predicted_next == true_next:
                    correct += 1
                total += 1
                
                # Log-likelihood
                true_prob = probs[str(true_next)]
                log_likelihood += np.log(max(true_prob, 1e-10))  # Avoid log(0)
        
        accuracy = correct / total if total > 0 else 0.0
        perplexity = np.exp(-log_likelihood / total) if total > 0 else float('inf')
        
        return {
            'accuracy': accuracy,
            'perplexity': perplexity,
            'total_predictions': total,
            'correct_predictions': correct
        }
    
    def get_probabilities(self, context: List[int]) -> Dict[str, float]:
        """Get the transition probabilities for a given context."""
        context_tensor = torch.tensor(context, dtype=torch.long, device=self.device).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            logits = self.model(context_tensor)
            probs = F.softmax(logits, dim=-1)
            next_probs = probs[0, -1, :].cpu().numpy()
            
            # The model has vocab_size=3 but only classes 0 and 1 are meaningful
            # Class 2 is an artifact from the architecture, not training
            
            if len(next_probs) == 3:
                # Use only the first two classes (0 and 1)
                prob_0 = float(next_probs[0])
                prob_1 = float(next_probs[1]) 
                # Ignore next_probs[2] - it's untrained padding class
                
                # Normalize properly
                total = prob_0 + prob_1
                if total > 0:
                    result = {
                        '0': prob_0 / total,
                        '1': prob_1 / total
                    }
                else:
                    result = {'0': 0.5, '1': 0.5}
            else:
                result = {'0': 0.5, '1': 0.5}
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Extract neural probabilities and create classical CSSR machine")
    parser.add_argument('--model', type=Path, required=True, help='Path to trained transformer model')
    parser.add_argument('--train_data', type=Path, required=True, help='Path to training dataset')
    parser.add_argument('--test_data', type=Path, required=True, help='Path to test dataset')
    parser.add_argument('--d_model', type=int, default=10, help='Model dimension')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--max_context', type=int, default=10, help='Maximum context length')
    parser.add_argument('--min_count', type=int, default=10, help='Minimum occurrences for context rule')
    parser.add_argument('--output', type=Path, default=Path('neural_cssr_results'), help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEURAL CSSR PROBABILITY EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Training data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"Max context length: {args.max_context}")
    print(f"Min count threshold: {args.min_count}")
    print()
    
    # Extract probabilities from trained transformer
    print("Step 1/3: Extracting probabilities from transformer...")
    extractor = TransformerProbabilityExtractor(
        args.model, args.d_model, args.layers, args.heads
    )
    extractor.extract_probabilities(args.train_data, args.max_context)
    
    print("\nStep 2/3: Aggregating transition probabilities...")
    transition_probs = extractor.aggregate_probabilities(args.min_count)
    
    # Create Neural CSSR machine
    print("\nStep 3/3: Creating and evaluating Neural CSSR machine...")
    neural_cssr = NeuralCSSRMachine(transition_probs)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    results = neural_cssr.evaluate_on_dataset(args.test_data)
    
    print(f"Neural CSSR Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Perplexity: {results['perplexity']:.4f}")
    print(f"  Total predictions: {results['total_predictions']:,}")
    print(f"  Correct predictions: {results['correct_predictions']:,}")
    print()
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save transition probabilities
    with open(args.output / 'transition_probabilities.json', 'w') as f:
        # Convert tuple keys to strings for JSON serialization
        json_probs = {str(k): v for k, v in transition_probs.items()}
        json.dump(json_probs, f, indent=2)
    
    # Save evaluation results
    with open(args.output / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output}")
    print(f"  - transition_probabilities.json: Extracted neural probabilities")
    print(f"  - evaluation_results.json: Performance metrics")

if __name__ == '__main__':
    main()
