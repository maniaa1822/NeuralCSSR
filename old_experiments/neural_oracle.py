#!/usr/bin/env python3
"""
Neural Probability Oracle for CSSR

Instead of pre-computing all probabilities, this creates an oracle that CSSR
can query in real-time for any context.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_delay_transformer import BinaryTransformer, SequenceDataset
from torch.utils.data import DataLoader

class NeuralProbabilityOracle:
    """Real-time neural probability oracle for CSSR."""
    
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
        
        # Cache for frequently queried contexts
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_probabilities(self, context: List[int]) -> Dict[str, float]:
        """Get next symbol probabilities for a given context."""
        context_tuple = tuple(context)
        
        # Check cache first
        if context_tuple in self.cache:
            self.cache_hits += 1
            return self.cache[context_tuple]
        
        self.cache_misses += 1
        
        # Handle empty context
        if not context:
            # Return uniform distribution for empty context
            return {'0': 0.5, '1': 0.5}
        
        # Create input tensor
        context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.device)  # [1, L]
        
        with torch.no_grad():
            logits = self.model(context_tensor)  # [1, L, 3]
            probs = F.softmax(logits, dim=-1)  # [1, L, 3]
            
            # Get probabilities for the last position (next symbol prediction)
            next_probs = probs[0, -1, :].cpu().numpy()  # [3]
            
            # DEBUG: Let's see what we're actually getting
            if len(context) <= 3:  # Only debug for short contexts
                print(f"DEBUG - Context: {context}")
                print(f"DEBUG - Raw logits: {logits[0, -1, :].cpu().numpy()}")
                print(f"DEBUG - Softmax probs: {next_probs}")
                print(f"DEBUG - Model vocab_size: {self.model.proj.out_features}")
            
            # CRITICAL QUESTION: What does vocab_size=3 actually represent?
            # From the SequenceDataset, we know:
            # - Original vocab: [PAD=0, UNK=1, '0'=2, '1'=3] 
            # - But SequenceDataset maps: 2→0 and 3→1
            # - So our binary sequences contain symbols 0 and 1
            # - But the model was trained with vocab_size=3, so it outputs 3 classes
            
            # The question is: what are these 3 classes?
            # Option 1: [PAD, '0', '1'] - but this doesn't make sense for binary sequences
            # Option 2: ['0', '1', padding] - this might be correct
            # Option 3: Something else entirely
            
            # Let's assume for now that indices 0 and 1 correspond to symbols '0' and '1'
            prob_0 = float(next_probs[0])  # Probability of symbol '0'
            prob_1 = float(next_probs[1])  # Probability of symbol '1'
            prob_2 = float(next_probs[2])  # Probability of ??? (padding?)
            
            if len(context) <= 3:  # Debug output
                print(f"DEBUG - Extracted: P(0)={prob_0:.4f}, P(1)={prob_1:.4f}, P(2)={prob_2:.4f}")
            
            # Normalize ignoring the third class (assuming it's padding)
            total = prob_0 + prob_1
            if total > 0:
                result = {
                    '0': prob_0 / total,
                    '1': prob_1 / total
                }
            else:
                result = {'0': 0.5, '1': 0.5}  # Fallback
    
        # Cache the result
        self.cache[context_tuple] = result
        return result
    
    def evaluate_on_dataset(self, dataset_path: Path) -> Dict[str, float]:
        """Evaluate oracle on dataset using on-demand probability queries."""
        print("Evaluating Neural Oracle...")
        
        dataset = SequenceDataset(dataset_path)
        correct = 0
        total = 0
        log_likelihood = 0.0
        
        for sequence in tqdm(dataset.data, desc="Evaluating sequences"):
            seq = sequence.numpy()
            
            for t in range(1, len(seq)):  # Skip first symbol
                # Use fixed context window (like training)
                context_window = 20  # Reasonable window size
                start_idx = max(0, t - context_window)
                context = seq[start_idx:t].tolist()
                true_next = seq[t]
                
                # Query oracle for probabilities
                probs = self.get_probabilities(context)
                
                # Hard prediction
                predicted_next = 0 if probs['0'] > probs['1'] else 1
                
                # Accuracy
                if predicted_next == true_next:
                    correct += 1
                total += 1
                
                # Log-likelihood
                true_prob = probs[str(true_next)]
                log_likelihood += np.log(max(true_prob, 1e-10))
    
        accuracy = correct / total if total > 0 else 0.0
        perplexity = np.exp(-log_likelihood / total) if total > 0 else float('inf')
        
        return {
            'accuracy': float(accuracy),  # Ensure Python float
            'perplexity': float(perplexity),  # Ensure Python float
            'total_predictions': int(total),
            'correct_predictions': int(correct),
            'cache_hits': int(self.cache_hits),
            'cache_misses': int(self.cache_misses),
            'cache_hit_rate': float(self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0)
        }

def main():
    parser = argparse.ArgumentParser(description="Neural Probability Oracle for CSSR")
    parser.add_argument('--model', type=Path, required=True, help='Path to trained transformer model')
    parser.add_argument('--test_data', type=Path, required=True, help='Path to test dataset')
    parser.add_argument('--d_model', type=int, default=10, help='Model dimension')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--output', type=Path, default=Path('neural_oracle_results'), help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEURAL PROBABILITY ORACLE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print()
    
    # Create neural oracle
    oracle = NeuralProbabilityOracle(args.model, args.d_model, args.layers, args.heads)
    
    # Evaluate on test set
    print("Evaluating Neural Oracle...")
    results = oracle.evaluate_on_dataset(args.test_data)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Neural Oracle Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Perplexity: {results['perplexity']:.4f}")
    print(f"  Total predictions: {results['total_predictions']:,}")
    print(f"  Correct predictions: {results['correct_predictions']:,}")
    print(f"  Cache hit rate: {results['cache_hit_rate']:.2f}% ({results['cache_hits']:,}/{results['cache_hits']+results['cache_misses']:,})")
    print()
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    with open(args.output / 'oracle_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output}/oracle_results.json")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i, context in enumerate([[0], [1], [0, 0], [1, 1], [0, 1], [1, 0]]):
        if i >= 5:
            break
        probs = oracle.get_probabilities(context)
        print(f"  Context {context} → P(0)={probs['0']:.3f}, P(1)={probs['1']:.3f}")

    print("\nDebug: Checking probability extraction...")
    # Test with a simple context
    test_context = [0, 0, 0]  # Three zeros
    probs = oracle.get_probabilities(test_context)
    print(f"Context [0,0,0] → P(0)={probs['0']:.4f}, P(1)={probs['1']:.4f}")

    # Test with alternating pattern
    test_context = [0, 1, 0, 1]
    probs = oracle.get_probabilities(test_context)
    print(f"Context [0,1,0,1] → P(0)={probs['0']:.4f}, P(1)={probs['1']:.4f}")

if __name__ == '__main__':
    main()