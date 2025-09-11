#!/usr/bin/env python3
"""
Inductive Bias Probe for Markov Machine Models

This script implements the inductive bias probe framework from 
"What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models"
adapted for Markov machines (Golden Mean and Even Process).

Usage:
    python markov_inductive_bias_probe.py --model_path path/to/model --data_path path/to/sequences --task golden_mean
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Tuple, Optional
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional: Only import if symbolic regression is needed
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not available. Symbolic regression will be skipped.")

class MarkovStateExtractor:
    """Extract true causal states from Markov machine sequences"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        
    def get_causal_state(self, sequence: List[int]) -> str:
        """Get the true causal state for a sequence"""
        if self.task_type == 'golden_mean':
            return self._golden_mean_state(sequence)
        elif self.task_type == 'even_process':
            return self._even_process_state(sequence)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _golden_mean_state(self, sequence: List[int]) -> str:
        """Golden Mean Machine causal states"""
        if len(sequence) == 0:
            return 'start'  # Can emit both 0 and 1
        
        last_symbol = sequence[-1]
        if last_symbol == 0:
            return 'after_0'  # Can emit both 0 and 1
        else:  # last_symbol == 1
            return 'after_1'  # Can only emit 0 (avoid 11)
    
    def _even_process_state(self, sequence: List[int]) -> str:
        """Even Process causal states"""
        ones_count = sum(sequence)
        return 'even' if ones_count % 2 == 0 else 'odd'
    
    def get_next_symbol_probabilities(self, sequence: List[int]) -> Dict[int, float]:
        """Get true next-symbol probabilities based on causal state"""
        state = self.get_causal_state(sequence)
        
        if self.task_type == 'golden_mean':
            if state in ['start', 'after_0']:
                return {0: 0.5, 1: 0.5}  # Can emit both
            else:  # after_1
                return {0: 1.0, 1: 0.0}  # Must emit 0
                
        elif self.task_type == 'even_process':
            # In even process, typically uniform distribution
            return {0: 0.5, 1: 0.5}
    
    def extract_features(self, sequence: List[int]) -> Dict[str, float]:
        """Extract both causal and potential heuristic features"""
        features = {}
        
        # True causal state features
        state = self.get_causal_state(sequence)
        features['causal_state_numeric'] = 0 if state in ['start', 'after_0', 'even'] else 1
        
        if self.task_type == 'golden_mean':
            features['last_symbol'] = sequence[-1] if sequence else 0
            features['can_emit_both'] = 1 if state in ['start', 'after_0'] else 0
        elif self.task_type == 'even_process':
            features['ones_parity'] = sum(sequence) % 2
            features['is_even_state'] = 1 if state == 'even' else 0
        
        # Potential heuristic features (non-Markovian)
        features['length'] = len(sequence)
        features['num_ones'] = sum(sequence)
        features['num_zeros'] = len(sequence) - sum(sequence)
        features['first_symbol'] = sequence[0] if sequence else 0
        features['ratio_ones'] = sum(sequence) / len(sequence) if sequence else 0
        
        # Recent history features
        features['last_two_sum'] = sum(sequence[-2:]) if len(sequence) >= 2 else 0
        features['last_three_sum'] = sum(sequence[-3:]) if len(sequence) >= 3 else 0
        
        return features

class InductiveBiasProbe:
    """Implementation of inductive bias probe for Markov machines"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def get_model_prediction(self, sequence: List[int]) -> float:
        """Get model's predicted probability of next symbol being 1"""
        if len(sequence) == 0:
            sequence = [0]  # Start token
            
        # Convert to tensor
        seq_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(seq_tensor)
            
            # Get logits for next token prediction
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[0, -1]  # Last position
            else:
                logits = outputs[0, -1]  # Assume raw logits
                
            # Convert to probabilities
            probs = torch.softmax(logits, dim=0)
            
            # Return probability of symbol 1
            if len(probs) > 1:
                return probs[1].item()
            else:
                return 0.5  # Default if binary classification
    
    def create_synthetic_datasets(self, sequences: List[List[int]], 
                                 state_extractor: MarkovStateExtractor, 
                                 num_datasets: int = 100) -> List[Tuple[List[List[int]], List[int]]]:
        """Create synthetic probe datasets where output depends only on causal state"""
        datasets = []
        
        # Get causal states for all sequences
        causal_states = [state_extractor.get_causal_state(seq) for seq in sequences]
        unique_states = list(set(causal_states))
        
        for dataset_idx in range(num_datasets):
            outputs = []
            
            for seq, state in zip(sequences, causal_states):
                # Create binary task: randomly assign each causal state to 0 or 1
                # Keep assignment consistent within this dataset
                np.random.seed(dataset_idx * 1000 + hash(state) % 1000)
                output = np.random.choice([0, 1])
                outputs.append(output)
            
            datasets.append((sequences.copy(), outputs))
        
        return datasets
    
    def fine_tune_on_dataset(self, sequences: List[List[int]], outputs: List[int], 
                           epochs: int = 10, lr: float = 1e-4) -> torch.nn.Module:
        """Fine-tune model on a synthetic dataset"""
        # Create a simple classification head
        model_copy = torch.nn.Sequential(
            self.model,
            torch.nn.Linear(self.model.config.n_embd if hasattr(self.model.config, 'n_embd') else 512, 2)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        model_copy.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for seq, target in zip(sequences, outputs):
                if len(seq) == 0:
                    seq = [0]
                    
                seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                target_tensor = torch.tensor([target], dtype=torch.long).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                hidden = self.model(seq_tensor, output_hidden_states=True).hidden_states[-1]
                logits = model_copy[-1](hidden[0, -1])  # Last position
                
                loss = criterion(logits.unsqueeze(0), target_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        model_copy.eval()
        return model_copy
    
    def compute_rib_dib_metrics(self, sequences: List[List[int]], 
                               state_extractor: MarkovStateExtractor,
                               num_probe_datasets: int = 20) -> Tuple[float, float]:
        """Compute R-IB and D-IB metrics"""
        
        # Create synthetic datasets
        probe_datasets = self.create_synthetic_datasets(sequences, state_extractor, num_probe_datasets)
        
        # Get causal states
        causal_states = [state_extractor.get_causal_state(seq) for seq in sequences]
        
        rib_scores = []
        dib_scores = []
        
        for dataset_sequences, dataset_outputs in probe_datasets[:num_probe_datasets]:
            # Fine-tune model on this dataset
            try:
                fine_tuned_model = self.fine_tune_on_dataset(dataset_sequences, dataset_outputs)
                
                # Get predictions from fine-tuned model
                predictions = []
                for seq in sequences:
                    if len(seq) == 0:
                        seq = [0]
                    seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        hidden = self.model(seq_tensor, output_hidden_states=True).hidden_states[-1]
                        logits = fine_tuned_model[-1](hidden[0, -1])
                        pred = torch.argmax(logits).item()
                        predictions.append(pred)
                
                # Compute R-IB: sequences with same causal state should have same prediction
                same_state_pairs = []
                diff_state_pairs = []
                
                for i in range(len(sequences)):
                    for j in range(i+1, len(sequences)):
                        if causal_states[i] == causal_states[j]:
                            same_state_pairs.append((predictions[i], predictions[j]))
                        else:
                            diff_state_pairs.append((predictions[i], predictions[j]))
                
                # R-IB: fraction of same-state pairs with same prediction
                if same_state_pairs:
                    rib = np.mean([pred1 == pred2 for pred1, pred2 in same_state_pairs])
                else:
                    rib = 1.0
                
                # D-IB: 1 - fraction of different-state pairs with same prediction
                if diff_state_pairs:
                    dib = 1 - np.mean([pred1 == pred2 for pred1, pred2 in diff_state_pairs])
                else:
                    dib = 1.0
                
                rib_scores.append(rib)
                dib_scores.append(dib)
                
            except Exception as e:
                print(f"Error in fine-tuning dataset: {e}")
                continue
        
        return np.mean(rib_scores), np.mean(dib_scores)

class SymbolicRegressionAnalyzer:
    """Analyze learned formulas using symbolic regression"""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        
    def analyze_model_formula(self, model_predictions: List[float], 
                            feature_vectors: List[Dict[str, float]],
                            max_iterations: int = 100) -> Optional[object]:
        """Run symbolic regression to discover learned formula"""
        
        if not PYSR_AVAILABLE:
            print("PySR not available, skipping symbolic regression")
            return None
        
        # Convert features to numpy array
        feature_names = list(feature_vectors[0].keys())
        X = np.array([[features[name] for name in feature_names] for features in feature_vectors])
        y = np.array(model_predictions)
        
        # Configure symbolic regression
        model_sr = PySRRegressor(
            populations=15,
            niterations=max_iterations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["abs"],
            constraints={"/": (1, 1)},  # Simple division only
            complexity_of_operators={"+": 1, "-": 1, "*": 1, "/": 2, "abs": 1},
            parsimony=0.01,
            feature_selection=True,
            timeout_in_seconds=180,
            feature_names=feature_names,
            random_state=42
        )
        
        try:
            model_sr.fit(X, y)
            return model_sr
        except Exception as e:
            print(f"Symbolic regression failed: {e}")
            return None
    
    def interpret_formula(self, sr_model, feature_names: List[str]) -> Dict[str, any]:
        """Interpret the discovered formula for Markovian behavior"""
        if sr_model is None:
            return {"error": "No symbolic regression model available"}
        
        try:
            best_equations = sr_model.get_best()
            best_eq = best_equations.iloc[0]
            
            equation_str = str(best_eq['equation'])
            complexity = best_eq['complexity']
            score = best_eq['score']
            
            # Analyze which features are used
            used_features = [f for f in feature_names if f in equation_str]
            
            # Categorize features
            markov_features = self._get_markov_features()
            non_markov_features = [f for f in used_features if f not in markov_features]
            
            analysis = {
                "equation": equation_str,
                "complexity": complexity,
                "score": score,
                "used_features": used_features,
                "markov_features_used": [f for f in used_features if f in markov_features],
                "non_markov_features_used": non_markov_features,
                "is_markovian": len(non_markov_features) == 0,
                "markov_score": len([f for f in used_features if f in markov_features]) / max(len(used_features), 1)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Failed to interpret formula: {e}"}
    
    def _get_markov_features(self) -> List[str]:
        """Get features that a true Markov model should use"""
        if self.task_type == 'golden_mean':
            return ['last_symbol', 'can_emit_both', 'causal_state_numeric']
        elif self.task_type == 'even_process':
            return ['ones_parity', 'is_even_state', 'causal_state_numeric']
        return []

def run_complete_analysis(model, sequences: List[List[int]], task_type: str, 
                         model_name: str = "model", num_probe_datasets: int = 20) -> Dict[str, any]:
    """Run complete inductive bias probe analysis"""
    
    print(f"\n{'='*60}")
    print(f"Running Inductive Bias Probe Analysis for {model_name}")
    print(f"Task: {task_type}")
    print(f"Number of sequences: {len(sequences)}")
    print(f"{'='*60}")
    
    # Initialize components
    state_extractor = MarkovStateExtractor(task_type)
    probe = InductiveBiasProbe(model, None)  # Assuming no tokenizer needed
    sr_analyzer = SymbolicRegressionAnalyzer(task_type)
    
    # 1. Analyze causal state alignment
    print("\n1. Analyzing causal state alignment...")
    causal_states = [state_extractor.get_causal_state(seq) for seq in sequences]
    unique_states = list(set(causal_states))
    print(f"Found {len(unique_states)} unique causal states: {unique_states}")
    
    # 2. Compute inductive bias metrics
    print("\n2. Computing inductive bias metrics...")
    try:
        rib_score, dib_score = probe.compute_rib_dib_metrics(sequences, state_extractor, num_probe_datasets)
        print(f"R-IB (Respecting State): {rib_score:.3f}")
        print(f"D-IB (Distinguishing State): {dib_score:.3f}")
    except Exception as e:
        print(f"Error computing IB metrics: {e}")
        rib_score, dib_score = 0.0, 0.0
    
    # 3. Symbolic regression analysis
    print("\n3. Running symbolic regression analysis...")
    
    # Extract model predictions and features
    model_predictions = []
    feature_vectors = []
    
    for seq in sequences:
        try:
            pred = probe.get_model_prediction(seq)
            features = state_extractor.extract_features(seq)
            
            model_predictions.append(pred)
            feature_vectors.append(features)
        except Exception as e:
            print(f"Error processing sequence {seq}: {e}")
            continue
    
    sr_model = sr_analyzer.analyze_model_formula(model_predictions, feature_vectors)
    formula_analysis = sr_analyzer.interpret_formula(sr_model, list(feature_vectors[0].keys()) if feature_vectors else [])
    
    if "error" not in formula_analysis:
        print(f"Discovered formula: {formula_analysis['equation']}")
        print(f"Formula complexity: {formula_analysis['complexity']}")
        print(f"Uses Markovian features: {formula_analysis['markov_features_used']}")
        print(f"Uses non-Markovian features: {formula_analysis['non_markov_features_used']}")
        print(f"Markov score: {formula_analysis['markov_score']:.3f}")
    else:
        print(f"Symbolic regression error: {formula_analysis['error']}")
    
    # 4. Compile results
    results = {
        "model_name": model_name,
        "task_type": task_type,
        "num_sequences": len(sequences),
        "unique_causal_states": unique_states,
        "rib_score": rib_score,
        "dib_score": dib_score,
        "formula_analysis": formula_analysis,
        "model_predictions_sample": model_predictions[:10],  # First 10 for inspection
    }
    
    return results

def compare_models(model1, model2, sequences: List[List[int]], task_type: str,
                  model1_name: str = "Model 1", model2_name: str = "Model 2") -> Dict[str, any]:
    """Compare two models using inductive bias probe"""
    
    print(f"\n{'='*80}")
    print(f"COMPARING MODELS: {model1_name} vs {model2_name}")
    print(f"{'='*80}")
    
    # Analyze both models
    results1 = run_complete_analysis(model1, sequences, task_type, model1_name)
    results2 = run_complete_analysis(model2, sequences, task_type, model2_name)
    
    # Create comparison
    comparison = {
        "model1_results": results1,
        "model2_results": results2,
        "rib_comparison": {
            "model1_rib": results1["rib_score"],
            "model2_rib": results2["rib_score"],
            "rib_difference": results1["rib_score"] - results2["rib_score"],
            "winner": model1_name if results1["rib_score"] > results2["rib_score"] else model2_name
        },
        "dib_comparison": {
            "model1_dib": results1["dib_score"],
            "model2_dib": results2["dib_score"], 
            "dib_difference": results1["dib_score"] - results2["dib_score"],
            "winner": model1_name if results1["dib_score"] > results2["dib_score"] else model2_name
        }
    }
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"R-IB (Respecting State):")
    print(f"  {model1_name}: {results1['rib_score']:.3f}")
    print(f"  {model2_name}: {results2['rib_score']:.3f}")
    print(f"  Winner: {comparison['rib_comparison']['winner']}")
    
    print(f"\nD-IB (Distinguishing State):")
    print(f"  {model1_name}: {results1['dib_score']:.3f}")
    print(f"  {model2_name}: {results2['dib_score']:.3f}")
    print(f"  Winner: {comparison['dib_comparison']['winner']}")
    
    # Formula comparison
    if "error" not in results1["formula_analysis"] and "error" not in results2["formula_analysis"]:
        print(f"\nFormula Analysis:")
        print(f"  {model1_name} formula: {results1['formula_analysis']['equation']}")
        print(f"  {model1_name} Markov score: {results1['formula_analysis']['markov_score']:.3f}")
        print(f"  {model2_name} formula: {results2['formula_analysis']['equation']}")
        print(f"  {model2_name} Markov score: {results2['formula_analysis']['markov_score']:.3f}")
    
    return comparison

def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Run Inductive Bias Probe on Markov Machine Models')
    parser.add_argument('--model1_path', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2_path', type=str, default=None, help='Path to second model (optional)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to sequence data')
    parser.add_argument('--task', choices=['golden_mean', 'even_process'], required=True, help='Task type')
    parser.add_argument('--output_path', type=str, default='probe_results.json', help='Output file for results')
    parser.add_argument('--num_probe_datasets', type=int, default=20, help='Number of probe datasets')
    parser.add_argument('--model1_name', type=str, default='Model1', help='Name for first model')
    parser.add_argument('--model2_name', type=str, default='Model2', help='Name for second model')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    # Implement your data loading logic here
    # sequences = load_sequences(args.data_path)
    
    # Load models
    print(f"Loading model from {args.model1_path}")
    # Implement your model loading logic here
    # model1 = load_model(args.model1_path)
    
    # For now, create placeholder
    sequences = [[0, 1, 0], [1, 0, 1], [0, 0, 1]]  # Replace with actual data loading
    model1 = None  # Replace with actual model loading
    
    if args.model2_path:
        print(f"Loading second model from {args.model2_path}")
        # model2 = load_model(args.model2_path)
        model2 = None  # Replace with actual model loading
        
        results = compare_models(model1, model2, sequences, args.task, 
                               args.model1_name, args.model2_name)
    else:
        results = run_complete_analysis(model1, sequences, args.task, args.model1_name)
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    # Example usage without command line args
    print("Markov Inductive Bias Probe - Example Usage")
    print("To use with your models, implement the model loading functions and run with appropriate arguments")
    
    # You can also run directly with your models:
    # sequences = your_load_sequences_function()
    # model = your_load_model_function() 
    # results = run_complete_analysis(model, sequences, 'golden_mean', 'nanoGPT')
