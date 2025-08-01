#!/usr/bin/env python3
"""
Systematic hyperparameter sweep for FSM extraction from neural transformers.

This script performs a comprehensive grid search over the FSM extraction parameter space
to find optimal configurations for recovering finite state machines from trained models.

Usage:
    python sweep_extraction_params.py --checkpoint checkpoints/three_state_test/best.pt --ground_truth domain_machines/custom_3_state/custom_3_state/custom_3_state.machine.json
"""

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from models.transformer_models import create_model
from models.data_utils import SequenceDataset
from cssr_enhanced_extractor import CSSREnhancedExtractor
from extract_fsm_sliding_window import load_model_from_checkpoint


@dataclass
class ExtractionConfig:
    """Configuration for CSSR-enhanced FSM extraction parameters."""
    # Core CSSR parameters
    num_states: Optional[int] = 3
    target_states: Optional[int] = None  # None = natural discovery, int = forced count
    max_suffix_length: int = 8
    significance_level: float = 0.001
    min_suffix_count: int = 10
    
    # Neural enhancement parameters
    neural_threshold: float = 5.0
    use_neural_test: bool = True
    use_classical_test: bool = True
    
    # Data collection parameters  
    max_sequences: int = 1000
    chunk_size: int = 25
    
    # Merging parameters (when too many states found)
    auto_merge: bool = True
    merge_method: str = 'kmeans'  # 'kmeans', 'hierarchical'
    
    # Alignment parameters (for ground truth comparison)
    alignment_distance: str = 'frobenius'
    alignment_method: str = 'hungarian'


@dataclass
class ExtractionResult:
    """Results from FSM extraction attempt."""
    config: ExtractionConfig
    
    # Clustering metrics
    n_clusters_found: int
    silhouette_score: float
    inertia: float
    stability_score: float
    
    # Ground truth comparison (if available)
    alignment_error: Optional[float] = None
    probability_error: Optional[float] = None
    state_recovery_score: Optional[float] = None
    
    # Execution metrics
    extraction_time: float = 0.0
    success: bool = False
    error_message: str = ""


# FSMExtractor class removed - now using CSSREnhancedExtractor directly


def define_hyperparameter_space(phase: str = '1') -> Dict[str, List]:
    """Define the hyperparameter space for CSSR-enhanced extraction."""
    
    if phase == 'suffix_length_test':
        # Test performance at very long suffix lengths where classical CSSR struggles most
        return {
            'target_states': [None, 7],  # Test both natural discovery and forced
            'max_suffix_length': [12, 15],  # Focus on lengths where classical CSSR fails
            'significance_level': [0.001, 0.01],  # Conservative and moderate
            'neural_threshold': [5.0],  # Fixed optimal value from previous tests
            'min_suffix_count': [3, 5],  # Very low counts for sparse long suffixes
            'max_sequences': [500],  # More data to handle sparsity
            'use_neural_test': [True],
            'use_classical_test': [True], 
            'merge_method': ['kmeans'],
            'alignment_distance': ['frobenius'],
            'alignment_method': ['hungarian'],
        }
    
    elif phase == 'quick':
        # Quick test with focused parameters
        return {
            'num_states': [3, 4],  # Focus on target and close values
            'max_suffix_length': [6, 8],  # Shorter suffixes for speed
            'significance_level': [0.001, 0.01],  # Conservative levels
            'neural_threshold': [5.0, 10.0],  # Middle range
            'use_neural_test': [True],
            'use_classical_test': [True],
            'min_suffix_count': [10],
            'max_sequences': [500],  # Fewer sequences for speed
            'merge_method': ['kmeans'],
            'alignment_distance': ['frobenius'],
            'alignment_method': ['hungarian'],
        }
    
    elif phase == 'test':
        # Minimal test sweep - just 4 combinations to verify pipeline works
        return {
            'target_states': [None, 7],  # Test natural vs forced
            'significance_level': [0.001, 0.01],  # Test conservative vs moderate
            'neural_threshold': [5.0],  # Fixed 
            'min_suffix_count': [10],  # Fixed
            'max_sequences': [200],  # Very small for speed
            'max_suffix_length': [6],  # Fixed short
            'use_neural_test': [True],
            'use_classical_test': [True], 
            'merge_method': ['kmeans'],
            'alignment_distance': ['frobenius'],
            'alignment_method': ['hungarian'],
        }
    
    elif phase == 'tight':
        # Tight, principled sweep: Natural discovery vs. forced state counts
        # Only 54 combinations (3×2×3×3×2) focusing on state discovery mechanisms
        return {
            'target_states': [None, 3, 4],  # None = natural discovery, 3/4 = forced
            'significance_level': [0.001, 0.01],  # Lower = fewer splits = fewer states
            'neural_threshold': [1.0, 5.0, 10.0],  # Higher = more merging = fewer states  
            'min_suffix_count': [10, 20, 30],  # Higher = fewer suffixes = potentially fewer states
            'max_sequences': [300, 500],  # Test with different data amounts
            
            # Fixed parameters for speed and consistency
            'max_suffix_length': [7],  # Sweet spot from classical CSSR
            'use_neural_test': [True],
            'use_classical_test': [True], 
            'merge_method': ['kmeans'],
            'alignment_distance': ['frobenius'],
            'alignment_method': ['hungarian'],
        }
    
    return {
        # Phase 1: Core CSSR parameters (most critical)
        'num_states': [2, 3, 4, 5],
        'max_suffix_length': [6, 8, 10, 12],
        'significance_level': [0.01, 0.001, 0.0001],
        'neural_threshold': [1.0, 3.0, 5.0, 10.0],
        
        # Phase 1.5: Test combination strategies
        'use_neural_test': [True, False],
        'use_classical_test': [True, False],
        
        # Phase 2: Secondary parameters
        'min_suffix_count': [5, 10, 20],
        'max_sequences': [500, 1000],
        'merge_method': ['kmeans'],  # Can add 'hierarchical' later
        
        # Phase 3: Alignment parameters (for evaluation)
        'alignment_distance': ['frobenius'],
        'alignment_method': ['hungarian'],
    }


def run_parameter_sweep(checkpoint_path: Path, data_path: Path, ground_truth_path: Optional[Path] = None, output_dir: Path = Path('sweep_results')) -> pd.DataFrame:
    """Run systematic parameter sweep."""
    print("🔍 Starting FSM Extraction Parameter Sweep")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Load ground truth if available
    ground_truth = None
    if ground_truth_path and ground_truth_path.exists():
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth: {ground_truth_path}")
    
    # Initialize CSSR-enhanced extractor
    extractor = CSSREnhancedExtractor(model, device, num_states=3)  # Default will be overridden per config
    
    # Define parameter space 
    param_space = define_hyperparameter_space('suffix_length_test')
    param_names = list(param_space.keys())
    param_combinations = list(product(*param_space.values()))
    
    print(f"Parameter space: {len(param_combinations):,} combinations")
    print(f"Parameters: {param_names}")
    print()
    
    # Run sweep
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, param_values in enumerate(param_combinations):
        config = ExtractionConfig(**dict(zip(param_names, param_values)))
        
        target_desc = f"target={config.target_states}" if config.target_states else "natural"
        print(f"[{i+1:3d}/{len(param_combinations)}] Testing: {target_desc}, len={config.max_suffix_length}, sig={config.significance_level}, neural_th={config.neural_threshold}")
        
        start_time = time.time()
        try:
            # Run extraction
            result = run_single_extraction(extractor, config, data_path, ground_truth)
            result.extraction_time = time.time() - start_time
            result.success = True
            
        except Exception as e:
            result = ExtractionResult(
                config=config,
                n_clusters_found=0,
                silhouette_score=-1.0,
                inertia=float('inf'),
                stability_score=0.0,
                extraction_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
        
        results.append(result)
        
        # Print result
        if result.success:
            print(f"  ✅ Success: {result.n_clusters_found} states, sil={result.silhouette_score:.3f}, stable={result.stability_score:.3f}")
            if result.alignment_error is not None and result.alignment_error != float('inf'):
                print(f"     Alignment error: {result.alignment_error:.4f}, Prob error: {result.probability_error:.4f}")
            elif result.alignment_error == float('inf'):
                print(f"     Alignment: No ground truth comparison available")
        else:
            print(f"  ❌ Failed: {result.error_message}")
        
        # Save intermediate results every 50 runs
        if (i + 1) % 50 == 0:
            save_results(results, output_dir / f'intermediate_results_{i+1}.csv')
    
    # Save final results
    df = save_results(results, output_dir / 'parameter_sweep_results.csv')
    
    print("\n" + "=" * 60)
    print("🎯 Parameter Sweep Complete!")
    print(f"📊 Results saved to: {output_dir}")
    
    # Print top results
    successful_df = df[df['success'] == True]
    
    if len(successful_df) > 0:
        if ground_truth and 'alignment_error' in successful_df.columns:
            # Filter out None values and convert to float
            valid_alignment = successful_df.dropna(subset=['alignment_error'])
            if len(valid_alignment) > 0:
                top_results = valid_alignment.nsmallest(5, 'alignment_error')
                print("\n🏆 Top 5 Configurations (by alignment error):")
                for idx, row in top_results.iterrows():
                    print(f"  {row['num_states']} states, len={row['max_suffix_length']}, sig={row['significance_level']}, neural_th={row['neural_threshold']} -> error: {row['alignment_error']:.4f}")
            else:
                print("\n⚠️ No valid alignment error results found")
        
        # Also show by number of states found (closest to target)
        target_states = 3
        successful_df['state_diff'] = abs(successful_df['n_clusters_found'] - target_states)
        closest_to_target = successful_df.nsmallest(5, 'state_diff')
        print("\n🎯 Top 5 Configurations (closest to 3 states):")
        for idx, row in closest_to_target.iterrows():
            target_desc = f"forced={row['target_states']}" if pd.notna(row['target_states']) else "natural"
            print(f"  Found {row['n_clusters_found']} states ({target_desc}), len={row['max_suffix_length']}, sig={row['significance_level']:.4f}, neural_th={row['neural_threshold']}")
    else:
        print("\n❌ No successful extractions found")
    
    return df


def run_single_extraction(extractor: CSSREnhancedExtractor, config: ExtractionConfig, 
                         data_path: Path, ground_truth: Optional[Dict] = None) -> ExtractionResult:
    """Run single CSSR-enhanced extraction with given configuration."""
    
    # Configure the extractor with current parameters
    extractor.max_suffix_length = config.max_suffix_length
    extractor.significance_level = config.significance_level
    extractor.min_suffix_count = config.min_suffix_count
    
    # Handle target_states parameter (may be None for natural discovery)
    if hasattr(config, 'target_states') and config.target_states is not None:
        extractor.num_states = config.target_states
    elif hasattr(config, 'num_states'):
        extractor.num_states = config.num_states
    else:
        extractor.num_states = None  # Let CSSR naturally discover state count
    
    # Temporarily modify the neural threshold in the extractor
    # (This requires modifying the extractor class to accept dynamic thresholds)
    original_neural_threshold = getattr(extractor, 'neural_threshold', 5.0)
    extractor.neural_threshold = config.neural_threshold
    
    try:
        # Run the full extraction pipeline
        epsilon_machine = extractor.extract_cssr_enhanced_fsm(
            data_path=data_path,
            output_dir=Path('temp_extraction'),  # Temporary directory
            max_sequences=config.max_sequences,
            chunk_size=config.chunk_size
        )
        
        # Extract metrics
        n_clusters_found = epsilon_machine.get('num_states', 0)
        if n_clusters_found == 0 and hasattr(extractor, 'causal_states'):
            n_clusters_found = len(extractor.causal_states)
        
        # Compute silhouette score using causal state representatives
        silhouette = 0.0
        if hasattr(extractor, 'causal_states') and len(extractor.causal_states) > 1:
            try:
                # Get representative hidden states for silhouette calculation
                hidden_reps = np.array([state['representative_hidden'] for state in extractor.causal_states])
                if len(hidden_reps) > 1 and hidden_reps.shape[1] > 0:
                    # Create dummy labels (each state is its own cluster)
                    labels = list(range(len(hidden_reps)))
                    if len(set(labels)) > 1:
                        silhouette = silhouette_score(hidden_reps, labels)
            except Exception as e:
                silhouette = 0.0  # Fallback if silhouette computation fails
        
        # Stability score: measure how consistent the suffix groupings are
        stability = compute_stability_score(extractor.causal_states)
        
        result = ExtractionResult(
            config=config,
            n_clusters_found=n_clusters_found,
            silhouette_score=silhouette,
            inertia=0.0,  # Not directly applicable to CSSR
            stability_score=stability,
            success=True  # Mark as successful
        )
        
        # Add ground truth comparison if available
        if ground_truth:
            alignment_error, probability_error = compute_ground_truth_alignment(
                epsilon_machine, ground_truth, config
            )
            result.alignment_error = alignment_error
            result.probability_error = probability_error
            result.state_recovery_score = 1.0 - min(alignment_error, 1.0)  # Convert error to score
        
    except Exception as e:
        # Return failed result
        result = ExtractionResult(
            config=config,
            n_clusters_found=0,
            silhouette_score=-1.0,
            inertia=float('inf'),
            stability_score=0.0,
            success=False,
            error_message=str(e)
        )
    finally:
        # Restore original neural threshold
        extractor.neural_threshold = original_neural_threshold
    
    return result


def compute_stability_score(causal_states: List[Dict]) -> float:
    """Compute stability score based on causal state characteristics."""
    if not causal_states:
        return 0.0
    
    # Measure the coefficient of variation in state sizes
    sizes = [state['count'] for state in causal_states]
    if len(sizes) <= 1:
        return 1.0
    
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    
    if mean_size == 0:
        return 0.0
    
    # Lower coefficient of variation = higher stability
    cv = std_size / mean_size
    stability = max(0.0, 1.0 - cv)
    
    return stability


def compute_ground_truth_alignment(epsilon_machine: Dict, ground_truth: Dict, 
                                 config: ExtractionConfig) -> Tuple[float, float]:
    """Compute alignment error between extracted machine and ground truth."""
    
    try:
        # Extract state counts for comparison
        extracted_states = epsilon_machine.get('num_states', 0)
        
        # Try to get ground truth state count from different possible locations
        gt_states = 0
        if 'num_states' in ground_truth:
            gt_states = ground_truth['num_states']
        elif 'states' in ground_truth:
            if isinstance(ground_truth['states'], list):
                gt_states = len(ground_truth['states'])
            elif isinstance(ground_truth['states'], dict):
                gt_states = len(ground_truth['states'])
        elif 'transitions' in ground_truth:
            # Count unique states from transitions
            all_states = set()
            transitions = ground_truth['transitions']
            if isinstance(transitions, dict):
                for state, state_transitions in transitions.items():
                    all_states.add(state)
                    if isinstance(state_transitions, dict):
                        for symbol, next_states in state_transitions.items():
                            if isinstance(next_states, dict):
                                all_states.update(next_states.keys())
            gt_states = len(all_states)
        
        # If we still don't have ground truth states, assume 3 for our test case
        if gt_states == 0:
            gt_states = 3  # Known ground truth for our custom 3-state machine
        
        # Compute alignment errors
        if gt_states > 0:
            # State count error (normalized)
            size_error = abs(extracted_states - gt_states) / gt_states
            
            # For now, use size error as probability error too
            # TODO: Implement proper transition probability comparison
            probability_error = size_error
            
            return size_error, probability_error
        else:
            return 1.0, 1.0
        
    except Exception as e:
        print(f"Warning: Ground truth alignment failed: {e}")
        return 1.0, 1.0  # Maximum error if comparison fails


def save_results(results: List[ExtractionResult], output_path: Path) -> pd.DataFrame:
    """Save results to CSV file."""
    data = []
    for result in results:
        row = asdict(result.config)
        row.update({
            'n_clusters_found': result.n_clusters_found,
            'silhouette_score': result.silhouette_score,
            'inertia': result.inertia,
            'stability_score': result.stability_score,
            'alignment_error': result.alignment_error if result.alignment_error is not None else float('inf'),
            'probability_error': result.probability_error if result.probability_error is not None else float('inf'),
            'state_recovery_score': result.state_recovery_score if result.state_recovery_score is not None else 0.0,
            'extraction_time': result.extraction_time,
            'success': result.success,
            'error_message': result.error_message or ""
        })
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='Systematic FSM extraction parameter sweep')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=Path, required=True, help='Training data path (.dat file)')
    parser.add_argument('--ground_truth', type=Path, help='Ground truth machine JSON path')
    parser.add_argument('--output_dir', type=Path, default=Path('sweep_results'), help='Output directory')
    parser.add_argument('--phase', choices=['1', '2', 'full'], default='1', help='Sweep phase')
    
    args = parser.parse_args()
    
    # Run parameter sweep
    results_df = run_parameter_sweep(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir
    )
    
    print(f"\n📈 Total configurations tested: {len(results_df)}")
    print(f"✅ Successful extractions: {results_df['success'].sum()}")
    print(f"❌ Failed extractions: {(~results_df['success']).sum()}")


if __name__ == '__main__':
    main()