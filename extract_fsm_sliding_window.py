#!/usr/bin/env python3
"""
Trajectory Dynamics FSM Extraction for Sliding Window Models

This script extracts finite state machines from sliding window transformers by analyzing
transition patterns in hidden state representations. Based on the principle that causal
states should exhibit similar future transition behaviors.

Key advantages for sliding window models:
- Context independence: Works with consistent 25-token windows
- Rich data: 25x more transition examples from overlapping windows  
- Unsupervised: Only needs number of states (K=7 for seven-state machine)
- Theoretically grounded: Based on causal state definition

Usage:
    python extract_fsm_sliding_window.py --checkpoint checkpoints/sliding_window_seven_state/best.pt
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from models.transformer_models import create_model
from models.data_utils import SequenceDataset


class SlidingWindowFSMExtractor:
    """Extract FSM from sliding window transformer using trajectory dynamics."""
    
    def __init__(self, model: nn.Module, device: torch.device, num_states: int = 7):
        self.model = model
        self.device = device
        self.num_states = num_states
        self.model.eval()
        
        # Storage for extracted data
        self.hidden_states = []
        self.tokens = []
        self.transition_vectors = []
        self.positions = []
        
        # Results
        self.clusters = None
        self.cluster_labels = None
        self.fsm_structure = None
        
    def extract_hidden_states(self, data_path: Path, max_sequences: int = 1000, 
                            chunk_size: int = 25) -> None:
        """Extract hidden states from sliding window model."""
        print(f"🔍 Extracting hidden states from {data_path}")
        print(f"Using sliding windows of size {chunk_size}")
        
        # Load data with sliding windows (same as training)
        dataset = SequenceDataset(data_path, chunk_size=chunk_size, sliding_window=True)
        
        # Limit sequences for computational efficiency
        num_sequences = min(len(dataset), max_sequences)
        print(f"Processing {num_sequences} sequences (of {len(dataset)} available)")
        
        hidden_states_batch = []
        tokens_batch = []
        
        with torch.no_grad():
            for i in range(num_sequences):
                if i % 100 == 0:
                    print(f"  Processing sequence {i}/{num_sequences}")
                
                sequence = dataset[i]  # [chunk_size]
                
                # Run through model to get hidden states
                input_ids = sequence.unsqueeze(0).to(self.device)  # [1, chunk_size]
                
                # Get hidden states (we want the final layer representations)
                # For sliding window transformer, we need to hook into the model
                hidden_states = self._get_hidden_states(input_ids)  # [1, chunk_size, d_model]
                
                # Store data
                hidden_states_batch.append(hidden_states.squeeze(0).cpu())  # [chunk_size, d_model]
                tokens_batch.append(sequence.cpu())
                
        self.hidden_states = hidden_states_batch
        self.tokens = tokens_batch
        print(f"✅ Extracted {len(self.hidden_states)} sequences")
    
    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from the final transformer layer."""
        B, T = input_ids.shape
        
        # Get embeddings + positional encoding
        x = self.model.embedding(input_ids) * (self.model.d_model ** 0.5)
        x = self.model.pos_encoder(x)
        
        # Pass through transformer layers
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        
        # Final transformer output contains the hidden states we want
        hidden_states = self.model.transformer(x, x, tgt_mask=mask)  # [B, T, d_model]
        
        return hidden_states
    
    def compute_transition_vectors(self) -> None:
        """Compute transition vectors: Δh_t = h_{t+1} - h_t for each token transition."""
        print("🔄 Computing transition vectors...")
        
        transition_data = []
        
        for seq_idx, (hidden_seq, token_seq) in enumerate(zip(self.hidden_states, self.tokens)):
            # hidden_seq: [chunk_size, d_model]
            # token_seq: [chunk_size]
            
            chunk_size, d_model = hidden_seq.shape
            
            # Compute transitions for each position except the last
            for t in range(chunk_size - 1):
                h_current = hidden_seq[t].numpy()  # [d_model]
                h_next = hidden_seq[t + 1].numpy()  # [d_model]
                
                # Transition vector
                delta_h = h_next - h_current  # [d_model]
                
                # Next token (what we're transitioning to)
                next_token = token_seq[t + 1].item()
                
                # Store transition data
                transition_data.append({
                    'delta_h': delta_h,
                    'next_token': next_token,
                    'position': t,
                    'sequence': seq_idx,
                    'current_state': h_current,
                    'next_state': h_next
                })
        
        self.transition_data = transition_data
        print(f"✅ Computed {len(transition_data)} transition vectors")
    
    def create_token_specific_features(self) -> np.ndarray:
        """Create features based on token-specific transition patterns."""
        print("🎯 Creating token-specific transition features...")
        
        # Group transitions by position and compute statistics
        position_transitions = defaultdict(lambda: {'0': [], '1': []})
        
        for trans in self.transition_data:
            pos = trans['position']
            token = str(trans['next_token'])
            if token in ['0', '1']:  # Only binary tokens
                position_transitions[pos][token].append(trans['delta_h'])
        
        # For each position, create feature vector [mean_delta_0, mean_delta_1]
        features = []
        positions = []
        
        for pos in sorted(position_transitions.keys()):
            if len(position_transitions[pos]['0']) > 0 and len(position_transitions[pos]['1']) > 0:
                # Mean transition vectors for each token
                mean_delta_0 = np.mean(position_transitions[pos]['0'], axis=0)
                mean_delta_1 = np.mean(position_transitions[pos]['1'], axis=0)
                
                # Concatenate to create feature vector
                feature_vector = np.concatenate([mean_delta_0, mean_delta_1])
                features.append(feature_vector)
                positions.append(pos)
        
        features = np.array(features)  # [n_positions, 2*d_model]
        self.feature_positions = positions
        
        print(f"✅ Created features for {len(features)} positions")
        print(f"Feature dimensionality: {features.shape[1]} (2 × {features.shape[1]//2} d_model)")
        
        return features
    
    def cluster_transition_patterns(self, features: np.ndarray) -> None:
        """Perform K-means clustering on transition features."""
        print(f"🎯 Clustering transition patterns into {self.num_states} states...")
        
        # Try multiple random seeds for stability
        best_score = -1
        best_kmeans = None
        
        for seed in range(10):
            kmeans = KMeans(n_clusters=self.num_states, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            
            if score > best_score:
                best_score = score
                best_kmeans = kmeans
        
        self.cluster_labels = best_kmeans.labels_
        self.cluster_centers = best_kmeans.cluster_centers_
        self.silhouette_score = best_score
        
        print(f"✅ Clustering complete!")
        print(f"   Silhouette score: {best_score:.3f}")
        print(f"   Cluster distribution: {np.bincount(self.cluster_labels)}")
    
    def build_fsm_structure(self) -> Dict:
        """Build FSM transition matrix from clustered states."""
        print("🏗️  Building FSM structure...")
        
        # Initialize transition matrix
        # transitions[state_i][state_j][token] = count
        transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Map positions to cluster labels
        pos_to_cluster = {pos: label for pos, label in zip(self.feature_positions, self.cluster_labels)}
        
        # Count transitions
        for trans in self.transition_data:
            pos = trans['position']
            next_pos = pos + 1
            next_token = str(trans['next_token'])
            
            if pos in pos_to_cluster and next_pos in pos_to_cluster and next_token in ['0', '1']:
                current_state = pos_to_cluster[pos]
                next_state = pos_to_cluster[next_pos]
                transitions[current_state][next_state][next_token] += 1
        
        # Convert to probability matrix
        fsm_structure = {
            'num_states': self.num_states,
            'alphabet': ['0', '1'],
            'transitions': {},
            'state_names': [f'S{i}' for i in range(self.num_states)]
        }
        
        for state_i in range(self.num_states):
            fsm_structure['transitions'][f'S{state_i}'] = {}
            
            for token in ['0', '1']:
                # Calculate transition probabilities for this token
                total_count = sum(transitions[state_i][state_j][token] 
                                for state_j in range(self.num_states))
                
                if total_count > 0:
                    probs = {}
                    for state_j in range(self.num_states):
                        count = transitions[state_i][state_j][token]
                        probs[f'S{state_j}'] = count / total_count
                    fsm_structure['transitions'][f'S{state_i}'][token] = probs
                else:
                    # Uniform distribution if no data
                    uniform_prob = 1.0 / self.num_states
                    fsm_structure['transitions'][f'S{state_i}'][token] = {
                        f'S{j}': uniform_prob for j in range(self.num_states)
                    }
        
        self.fsm_structure = fsm_structure
        print("✅ FSM structure built!")
        return fsm_structure
    
    def visualize_results(self, output_dir: Path) -> None:
        """Create visualizations of the extraction results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 Creating visualizations in {output_dir}")
        
        # 1. Cluster visualization with t-SNE
        if hasattr(self, 'cluster_labels') and len(self.transition_data) > 0:
            # Get features for visualization  
            features = self.create_token_specific_features()
            
            # t-SNE embedding
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            features_2d = tsne.fit_transform(features)
            
            plt.figure(figsize=(12, 5))
            
            # Plot clusters
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=self.cluster_labels, cmap='tab10', alpha=0.7)
            plt.title(f'Transition Pattern Clusters (K={self.num_states})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.colorbar(scatter, label='Cluster')
            
            # Plot PCA
            plt.subplot(1, 2, 2)
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                                c=self.cluster_labels, cmap='tab10', alpha=0.7)
            plt.title('PCA Projection')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            plt.colorbar(scatter, label='Cluster')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Transition matrix heatmap
        if self.fsm_structure:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for i, token in enumerate(['0', '1']):
                # Build transition matrix for this token
                matrix = np.zeros((self.num_states, self.num_states))
                for state_i in range(self.num_states):
                    state_name = f'S{state_i}'
                    if state_name in self.fsm_structure['transitions']:
                        if token in self.fsm_structure['transitions'][state_name]:
                            for state_j in range(self.num_states):
                                target_name = f'S{state_j}'
                                if target_name in self.fsm_structure['transitions'][state_name][token]:
                                    matrix[state_i, state_j] = self.fsm_structure['transitions'][state_name][token][target_name]
                
                # Plot heatmap
                sns.heatmap(matrix, annot=True, fmt='.3f', cmap='Blues', 
                           xticklabels=[f'S{j}' for j in range(self.num_states)],
                           yticklabels=[f'S{i}' for i in range(self.num_states)],
                           ax=axes[i])
                axes[i].set_title(f'Transition Matrix for Token "{token}"')
                axes[i].set_xlabel('Next State')
                axes[i].set_ylabel('Current State')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'transition_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations saved!")
    
    def save_results(self, output_dir: Path) -> None:
        """Save extraction results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'extraction_method': 'sliding_window_trajectory_dynamics',
            'model_info': {
                'num_states_extracted': self.num_states,
                'num_transition_vectors': len(self.transition_data) if hasattr(self, 'transition_data') else 0,
                'num_sequences_processed': len(self.hidden_states)
            },
            'clustering_metrics': {
                'silhouette_score': float(self.silhouette_score) if hasattr(self, 'silhouette_score') else None,
                'cluster_distribution': self.cluster_labels.tolist() if hasattr(self, 'cluster_labels') else None
            },
            'fsm_structure': self.fsm_structure
        }
        
        output_file = output_dir / 'fsm_extraction_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")
    
    def extract_fsm(self, data_path: Path, output_dir: Path, 
                   max_sequences: int = 1000, chunk_size: int = 25) -> Dict:
        """Complete FSM extraction pipeline."""
        print("🚀 Starting FSM extraction from sliding window model")
        print("=" * 70)
        
        # Step 1: Extract hidden states
        self.extract_hidden_states(data_path, max_sequences, chunk_size)
        
        # Step 2: Compute transition vectors  
        self.compute_transition_vectors()
        
        # Step 3: Create features and cluster
        features = self.create_token_specific_features()
        self.cluster_transition_patterns(features)
        
        # Step 4: Build FSM structure
        fsm_structure = self.build_fsm_structure()
        
        # Step 5: Save results and visualizations
        self.save_results(output_dir)
        self.visualize_results(output_dir)
        
        print("=" * 70)
        print("🎉 FSM extraction complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🎯 Extracted {self.num_states}-state FSM")
        print(f"📊 Silhouette score: {self.silhouette_score:.3f}")
        
        return fsm_structure


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load sliding window model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    model_config = checkpoint['model_config'].copy()
    
    # Map class names to factory names
    model_type_mapping = {
        'StreamlinedTransformer': 'streamlined',
        'TimeDelayTransformer': 'time_delay',
        'SlidingWindowTransformer': 'sliding_window'
    }
    
    if model_config['model_type'] in model_type_mapping:
        model_config['model_type'] = model_type_mapping[model_config['model_type']]
    
    # Create and load model
    model = create_model(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ Loaded {model_config['model_type']} model from {checkpoint_path}")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Extract FSM from sliding window transformer")
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=Path, 
                       default=Path('domain_machines/seven_state_human/seven_state_human/seven_state_human.dat'),
                       help='Path to sequence data file')
    parser.add_argument('--output', type=Path, 
                       default=Path('results/fsm_extraction_sliding_window'),
                       help='Output directory for results')
    parser.add_argument('--num-states', type=int, default=7,
                       help='Number of FSM states to extract')
    parser.add_argument('--max-sequences', type=int, default=1000,
                       help='Maximum sequences to process')
    parser.add_argument('--chunk-size', type=int, default=25,
                       help='Chunk size (should match training)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create extractor
    extractor = SlidingWindowFSMExtractor(model, device, args.num_states)
    
    # Extract FSM
    fsm_structure = extractor.extract_fsm(
        args.data, 
        args.output,
        args.max_sequences,
        args.chunk_size
    )
    
    return fsm_structure


if __name__ == '__main__':
    main()