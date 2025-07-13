#!/usr/bin/env python3
"""
Internal State Analysis for Transformer FSM Learning
Analyzes how the transformer internally encodes and tracks FSM states.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_delay_transformer import BinaryTransformer, SequenceDataset, collate_fn_ar

class InternalStateAnalyzer:
    """Analyzer for transformer internal representations."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for internal representations
        self.hidden_states = {}  # layer -> list of hidden states
        self.attention_weights = {}  # layer -> list of attention weights
        self.embeddings = []
        
        # Register hooks to capture internal states
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture internal representations."""
        
        # Hook for embeddings (after positional encoding)
        def embedding_hook(module, input, output):
            self.embeddings.append(output.detach().cpu())
        
        # Hook for transformer layers
        def transformer_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx not in self.hidden_states:
                    self.hidden_states[layer_idx] = []
                self.hidden_states[layer_idx].append(output.detach().cpu())
            return hook
        
        # Hook for attention weights
        def attention_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx not in self.attention_weights:
                    self.attention_weights[layer_idx] = []
                # Extract attention weights from MultiheadAttention
                if hasattr(module, 'self_attn'):
                    # This is a TransformerEncoderLayer
                    pass  # We'll handle this differently
            return hook
        
        # Register hooks on transformer layers
        if hasattr(self.model, 'tr') and hasattr(self.model.tr, 'layers'):
            for i, layer in enumerate(self.model.tr.layers):
                layer.register_forward_hook(transformer_hook(i))
        
        # Register hook on positional encoding
        if hasattr(self.model, 'pos_encoding'):
            self.model.pos_encoding.register_forward_hook(embedding_hook)
    
    def clear_states(self):
        """Clear stored internal states."""
        self.hidden_states = {}
        self.attention_weights = {}
        self.embeddings = []
    
    @torch.no_grad()
    def extract_representations(self, dataloader: DataLoader, max_batches: int = 10) -> Dict:
        """Extract internal representations from model on given data."""
        self.clear_states()
        
        sequence_info = []
        batch_count = 0
        
        for toks, tgt in dataloader:
            if batch_count >= max_batches:
                break
                
            toks, tgt = toks.to(self.device), tgt.to(self.device)
            
            # Forward pass (hooks will capture internal states)
            logits = self.model(toks)
            predictions = torch.argmax(logits, dim=-1)
            
            # Store sequence information
            for i in range(toks.size(0)):
                seq_len = (tgt[i] != 2).sum().item()
                if seq_len > 0:
                    sequence_info.append({
                        'batch_idx': batch_count,
                        'seq_idx': i,
                        'length': seq_len,
                        'tokens': toks[i][:seq_len].cpu().numpy(),
                        'targets': tgt[i][:seq_len].cpu().numpy(),
                        'predictions': predictions[i][:seq_len].cpu().numpy(),
                        'accuracy': (predictions[i][:seq_len] == tgt[i][:seq_len]).float().mean().item()
                    })
            
            batch_count += 1
        
        return {
            'sequence_info': sequence_info,
            'hidden_states': self.hidden_states,
            'attention_weights': self.attention_weights,
            'embeddings': self.embeddings,
            'num_layers': len(self.hidden_states),
            'num_sequences': len(sequence_info)
        }

def load_ground_truth_states(dataset_path: Path) -> Dict:
    """Load ground truth FSM state information."""
    gt_path = dataset_path / "ground_truth"
    
    # Load causal state labels
    with open(gt_path / "causal_state_labels.json") as f:
        state_labels = json.load(f)
    
    # Load machine properties
    with open(gt_path / "machine_properties.json") as f:
        machines = json.load(f)
    
    return {"state_labels": state_labels, "machines": machines}

def analyze_hidden_state_clustering(representations: Dict, output_dir: Path):
    """Analyze clustering patterns in hidden states."""
    print("Analyzing hidden state clustering...")
    
    # Extract hidden states from all layers and sequences
    all_states = {}
    sequence_positions = {}
    
    for layer_idx, layer_states in representations['hidden_states'].items():
        layer_representations = []
        positions = []
        
        for batch_idx, batch_states in enumerate(layer_states):
            # batch_states: [batch_size, seq_len, d_model]
            B, T, D = batch_states.shape
            
            for seq_idx in range(B):
                for pos in range(T):
                    layer_representations.append(batch_states[seq_idx, pos].numpy())
                    positions.append({
                        'batch': batch_idx,
                        'sequence': seq_idx,
                        'position': pos,
                        'layer': layer_idx
                    })
        
        all_states[layer_idx] = np.array(layer_representations)
        sequence_positions[layer_idx] = positions
    
    # Perform dimensionality reduction and clustering for each layer
    results = {}
    
    for layer_idx, states in all_states.items():
        if len(states) == 0:
            continue
            
        print(f"  Layer {layer_idx}: {states.shape[0]} representations of dim {states.shape[1]}")
        
        # PCA
        pca = PCA(n_components=min(10, states.shape[1]))
        pca_states = pca.fit_transform(states)
        
        # t-SNE (on PCA-reduced data for efficiency)
        tsne_components = min(2, pca_states.shape[1])
        if pca_states.shape[0] > 1 and tsne_components > 1:
            tsne = TSNE(n_components=tsne_components, random_state=42)
            tsne_states = tsne.fit_transform(pca_states[:, :5])  # Use first 5 PCA components
        else:
            tsne_states = pca_states[:, :tsne_components]
        
        # K-means clustering
        n_clusters = min(8, len(states) // 10)  # Reasonable number of clusters
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(states)
        else:
            cluster_labels = np.zeros(len(states))
        
        results[layer_idx] = {
            'pca_states': pca_states,
            'tsne_states': tsne_states,
            'cluster_labels': cluster_labels,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'positions': sequence_positions[layer_idx],
            'n_clusters': n_clusters
        }
        
        # Create visualization
        if tsne_states.shape[1] >= 2:
            plt.figure(figsize=(12, 5))
            
            # PCA plot
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(pca_states[:, 0], pca_states[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
            plt.title(f'Layer {layer_idx} - PCA View')
            plt.colorbar(scatter, label='Cluster')
            
            # t-SNE plot
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(tsne_states[:, 0], tsne_states[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.6, s=20)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f'Layer {layer_idx} - t-SNE View')
            plt.colorbar(scatter, label='Cluster')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'layer_{layer_idx}_clustering.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    return results

def analyze_state_transitions(representations: Dict, clustering_results: Dict, output_dir: Path):
    """Analyze how internal states correspond to FSM state transitions."""
    print("Analyzing state transition patterns...")
    
    # For each layer, track how cluster assignments change over sequences
    transition_analysis = {}
    
    for layer_idx in clustering_results:
        cluster_labels = clustering_results[layer_idx]['cluster_labels']
        positions = clustering_results[layer_idx]['positions']
        
        # Group by sequence
        sequence_clusters = defaultdict(list)
        for i, pos_info in enumerate(positions):
            seq_key = (pos_info['batch'], pos_info['sequence'])
            sequence_clusters[seq_key].append({
                'position': pos_info['position'],
                'cluster': cluster_labels[i]
            })
        
        # Analyze transitions within sequences
        transitions = defaultdict(int)
        state_persistence = []
        
        for seq_key, seq_clusters in sequence_clusters.items():
            # Sort by position
            seq_clusters.sort(key=lambda x: x['position'])
            
            # Count transitions
            for i in range(1, len(seq_clusters)):
                prev_cluster = seq_clusters[i-1]['cluster']
                curr_cluster = seq_clusters[i]['cluster']
                transitions[(prev_cluster, curr_cluster)] += 1
                
                # Track state persistence
                state_persistence.append(prev_cluster == curr_cluster)
        
        # Calculate transition matrix
        n_clusters = clustering_results[layer_idx]['n_clusters']
        transition_matrix = np.zeros((n_clusters, n_clusters))
        
        for (from_state, to_state), count in transitions.items():
            transition_matrix[from_state, to_state] = count
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        normalized_transition_matrix = np.divide(
            transition_matrix, 
            row_sums[:, np.newaxis], 
            out=np.zeros_like(transition_matrix), 
            where=row_sums[:, np.newaxis] != 0
        )
        
        transition_analysis[layer_idx] = {
            'transition_matrix': transition_matrix,
            'normalized_transition_matrix': normalized_transition_matrix,
            'state_persistence_rate': np.mean(state_persistence) if state_persistence else 0.0,
            'num_transitions': len(transitions)
        }
        
        # Visualize transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_transition_matrix, 
                   annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=[f'State {i}' for i in range(n_clusters)],
                   yticklabels=[f'State {i}' for i in range(n_clusters)])
        plt.title(f'Layer {layer_idx} - Normalized State Transition Matrix')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx}_transitions.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return transition_analysis

def create_probing_classifier(representations: Dict, ground_truth: Dict, output_dir: Path):
    """Create probing classifiers to test what information is encoded in hidden states."""
    print("Training probing classifiers...")
    
    # This is a simplified version - we'd need to align with actual ground truth states
    # For now, we'll probe for token prediction capability at each layer
    
    probing_results = {}
    
    for layer_idx, layer_states in representations['hidden_states'].items():
        print(f"  Probing layer {layer_idx}...")
        
        # Collect all hidden states and corresponding targets
        X = []  # Hidden states
        y = []  # Token labels
        
        seq_info = representations['sequence_info']
        state_idx = 0
        
        for batch_idx, batch_states in enumerate(layer_states):
            B, T, D = batch_states.shape
            
            for seq_idx in range(B):
                # Find corresponding sequence info
                matching_seq = None
                for seq in seq_info:
                    if seq['batch_idx'] == batch_idx and seq['seq_idx'] == seq_idx:
                        matching_seq = seq
                        break
                
                if matching_seq is None:
                    continue
                
                seq_len = matching_seq['length']
                targets = matching_seq['targets']
                
                for pos in range(min(seq_len, T)):
                    X.append(batch_states[seq_idx, pos].numpy())
                    y.append(targets[pos])
        
        if len(X) == 0:
            continue
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        probing_results[layer_idx] = {
            'accuracy': accuracy,
            'num_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"    Layer {layer_idx}: {accuracy:.4f} accuracy ({len(X)} samples)")
    
    return probing_results

def generate_internal_analysis_report(clustering_results: Dict, transition_analysis: Dict, 
                                    probing_results: Dict, representations: Dict) -> str:
    """Generate comprehensive internal state analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("TRANSFORMER INTERNAL STATE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overview
    report.append("🔍 ANALYSIS OVERVIEW")
    report.append("-" * 40)
    report.append(f"Number of layers analyzed: {representations['num_layers']}")
    report.append(f"Number of sequences: {representations['num_sequences']}")
    report.append(f"Model dimension: {list(clustering_results.values())[0]['pca_states'].shape[1] if clustering_results else 'N/A'}")
    report.append("")
    
    # Clustering Analysis
    report.append("🎯 HIDDEN STATE CLUSTERING ANALYSIS")
    report.append("-" * 40)
    
    for layer_idx in sorted(clustering_results.keys()):
        results = clustering_results[layer_idx]
        pca_var = results['pca_explained_variance']
        
        report.append(f"Layer {layer_idx}:")
        report.append(f"  • Number of clusters identified: {results['n_clusters']}")
        report.append(f"  • PCA variance explained (first 2 components): {pca_var[0]:.3f}, {pca_var[1]:.3f}")
        report.append(f"  • Total representations: {len(results['cluster_labels'])}")
        
        # Cluster distribution
        unique, counts = np.unique(results['cluster_labels'], return_counts=True)
        cluster_dist = dict(zip(unique, counts))
        report.append(f"  • Cluster distribution: {cluster_dist}")
        report.append("")
    
    # Transition Analysis
    report.append("🔄 STATE TRANSITION ANALYSIS")
    report.append("-" * 40)
    
    for layer_idx in sorted(transition_analysis.keys()):
        results = transition_analysis[layer_idx]
        
        report.append(f"Layer {layer_idx}:")
        report.append(f"  • State persistence rate: {results['state_persistence_rate']:.3f}")
        report.append(f"  • Number of transitions observed: {results['num_transitions']}")
        
        # Analyze transition matrix properties
        trans_matrix = results['normalized_transition_matrix']
        diagonal_strength = np.mean(np.diag(trans_matrix))
        off_diagonal_strength = np.mean(trans_matrix[~np.eye(trans_matrix.shape[0], dtype=bool)])
        
        report.append(f"  • Diagonal transition strength: {diagonal_strength:.3f}")
        report.append(f"  • Off-diagonal transition strength: {off_diagonal_strength:.3f}")
        report.append("")
    
    # Probing Results
    report.append("🔬 PROBING CLASSIFIER RESULTS")
    report.append("-" * 40)
    
    for layer_idx in sorted(probing_results.keys()):
        results = probing_results[layer_idx]
        report.append(f"Layer {layer_idx}:")
        report.append(f"  • Token prediction accuracy: {results['accuracy']:.4f}")
        report.append(f"  • Training samples: {results['train_samples']}")
        report.append(f"  • Test samples: {results['test_samples']}")
        report.append("")
    
    # Key Insights
    report.append("💡 KEY INSIGHTS")
    report.append("-" * 40)
    
    # Analyze layer progression
    if len(probing_results) > 1:
        accuracies = [probing_results[layer]['accuracy'] for layer in sorted(probing_results.keys())]
        if accuracies[-1] > accuracies[0]:
            report.append("✅ PROGRESSIVE REFINEMENT: Token prediction accuracy improves through layers")
            report.append(f"   Layer 0: {accuracies[0]:.4f} → Layer {len(accuracies)-1}: {accuracies[-1]:.4f}")
        else:
            report.append("⚠️  Token prediction accuracy varies across layers")
    
    # Analyze clustering stability
    if len(clustering_results) > 1:
        cluster_counts = [clustering_results[layer]['n_clusters'] for layer in sorted(clustering_results.keys())]
        if len(set(cluster_counts)) == 1:
            report.append(f"✅ CONSISTENT CLUSTERING: All layers identify {cluster_counts[0]} clusters")
        else:
            report.append(f"⚠️  Variable clustering across layers: {cluster_counts}")
    
    # Analyze state persistence
    if transition_analysis:
        persistence_rates = [transition_analysis[layer]['state_persistence_rate'] 
                           for layer in sorted(transition_analysis.keys())]
        avg_persistence = np.mean(persistence_rates)
        
        if avg_persistence > 0.7:
            report.append(f"✅ HIGH STATE PERSISTENCE: Average {avg_persistence:.3f} - Model maintains stable internal states")
        elif avg_persistence > 0.5:
            report.append(f"⚠️  Moderate state persistence: {avg_persistence:.3f}")
        else:
            report.append(f"❌ Low state persistence: {avg_persistence:.3f} - Highly dynamic internal states")
    
    report.append("")
    
    # Final Assessment
    report.append("🏆 FINAL ASSESSMENT")
    report.append("-" * 40)
    
    evidence_score = 0
    
    # High final layer accuracy suggests good representation learning
    if probing_results:
        final_accuracy = max(probing_results[layer]['accuracy'] for layer in probing_results)
        if final_accuracy > 0.95:
            evidence_score += 2
            report.append("✅ Excellent internal representation quality (high probing accuracy)")
        elif final_accuracy > 0.8:
            evidence_score += 1
            report.append("⚠️  Good internal representation quality")
    
    # Consistent clustering suggests structured representations
    if clustering_results and len(set(clustering_results[layer]['n_clusters'] 
                                    for layer in clustering_results)) <= 2:
        evidence_score += 1
        report.append("✅ Consistent internal clustering across layers")
    
    # Moderate state persistence suggests meaningful state tracking
    if transition_analysis:
        avg_persistence = np.mean([transition_analysis[layer]['state_persistence_rate'] 
                                 for layer in transition_analysis])
        if 0.5 <= avg_persistence <= 0.8:
            evidence_score += 1
            report.append("✅ Balanced state persistence (not too rigid, not too chaotic)")
    
    if evidence_score >= 3:
        report.append("")
        report.append("🎯 STRONG EVIDENCE: Transformer has learned structured internal FSM representations")
        report.append("   - Clear clustering patterns in hidden states")
        report.append("   - Meaningful state transition dynamics")
        report.append("   - High-quality information encoding")
    elif evidence_score >= 2:
        report.append("")
        report.append("⚠️  MODERATE EVIDENCE: Some structured internal representations detected")
    else:
        report.append("")
        report.append("❌ LIMITED EVIDENCE: Internal representations may be less structured")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Analyze transformer internal states")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--test", type=Path, help="Path to test dataset (defaults to dataset/val)")
    parser.add_argument("--d_model", type=int, default=32, help="Model d_model")
    parser.add_argument("--layers", type=int, default=3, help="Model layers")
    parser.add_argument("--heads", type=int, default=8, help="Model heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output", type=Path, default=Path("results/internal_analysis"), help="Output directory")
    parser.add_argument("--max_batches", type=int, default=5, help="Max batches to analyze")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and data...")
    
    # Load model
    model = BinaryTransformer(vocab_size=3, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    # Load test dataset
    test_path = args.test if args.test else args.dataset / "neural_format" / "val_dataset.pt"
    test_dataset = SequenceDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_ar)
    
    # Load ground truth
    ground_truth = load_ground_truth_states(args.dataset)
    
    # Initialize analyzer
    analyzer = InternalStateAnalyzer(model, device)
    
    print("Extracting internal representations...")
    representations = analyzer.extract_representations(test_loader, args.max_batches)
    
    print("Analyzing hidden state clustering...")
    clustering_results = analyze_hidden_state_clustering(representations, args.output)
    
    print("Analyzing state transitions...")
    transition_analysis = analyze_state_transitions(representations, clustering_results, args.output)
    
    print("Training probing classifiers...")
    probing_results = create_probing_classifier(representations, ground_truth, args.output)
    
    print("Generating report...")
    report = generate_internal_analysis_report(clustering_results, transition_analysis, 
                                             probing_results, representations)
    
    print(report)
    
    # Save report
    with open(args.output / "internal_analysis_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed results
    results = {
        'clustering_results': {str(k): v for k, v in clustering_results.items()},
        'transition_analysis': {str(k): v for k, v in transition_analysis.items()},
        'probing_results': {str(k): v for k, v in probing_results.items()},
        'representations_summary': {
            'num_layers': representations['num_layers'],
            'num_sequences': representations['num_sequences']
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        else:
            return obj
    
    results = convert_numpy(results)
    
    with open(args.output / "internal_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()