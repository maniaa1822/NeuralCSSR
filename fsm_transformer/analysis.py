import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

from .transformer import AutoregressiveTransformer
from .epsilon_machine import EpsilonMachine
from .data_generator import load_dataset


class CausalStateAnalyzer:
    """Analyze learned causal states in transformer representations."""
    
    def __init__(self, model: AutoregressiveTransformer, epsilon_machine: EpsilonMachine):
        self.model = model
        self.epsilon_machine = epsilon_machine
        self.model.eval()
        
    def extract_context_representations(
        self,
        sequences: List[List[str]],
        state_trajectories: List[List[str]],
        layer: int = -1,
        max_contexts: int = 10000
    ) -> Tuple[np.ndarray, List[str], List[List[str]]]:
        """
        Extract transformer representations for sequence contexts.
        
        Returns:
            (representations, true_states, contexts)
        """
        representations = []
        true_states = []
        contexts = []
        
        count = 0
        for sequence, states in zip(sequences, state_trajectories):
            if count >= max_contexts:
                break
                
            for i in range(1, len(sequence)):  # Skip empty context
                if count >= max_contexts:
                    break
                    
                context = sequence[:i]
                true_state = states[i-1]  # State before generating next token
                
                # Convert to tensor
                input_ids = torch.tensor([[self.model.token_embedding.weight.data.shape[0] if token == '<PAD>' else int(token) for token in context]], dtype=torch.long)
                
                # Pad to reasonable length
                max_len = 128
                if input_ids.shape[1] > max_len:
                    input_ids = input_ids[:, -max_len:]
                
                attention_mask = torch.ones_like(input_ids)
                
                # Extract representation
                with torch.no_grad():
                    repr_tensor = self.model.extract_representations(input_ids, attention_mask, layer)
                    # Use last position representation
                    repr_vec = repr_tensor[0, -1, :].cpu().numpy()
                
                representations.append(repr_vec)
                true_states.append(true_state)
                contexts.append(context)
                count += 1
        
        return np.array(representations), true_states, contexts
    
    def cluster_representations(
        self,
        representations: np.ndarray,
        n_clusters: int = 3,
        method: str = 'kmeans'
    ) -> Tuple[np.ndarray, float]:
        """
        Cluster transformer representations to discover learned states.
        
        Returns:
            (cluster_labels, silhouette_score)
        """
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(representations)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        sil_score = silhouette_score(representations, labels)
        return labels, sil_score
    
    def evaluate_state_recovery(
        self,
        true_states: List[str],
        predicted_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate how well clustering recovers true causal states."""
        
        # Convert state names to indices
        state_to_idx = {state: i for i, state in enumerate(self.epsilon_machine.states)}
        true_labels = np.array([state_to_idx[state] for state in true_states])
        
        # Compute metrics
        ari = adjusted_rand_score(true_labels, predicted_labels)
        
        # Compute purity (best alignment between clusters and true states)
        n_samples = len(true_labels)
        n_clusters = len(np.unique(predicted_labels))
        n_states = len(self.epsilon_machine.states)
        
        # Create confusion matrix
        confusion = np.zeros((n_clusters, n_states))
        for pred, true in zip(predicted_labels, true_labels):
            confusion[pred, true] += 1
        
        # Compute purity
        purity = np.sum(np.max(confusion, axis=1)) / n_samples
        
        return {
            'adjusted_rand_score': ari,
            'purity': purity,
            'n_clusters_found': n_clusters,
            'n_true_states': n_states,
            'confusion_matrix': confusion.tolist()
        }
    
    def visualize_representations(
        self,
        representations: np.ndarray,
        true_states: List[str],
        predicted_labels: Optional[np.ndarray] = None,
        method: str = 'umap',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize transformer representations in 2D."""
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
        
        coords_2d = reducer.fit_transform(representations)
        
        # Create plot
        fig, axes = plt.subplots(1, 2 if predicted_labels is not None else 1, figsize=(15, 6))
        if predicted_labels is None:
            axes = [axes]
        
        # Plot true states
        ax = axes[0]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, state in enumerate(self.epsilon_machine.states):
            mask = np.array(true_states) == state
            if np.any(mask):
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c=colors[i], label=f'True State {state}', alpha=0.6)
        ax.set_title(f'True Causal States ({method.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot predicted clusters
        if predicted_labels is not None:
            ax = axes[1]
            for cluster in np.unique(predicted_labels):
                mask = predicted_labels == cluster
                ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                          c=colors[cluster % len(colors)], label=f'Cluster {cluster}', alpha=0.6)
            ax.set_title(f'Predicted Clusters ({method.upper()})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_attention_patterns(
        self,
        sequences: List[List[str]],
        contexts_of_interest: List[List[str]],
        layer: int = -1
    ) -> Dict[str, np.ndarray]:
        """Analyze attention patterns for specific contexts."""
        
        attention_patterns = {}
        
        for context in contexts_of_interest:
            if len(context) < 2:
                continue
                
            # Convert to tensor
            input_ids = torch.tensor([[int(token) for token in context]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            # Get attention weights
            with torch.no_grad():
                outputs = self.model.forward(input_ids, attention_mask, return_hidden_states=True)
                
                if layer == -1:
                    layer_idx = -1
                else:
                    layer_idx = layer
                    
                attn_weights = outputs['attention_weights'][layer_idx]  # [batch, heads, seq, seq]
                
                # Average over heads and take last position (what it attends to)
                avg_attention = attn_weights[0].mean(dim=0)[-1, :].cpu().numpy()
                
                context_str = ''.join(context)
                attention_patterns[context_str] = avg_attention
        
        return attention_patterns
    
    def compute_predictive_accuracy(
        self,
        test_sequences: List[List[str]],
        test_states: List[List[str]]
    ) -> Dict[str, float]:
        """Compute predictive accuracy on test sequences."""
        
        correct_predictions = 0
        total_predictions = 0
        state_accuracies = {state: {'correct': 0, 'total': 0} for state in self.epsilon_machine.states}
        
        for sequence, states in zip(test_sequences, test_states):
            for i in range(len(sequence) - 1):
                context = sequence[:i+1]
                true_next = sequence[i+1]
                true_state = states[i]
                
                # Get model prediction
                input_ids = torch.tensor([[int(token) for token in context]], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                
                with torch.no_grad():
                    probs = self.model.get_next_token_probabilities(input_ids, attention_mask)
                    predicted = torch.argmax(probs, dim=-1).item()
                
                # Check accuracy
                is_correct = (predicted == int(true_next))
                correct_predictions += is_correct
                total_predictions += 1
                
                # State-specific accuracy
                state_accuracies[true_state]['correct'] += is_correct
                state_accuracies[true_state]['total'] += 1
        
        # Compute overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Compute per-state accuracy
        per_state_accuracy = {}
        for state, counts in state_accuracies.items():
            if counts['total'] > 0:
                per_state_accuracy[state] = counts['correct'] / counts['total']
            else:
                per_state_accuracy[state] = 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'per_state_accuracy': per_state_accuracy,
            'total_predictions': total_predictions
        }


def run_full_analysis(
    model_path: str,
    data_path: str,
    output_dir: str = "analysis_results",
    layer: int = -1,
    max_contexts: int = 5000
) -> Dict:
    """Run complete causal state analysis."""
    
    # Load model and data
    model = torch.load(model_path, map_location='cpu')
    dataset, metadata = load_dataset(data_path)
    
    # Load raw sequences
    with open(Path(data_path) / 'raw_data.json', 'r') as f:
        raw_data = json.load(f)
    
    sequences = raw_data['sequences']
    state_trajectories = raw_data['state_trajectories']
    
    # Initialize analyzer
    epsilon_machine = EpsilonMachine()
    analyzer = CausalStateAnalyzer(model, epsilon_machine)
    
    # Extract representations
    print("Extracting representations...")
    representations, true_states, contexts = analyzer.extract_context_representations(
        sequences, state_trajectories, layer, max_contexts
    )
    
    # Cluster representations
    print("Clustering representations...")
    predicted_labels, sil_score = analyzer.cluster_representations(representations, n_clusters=3)
    
    # Evaluate state recovery
    print("Evaluating state recovery...")
    recovery_metrics = analyzer.evaluate_state_recovery(true_states, predicted_labels)
    
    # Visualize
    print("Creating visualizations...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig = analyzer.visualize_representations(
        representations, true_states, predicted_labels, 
        method='pca', save_path=str(output_path / 'state_visualization.png')
    )
    plt.close(fig)
    
    # Compute predictive accuracy
    print("Computing predictive accuracy...")
    test_sequences = sequences[:1000]  # Use first 1000 for testing
    test_states = state_trajectories[:1000]
    accuracy_metrics = analyzer.compute_predictive_accuracy(test_sequences, test_states)
    
    # Compile results
    results = {
        'clustering': {
            'silhouette_score': sil_score,
            'n_representations': len(representations)
        },
        'state_recovery': recovery_metrics,
        'predictive_accuracy': accuracy_metrics,
        'analysis_params': {
            'layer': layer,
            'max_contexts': max_contexts,
            'model_path': model_path,
            'data_path': data_path
        }
    }
    
    # Save results
    with open(output_path / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_path}")
    return results