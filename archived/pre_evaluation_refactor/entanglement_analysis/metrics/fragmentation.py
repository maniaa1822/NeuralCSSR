"""
Fragmentation analysis for ε-state representations.

Analyzes how ε-states fragment across the latent space.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict


class FragmentationAnalyzer:
    """
    Analyzes fragmentation patterns in latent representations of ε-states.
    """

    @staticmethod
    def analyze_state_fragmentation(activations: np.ndarray,
                                  epsilon_states: np.ndarray,
                                  n_clusters_range: List[int] = [3, 5, 7, 10, 15]) -> Dict[str, Any]:
        """
        Comprehensive fragmentation analysis.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state labels
            n_clusters_range: Range of cluster numbers to try

        Returns:
            Fragmentation analysis results
        """
        n_states = len(np.unique(epsilon_states))
        results = {}

        # Try different numbers of clusters
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(activations)

            # Analyze fragmentation
            fragmentation = FragmentationAnalyzer._compute_fragmentation_metrics(
                epsilon_states, cluster_labels, n_states
            )

            results[f'k_{n_clusters}'] = {
                'fragmentation_score': fragmentation['median_clusters_per_state'] - 1.0,
                'ari': fragmentation['ari_clusters_vs_states'],
                'nmi': fragmentation['nmi_clusters_vs_states'],
                'clusters_per_state': fragmentation['clusters_per_state'],
                'states_per_cluster': fragmentation['states_per_cluster']
            }

        # Find optimal clustering
        best_k = max(results.keys(),
                    key=lambda k: results[k]['ari'] + results[k]['nmi'])

        results['optimal_clustering'] = best_k
        results['best_fragmentation_score'] = results[best_k]['fragmentation_score']

        return results

    @staticmethod
    def _compute_fragmentation_metrics(epsilon_states: np.ndarray,
                                     cluster_labels: np.ndarray,
                                     n_states: int) -> Dict[str, Any]:
        """Compute detailed fragmentation metrics."""
        state_to_clusters = defaultdict(set)
        cluster_to_states = defaultdict(set)

        for state, cluster in zip(epsilon_states, cluster_labels):
            state_to_clusters[state].add(cluster)
            cluster_to_states[cluster].add(state)

        clusters_per_state = [len(clusters) for clusters in state_to_clusters.values()]
        states_per_cluster = [len(states) for states in cluster_to_states.values()]

        # Handle missing states (if some states have no examples)
        while len(clusters_per_state) < n_states:
            clusters_per_state.append(0)

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(epsilon_states, cluster_labels)
        nmi = normalized_mutual_info_score(epsilon_states, cluster_labels)

        return {
            'clusters_per_state': clusters_per_state,
            'states_per_cluster': states_per_cluster,
            'median_clusters_per_state': np.median(clusters_per_state),
            'mean_clusters_per_state': np.mean(clusters_per_state),
            'ari_clusters_vs_states': ari,
            'nmi_clusters_vs_states': nmi
        }

    @staticmethod
    def compute_fragmentation_histogram(activations: np.ndarray,
                                      epsilon_states: np.ndarray,
                                      n_clusters: int = 7) -> Dict[str, Any]:
        """
        Compute fragmentation histogram data for plotting.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state labels
            n_clusters: Number of clusters

        Returns:
            Histogram data for plotting
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(activations)

        state_to_clusters = defaultdict(set)
        for state, cluster in zip(epsilon_states, cluster_labels):
            state_to_clusters[state].add(cluster)

        # Create histogram data
        clusters_per_state = []
        state_labels = []

        unique_states = sorted(np.unique(epsilon_states))
        for state in unique_states:
            if state in state_to_clusters:
                n_clusters_state = len(state_to_clusters[state])
            else:
                n_clusters_state = 0

            clusters_per_state.append(n_clusters_state)
            state_labels.append(f'State {state}')

        return {
            'states': state_labels,
            'clusters_per_state': clusters_per_state,
            'median_fragmentation': np.median(clusters_per_state),
            'mean_fragmentation': np.mean(clusters_per_state)
        }

    @staticmethod
    def analyze_cluster_purity(activations: np.ndarray,
                             epsilon_states: np.ndarray,
                             n_clusters: int = 7) -> Dict[str, Any]:
        """
        Analyze cluster purity and state concentration.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state labels
            n_clusters: Number of clusters

        Returns:
            Purity analysis
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(activations)

        # Compute purity for each cluster
        cluster_purities = []
        cluster_sizes = []
        dominant_states = []

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_states = epsilon_states[cluster_mask]

            if len(cluster_states) == 0:
                continue

            # Count state frequencies in this cluster
            unique_states, counts = np.unique(cluster_states, return_counts=True)
            dominant_state = unique_states[np.argmax(counts)]
            purity = counts[np.argmax(counts)] / len(cluster_states)

            cluster_purities.append(purity)
            cluster_sizes.append(len(cluster_states))
            dominant_states.append(dominant_state)

        return {
            'mean_cluster_purity': np.mean(cluster_purities),
            'median_cluster_purity': np.median(cluster_purities),
            'cluster_purities': cluster_purities,
            'cluster_sizes': cluster_sizes,
            'dominant_states': dominant_states,
            'purity_weighted_by_size': np.average(cluster_purities, weights=cluster_sizes)
        }

    @staticmethod
    def detect_polysemantic_clusters(activations: np.ndarray,
                                   epsilon_states: np.ndarray,
                                   n_clusters: int = 7,
                                   purity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect clusters that contain multiple ε-states (polysemantic).

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state labels
            n_clusters: Number of clusters
            purity_threshold: Purity threshold below which cluster is polysemantic

        Returns:
            Polysemantic cluster analysis
        """
        purity_analysis = FragmentationAnalyzer.analyze_cluster_purity(
            activations, epsilon_states, n_clusters
        )

        polysemantic_clusters = []
        clean_clusters = []

        for i, purity in enumerate(purity_analysis['cluster_purities']):
            if purity < purity_threshold:
                polysemantic_clusters.append({
                    'cluster_id': i,
                    'purity': purity,
                    'size': purity_analysis['cluster_sizes'][i],
                    'dominant_state': purity_analysis['dominant_states'][i]
                })
            else:
                clean_clusters.append({
                    'cluster_id': i,
                    'purity': purity,
                    'size': purity_analysis['cluster_sizes'][i],
                    'dominant_state': purity_analysis['dominant_states'][i]
                })

        return {
            'polysemantic_clusters': polysemantic_clusters,
            'clean_clusters': clean_clusters,
            'fraction_polysemantic': len(polysemantic_clusters) / n_clusters,
            'total_polysemantic_mass': sum(c['size'] for c in polysemantic_clusters),
            'total_clean_mass': sum(c['size'] for c in clean_clusters)
        }
