"""
Core entanglement metrics for ε-state analysis.

Computes linearity, synergy, fragmentation, and other entanglement measures.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict


class EntanglementMetrics:
    """
    Computes various entanglement metrics for latent representations.
    """

    @staticmethod
    def compute_epsilon_state_linearity(activations: np.ndarray,
                                       epsilon_states: np.ndarray,
                                       train_size: float = 0.7) -> Dict[str, float]:
        """
        Assess how linearly separable ε-states are.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state labels
            train_size: Fraction for training

        Returns:
            Linearity metrics
        """
        n_classes = len(np.unique(epsilon_states))

        # Train linear probe
        probe = LogisticRegression(random_state=42, max_iter=1000,
                                 multi_class='multinomial', solver='lbfgs')
        probe.fit(activations[:int(train_size * len(activations))],
                 epsilon_states[:int(train_size * len(epsilon_states))])

        # Evaluate
        test_activations = activations[int(train_size * len(activations)):]
        test_states = epsilon_states[int(train_size * len(epsilon_states)):]

        accuracy = probe.score(test_activations, test_states)

        # Margin analysis
        logits = probe.decision_function(test_activations)
        if logits.ndim == 1:
            margins = np.abs(logits)
        else:
            sorted_logits = np.sort(logits, axis=1)
            margins = sorted_logits[:, -1] - sorted_logits[:, -2]

        mean_margin = np.mean(margins)

        return {
            'accuracy': accuracy,
            'mean_margin': mean_margin,
            'n_classes': n_classes,
            'is_well_separable': accuracy >= 0.9
        }

    @staticmethod
    def compute_primitive_synergy(activations: np.ndarray,
                                 epsilon_states: np.ndarray,
                                 primitive_factors: Dict[str, np.ndarray],
                                 train_size: float = 0.7) -> Dict[str, float]:
        """
        Compute synergy between primitives and ε-states.

        Synergy = Acc_joint(ε) - Acc_compose_from_primitives
        Large positive synergy ⇒ information not distributed as separable primitives

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) ε-state labels
            primitive_factors: primitive_name -> (N,) factor values
            train_size: Training fraction

        Returns:
            Synergy metrics
        """
        n_train = int(train_size * len(activations))

        # Joint accuracy on ε-states
        joint_probe = LogisticRegression(random_state=42, max_iter=1000,
                                       multi_class='multinomial', solver='lbfgs')
        joint_probe.fit(activations[:n_train], epsilon_states[:n_train])
        joint_acc = joint_probe.score(activations[n_train:], epsilon_states[n_train:])

        # Compose from primitives: predict ε-state from primitive combination
        # For simplicity, concatenate primitive features
        primitive_features = np.column_stack(list(primitive_factors.values()))

        compose_probe = LogisticRegression(random_state=42, max_iter=1000,
                                         multi_class='multinomial', solver='lbfgs')
        compose_probe.fit(primitive_features[:n_train], epsilon_states[:n_train])
        compose_acc = compose_probe.score(primitive_features[n_train:], epsilon_states[n_train:])

        synergy = joint_acc - compose_acc

        return {
            'joint_accuracy': joint_acc,
            'compose_accuracy': compose_acc,
            'synergy': synergy,
            'relative_synergy': synergy / max(joint_acc, 1e-6)
        }

    @staticmethod
    def compute_fragmentation_analysis(activations: np.ndarray,
                                     epsilon_states: np.ndarray,
                                     n_clusters: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze fragmentation: how many clusters per ε-state.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state labels
            n_clusters: Number of clusters (default: n_states)

        Returns:
            Fragmentation metrics
        """
        if n_clusters is None:
            n_clusters = len(np.unique(epsilon_states))

        # Cluster representations
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(activations)

        # Analyze cluster distribution per state
        state_to_clusters = defaultdict(set)
        cluster_to_states = defaultdict(set)

        for state, cluster in zip(epsilon_states, cluster_labels):
            state_to_clusters[state].add(cluster)
            cluster_to_states[cluster].add(state)

        # Compute fragmentation metrics
        clusters_per_state = [len(clusters) for clusters in state_to_clusters.values()]
        states_per_cluster = [len(states) for states in cluster_to_states.values()]

        # ARI and NMI between clusters and true states
        ari = adjusted_rand_score(epsilon_states, cluster_labels)
        nmi = normalized_mutual_info_score(epsilon_states, cluster_labels)

        return {
            'median_clusters_per_state': np.median(clusters_per_state),
            'mean_clusters_per_state': np.mean(clusters_per_state),
            'max_clusters_per_state': max(clusters_per_state),
            'median_states_per_cluster': np.median(states_per_cluster),
            'ari_clusters_vs_states': ari,
            'nmi_clusters_vs_states': nmi,
            'fragmentation_score': np.median(clusters_per_state) - 1.0  # Ideal is 1.0
        }

    @staticmethod
    def compute_reachability_extrapolation(activations: Dict[str, np.ndarray],
                                         reachability_info: Dict[str, Dict[str, List[str]]],
                                         epsilon_states: np.ndarray,
                                         k_train: List[int] = [1, 2],
                                         k_test: List[int] = [3, 4, 5]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate k-step reachability extrapolation.

        Train probes for k∈{1,2}, test on k∈{3,4,5}.
        Slower decay ⇒ geometry encoded rather than label tables.

        Args:
            activations: layer -> (N, hidden_dim) features
            reachability_info: state -> k_step -> reachable_states
            epsilon_states: (N,) current state labels
            k_train: k values to train on
            k_test: k values to test on

        Returns:
            layer -> extrapolation_metrics
        """
        results = {}

        for layer_name, features in activations.items():
            # For each test k, train on smaller k and evaluate
            layer_results = {}

            for test_k in k_test:
                # Get samples that can reach test_k steps
                test_mask = np.zeros(len(epsilon_states), dtype=bool)

                for i, state in enumerate(epsilon_states):
                    reachable = reachability_info.get(str(state), {}).get(f'k_{test_k}', [])
                    test_mask[i] = len(reachable) > 0

                if not np.any(test_mask):
                    continue

                # For training, use samples from smaller k
                train_masks = []
                for train_k in k_train:
                    train_mask = np.zeros(len(epsilon_states), dtype=bool)
                    for i, state in enumerate(epsilon_states):
                        reachable = reachability_info.get(str(state), {}).get(f'k_{train_k}', [])
                        train_mask[i] = len(reachable) > 0
                    train_masks.append(train_mask)

                # Combine training masks
                combined_train_mask = np.any(train_masks, axis=0)

                if not np.any(combined_train_mask):
                    continue

                # Create reachability classification task
                # Positive: can reach within test_k steps
                y_train = combined_train_mask[combined_train_mask].astype(int)
                y_test = test_mask[test_mask].astype(int)

                X_train = features[combined_train_mask]
                X_test = features[test_mask]

                # Train probe
                probe = LogisticRegression(random_state=42, max_iter=1000)
                probe.fit(X_train, y_train)

                # Evaluate
                accuracy = probe.score(X_test, y_test)

                layer_results[f'k_{test_k}_extrapolation'] = accuracy

            # Compute decay rate
            if layer_results:
                accuracies = list(layer_results.values())
                if len(accuracies) > 1:
                    # Exponential decay fit
                    k_values = np.array(k_test[:len(accuracies)])
                    acc_values = np.array(accuracies)

                    # Simple decay metric: ratio of accuracies
                    decay_rate = acc_values[0] / max(acc_values[-1], 1e-6) if len(acc_values) > 1 else 1.0
                    layer_results['decay_rate'] = decay_rate

            results[layer_name] = layer_results

        return results

    @staticmethod
    def compute_similarity_structure(activations: Dict[str, np.ndarray],
                                   primitive_factors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Analyze similarity structure using CCA and PWCCA.

        Args:
            activations: layer -> (N, hidden_dim) features
            primitive_factors: primitive_name -> (N,) values

        Returns:
            layer -> similarity_metrics
        """
        results = {}

        # Simple correlation-based analysis (full CCA would require more dependencies)
        for layer_name, features in activations.items():
            layer_results = {}

            # Correlations between primitives
            primitive_names = list(primitive_factors.keys())
            primitive_matrix = np.column_stack([primitive_factors[name] for name in primitive_names])

            # Correlations between representations and primitives
            rep_primitive_corr = []
            for i in range(features.shape[1]):
                for j in range(primitive_matrix.shape[1]):
                    corr = np.corrcoef(features[:, i], primitive_matrix[:, j])[0, 1]
                    rep_primitive_corr.append(abs(corr))

            # Average absolute correlation
            layer_results['mean_abs_corr_rep_primitives'] = np.mean(rep_primitive_corr)

            # Correlations between different primitives in representations
            primitive_corr_matrix = np.corrcoef(primitive_matrix.T)
            layer_results['mean_primitive_correlation'] = np.mean(np.abs(primitive_corr_matrix))

            results[layer_name] = layer_results

        return results
