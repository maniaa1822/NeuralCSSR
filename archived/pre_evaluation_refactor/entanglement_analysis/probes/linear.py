"""
Linear probe suite for entanglement analysis.

Implements multiclass logistic regression probes with L2 regularization
and comprehensive evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class LinearProbeSuite:
    """
    Suite of linear probes for analyzing representational linearity.

    Trains separate probes for ε-states and each primitive factor,
    with comprehensive evaluation metrics.
    """

    def __init__(self, l2_penalty: float = 1.0, random_state: int = 42):
        """
        Initialize probe suite.

        Args:
            l2_penalty: L2 regularization strength (C=1/lambda)
            random_state: Random seed for reproducibility
        """
        self.l2_penalty = l2_penalty
        self.random_state = random_state
        self.probes = {}  # layer -> label_name -> probe
        self.label_names = []

    def fit_probes(self, activations: Dict[str, np.ndarray],
                   labels: Dict[str, np.ndarray],
                   train_size: float = 0.7,
                   stratify: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Fit linear probes for all layers and labels.

        Args:
            activations: layer_name -> (N, hidden_dim) features
            labels: label_name -> (N,) targets
            train_size: Fraction of data for training
            stratify: Whether to stratify splits by label

        Returns:
            Nested dict: layer -> label -> metrics_dict
        """
        self.label_names = list(labels.keys())
        results = {}

        for layer_name, features in activations.items():
            results[layer_name] = {}

            for label_name, targets in labels.items():
                # Check if target has enough classes
                n_classes = len(np.unique(targets))
                if n_classes < 2:
                    print(f"Warning: Primitive {label_name} has only {n_classes} unique values, skipping")
                    results[layer_name][label_name] = {
                        'accuracy': 0.5,  # Chance level
                        'macro_f1': 0.0,
                        'mean_margin': 0.0,
                        'ece': 0.0,
                        'n_classes': n_classes,
                        'skipped': True
                    }
                    continue

                # Train/val split
                stratify_targets = targets if stratify and n_classes > 1 else None
                X_train, X_val, y_train, y_val = train_test_split(
                    features, targets,
                    train_size=train_size,
                    random_state=self.random_state,
                    stratify=stratify_targets
                )

                # Check training set has enough classes
                train_classes = len(np.unique(y_train))
                if train_classes < 2:
                    print(f"Warning: Primitive {label_name} training set has only {train_classes} classes, skipping")
                    results[layer_name][label_name] = {
                        'accuracy': 0.5,
                        'macro_f1': 0.0,
                        'mean_margin': 0.0,
                        'ece': 0.0,
                        'n_classes': n_classes,
                        'skipped': True
                    }
                    continue

                # Train probe
                probe = LogisticRegression(
                    C=self.l2_penalty,
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='multinomial' if n_classes > 2 else 'ovr',
                    solver='lbfgs'
                )

                probe.fit(X_train, y_train)

                # Evaluate
                metrics = self._evaluate_probe(probe, X_val, y_val)

                # Store
                if layer_name not in self.probes:
                    self.probes[layer_name] = {}
                self.probes[layer_name][label_name] = probe
                results[layer_name][label_name] = metrics

        return results

    def _evaluate_probe(self, probe: LogisticRegression,
                        X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate a trained probe with comprehensive metrics."""
        # Predictions and probabilities
        y_pred = probe.predict(X)
        y_prob = probe.predict_proba(X)

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average='macro')

        # Margin analysis (mean logit gap)
        logits = probe.decision_function(X)
        if logits.ndim == 1:
            # Binary case
            margins = np.abs(logits)
        else:
            # Multiclass: gap between top two logits
            sorted_logits = np.sort(logits, axis=1)
            margins = sorted_logits[:, -1] - sorted_logits[:, -2]

        mean_margin = np.mean(margins)

        # Calibration (Expected Calibration Error)
        if y_prob.shape[1] == 2:
            # Binary calibration
            prob_true, prob_pred = calibration_curve(y, y_prob[:, 1], n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred))
        else:
            # Multiclass: use confidence as proxy
            confidence = np.max(y_prob, axis=1)
            ece = np.mean(np.abs((y == y_pred).astype(float) - confidence))

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'mean_margin': mean_margin,
            'ece': ece
        }

    def few_shot_evaluation(self, activations: Dict[str, np.ndarray],
                           labels: Dict[str, np.ndarray],
                           shots: List[int] = [8, 16, 32],
                           n_trials: int = 5) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
        """
        Evaluate probes in few-shot setting.

        Args:
            activations: layer_name -> (N, hidden_dim) features
            labels: label_name -> (N,) targets
            shots: Number of examples per class for training
            n_trials: Number of random trials

        Returns:
            layer -> label -> shots -> metrics_dict
        """
        results = {}

        for layer_name, features in activations.items():
            results[layer_name] = {}

            for label_name, targets in labels.items():
                results[layer_name][label_name] = {}

                n_classes = len(np.unique(targets))
                if n_classes < 2:
                    # Skip primitives with insufficient classes
                    for n_shot in shots:
                        results[layer_name][label_name][n_shot] = {
                            'accuracy': 0.5,
                            'macro_f1': 0.0,
                            'mean_margin': 0.0,
                            'ece': 0.0,
                            'skipped': True
                        }
                    continue

                for n_shot in shots:
                    trial_metrics = []

                    for trial in range(n_trials):
                        # Sample few-shot training set
                        train_indices = self._sample_few_shot_indices(
                            targets, n_shot, n_classes
                        )

                        X_train = features[train_indices]
                        y_train = targets[train_indices]

                        # Check if training set has enough classes
                        train_classes = len(np.unique(y_train))
                        if train_classes < 2:
                            metrics = {
                                'accuracy': 0.5,
                                'macro_f1': 0.0,
                                'mean_margin': 0.0,
                                'ece': 0.0,
                                'skipped': True
                            }
                        else:
                            # Use remaining data for validation
                            val_mask = np.ones(len(features), dtype=bool)
                            val_mask[train_indices] = False
                            X_val = features[val_mask]
                            y_val = targets[val_mask]

                            # If no validation data, skip this trial
                            if len(X_val) == 0:
                                metrics = {
                                    'accuracy': 0.5,
                                    'macro_f1': 0.0,
                                    'mean_margin': 0.0,
                                    'ece': 0.0,
                                    'skipped': True
                                }
                            else:
                                # Train and evaluate
                                probe = LogisticRegression(
                                    C=self.l2_penalty,
                                    random_state=self.random_state + trial,
                                    max_iter=1000,
                                    multi_class='multinomial' if train_classes > 2 else 'ovr',
                                    solver='lbfgs'
                                )

                                probe.fit(X_train, y_train)
                                metrics = self._evaluate_probe(probe, X_val, y_val)

                        trial_metrics.append(metrics)

                    # Average across trials
                    avg_metrics = {}
                    for key in trial_metrics[0].keys():
                        avg_metrics[key] = np.mean([m[key] for m in trial_metrics])

                    results[layer_name][label_name][n_shot] = avg_metrics

        return results

    def _sample_few_shot_indices(self, targets: np.ndarray,
                                n_shot: int, n_classes: int) -> np.ndarray:
        """Sample balanced few-shot indices."""
        indices = []

        for class_idx in range(n_classes):
            class_mask = targets == class_idx
            class_indices = np.where(class_mask)[0]

            # Sample n_shot examples (or all if fewer available)
            n_sample = min(n_shot, len(class_indices))
            sampled = np.random.choice(class_indices, n_sample, replace=False)
            indices.extend(sampled)

        return np.array(indices)

    def predict_logits(self, layer_name: str, label_name: str,
                      features: np.ndarray) -> np.ndarray:
        """Get logits for a specific probe."""
        if layer_name not in self.probes or label_name not in self.probes[layer_name]:
            # Return zeros if probe wasn't trained (e.g., insufficient classes)
            n_samples = features.shape[0]
            # For binary classification fallback - return array of shape (n_samples,)
            # This will be treated as binary logits
            return np.zeros(n_samples)

        probe = self.probes[layer_name][label_name]
        return probe.decision_function(features)
