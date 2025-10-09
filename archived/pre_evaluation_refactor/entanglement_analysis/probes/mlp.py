"""
MLP probe suite for non-linear analysis.

Implements small 1-hidden-layer MLPs with ReLU activation for
comparing against linear probes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SimpleMLP(nn.Module):
    """Simple 1-hidden-layer MLP for probing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MLPProbeSuite:
    """
    Suite of small MLP probes for non-linear analysis.

    Trains 1-hidden-layer MLPs with ReLU activation, testing different
    hidden dimensions (4-16) to find minimal non-linearity needed.
    """

    def __init__(self, hidden_dims: List[int] = [4, 8, 16],
                 learning_rate: float = 1e-3,
                 max_epochs: int = 100,
                 patience: int = 10,
                 random_state: int = 42):
        """
        Initialize MLP probe suite.

        Args:
            hidden_dims: Hidden layer dimensions to try
            learning_rate: Learning rate for Adam
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            random_state: Random seed
        """
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        # Set deterministic behavior
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.probes = {}  # layer -> label -> hidden_dim -> probe

    def fit_probes(self, activations: Dict[str, np.ndarray],
                   labels: Dict[str, np.ndarray],
                   train_size: float = 0.7,
                   batch_size: int = 32) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Fit MLP probes for all layers and labels.

        Args:
            activations: layer_name -> (N, hidden_dim) features
            labels: label_name -> (N,) targets
            train_size: Fraction of data for training
            batch_size: Batch size for training

        Returns:
            Nested dict: layer -> label -> metrics_dict
        """
        results = {}

        for layer_name, features in activations.items():
            results[layer_name] = {}

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            for label_name, targets in labels.items():
                results[layer_name][label_name] = {}

                # Check if target has enough classes
                n_classes = len(np.unique(targets))
                if n_classes < 2:
                    print(f"Warning: Primitive {label_name} has only {n_classes} unique values, skipping MLP")
                    results[layer_name][label_name] = {
                        'accuracy': 0.5,
                        'macro_f1': 0.0,
                        'best_hidden_dim': 4,
                        'n_classes': n_classes,
                        'skipped': True
                    }
                    continue

                # Train/val split
                stratify_targets = targets if n_classes > 1 else None
                X_train, X_val, y_train, y_val = train_test_split(
                    features_scaled, targets,
                    train_size=train_size,
                    random_state=self.random_state,
                    stratify=stratify_targets
                )

                # Check training set has enough classes
                train_classes = len(np.unique(y_train))
                if train_classes < 2:
                    print(f"Warning: Primitive {label_name} training set has only {train_classes} classes, skipping MLP")
                    results[layer_name][label_name] = {
                        'accuracy': 0.5,
                        'macro_f1': 0.0,
                        'best_hidden_dim': 4,
                        'n_classes': n_classes,
                        'skipped': True
                    }
                    continue

                # Try different hidden dimensions
                best_metrics = None
                best_hidden = None

                for hidden_dim in self.hidden_dims:
                    metrics = self._train_mlp_probe(
                        X_train, y_train, X_val, y_val,
                        features.shape[1], hidden_dim, len(np.unique(targets)),
                        batch_size
                    )

                    if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
                        best_metrics = metrics
                        best_hidden = hidden_dim

                # Store best probe
                if layer_name not in self.probes:
                    self.probes[layer_name] = {}
                if label_name not in self.probes[layer_name]:
                    self.probes[layer_name][label_name] = {}

                # Re-train on full training set with best hidden dim
                final_metrics = self._train_mlp_probe(
                    X_train, y_train, X_val, y_val,
                    features.shape[1], best_hidden, len(np.unique(targets)),
                    batch_size, store_probe=True,
                    probe_key=(layer_name, label_name, best_hidden)
                )

                results[layer_name][label_name] = {
                    **final_metrics,
                    'best_hidden_dim': best_hidden
                }

        return results

    def _train_mlp_probe(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        input_dim: int, hidden_dim: int, output_dim: int,
                        batch_size: int, store_probe: bool = False,
                        probe_key: Optional[Tuple] = None) -> Dict[str, float]:
        """Train a single MLP probe."""

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = SimpleMLP(input_dim, hidden_dim, output_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop with early stopping
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.max_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                _, val_pred = torch.max(val_outputs, 1)
                val_acc = (val_pred == y_val_t).float().mean().item()

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            _, val_pred = torch.max(val_outputs, 1)

            val_pred_np = val_pred.cpu().numpy()
            accuracy = accuracy_score(y_val, val_pred_np)
            macro_f1 = f1_score(y_val, val_pred_np, average='macro')

        # Store probe if requested
        if store_probe and probe_key:
            layer_name, label_name, hidden_dim = probe_key
            if layer_name not in self.probes:
                self.probes[layer_name] = {}
            if label_name not in self.probes[layer_name]:
                self.probes[layer_name][label_name] = {}
            self.probes[layer_name][label_name][hidden_dim] = model

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1
        }

    def few_shot_evaluation(self, activations: Dict[str, np.ndarray],
                           labels: Dict[str, np.ndarray],
                           shots: List[int] = [8, 16, 32],
                           n_trials: int = 5,
                           batch_size: int = 16) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
        """
        Evaluate MLP probes in few-shot setting.

        Args:
            activations: layer_name -> (N, hidden_dim) features
            labels: label_name -> (N,) targets
            shots: Number of examples per class
            n_trials: Number of random trials
            batch_size: Training batch size

        Returns:
            layer -> label -> shots -> metrics_dict
        """
        results = {}

        for layer_name, features in activations.items():
            results[layer_name] = {}

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            for label_name, targets in labels.items():
                results[layer_name][label_name] = {}

                n_classes = len(np.unique(targets))
                if n_classes < 2:
                    # Skip primitives with insufficient classes
                    for n_shot in shots:
                        results[layer_name][label_name][n_shot] = {
                            'accuracy': 0.5,
                            'macro_f1': 0.0,
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

                        X_train = features_scaled[train_indices]
                        y_train = targets[train_indices]

                        # Use remaining data for validation
                        val_mask = np.ones(len(features_scaled), dtype=bool)
                        val_mask[train_indices] = False
                        X_val = features_scaled[val_mask]
                        y_val = targets[val_mask]

                        # If no validation data, skip this trial
                        if len(X_val) == 0:
                            metrics = {
                                'accuracy': 0.5,
                                'macro_f1': 0.0,
                                'skipped': True
                            }
                        else:
                            # Train and evaluate
                            metrics = self._train_mlp_probe(
                                X_train, y_train, X_val, y_val,
                                features.shape[1], 8, n_classes,  # Use 8 hidden as default
                                batch_size
                            )
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
