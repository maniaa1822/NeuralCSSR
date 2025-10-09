"""
Composition evaluator for zero-shot and few-shot logical composition tests.

Generates DNF/CNF formulas over primitives and evaluates compositionality
through logit combination vs learned composition.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import itertools
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class CompositionEvaluator:
    """
    Evaluates compositionality through logical formulas over primitives.

    Tests whether complex logical combinations of primitives can be
    predicted through logit arithmetic vs requiring learned composition.
    """

    def __init__(self, primitive_info: Dict[str, Dict[str, Any]],
                 random_state: int = 42):
        """
        Initialize composition evaluator.

        Args:
            primitive_info: Metadata about primitive factors
            random_state: Random seed
        """
        self.primitive_info = primitive_info
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

    def generate_formulas(self, n_formulas: int = 200,
                         max_terms: int = 3) -> List[Dict[str, Any]]:
        """
        Generate random DNF/CNF formulas over primitives.

        Args:
            n_formulas: Number of formulas to generate
            max_terms: Maximum terms per formula

        Returns:
            List of formula specifications
        """
        formulas = []
        primitive_names = list(self.primitive_info.keys())

        for _ in range(n_formulas):
            # Randomly choose DNF or CNF
            is_dnf = random.choice([True, False])

            # Randomly select primitives (2-4)
            n_prims = random.randint(2, min(4, len(primitive_names)))
            selected_prims = random.sample(primitive_names, n_prims)

            # Generate terms
            terms = []
            for _ in range(random.randint(1, max_terms)):
                term = {}
                for prim in selected_prims:
                    # Randomly choose a value for this primitive
                    if self.primitive_info[prim]['type'] == 'categorical':
                        classes = self.primitive_info[prim]['classes']
                        if isinstance(classes[0], str):
                            term[prim] = random.choice(classes)
                        else:
                            term[prim] = random.choice(classes)
                    else:
                        # For numerical, randomly choose threshold
                        term[prim] = random.choice([0, 1])  # Binary for simplicity

                terms.append(term)

            formula = {
                'type': 'dnf' if is_dnf else 'cnf',
                'primitives': selected_prims,
                'terms': terms,
                'target_name': f"{'∨' if is_dnf else '∧'}_".join(
                    [f"{'∧' if is_dnf else '∨'}_".join([f"{k}={v}" for k,v in term.items()])
                     for term in terms]
                )
            }
            formulas.append(formula)

        return formulas

    def evaluate_formulas(self, formulas: List[Dict[str, Any]],
                         primitive_logits: Dict[str, np.ndarray],
                         features: np.ndarray,
                         labels: Dict[str, np.ndarray],
                         few_shot_shots: int = 16) -> Dict[str, Dict[str, float]]:
        """
        Evaluate compositionality for a set of formulas.

        Args:
            formulas: Generated formula specifications
            primitive_logits: primitive_name -> (N, n_classes) logits
            features: (N, hidden_dim) representations for few-shot learning
            labels: primitive_name -> (N,) ground truth labels
            few_shot_shots: Examples per class for few-shot composition

        Returns:
            Formula results with zero-shot and few-shot accuracies
        """
        results = {}

        for formula in formulas:
            formula_key = formula['target_name']

            # Compute ground truth for this formula
            y_true = self._evaluate_formula_ground_truth(formula, labels)

            # Skip if too imbalanced or constant
            if len(np.unique(y_true)) < 2:
                continue

            # Zero-shot composition via logit arithmetic
            y_pred_zero = self._zero_shot_composition(formula, primitive_logits)

            # Few-shot learned composition
            y_pred_few = self._few_shot_composition(
                formula, features, y_true, few_shot_shots
            )

            # Compute accuracies
            acc_zero = accuracy_score(y_true, y_pred_zero)
            acc_few = accuracy_score(y_true, y_pred_few)
            acc_chance = max(np.mean(y_true), 1 - np.mean(y_true))  # Chance accuracy

            results[formula_key] = {
                'zero_shot_accuracy': acc_zero,
                'few_shot_accuracy': acc_few,
                'chance_accuracy': acc_chance,
                'composition_gap_zero': acc_zero - acc_chance,
                'composition_gap_few': acc_few - acc_zero,
                'formula_type': formula['type'],
                'n_primitives': len(formula['primitives'])
            }

        return results

    def _evaluate_formula_ground_truth(self, formula: Dict[str, Any],
                                      labels: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute ground truth labels for a logical formula."""
        n_samples = len(next(iter(labels.values())))
        results = np.zeros(n_samples, dtype=bool)

        for term in formula['terms']:
            term_results = np.ones(n_samples, dtype=bool)

            for prim_name, prim_value in term.items():
                prim_labels = labels[prim_name]

                if self.primitive_info[prim_name]['type'] == 'categorical':
                    # Exact match for categorical
                    term_results &= (prim_labels == prim_value)
                else:
                    # Threshold for numerical
                    if prim_value == 0:
                        term_results &= (prim_labels == 0)
                    else:
                        term_results &= (prim_labels > 0)

            if formula['type'] == 'dnf':
                results |= term_results  # OR for DNF
            else:
                results &= term_results  # AND for CNF

        return results.astype(int)

    def _zero_shot_composition(self, formula: Dict[str, Any],
                              primitive_logits: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform zero-shot composition using logit arithmetic."""
        n_samples = len(next(iter(primitive_logits.values())))

        # For each sample, compute composite logit
        composite_logits = np.zeros((n_samples, 2))  # Binary classification

        for sample_idx in range(n_samples):
            sample_composite = 0.0

            for term in formula['terms']:
                term_logit = 0.0

                for prim_name, prim_value in term.items():
                    if prim_name in primitive_logits:
                        # Get logits for this primitive
                        logits = primitive_logits[prim_name]

                        if np.isscalar(logits[sample_idx]) or logits[sample_idx].shape == ():
                            # Untrained probe - use default logit of 0
                            term_logit += 0.0
                        else:
                            # Trained probe - use actual logits
                            logit_values = logits[sample_idx]

                            if self.primitive_info[prim_name]['type'] == 'categorical':
                                # Convert to binary logit (positive class vs others)
                                if len(logit_values) == 2:
                                    # Binary case
                                    term_logit += logit_values[1] if prim_value == 1 else logit_values[0]
                                else:
                                    # Multiclass: use logit for target class vs mean of others
                                    target_logit = logit_values[prim_value]
                                    other_logits = np.delete(logit_values, prim_value)
                                    term_logit += target_logit - np.mean(other_logits)
                            else:
                                # For numerical, treat as binary
                                term_logit += logit_values[prim_value] if prim_value < len(logit_values) else logit_values[0]

                # Combine terms
                if formula['type'] == 'dnf':
                    sample_composite = max(sample_composite, term_logit)  # OR ≈ max
                else:
                    sample_composite += term_logit  # AND ≈ sum

            # Convert to binary prediction
            composite_logits[sample_idx] = [-sample_composite, sample_composite]

        return np.argmax(composite_logits, axis=1)

    def _few_shot_composition(self, formula: Dict[str, Any],
                             features: np.ndarray, y_true: np.ndarray,
                             n_shots: int) -> np.ndarray:
        """Learn composition with few-shot examples."""
        n_classes = len(np.unique(y_true))

        # Sample few-shot examples
        train_indices = []
        for class_idx in range(n_classes):
            class_mask = y_true == class_idx
            class_indices = np.where(class_mask)[0]
            n_sample = min(n_shots, len(class_indices))
            sampled = np.random.choice(class_indices, n_sample, replace=False)
            train_indices.extend(sampled)

        train_indices = np.array(train_indices)

        # Train linear probe on composition
        X_train = features[train_indices]
        y_train = y_true[train_indices]

        probe = LogisticRegression(random_state=self.random_state, max_iter=1000)
        probe.fit(X_train, y_train)

        # Predict on all data
        return probe.predict(features)

    def aggregate_results(self, formula_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate composition results across formulas."""
        if not formula_results:
            return {}

        metrics = ['zero_shot_accuracy', 'few_shot_accuracy', 'chance_accuracy',
                  'composition_gap_zero', 'composition_gap_few']

        aggregated = {}
        for metric in metrics:
            values = [r[metric] for r in formula_results.values()]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)

        # Additional summary statistics
        aggregated['n_formulas'] = len(formula_results)
        aggregated['zero_shot_success_rate'] = np.mean([
            r['zero_shot_accuracy'] > r['chance_accuracy']
            for r in formula_results.values()
        ])
        aggregated['few_shot_improvement'] = np.mean([
            r['few_shot_accuracy'] > r['zero_shot_accuracy']
            for r in formula_results.values()
        ])

        return aggregated
