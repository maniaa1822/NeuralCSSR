"""
Probe evaluation utilities for entanglement analysis.

Computes nonlinearity index, composition gaps, and other comparative metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ProbeEvaluator:
    """
    Evaluates and compares linear vs MLP probes to quantify entanglement.
    """

    @staticmethod
    def compute_nonlinearity_index(linear_results: Dict[str, Dict[str, Dict[str, float]]],
                                  mlp_results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        """
        Compute nonlinearity index (NLI) for each label at each layer.

        NLI = Acc_MLP - Acc_Linear
        High NLI ⇒ latent needs non-linear readout ⇒ entanglement

        Args:
            linear_results: layer -> label -> {'accuracy': float, ...}
            mlp_results: layer -> label -> {'accuracy': float, ...}

        Returns:
            layer -> label -> {'nli': float, 'linear_acc': float, 'mlp_acc': float}
        """
        nli_results = {}

        for layer in linear_results.keys():
            nli_results[layer] = {}

            for label in linear_results[layer].keys():
                if label in mlp_results[layer]:
                    linear_acc = linear_results[layer][label]['accuracy']
                    mlp_acc = mlp_results[layer][label]['accuracy']

                    nli = mlp_acc - linear_acc

                    nli_results[layer][label] = {
                        'nli': nli,
                        'linear_accuracy': linear_acc,
                        'mlp_accuracy': mlp_acc
                    }

        return nli_results

    @staticmethod
    def compute_composition_gaps(composition_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute composition gaps from formula evaluation results.

        Args:
            composition_results: Formula results from CompositionEvaluator

        Returns:
            Aggregated gap metrics
        """
        if not composition_results:
            return {}

        zero_shot_gaps = [r['composition_gap_zero'] for r in composition_results.values()]
        few_shot_gaps = [r['composition_gap_few'] for r in composition_results.values()]

        return {
            'zero_shot_gap_mean': np.mean(zero_shot_gaps),
            'zero_shot_gap_std': np.std(zero_shot_gaps),
            'few_shot_gap_mean': np.mean(few_shot_gaps),
            'few_shot_gap_std': np.std(few_shot_gaps),
            'zero_shot_success_rate': np.mean([
                r['zero_shot_accuracy'] > r['chance_accuracy']
                for r in composition_results.values()
            ]),
            'few_shot_success_rate': np.mean([
                r['few_shot_accuracy'] > r['zero_shot_accuracy']
                for r in composition_results.values()
            ])
        }

    @staticmethod
    def compute_few_shot_curves(linear_few_shot: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
                               mlp_few_shot: Optional[Dict] = None,
                               shots: List[int] = [8, 16, 32, 64]) -> Dict[str, Dict[str, List[float]]]:
        """
        Compute few-shot learning curves.

        Args:
            linear_few_shot: Few-shot results from LinearProbeSuite
            mlp_few_shot: Few-shot results from MLPProbeSuite (optional)
            shots: Shot counts to include

        Returns:
            layer -> label -> [accuracy_at_shot_1, accuracy_at_shot_2, ...]
        """
        curves = {}

        for layer in linear_few_shot.keys():
            curves[layer] = {}

            for label in linear_few_shot[layer].keys():
                linear_curve = []

                for shot in shots:
                    if shot in linear_few_shot[layer][label]:
                        linear_curve.append(
                            linear_few_shot[layer][label][shot]['accuracy']
                        )
                    else:
                        linear_curve.append(np.nan)

                curves[layer][label] = {
                    'linear': linear_curve,
                    'shots': shots
                }

                if mlp_few_shot and layer in mlp_few_shot and label in mlp_few_shot[layer]:
                    mlp_curve = []
                    for shot in shots:
                        if shot in mlp_few_shot[layer][label]:
                            mlp_curve.append(
                                mlp_few_shot[layer][label][shot]['accuracy']
                            )
                        else:
                            mlp_curve.append(np.nan)

                    curves[layer][label]['mlp'] = mlp_curve

        return curves

    @staticmethod
    def summarize_probe_performance(linear_results: Dict[str, Dict[str, Dict[str, float]]],
                                   mlp_results: Dict[str, Dict[str, Dict[str, float]]],
                                   target_labels: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Summarize probe performance across layers.

        Args:
            linear_results: Linear probe results
            mlp_results: MLP probe results
            target_labels: Labels to include (default: all)

        Returns:
            layer -> summary_metrics
        """
        if target_labels is None:
            target_labels = list(next(iter(linear_results.values())).keys())

        summaries = {}

        for layer in linear_results.keys():
            layer_metrics = []

            for label in target_labels:
                if label in linear_results[layer] and label in mlp_results[layer]:
                    linear_acc = linear_results[layer][label]['accuracy']
                    mlp_acc = mlp_results[layer][label]['accuracy']
                    macro_f1 = linear_results[layer][label]['macro_f1']
                    nli = mlp_acc - linear_acc

                    layer_metrics.append({
                        'linear_accuracy': linear_acc,
                        'mlp_accuracy': mlp_acc,
                        'macro_f1': macro_f1,
                        'nli': nli
                    })

            if layer_metrics:
                summaries[layer] = {
                    'mean_linear_accuracy': np.mean([m['linear_accuracy'] for m in layer_metrics]),
                    'mean_mlp_accuracy': np.mean([m['mlp_accuracy'] for m in layer_metrics]),
                    'mean_nli': np.mean([m['nli'] for m in layer_metrics]),
                    'mean_macro_f1': np.mean([m['macro_f1'] for m in layer_metrics]),
                    'n_labels': len(layer_metrics)
                }

        return summaries

    @staticmethod
    def detect_layer_transitions(probe_summaries: Dict[str, Dict[str, float]],
                                threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect where information becomes linearly accessible.

        Args:
            probe_summaries: Layer summaries from summarize_probe_performance
            threshold: Accuracy threshold for "accessible"

        Returns:
            Transition analysis
        """
        if not probe_summaries:
            return {}

        # Sort layers by depth (assuming naming convention)
        layer_names = sorted(probe_summaries.keys(),
                           key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

        linear_accs = [probe_summaries[layer]['mean_linear_accuracy'] for layer in layer_names]

        # Find first layer above threshold
        transition_layer = None
        for i, acc in enumerate(linear_accs):
            if acc >= threshold:
                transition_layer = layer_names[i]
                break

        # Compute improvement rates
        improvements = []
        for i in range(1, len(linear_accs)):
            improvement = linear_accs[i] - linear_accs[i-1]
            improvements.append(improvement)

        return {
            'transition_layer': transition_layer,
            'linear_accuracies': dict(zip(layer_names, linear_accs)),
            'max_linear_accuracy': max(linear_accs),
            'mean_improvement': np.mean(improvements) if improvements else 0,
            'early_layers_low_accuracy': linear_accs[0] < threshold
        }
