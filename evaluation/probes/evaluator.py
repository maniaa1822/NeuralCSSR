"""Probe evaluation and comparison utilities."""

from __future__ import annotations

from typing import Dict

from .linear import ProbeMetrics


class ProbeEvaluator:
    """Evaluate and compare probe performance across layers and types."""

    @staticmethod
    def compute_nonlinearity_index(
        linear_results: Dict[str, Dict[str, ProbeMetrics]],
        mlp_results: Dict[str, Dict[str, ProbeMetrics]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute nonlinearity index (NLI) for each layer and label.

        NLI = (mlp_acc - linear_acc) / (1 - linear_acc)
        Measures how much non-linearity helps beyond linear probes.

        Args:
            linear_results: {layer: {label: metrics}}
            mlp_results: {layer: {label: metrics}}

        Returns:
            {layer: {label: nli_value}}
        """
        nli_results = {}

        for layer_name in linear_results.keys():
            if layer_name not in mlp_results:
                continue

            nli_results[layer_name] = {}

            for label_name in linear_results[layer_name].keys():
                if label_name not in mlp_results[layer_name]:
                    continue

                linear_acc = linear_results[layer_name][label_name].accuracy
                mlp_acc = mlp_results[layer_name][label_name].accuracy

                if linear_acc >= 1.0:
                    # Perfect linear accuracy: NLI = 0
                    nli = 0.0
                else:
                    nli = (mlp_acc - linear_acc) / (1.0 - linear_acc)

                nli_results[layer_name][label_name] = nli

        return nli_results

    @staticmethod
    def summarize_probe_performance(
        linear_results: Dict[str, Dict[str, ProbeMetrics]],
        mlp_results: Dict[str, Dict[str, ProbeMetrics]],
    ) -> Dict[str, Dict[str, float]]:
        """Summarize probe performance per layer.

        Args:
            linear_results: {layer: {label: metrics}}
            mlp_results: {layer: {label: metrics}}

        Returns:
            {layer: {
                "mean_linear_accuracy": float,
                "mean_mlp_accuracy": float,
                "mean_nli": float,
            }}
        """
        summary = {}
        nli_results = ProbeEvaluator.compute_nonlinearity_index(linear_results, mlp_results)

        for layer_name in linear_results.keys():
            linear_accs = [m.accuracy for m in linear_results[layer_name].values()]
            mlp_accs = [m.accuracy for m in mlp_results[layer_name].values() if layer_name in mlp_results]
            nli_vals = list(nli_results.get(layer_name, {}).values())

            summary[layer_name] = {
                "mean_linear_accuracy": sum(linear_accs) / len(linear_accs) if linear_accs else 0.0,
                "mean_mlp_accuracy": sum(mlp_accs) / len(mlp_accs) if mlp_accs else 0.0,
                "mean_nli": sum(nli_vals) / len(nli_vals) if nli_vals else 0.0,
            }

        return summary
