"""
FER (Fractured Entangled Representation) score calculator.

Aggregates multiple entanglement metrics into a single score.
"""

import numpy as np
from typing import Dict, List, Optional, Any


class FERCalculator:
    """
    Computes the FER Index: a composite score measuring representation quality.

    FER = w₁(1 - LinAcc_prims) + w₂ NLI_avg + w₃ (Frag - 1) + w₄ Synergy + w₅ Curvature
    Lower is better (less fractured/entangled).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize FER calculator.

        Args:
            weights: Custom weights for FER components (optional)
        """
        self.default_weights = {
            'primitive_linearity': 1.0,    # 1 - LinAcc_prims
            'nonlinearity_index': 1.0,     # NLI_avg
            'fragmentation': 1.0,          # Frag - 1
            'synergy': 1.0,                # Synergy
            'curvature': 1.0               # Curvature
        }
        self.weights = weights if weights else self.default_weights

    def compute_fer_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute FER score from collected metrics.

        Args:
            metrics: Dictionary containing all computed metrics

        Returns:
            FER score (lower is better)
        """
        components = self._extract_components(metrics)
        normalized_components = self._normalize_components(components)

        fer_score = 0.0
        for component_name, weight in self.weights.items():
            if component_name in normalized_components:
                fer_score += weight * normalized_components[component_name]

        return fer_score

    def _extract_components(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract FER components from metrics dictionary."""
        components = {}

        # Primitive linearity: 1 - mean linear accuracy on primitives
        if 'primitive_probe_summaries' in metrics:
            prim_summaries = metrics['primitive_probe_summaries']
            prim_accuracies = [summary['mean_linear_accuracy']
                             for summary in prim_summaries.values()]
            components['primitive_linearity'] = 1.0 - np.mean(prim_accuracies)

        # Nonlinearity index: average NLI across primitives
        if 'nli_results' in metrics:
            nli_values = []
            for layer_results in metrics['nli_results'].values():
                for label_result in layer_results.values():
                    nli_values.append(label_result['nli'])
            components['nonlinearity_index'] = np.mean(nli_values) if nli_values else 0.0

        # Fragmentation: median clusters per state - 1
        if 'fragmentation_analysis' in metrics:
            frag_scores = [result['median_clusters_per_state']
                          for result in metrics['fragmentation_analysis'].values()]
            components['fragmentation'] = np.mean(frag_scores) - 1.0 if frag_scores else 0.0

        # Synergy: average synergy across layers
        if 'entanglement_metrics' in metrics:
            synergy_values = [result['synergy']
                            for result in metrics['entanglement_metrics'].values()
                            if 'synergy' in result]
            components['synergy'] = np.mean(synergy_values) if synergy_values else 0.0

        # Curvature: average trajectory curvature
        if 'trajectory_analysis' in metrics:
            curvature_values = [result['mean_curvature']
                              for result in metrics['trajectory_analysis'].values()
                              if 'mean_curvature' in result]
            components['curvature'] = np.mean(curvature_values) if curvature_values else 0.0

        return components

    def _normalize_components(self, components: Dict[str, float]) -> Dict[str, float]:
        """Normalize components to [0,1] range using baselines."""
        # Baselines based on typical ranges observed in the paper/thresholds
        baselines = {
            'primitive_linearity': {'min': 0.0, 'max': 0.5},  # 0.5 = 50% error
            'nonlinearity_index': {'min': 0.0, 'max': 0.15},  # 15% NLI
            'fragmentation': {'min': 0.0, 'max': 1.6},        # 1.6 clusters/state
            'synergy': {'min': 0.0, 'max': 0.2},              # 20% synergy
            'curvature': {'min': 0.0, 'max': 1.5}             # π/2 radians
        }

        normalized = {}
        for component_name, value in components.items():
            if component_name in baselines:
                baseline = baselines[component_name]
                # Clip to baseline range and normalize
                clipped_value = np.clip(value, baseline['min'], baseline['max'])
                normalized_value = (clipped_value - baseline['min']) / (baseline['max'] - baseline['min'])
                normalized[component_name] = normalized_value
            else:
                # If no baseline, use value as-is (assuming it's already [0,1])
                normalized[component_name] = value

        return normalized

    def compute_layer_wise_fer(self, all_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute FER score for each layer.

        Args:
            all_metrics: layer_name -> metrics_dict

        Returns:
            layer_name -> fer_score
        """
        layer_fer_scores = {}

        for layer_name, metrics in all_metrics.items():
            fer_score = self.compute_fer_score(metrics)
            layer_fer_scores[layer_name] = fer_score

        return layer_fer_scores

    def get_fer_interpretation(self, fer_score: float) -> str:
        """Interpret FER score qualitatively."""
        if fer_score < 0.2:
            return "Excellent: Well-factorized representations"
        elif fer_score < 0.4:
            return "Good: Mostly disentangled with minor entanglement"
        elif fer_score < 0.6:
            return "Moderate: Some entanglement present"
        elif fer_score < 0.8:
            return "Poor: Significant entanglement"
        else:
            return "Very Poor: Highly entangled representations"

    def get_component_contributions(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Get contribution of each component to total FER score."""
        components = self._extract_components(metrics)
        normalized_components = self._normalize_components(components)

        contributions = {}
        total_weighted_score = 0.0

        for component_name, weight in self.weights.items():
            if component_name in normalized_components:
                contribution = weight * normalized_components[component_name]
                contributions[component_name] = contribution
                total_weighted_score += contribution

        # Convert to percentages
        if total_weighted_score > 0:
            for component_name in contributions:
                contributions[component_name] /= total_weighted_score

        return contributions
