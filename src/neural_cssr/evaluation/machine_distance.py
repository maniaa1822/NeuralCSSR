"""Main machine distance calculator integrating all distance metrics."""

from typing import Dict, List, Any
import json
import os
from datetime import datetime

from .metrics.state_mapping import StateMappingDistance
from .metrics.symbol_distribution import SymbolDistributionDistance
from .metrics.transition_structure import TransitionStructureDistance


class MachineDistanceCalculator:
    """Main interface for computing all distance metrics between discovered and ground truth machines."""
    
    def __init__(self):
        """Initialize all distance metric calculators."""
        self.state_mapping = StateMappingDistance()
        self.symbol_distribution = SymbolDistributionDistance()
        self.transition_structure = TransitionStructureDistance()
    
    def compute_all_distances(self, discovered_machine: Dict[str, Any], 
                            ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute all three distance metrics.
        
        Args:
            discovered_machine: CSSR discovered machine results
            ground_truth_machines: List of ground truth machines from dataset
            
        Returns:
            Dictionary containing all distance metrics and summary analysis
        """
        # Compute individual distance metrics
        state_mapping_distance = self.state_mapping.compute(discovered_machine, ground_truth_machines)
        symbol_distribution_distance = self.symbol_distribution.compute(discovered_machine, ground_truth_machines)
        transition_structure_distance = self.transition_structure.compute(discovered_machine, ground_truth_machines)
        
        # Compute summary metrics
        summary = self._compute_summary(
            state_mapping_distance, 
            symbol_distribution_distance, 
            transition_structure_distance
        )
        
        return {
            'state_mapping_distance': state_mapping_distance,
            'symbol_distribution_distance': symbol_distribution_distance,
            'transition_structure_distance': transition_structure_distance,
            'summary': summary,
            'metadata': {
                'computation_timestamp': datetime.now().isoformat(),
                'discovered_machine_info': self._extract_machine_info(discovered_machine),
                'ground_truth_machine_count': len(ground_truth_machines),
                'ground_truth_machine_ids': [m.get('machine_id', 'unknown') for m in ground_truth_machines]
            }
        }
    
    def _compute_summary(self, state_mapping: Dict[str, Any], 
                        symbol_distribution: Dict[str, Any],
                        transition_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall summary metrics combining all distance measures."""
        
        # Extract key quality scores
        state_quality = state_mapping.get('assignment_quality', {}).get('quality_score', 0.0)
        symbol_quality = symbol_distribution.get('quality_assessment', {}).get('overall_quality_score', 0.0)
        
        # Transition structure quality (convert distances to quality scores)
        transition_quality = 1.0 - min(
            transition_structure.get('graph_edit_distance', 1.0),
            transition_structure.get('spectral_distance', 1.0)
        )
        
        # Overall distance score (weighted average)
        # Give more weight to symbol distribution as it's most direct
        weights = {'symbol': 0.5, 'state': 0.3, 'transition': 0.2}
        overall_quality = (
            weights['symbol'] * symbol_quality +
            weights['state'] * state_quality +
            weights['transition'] * transition_quality
        )
        
        # Determine best metric based on individual scores
        metric_scores = {
            'symbol_distribution': symbol_quality,
            'state_mapping': state_quality,
            'transition_structure': transition_quality
        }
        best_metric = max(metric_scores, key=metric_scores.get)
        
        # Compute confidence based on agreement between metrics
        metric_values = list(metric_scores.values())
        confidence = 1.0 - (max(metric_values) - min(metric_values)) if metric_values else 0.0
        
        return {
            'overall_distance_score': 1.0 - overall_quality,  # Convert quality to distance
            'overall_quality_score': overall_quality,
            'best_metric': best_metric,
            'confidence': confidence,
            'metric_scores': metric_scores,
            'interpretation': self._interpret_results(overall_quality, confidence, best_metric),
            'recommendations': self._generate_recommendations(state_mapping, symbol_distribution, transition_structure)
        }
    
    def _extract_machine_info(self, discovered_machine: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic information about the discovered machine."""
        info = {
            'format': 'unknown',
            'num_states': 0,
            'has_transitions': False,
            'parameter_sets': 0
        }
        
        if 'cssr_results' in discovered_machine:
            info['format'] = 'cssr_results'
            info['parameter_sets'] = len(discovered_machine['cssr_results'].get('parameter_results', {}))
            
            # Get info from first parameter set
            for param_data in discovered_machine['cssr_results']['parameter_results'].values():
                if 'discovered_structure' in param_data:
                    structure = param_data['discovered_structure']
                    info['num_states'] = structure.get('num_states', len(structure.get('states', {})))
                    info['has_transitions'] = 'transitions' in structure
                    break
        
        elif 'states' in discovered_machine:
            info['format'] = 'direct_states'
            info['num_states'] = len(discovered_machine['states'])
            info['has_transitions'] = 'transitions' in discovered_machine
        
        return info
    
    def _interpret_results(self, overall_quality: float, confidence: float, best_metric: str) -> Dict[str, str]:
        """Provide interpretive analysis of the results."""
        # Quality interpretation
        if overall_quality >= 0.8:
            quality_text = "Excellent match - CSSR discovered machine closely matches ground truth"
        elif overall_quality >= 0.6:
            quality_text = "Good match - CSSR shows strong correspondence with ground truth"
        elif overall_quality >= 0.4:
            quality_text = "Fair match - Some correspondence but significant differences exist"
        elif overall_quality >= 0.2:
            quality_text = "Poor match - Limited correspondence with ground truth"
        else:
            quality_text = "Very poor match - Little to no correspondence with ground truth"
        
        # Confidence interpretation
        if confidence >= 0.8:
            confidence_text = "High confidence - All metrics agree on the assessment"
        elif confidence >= 0.6:
            confidence_text = "Moderate confidence - Most metrics are in agreement"
        elif confidence >= 0.4:
            confidence_text = "Low confidence - Mixed results across different metrics"
        else:
            confidence_text = "Very low confidence - Conflicting results across metrics"
        
        # Best metric interpretation
        metric_explanations = {
            'symbol_distribution': "Symbol emission patterns show the strongest correspondence",
            'state_mapping': "State-to-state mappings show the strongest correspondence", 
            'transition_structure': "Graph connectivity patterns show the strongest correspondence"
        }
        
        return {
            'overall_quality': quality_text,
            'confidence_level': confidence_text,
            'strongest_correspondence': metric_explanations.get(best_metric, "Unknown metric type")
        }
    
    def _generate_recommendations(self, state_mapping: Dict[str, Any],
                                symbol_distribution: Dict[str, Any],
                                transition_structure: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        # Check symbol distribution quality
        symbol_quality = symbol_distribution.get('quality_assessment', {}).get('overall_quality_score', 0.0)
        if symbol_quality < 0.5:
            recommendations.append(
                "Poor symbol distribution match suggests CSSR parameters may need adjustment "
                "(try different significance levels or history lengths)"
            )
        
        # Check coverage
        coverage = symbol_distribution.get('coverage_score', {}).get('coverage_fraction', 0.0)
        if coverage < 0.7:
            recommendations.append(
                f"Low coverage ({coverage:.1%}) indicates CSSR may have missed some ground truth states - "
                "consider increasing sequence length or adjusting parameters"
            )
        
        # Check state count mismatch
        discovered_states = len(state_mapping.get('optimal_assignment', []))
        unmatched_discovered = len(state_mapping.get('unmatched_discovered_states', []))
        if unmatched_discovered > 0:
            recommendations.append(
                f"CSSR discovered {unmatched_discovered} extra states - "
                "may indicate overfitting or need for higher significance level"
            )
        
        # Check transition structure
        connectivity_sim = transition_structure.get('connectivity_similarity', 0.0)
        if connectivity_sim < 0.5:
            recommendations.append(
                "Poor transition structure match - consider analyzing longer sequences "
                "to better capture state transition patterns"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Results look good - CSSR parameters appear well-tuned for this dataset")
        
        return recommendations
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Generate comprehensive markdown report with analysis.
        
        Args:
            results: Results from compute_all_distances
            output_path: Path to save the markdown report
            
        Returns:
            Path to the generated report file
        """
        report_content = self._create_markdown_report(results)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        # Also save JSON results
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return output_path
    
    def _create_markdown_report(self, results: Dict[str, Any]) -> str:
        """Create detailed markdown report from results."""
        summary = results.get('summary', {})
        metadata = results.get('metadata', {})
        
        report = f"""# Machine Distance Analysis Report

Generated: {metadata.get('computation_timestamp', 'Unknown')}

## Executive Summary

**Overall Quality Score**: {summary.get('overall_quality_score', 0.0):.3f}
**Overall Distance Score**: {summary.get('overall_distance_score', 0.0):.3f}
**Confidence Level**: {summary.get('confidence', 0.0):.3f}
**Best Performing Metric**: {summary.get('best_metric', 'Unknown')}

### Interpretation
{summary.get('interpretation', {}).get('overall_quality', 'No interpretation available')}

{summary.get('interpretation', {}).get('confidence_level', '')}

{summary.get('interpretation', {}).get('strongest_correspondence', '')}

## Machine Information

**Discovered Machine**: {metadata.get('discovered_machine_info', {}).get('num_states', 0)} states
**Ground Truth Machines**: {metadata.get('ground_truth_machine_count', 0)} machines
**Machine IDs**: {', '.join(metadata.get('ground_truth_machine_ids', []))}

## Detailed Metrics

### 1. Symbol Distribution Distance
{self._format_symbol_distribution_section(results.get('symbol_distribution_distance', {}))}

### 2. State Mapping Distance  
{self._format_state_mapping_section(results.get('state_mapping_distance', {}))}

### 3. Transition Structure Distance
{self._format_transition_structure_section(results.get('transition_structure_distance', {}))}

## Recommendations

{chr(10).join([f'- {rec}' for rec in summary.get('recommendations', [])])}

## Technical Details

### Metric Scores
{chr(10).join([f'- **{metric}**: {score:.3f}' for metric, score in summary.get('metric_scores', {}).items()])}

---
*Report generated by Neural CSSR Machine Distance Calculator*
"""
        return report
    
    def _format_symbol_distribution_section(self, results: Dict[str, Any]) -> str:
        """Format symbol distribution results for markdown."""
        if not results:
            return "No results available"
        
        quality = results.get('quality_assessment', {})
        coverage = results.get('coverage_score', {})
        
        section = f"""
**Average JS Divergence**: {results.get('average_js_divergence', 0.0):.4f}
**Quality Score**: {quality.get('overall_quality_score', 0.0):.3f} ({quality.get('assessment', 'Unknown')})
**Coverage**: {coverage.get('coverage_fraction', 0.0):.1%} ({coverage.get('well_matched_states', 0)}/{coverage.get('total_ground_truth_states', 0)} states)

#### Best State Matches
"""
        
        # Add top 3 best matches
        mappings = results.get('state_mappings', [])[:3]
        for mapping in mappings:
            section += f"- **{mapping.get('discovered_state', 'Unknown')}** → **{mapping.get('best_match', {}).get('full_name', 'Unknown')}** (JS: {mapping.get('js_divergence', 0.0):.4f})\n"
        
        return section
    
    def _format_state_mapping_section(self, results: Dict[str, Any]) -> str:
        """Format state mapping results for markdown."""
        if not results:
            return "No results available"
        
        quality = results.get('assignment_quality', {})
        
        section = f"""
**Total Assignment Cost**: {results.get('total_cost', 0.0):.4f}
**Average Cost per State**: {results.get('average_cost', 0.0):.4f}
**Quality Score**: {quality.get('quality_score', 0.0):.3f}
**Unmatched Discovered States**: {len(results.get('unmatched_discovered_states', []))}
**Unmatched True States**: {len(results.get('unmatched_true_states', []))}

#### Optimal Assignments
"""
        
        # Add all assignments
        assignments = results.get('optimal_assignment', [])
        for assignment in assignments:
            section += f"- **{assignment.get('discovered_state', 'Unknown')}** → **{assignment.get('ground_truth_state', 'Unknown')}** (Cost: {assignment.get('cost', 0.0):.4f})\n"
        
        return section
    
    def _format_transition_structure_section(self, results: Dict[str, Any]) -> str:
        """Format transition structure results for markdown."""
        if not results:
            return "No results available"
        
        coverage = results.get('transition_coverage', {})
        
        section = f"""
**Graph Edit Distance**: {results.get('graph_edit_distance', 0.0):.4f}
**Spectral Distance**: {results.get('spectral_distance', 0.0):.4f}
**Connectivity Similarity**: {results.get('connectivity_similarity', 0.0):.4f}
**Transition Coverage**: {coverage.get('coverage_ratio', 0.0):.1%}
**Precision**: {coverage.get('precision', 0.0):.3f}
**Recall**: {coverage.get('recall', 0.0):.3f}

#### Structural Properties
"""
        
        # Add structural comparison
        structural = results.get('structural_comparison', {})
        for prop, comparison in structural.items():
            if isinstance(comparison, dict):
                section += f"- **{prop}**: Discovered={comparison.get('discovered', 'N/A')}, Ground Truth={comparison.get('ground_truth', 'N/A')}, Similarity={comparison.get('similarity', 0.0):.3f}\n"
        
        return section