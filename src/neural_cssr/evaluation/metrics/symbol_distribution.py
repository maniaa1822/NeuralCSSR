"""Symbol distribution distance comparing state-wise emission distributions."""

from typing import Dict, List, Any
import numpy as np
from scipy.spatial.distance import jensenshannon


class SymbolDistributionDistance:
    """Compare state-wise symbol emission distributions between discovered and ground truth machines."""
    
    def compute(self, discovered_machine: Dict[str, Any], ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare symbol distributions finding optimal matching between discovered and true states.
        
        Args:
            discovered_machine: CSSR discovered machine with states and symbol distributions
            ground_truth_machines: List of ground truth machines from dataset
            
        Returns:
            Dictionary containing distribution distance metrics and state mappings
        """
        # Extract discovered states and their distributions
        discovered_states = self._extract_discovered_states(discovered_machine)
        
        # Extract all ground truth states from all machines
        ground_truth_states = self._extract_ground_truth_states(ground_truth_machines)
        
        if not discovered_states or not ground_truth_states:
            return self._empty_result()
        
        # Find best matches for each discovered state
        state_mappings = []
        js_divergences = []
        
        for disc_state in discovered_states:
            best_match = self._find_best_match(disc_state, ground_truth_states)
            state_mappings.append(best_match)
            js_divergences.append(best_match['js_divergence'])
        
        # Compute aggregate metrics
        return self._compute_aggregate_metrics(state_mappings, js_divergences, ground_truth_states)
    
    def _extract_discovered_states(self, discovered_machine: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract state information from discovered machine."""
        states = []
        
        # Handle different possible formats of CSSR results
        if 'cssr_results' in discovered_machine:
            # Navigate to actual states data
            for param_key, param_data in discovered_machine['cssr_results']['parameter_results'].items():
                if 'discovered_structure' in param_data and 'states' in param_data['discovered_structure']:
                    state_data = param_data['discovered_structure']['states']
                    for state_name, state_info in state_data.items():
                        if 'symbol_distribution' in state_info:
                            # Convert counts to probabilities
                            symbol_counts = state_info['symbol_distribution']
                            total_count = sum(symbol_counts.values())
                            if total_count > 0:
                                symbol_probs = {
                                    symbol: count / total_count 
                                    for symbol, count in symbol_counts.items()
                                }
                                states.append({
                                    'state_name': state_name,
                                    'distribution': symbol_probs,
                                    'observations': state_info.get('total_observations', 0),
                                    'entropy': state_info.get('entropy', 0.0)
                                })
                    break  # Use first parameter set found
        elif 'states' in discovered_machine:
            # Direct states format
            for state_name, state_info in discovered_machine['states'].items():
                if 'symbol_distribution' in state_info:
                    states.append({
                        'state_name': state_name,
                        'distribution': state_info['symbol_distribution'],
                        'observations': state_info.get('observations', 0),
                        'entropy': state_info.get('entropy', 0.0)
                    })
        
        return states
    
    def _extract_ground_truth_states(self, ground_truth_machines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all states from ground truth machines with proper machine tracking."""
        states = []
        
        for machine in ground_truth_machines:
            machine_id = machine.get('machine_id', 'unknown')
            machine_states = machine.get('states', {})
            
            for state_id, state_info in machine_states.items():
                distribution = state_info.get('distribution', {})
                states.append({
                    'machine_id': machine_id,
                    'state_id': state_id,
                    'distribution': distribution,
                    'full_state_name': f"{machine_id}_{state_id}",
                    'properties': machine.get('properties', {})
                })
        
        return states
    
    def _find_best_match(self, discovered_state: Dict[str, Any], 
                        ground_truth_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best matching ground truth state for a discovered state."""
        best_match = None
        min_divergence = float('inf')
        
        for gt_state in ground_truth_states:
            divergence = self._jensen_shannon_divergence(
                discovered_state['distribution'],
                gt_state['distribution']
            )
            
            if divergence < min_divergence:
                min_divergence = divergence
                best_match = gt_state
        
        # Compute distance rank (how many states have smaller divergence)
        divergences = [
            self._jensen_shannon_divergence(discovered_state['distribution'], gt_state['distribution'])
            for gt_state in ground_truth_states
        ]
        distance_rank = sum(1 for d in divergences if d < min_divergence) + 1
        
        return {
            'discovered_state': discovered_state['state_name'],
            'best_match': {
                'machine_id': best_match['machine_id'],
                'state_id': best_match['state_id'],
                'full_name': best_match['full_state_name']
            },
            'js_divergence': min_divergence,
            'distance_rank': distance_rank,
            'discovered_distribution': discovered_state['distribution'],
            'matched_distribution': best_match['distribution'],
            'discovered_observations': discovered_state.get('observations', 0),
            'discovered_entropy': discovered_state.get('entropy', 0.0)
        }
    
    def _jensen_shannon_divergence(self, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
        """Compute Jensen-Shannon divergence between two probability distributions."""
        # Get all symbols from both distributions
        all_symbols = set(dist1.keys()) | set(dist2.keys())
        
        if not all_symbols:
            return 0.0
        
        # Convert to arrays with consistent ordering
        symbols = sorted(all_symbols)
        p = np.array([dist1.get(symbol, 0.0) for symbol in symbols])
        q = np.array([dist2.get(symbol, 0.0) for symbol in symbols])
        
        # Normalize to ensure they sum to 1
        if p.sum() > 0:
            p = p / p.sum()
        if q.sum() > 0:
            q = q / q.sum()
        
        # Compute Jensen-Shannon divergence
        try:
            return jensenshannon(p, q)
        except (ValueError, RuntimeWarning):
            # Handle edge cases
            return 1.0
    
    def _compute_aggregate_metrics(self, state_mappings: List[Dict[str, Any]], 
                                  js_divergences: List[float],
                                  ground_truth_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate distance metrics."""
        if not js_divergences:
            return self._empty_result()
        
        js_array = np.array(js_divergences)
        
        # Basic statistics
        average_js = np.mean(js_array)
        max_js = np.max(js_array)
        min_js = np.min(js_array)
        std_js = np.std(js_array)
        
        # Coverage analysis
        coverage_score = self._compute_coverage_score(state_mappings, ground_truth_states)
        
        # Bidirectional analysis: find best discovered match for each true state
        reverse_mappings = self._compute_reverse_mappings(state_mappings, ground_truth_states)
        
        return {
            'average_js_divergence': float(average_js),
            'max_js_divergence': float(max_js),
            'min_js_divergence': float(min_js),
            'std_js_divergence': float(std_js),
            'state_mappings': state_mappings,
            'coverage_score': coverage_score,
            'reverse_mappings': reverse_mappings,
            'quality_assessment': self._assess_quality(js_divergences, coverage_score)
        }
    
    def _compute_coverage_score(self, state_mappings: List[Dict[str, Any]], 
                               ground_truth_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute how well ground truth states are covered by discovered states."""
        # Track which ground truth states have good matches
        matched_gt_states = set()
        good_match_threshold = 0.3  # JS divergence threshold for "good" match
        
        for mapping in state_mappings:
            if mapping['js_divergence'] < good_match_threshold:
                matched_gt_states.add(mapping['best_match']['full_name'])
        
        total_gt_states = len(ground_truth_states)
        well_matched_count = len(matched_gt_states)
        
        return {
            'total_ground_truth_states': total_gt_states,
            'well_matched_states': well_matched_count,
            'coverage_fraction': well_matched_count / total_gt_states if total_gt_states > 0 else 0.0,
            'unmatched_states': [
                gt_state['full_state_name'] 
                for gt_state in ground_truth_states 
                if gt_state['full_state_name'] not in matched_gt_states
            ]
        }
    
    def _compute_reverse_mappings(self, state_mappings: List[Dict[str, Any]], 
                                 ground_truth_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find best discovered state match for each ground truth state."""
        reverse_mappings = []
        
        for gt_state in ground_truth_states:
            best_discovered = None
            min_divergence = float('inf')
            
            for mapping in state_mappings:
                if mapping['best_match']['full_name'] == gt_state['full_state_name']:
                    if mapping['js_divergence'] < min_divergence:
                        min_divergence = mapping['js_divergence']
                        best_discovered = mapping['discovered_state']
            
            reverse_mappings.append({
                'ground_truth_state': gt_state['full_state_name'],
                'machine_id': gt_state['machine_id'],
                'state_id': gt_state['state_id'],
                'best_discovered_match': best_discovered,
                'js_divergence': min_divergence if best_discovered else float('inf'),
                'has_match': best_discovered is not None
            })
        
        return reverse_mappings
    
    def _assess_quality(self, js_divergences: List[float], coverage_score: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of the symbol distribution matching."""
        avg_divergence = np.mean(js_divergences)
        
        # Quality score: combination of low divergence and good coverage
        divergence_quality = 1.0 - avg_divergence  # JS divergence is [0,1]
        coverage_quality = coverage_score['coverage_fraction']
        
        overall_quality = (divergence_quality + coverage_quality) / 2.0
        
        # Confidence based on consistency
        consistency = 1.0 / (1.0 + np.std(js_divergences))
        
        return {
            'overall_quality_score': max(0.0, overall_quality),
            'divergence_quality': max(0.0, divergence_quality),
            'coverage_quality': coverage_quality,
            'consistency_score': consistency,
            'assessment': self._quality_assessment_text(overall_quality)
        }
    
    def _quality_assessment_text(self, quality_score: float) -> str:
        """Provide textual assessment of quality score."""
        if quality_score >= 0.8:
            return "Excellent match"
        elif quality_score >= 0.6:
            return "Good match"
        elif quality_score >= 0.4:
            return "Fair match"
        elif quality_score >= 0.2:
            return "Poor match"
        else:
            return "Very poor match"
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'average_js_divergence': 0.0,
            'max_js_divergence': 0.0,
            'min_js_divergence': 0.0,
            'std_js_divergence': 0.0,
            'state_mappings': [],
            'coverage_score': {
                'total_ground_truth_states': 0,
                'well_matched_states': 0,
                'coverage_fraction': 0.0,
                'unmatched_states': []
            },
            'reverse_mappings': [],
            'quality_assessment': {
                'overall_quality_score': 0.0,
                'divergence_quality': 0.0,
                'coverage_quality': 0.0,
                'consistency_score': 0.0,
                'assessment': "No data"
            }
        }