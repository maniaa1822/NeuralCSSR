"""Causal equivalence distance using computational mechanics equivalence relations."""

from typing import Dict, List, Any
import math
from scipy.spatial.distance import jensenshannon


class CausalEquivalenceDistance:
    """
    Analyze causal state structure using equivalence relations.
    Based on Definition 1: ←y ∼ ←y' ⟺ P(→Y|←Y = ←y) = P(→Y|←Y = ←y')
    
    This metric evaluates how well CSSR discovered states correspond to 
    theoretically optimal causal equivalence classes.
    """
    
    def compute(self, discovered_machine: Dict[str, Any], 
                ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze causal equivalence between discovered and ground truth machines.
        
        Args:
            discovered_machine: CSSR results with state distributions
            ground_truth_machines: Theoretical ε-machines from dataset
            
        Returns:
            Dictionary containing causal equivalence analysis
        """
        # Extract state information
        discovered_states = self._extract_discovered_states(discovered_machine)
        ground_truth_states = self._extract_ground_truth_states(ground_truth_machines)
        
        if not discovered_states or not ground_truth_states:
            return self._empty_result()
        
        # Compute equivalence mappings
        equivalence_mappings = self._compute_equivalence_mappings(discovered_states, ground_truth_states)
        
        # Analyze state refinement patterns
        refinement_analysis = self._analyze_state_refinement(discovered_states, ground_truth_states, equivalence_mappings)
        
        # Compute overall causal equivalence score
        causal_score = self._compute_causal_equivalence_score(equivalence_mappings, refinement_analysis)
        
        # Analyze future prediction consistency
        prediction_consistency = self._analyze_prediction_consistency(discovered_states, ground_truth_states)
        
        return {
            'causal_equivalence_score': causal_score,
            'future_prediction_consistency': prediction_consistency,
            'state_refinement_analysis': refinement_analysis,
            'equivalence_class_mapping': equivalence_mappings,
            'quality_assessment': self._assess_equivalence_quality(causal_score, prediction_consistency, refinement_analysis)
        }
    
    def _extract_discovered_states(self, discovered_machine: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract state information from discovered machine."""
        states = []
        
        if 'cssr_results' in discovered_machine:
            # Navigate to actual states data
            for param_data in discovered_machine['cssr_results']['parameter_results'].values():
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
                                    'observations': total_count
                                })
                    break  # Use first parameter set found
                    
        elif 'states' in discovered_machine:
            # Direct states format
            for state_name, state_info in discovered_machine['states'].items():
                if 'symbol_distribution' in state_info:
                    states.append({
                        'state_name': state_name,
                        'distribution': state_info['symbol_distribution'],
                        'observations': state_info.get('observations', 0)
                    })
        
        return states
    
    def _extract_ground_truth_states(self, ground_truth_machines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all states from ground truth machines."""
        states = []
        
        for machine in ground_truth_machines:
            machine_id = machine.get('machine_id', 'unknown')
            machine_states = machine.get('states', {})
            
            for state_id, state_info in machine_states.items():
                distribution = state_info.get('distribution', {})
                states.append({
                    'state_name': f"{machine_id}_{state_id}",
                    'machine_id': machine_id,
                    'state_id': state_id,
                    'distribution': distribution
                })
        
        return states
    
    def _compute_equivalence_mappings(self, discovered_states: List[Dict[str, Any]], 
                                    ground_truth_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute equivalence class mappings between discovered and ground truth states.
        
        For each discovered state, find the ground truth states that are 
        causally equivalent (similar future prediction distributions).
        """
        mappings = []
        
        for disc_state in discovered_states:
            equivalent_states = []
            
            for gt_state in ground_truth_states:
                # Compute similarity in future prediction (symbol distribution)
                similarity = self._compute_future_distribution_similarity(
                    disc_state['distribution'], 
                    gt_state['distribution']
                )
                
                equivalent_states.append({
                    'ground_truth_state': gt_state['state_name'],
                    'machine_id': gt_state['machine_id'],
                    'state_id': gt_state['state_id'],
                    'equivalence_strength': similarity,
                    'js_divergence': 1.0 - similarity  # Convert similarity to divergence
                })
            
            # Sort by equivalence strength (highest first)
            equivalent_states.sort(key=lambda x: x['equivalence_strength'], reverse=True)
            
            mappings.append({
                'discovered_state': disc_state['state_name'],
                'equivalent_ground_truth_states': equivalent_states,
                'best_match': equivalent_states[0] if equivalent_states else None,
                'num_strong_equivalences': sum(1 for s in equivalent_states if s['equivalence_strength'] > 0.8)
            })
        
        return mappings
    
    def _compute_future_distribution_similarity(self, dist1: Dict[str, float], 
                                              dist2: Dict[str, float]) -> float:
        """
        Compare P(→Y|state1) vs P(→Y|state2) using Jensen-Shannon divergence.
        
        If JS divergence ≈ 0, then states are causally equivalent.
        Returns similarity score (1 = identical, 0 = completely different).
        """
        # Get all symbols from both distributions
        all_symbols = set(dist1.keys()) | set(dist2.keys())
        
        if not all_symbols:
            return 1.0  # Both empty = equivalent
        
        # Convert to arrays with consistent ordering
        symbols = sorted(all_symbols)
        p = [dist1.get(symbol, 0.0) for symbol in symbols]
        q = [dist2.get(symbol, 0.0) for symbol in symbols]
        
        # Normalize to ensure they sum to 1
        p_sum = sum(p)
        q_sum = sum(q)
        if p_sum > 0:
            p = [x / p_sum for x in p]
        if q_sum > 0:
            q = [x / q_sum for x in q]
        
        # Compute Jensen-Shannon divergence
        try:
            js_div = jensenshannon(p, q)
            # Convert to similarity (1 - divergence)
            similarity = 1.0 - js_div
            return max(0.0, similarity)
        except (ValueError, RuntimeWarning):
            return 0.0
    
    def _analyze_state_refinement(self, discovered_states: List[Dict[str, Any]], 
                                 ground_truth_states: List[Dict[str, Any]],
                                 equivalence_mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze whether CSSR over-split, under-split, or correctly identified causal states.
        """
        # Count strong matches (equivalence strength > 0.8)
        correctly_identified = []
        over_split_states = []
        
        for mapping in equivalence_mappings:
            best_match = mapping.get('best_match')
            if best_match and best_match['equivalence_strength'] > 0.8:
                correctly_identified.append({
                    'discovered_state': mapping['discovered_state'],
                    'matched_ground_truth': best_match['ground_truth_state'],
                    'equivalence_strength': best_match['equivalence_strength']
                })
            elif best_match and best_match['equivalence_strength'] < 0.5:
                over_split_states.append(mapping['discovered_state'])
        
        # Check for under-splitting (multiple ground truth states mapping to same discovered state)
        gt_to_disc_mapping = {}
        for mapping in equivalence_mappings:
            best_match = mapping.get('best_match')
            if best_match and best_match['equivalence_strength'] > 0.7:
                gt_state = best_match['ground_truth_state']
                disc_state = mapping['discovered_state']
                
                if gt_state not in gt_to_disc_mapping:
                    gt_to_disc_mapping[gt_state] = []
                gt_to_disc_mapping[gt_state].append(disc_state)
        
        # Find ground truth states with no good matches (potential under-splitting)
        well_matched_gt_states = set(gt_to_disc_mapping.keys())
        all_gt_states = set(state['state_name'] for state in ground_truth_states)
        under_split_states = list(all_gt_states - well_matched_gt_states)
        
        return {
            'correctly_identified_states': correctly_identified,
            'over_split_states': over_split_states,
            'under_split_states': under_split_states,
            'num_discovered_states': len(discovered_states),
            'num_ground_truth_states': len(ground_truth_states),
            'state_count_ratio': len(discovered_states) / len(ground_truth_states) if ground_truth_states else 0,
            'refinement_summary': self._summarize_refinement(
                len(correctly_identified), len(over_split_states), len(under_split_states),
                len(discovered_states), len(ground_truth_states)
            )
        }
    
    def _summarize_refinement(self, correct: int, over_split: int, under_split: int,
                            num_discovered: int, num_ground_truth: int) -> Dict[str, Any]:
        """Summarize the state refinement analysis."""
        
        if num_discovered == num_ground_truth and correct == num_discovered:
            refinement_type = "optimal"
            description = "Perfect state identification - correct number and mapping"
        elif num_discovered > num_ground_truth:
            refinement_type = "over_refined"
            description = f"Over-splitting detected - {num_discovered} discovered vs {num_ground_truth} true states"
        elif num_discovered < num_ground_truth:
            refinement_type = "under_refined"  
            description = f"Under-splitting detected - {num_discovered} discovered vs {num_ground_truth} true states"
        else:
            refinement_type = "mixed"
            description = "Mixed refinement - some correct, some incorrect state identification"
        
        accuracy = correct / max(num_discovered, num_ground_truth) if max(num_discovered, num_ground_truth) > 0 else 0
        
        return {
            'refinement_type': refinement_type,
            'description': description,
            'accuracy': accuracy,
            'correct_identifications': correct,
            'total_states': max(num_discovered, num_ground_truth)
        }
    
    def _compute_causal_equivalence_score(self, equivalence_mappings: List[Dict[str, Any]],
                                        refinement_analysis: Dict[str, Any]) -> float:
        """Compute overall causal equivalence score."""
        
        if not equivalence_mappings:
            return 0.0
        
        # Average equivalence strength of best matches
        best_match_strengths = []
        for mapping in equivalence_mappings:
            best_match = mapping.get('best_match')
            if best_match:
                best_match_strengths.append(best_match['equivalence_strength'])
        
        avg_equivalence_strength = sum(best_match_strengths) / len(best_match_strengths) if best_match_strengths else 0.0
        
        # Refinement accuracy
        refinement_accuracy = refinement_analysis['refinement_summary']['accuracy']
        
        # Combined score (weighted average)
        causal_score = 0.7 * avg_equivalence_strength + 0.3 * refinement_accuracy
        
        return causal_score
    
    def _analyze_prediction_consistency(self, discovered_states: List[Dict[str, Any]], 
                                      ground_truth_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how consistently states predict future symbols."""
        
        # Compute prediction entropy for each state set
        discovered_entropy = self._compute_prediction_entropy(discovered_states)
        ground_truth_entropy = self._compute_prediction_entropy(ground_truth_states)
        
        # Consistency is measured by similarity in prediction entropy
        entropy_similarity = 1.0 - abs(discovered_entropy - ground_truth_entropy) / max(discovered_entropy, ground_truth_entropy, 1.0)
        
        return {
            'discovered_prediction_entropy': discovered_entropy,
            'ground_truth_prediction_entropy': ground_truth_entropy,
            'entropy_similarity': entropy_similarity,
            'consistency_score': entropy_similarity
        }
    
    def _compute_prediction_entropy(self, states: List[Dict[str, Any]]) -> float:
        """Compute average prediction entropy across states."""
        if not states:
            return 0.0
        
        entropies = []
        for state in states:
            distribution = state['distribution']
            entropy = 0.0
            for prob in distribution.values():
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            entropies.append(entropy)
        
        return sum(entropies) / len(entropies)
    
    def _assess_equivalence_quality(self, causal_score: float, prediction_consistency: Dict[str, Any],
                                   refinement_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of causal equivalence detection."""
        
        # Quality levels based on causal equivalence score
        if causal_score >= 0.9:
            quality_level = "excellent"
            description = "Excellent causal state identification - strong equivalence relationships detected"
        elif causal_score >= 0.75:
            quality_level = "good"
            description = "Good causal state identification - most equivalence relationships captured"
        elif causal_score >= 0.6:
            quality_level = "fair"
            description = "Fair causal state identification - some equivalence relationships detected"
        elif causal_score >= 0.4:
            quality_level = "poor"
            description = "Poor causal state identification - limited equivalence relationships"
        else:
            quality_level = "very_poor"
            description = "Very poor causal state identification - minimal equivalence detection"
        
        # Specific insights
        refinement_type = refinement_analysis['refinement_summary']['refinement_type']
        consistency_score = prediction_consistency['consistency_score']
        
        insights = []
        if refinement_type == "over_refined":
            insights.append("CSSR may have over-split states - consider higher significance levels")
        elif refinement_type == "under_refined":
            insights.append("CSSR may have under-split states - consider longer history lengths")
        
        if consistency_score < 0.7:
            insights.append("Low prediction consistency - discovered states may not capture optimal causal structure")
        
        return {
            'quality_score': causal_score,
            'quality_level': quality_level,
            'description': description,
            'refinement_type': refinement_type,
            'prediction_consistency': consistency_score,
            'insights': insights
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'causal_equivalence_score': 0.0,
            'future_prediction_consistency': {
                'discovered_prediction_entropy': 0.0,
                'ground_truth_prediction_entropy': 0.0,
                'entropy_similarity': 0.0,
                'consistency_score': 0.0
            },
            'state_refinement_analysis': {
                'correctly_identified_states': [],
                'over_split_states': [],
                'under_split_states': [],
                'num_discovered_states': 0,
                'num_ground_truth_states': 0,
                'state_count_ratio': 0.0,
                'refinement_summary': {
                    'refinement_type': 'unknown',
                    'description': 'No states found',
                    'accuracy': 0.0,
                    'correct_identifications': 0,
                    'total_states': 0
                }
            },
            'equivalence_class_mapping': [],
            'quality_assessment': {
                'quality_score': 0.0,
                'quality_level': 'no_data',
                'description': 'No data available for analysis',
                'refinement_type': 'unknown',
                'prediction_consistency': 0.0,
                'insights': ['No data available for causal equivalence analysis']
            }
        }