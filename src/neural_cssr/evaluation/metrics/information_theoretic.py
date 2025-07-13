"""Information-theoretic distance using fundamental ε-machine measures."""

import math
from typing import Dict, List, Any, Tuple


class InformationTheoreticDistance:
    """
    Compute distance using fundamental ε-machine information measures.
    Based on C_μ, h_μ, and E from Crutchfield & Barnett computational mechanics theory.
    
    Mathematical Foundation:
    - Statistical Complexity: C_μ = H[S] = -Σ π(σ_i) log₂ π(σ_i)
    - Entropy Rate: h_μ = H[Y₀|S₀] = -Σᵢ π(σᵢ) Σᵧ P(y|σᵢ) log₂ P(y|σᵢ)
    - Excess Entropy: E = C_μ - h_μ
    """
    
    def compute(self, discovered_machine: Dict[str, Any], 
                ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute information-theoretic distances between discovered and ground truth machines.
        
        Args:
            discovered_machine: CSSR results with state distributions
            ground_truth_machines: Theoretical ε-machines from dataset
            
        Returns:
            Dictionary containing all information-theoretic distance measures
        """
        # Extract discovered machine measures
        discovered_measures = self._compute_machine_measures(discovered_machine, is_discovered=True)
        
        # Compute measures for all ground truth machines
        gt_measures_list = []
        for gt_machine in ground_truth_machines:
            gt_measures = self._compute_machine_measures(gt_machine, is_discovered=False)
            gt_measures['machine_id'] = gt_machine.get('machine_id', 'unknown')
            gt_measures_list.append(gt_measures)
        
        # Find best matching ground truth machine
        best_match, distances = self._find_best_match(discovered_measures, gt_measures_list)
        
        # Compute combined measures
        combined_measures = self._compute_combined_measures(discovered_measures, best_match, distances)
        
        return {
            'discovered_measures': discovered_measures,
            'ground_truth_measures': gt_measures_list,
            'best_match': best_match,
            'distances': distances,
            'combined_information_distance': combined_measures['combined_distance'],
            'theoretical_optimality_ratio': combined_measures['optimality_ratio'],
            'component_measures': {
                'discovered': {
                    'C_mu': discovered_measures['statistical_complexity'],
                    'h_mu': discovered_measures['entropy_rate'],
                    'E': discovered_measures['excess_entropy']
                },
                'ground_truth': {
                    'C_mu': best_match['statistical_complexity'],
                    'h_mu': best_match['entropy_rate'],
                    'E': best_match['excess_entropy']
                }
            },
            'quality_assessment': self._assess_quality(distances, combined_measures)
        }
    
    def _compute_machine_measures(self, machine_data: Dict[str, Any], 
                                 is_discovered: bool = True) -> Dict[str, Any]:
        """Compute fundamental information measures for a machine."""
        
        if is_discovered:
            states_data = self._extract_discovered_states(machine_data)
        else:
            states_data = self._extract_ground_truth_states(machine_data)
        
        if not states_data:
            return self._empty_measures()
        
        # Compute statistical complexity C_μ = H[S]
        statistical_complexity = self._compute_statistical_complexity(states_data)
        
        # Compute entropy rate h_μ = H[Y|S]
        entropy_rate = self._compute_entropy_rate(states_data)
        
        # Compute excess entropy E = C_μ - h_μ
        excess_entropy = statistical_complexity - entropy_rate
        
        return {
            'statistical_complexity': statistical_complexity,
            'entropy_rate': entropy_rate,
            'excess_entropy': excess_entropy,
            'num_states': len(states_data),
            'total_observations': sum(state['observations'] for state in states_data)
        }
    
    def _extract_discovered_states(self, discovered_machine: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract state information from discovered machine (CSSR results)."""
        states = []
        
        if 'cssr_results' in discovered_machine:
            # Navigate to actual states data from CSSR results
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
                                    'observations': total_count,
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
    
    def _extract_ground_truth_states(self, ground_truth_machine: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract state information from ground truth machine."""
        states = []
        
        machine_states = ground_truth_machine.get('states', {})
        samples_count = ground_truth_machine.get('samples_count', 1000)
        
        # Estimate state probabilities (uniform distribution as default)
        num_states = len(machine_states)
        state_prob = 1.0 / num_states if num_states > 0 else 0.0
        estimated_observations = int(samples_count * state_prob)
        
        for state_id, state_info in machine_states.items():
            distribution = state_info.get('distribution', {})
            
            states.append({
                'state_name': f"{ground_truth_machine.get('machine_id', 'unknown')}_{state_id}",
                'distribution': distribution,
                'observations': estimated_observations,  # Estimated
                'entropy': self._compute_entropy(distribution)
            })
        
        return states
    
    def _compute_statistical_complexity(self, states_data: List[Dict[str, Any]]) -> float:
        """
        Compute statistical complexity C_μ = H[S] = -Σ π(σᵢ) log₂ π(σᵢ)
        
        This measures the entropy of the causal state distribution.
        """
        total_observations = sum(state['observations'] for state in states_data)
        
        if total_observations == 0:
            return 0.0
        
        # Compute stationary probabilities
        stationary_probs = [
            state['observations'] / total_observations 
            for state in states_data
        ]
        
        # Compute entropy
        complexity = 0.0
        for prob in stationary_probs:
            if prob > 0:
                complexity -= prob * math.log2(prob)
        
        return complexity
    
    def _compute_entropy_rate(self, states_data: List[Dict[str, Any]]) -> float:
        """
        Compute entropy rate h_μ = H[Y₀|S₀] = -Σᵢ π(σᵢ) Σᵧ P(y|σᵢ) log₂ P(y|σᵢ)
        
        This measures the conditional entropy of symbols given causal states.
        """
        total_observations = sum(state['observations'] for state in states_data)
        
        if total_observations == 0:
            return 0.0
        
        entropy_rate = 0.0
        
        for state in states_data:
            # State probability in stationary distribution
            state_prob = state['observations'] / total_observations
            
            # Entropy of symbol distribution for this state
            symbol_entropy = self._compute_entropy(state['distribution'])
            
            # Weighted contribution to entropy rate
            entropy_rate += state_prob * symbol_entropy
        
        return entropy_rate
    
    def _compute_entropy(self, distribution: Dict[str, float]) -> float:
        """Compute entropy of a probability distribution."""
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def _find_best_match(self, discovered_measures: Dict[str, Any], 
                        gt_measures_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Find the ground truth machine that best matches the discovered machine."""
        
        best_match = None
        best_distances = None
        min_combined_distance = float('inf')
        
        for gt_measures in gt_measures_list:
            distances = {
                'statistical_complexity_distance': abs(
                    discovered_measures['statistical_complexity'] - gt_measures['statistical_complexity']
                ),
                'entropy_rate_distance': abs(
                    discovered_measures['entropy_rate'] - gt_measures['entropy_rate']
                ),
                'excess_entropy_distance': abs(
                    discovered_measures['excess_entropy'] - gt_measures['excess_entropy']
                )
            }
            
            # Combined distance (weighted)
            combined_distance = (
                0.4 * distances['statistical_complexity_distance'] +
                0.4 * distances['entropy_rate_distance'] +
                0.2 * distances['excess_entropy_distance']
            )
            
            if combined_distance < min_combined_distance:
                min_combined_distance = combined_distance
                best_match = gt_measures
                best_distances = distances
                best_distances['combined_distance'] = combined_distance
        
        return best_match or gt_measures_list[0], best_distances or {}
    
    def _compute_combined_measures(self, discovered_measures: Dict[str, Any], 
                                  best_match: Dict[str, Any], 
                                  distances: Dict[str, float]) -> Dict[str, Any]:
        """Compute combined quality and optimality measures."""
        
        # Optimality ratio: how close to theoretical optimum
        gt_complexity = best_match['statistical_complexity']
        discovered_complexity = discovered_measures['statistical_complexity']
        
        # Optimality ratio (1.0 = perfect, <1.0 = suboptimal, >1.0 = over-complex)
        optimality_ratio = discovered_complexity / gt_complexity if gt_complexity > 0 else 1.0
        
        # Combined distance (normalized)
        max_possible_distance = max(gt_complexity, discovered_complexity, 1.0)
        normalized_distance = distances.get('combined_distance', 0.0) / max_possible_distance
        
        return {
            'combined_distance': distances.get('combined_distance', 0.0),
            'normalized_distance': normalized_distance,
            'optimality_ratio': optimality_ratio
        }
    
    def _assess_quality(self, distances: Dict[str, float], 
                       combined_measures: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the information-theoretic match."""
        
        # Quality based on combined distance (lower is better)
        combined_distance = distances.get('combined_distance', float('inf'))
        
        if combined_distance < 0.1:
            quality_level = "excellent"
            quality_score = 0.95
        elif combined_distance < 0.3:
            quality_level = "good"  
            quality_score = 0.8
        elif combined_distance < 0.6:
            quality_level = "fair"
            quality_score = 0.6
        elif combined_distance < 1.0:
            quality_level = "poor"
            quality_score = 0.3
        else:
            quality_level = "very_poor"
            quality_score = 0.1
        
        # Optimality assessment
        optimality_ratio = combined_measures.get('optimality_ratio', 1.0)
        if 0.9 <= optimality_ratio <= 1.1:
            optimality_assessment = "optimal complexity"
        elif optimality_ratio < 0.9:
            optimality_assessment = "under-complex (missing states)"
        else:
            optimality_assessment = "over-complex (extra states)"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'optimality_assessment': optimality_assessment,
            'distances_summary': {
                'C_mu_distance': distances.get('statistical_complexity_distance', 0.0),
                'h_mu_distance': distances.get('entropy_rate_distance', 0.0),
                'E_distance': distances.get('excess_entropy_distance', 0.0)
            }
        }
    
    def _empty_measures(self) -> Dict[str, Any]:
        """Return empty measures structure."""
        return {
            'statistical_complexity': 0.0,
            'entropy_rate': 0.0,
            'excess_entropy': 0.0,
            'num_states': 0,
            'total_observations': 0
        }