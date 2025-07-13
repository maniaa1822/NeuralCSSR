"""Optimality analysis for ε-machine properties: minimality, unifilarity, causal sufficiency."""

import math
from typing import Dict, List, Any


class OptimalityAnalysis:
    """
    Assess optimality properties: minimality, unifilarity, causal sufficiency.
    Based on theoretical optimality criteria from computational mechanics.
    
    Key Properties:
    - Minimality: Fewest states among all equally predictive models
    - Unifilarity: H[S_{t+1}|Y_t, S_t] = 0 (deterministic state transitions)
    - Causal Sufficiency: P(→Y|←Y) = P(→Y|S) (causal states contain all predictive information)
    """
    
    def compute(self, discovered_machine: Dict[str, Any], 
                ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess optimality properties of discovered machine against ground truth.
        
        Args:
            discovered_machine: CSSR results with state distributions
            ground_truth_machines: Theoretical ε-machines from dataset
            
        Returns:
            Dictionary containing optimality analysis
        """
        # Extract machine information
        discovered_info = self._extract_machine_info(discovered_machine, is_discovered=True)
        ground_truth_info = [self._extract_machine_info(gt, is_discovered=False) for gt in ground_truth_machines]
        
        if not discovered_info or not ground_truth_info:
            return self._empty_result()
        
        # Find best matching ground truth machine
        best_gt_match = self._find_best_ground_truth_match(discovered_info, ground_truth_info)
        
        # Compute optimality metrics
        minimality_score = self._assess_minimality(discovered_info, best_gt_match)
        unifilarity_score = self._assess_unifilarity(discovered_info)
        causal_sufficiency_score = self._assess_causal_sufficiency(discovered_info, best_gt_match)
        prediction_optimality = self._assess_prediction_optimality(discovered_info, best_gt_match)
        
        # Compute theoretical bounds
        theoretical_bounds = self._compute_theoretical_bounds(ground_truth_info)
        
        return {
            'minimality_score': minimality_score,
            'unifilarity_score': unifilarity_score,
            'causal_sufficiency_score': causal_sufficiency_score,
            'prediction_optimality': prediction_optimality,
            'theoretical_bounds': theoretical_bounds,
            'discovered_machine_info': discovered_info,
            'best_ground_truth_match': best_gt_match,
            'overall_optimality_score': self._compute_overall_optimality(
                minimality_score, unifilarity_score, causal_sufficiency_score, prediction_optimality
            ),
            'optimality_assessment': self._assess_overall_optimality(
                minimality_score, unifilarity_score, causal_sufficiency_score, prediction_optimality
            )
        }
    
    def _extract_machine_info(self, machine_data: Dict[str, Any], is_discovered: bool = True) -> Dict[str, Any]:
        """Extract key information about a machine."""
        
        if is_discovered:
            return self._extract_discovered_info(machine_data)
        else:
            return self._extract_ground_truth_info(machine_data)
    
    def _extract_discovered_info(self, discovered_machine: Dict[str, Any]) -> Dict[str, Any]:
        """Extract information from discovered machine (CSSR results)."""
        info = {
            'states': {},
            'num_states': 0,
            'total_observations': 0,
            'statistical_complexity': 0.0,
            'entropy_rate': 0.0,
            'state_entropies': {},
            'prediction_entropies': {}
        }
        
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
                                
                                # Compute entropy for this state's predictions
                                prediction_entropy = self._compute_entropy(symbol_probs)
                                
                                info['states'][state_name] = {
                                    'distribution': symbol_probs,
                                    'observations': total_count,
                                    'prediction_entropy': prediction_entropy
                                }
                                info['total_observations'] += total_count
                                info['prediction_entropies'][state_name] = prediction_entropy
                    
                    info['num_states'] = len(info['states'])
                    
                    # Compute overall measures
                    if info['total_observations'] > 0:
                        info['statistical_complexity'] = self._compute_statistical_complexity(info['states'], info['total_observations'])
                        info['entropy_rate'] = self._compute_entropy_rate(info['states'], info['total_observations'])
                    
                    break  # Use first parameter set found
                    
        elif 'states' in discovered_machine:
            # Direct states format
            for state_name, state_info in discovered_machine['states'].items():
                if 'symbol_distribution' in state_info:
                    distribution = state_info['symbol_distribution']
                    observations = state_info.get('observations', 0)
                    prediction_entropy = self._compute_entropy(distribution)
                    
                    info['states'][state_name] = {
                        'distribution': distribution,
                        'observations': observations,
                        'prediction_entropy': prediction_entropy
                    }
                    info['total_observations'] += observations
                    info['prediction_entropies'][state_name] = prediction_entropy
            
            info['num_states'] = len(info['states'])
            if info['total_observations'] > 0:
                info['statistical_complexity'] = self._compute_statistical_complexity(info['states'], info['total_observations'])
                info['entropy_rate'] = self._compute_entropy_rate(info['states'], info['total_observations'])
        
        return info
    
    def _extract_ground_truth_info(self, ground_truth_machine: Dict[str, Any]) -> Dict[str, Any]:
        """Extract information from ground truth machine."""
        
        info = {
            'machine_id': ground_truth_machine.get('machine_id', 'unknown'),
            'states': {},
            'num_states': 0,
            'statistical_complexity': 0.0,
            'entropy_rate': 0.0,
            'prediction_entropies': {}
        }
        
        # Get properties if available
        properties = ground_truth_machine.get('properties', {})
        info['statistical_complexity'] = properties.get('statistical_complexity', 0.0)
        info['entropy_rate'] = properties.get('entropy_rate', 0.0)
        
        # Process states
        machine_states = ground_truth_machine.get('states', {})
        samples_count = ground_truth_machine.get('samples_count', 1000)
        
        for state_id, state_info in machine_states.items():
            distribution = state_info.get('distribution', {})
            prediction_entropy = self._compute_entropy(distribution)
            
            # Estimate observations (uniform distribution across states)
            estimated_observations = samples_count // len(machine_states) if machine_states else 0
            
            info['states'][f"{info['machine_id']}_{state_id}"] = {
                'distribution': distribution,
                'observations': estimated_observations,
                'prediction_entropy': prediction_entropy
            }
            info['prediction_entropies'][f"{info['machine_id']}_{state_id}"] = prediction_entropy
        
        info['num_states'] = len(info['states'])
        
        # If not provided, compute from states
        if info['statistical_complexity'] == 0.0 and info['states']:
            info['statistical_complexity'] = math.log2(info['num_states']) if info['num_states'] > 0 else 0.0
        
        if info['entropy_rate'] == 0.0 and info['states']:
            # Uniform state probabilities assumed
            total_entropy = sum(info['prediction_entropies'].values())
            info['entropy_rate'] = total_entropy / info['num_states'] if info['num_states'] > 0 else 0.0
        
        return info
    
    def _find_best_ground_truth_match(self, discovered_info: Dict[str, Any], 
                                     ground_truth_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find ground truth machine that best matches discovered machine."""
        
        best_match = None
        best_score = float('inf')
        
        for gt_info in ground_truth_info:
            # Compute distance based on fundamental measures
            complexity_diff = abs(discovered_info['statistical_complexity'] - gt_info['statistical_complexity'])
            entropy_rate_diff = abs(discovered_info['entropy_rate'] - gt_info['entropy_rate'])
            state_count_diff = abs(discovered_info['num_states'] - gt_info['num_states'])
            
            # Combined distance
            total_distance = complexity_diff + entropy_rate_diff + 0.1 * state_count_diff
            
            if total_distance < best_score:
                best_score = total_distance
                best_match = gt_info
        
        return best_match or ground_truth_info[0] if ground_truth_info else {}
    
    def _assess_minimality(self, discovered_info: Dict[str, Any], 
                          best_gt_match: Dict[str, Any]) -> float:
        """
        Assess minimality: fewest states for given prediction quality.
        
        Perfect minimality score = 1.0 when discovered states == optimal states.
        Score decreases with excess states.
        """
        if not best_gt_match:
            return 0.0
        
        optimal_states = best_gt_match.get('num_states', 1)
        discovered_states = discovered_info.get('num_states', 0)
        
        if optimal_states == 0:
            return 1.0 if discovered_states == 0 else 0.0
        
        # Minimality score: penalize excess states
        if discovered_states <= optimal_states:
            # Perfect or under-specified
            minimality_score = discovered_states / optimal_states
        else:
            # Over-specified: penalize excess states
            excess_penalty = (discovered_states - optimal_states) / optimal_states
            minimality_score = 1.0 / (1.0 + excess_penalty)
        
        return min(1.0, max(0.0, minimality_score))
    
    def _assess_unifilarity(self, discovered_info: Dict[str, Any]) -> float:
        """
        Assess unifilarity: H[S_{t+1}|Y_t, S_t] ≈ 0.
        
        True ε-machines have deterministic state transitions.
        We approximate this by measuring consistency in state predictions.
        """
        states = discovered_info.get('states', {})
        
        if not states:
            return 0.0
        
        # Measure prediction consistency across states
        prediction_entropies = [state_info['prediction_entropy'] for state_info in states.values()]
        
        if not prediction_entropies:
            return 1.0
        
        # Unifilarity is high when prediction entropies are consistent
        # (indicating deterministic structure)
        avg_entropy = sum(prediction_entropies) / len(prediction_entropies)
        entropy_variance = sum((h - avg_entropy) ** 2 for h in prediction_entropies) / len(prediction_entropies)
        
        # Unifilarity score: high when variance is low
        unifilarity_score = 1.0 / (1.0 + entropy_variance)
        
        return min(1.0, max(0.0, unifilarity_score))
    
    def _assess_causal_sufficiency(self, discovered_info: Dict[str, Any], 
                                  best_gt_match: Dict[str, Any]) -> float:
        """
        Assess causal sufficiency: P(→Y|←Y) ≈ P(→Y|S).
        
        Causal states should capture all predictive information.
        """
        if not best_gt_match:
            return 0.0
        
        # Compare entropy rates as a proxy for causal sufficiency
        discovered_entropy_rate = discovered_info.get('entropy_rate', 0.0)
        optimal_entropy_rate = best_gt_match.get('entropy_rate', 0.0)
        
        if optimal_entropy_rate == 0.0:
            return 1.0 if discovered_entropy_rate == 0.0 else 0.0
        
        # Causal sufficiency score based on entropy rate match
        entropy_rate_ratio = discovered_entropy_rate / optimal_entropy_rate
        
        # Score is high when entropy rates match closely
        if 0.9 <= entropy_rate_ratio <= 1.1:
            causal_sufficiency_score = 1.0
        else:
            difference = abs(entropy_rate_ratio - 1.0)
            causal_sufficiency_score = 1.0 / (1.0 + difference)
        
        return min(1.0, max(0.0, causal_sufficiency_score))
    
    def _assess_prediction_optimality(self, discovered_info: Dict[str, Any], 
                                    best_gt_match: Dict[str, Any]) -> float:
        """Assess how well the discovered machine predicts compared to optimal."""
        
        if not best_gt_match:
            return 0.0
        
        # Compare statistical complexities (information storage efficiency)
        discovered_complexity = discovered_info.get('statistical_complexity', 0.0)
        optimal_complexity = best_gt_match.get('statistical_complexity', 0.0)
        
        if optimal_complexity == 0.0:
            return 1.0 if discovered_complexity == 0.0 else 0.0
        
        # Prediction optimality: closer to optimal complexity is better
        complexity_ratio = discovered_complexity / optimal_complexity
        
        if 0.9 <= complexity_ratio <= 1.1:
            prediction_optimality = 1.0
        else:
            difference = abs(complexity_ratio - 1.0)
            prediction_optimality = 1.0 / (1.0 + difference)
        
        return min(1.0, max(0.0, prediction_optimality))
    
    def _compute_theoretical_bounds(self, ground_truth_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute theoretical bounds from ground truth machines."""
        
        if not ground_truth_info:
            return {
                'minimum_possible_states': 1,
                'optimal_statistical_complexity': 0.0,
                'optimal_entropy_rate': 0.0,
                'complexity_range': [0.0, 0.0],
                'entropy_rate_range': [0.0, 0.0]
            }
        
        # Extract measures from all ground truth machines
        complexities = [gt['statistical_complexity'] for gt in ground_truth_info]
        entropy_rates = [gt['entropy_rate'] for gt in ground_truth_info]
        state_counts = [gt['num_states'] for gt in ground_truth_info]
        
        return {
            'minimum_possible_states': min(state_counts) if state_counts else 1,
            'optimal_statistical_complexity': min(complexities) if complexities else 0.0,
            'optimal_entropy_rate': min(entropy_rates) if entropy_rates else 0.0,
            'complexity_range': [min(complexities), max(complexities)] if complexities else [0.0, 0.0],
            'entropy_rate_range': [min(entropy_rates), max(entropy_rates)] if entropy_rates else [0.0, 0.0],
            'state_count_range': [min(state_counts), max(state_counts)] if state_counts else [1, 1]
        }
    
    def _compute_overall_optimality(self, minimality: float, unifilarity: float, 
                                   causal_sufficiency: float, prediction_optimality: float) -> float:
        """Compute weighted overall optimality score."""
        
        # Weighted combination of optimality measures
        weights = {
            'minimality': 0.3,
            'unifilarity': 0.2,
            'causal_sufficiency': 0.3,
            'prediction_optimality': 0.2
        }
        
        overall_score = (
            weights['minimality'] * minimality +
            weights['unifilarity'] * unifilarity +
            weights['causal_sufficiency'] * causal_sufficiency +
            weights['prediction_optimality'] * prediction_optimality
        )
        
        return overall_score
    
    def _assess_overall_optimality(self, minimality: float, unifilarity: float,
                                  causal_sufficiency: float, prediction_optimality: float) -> Dict[str, Any]:
        """Provide qualitative assessment of optimality."""
        
        overall_score = self._compute_overall_optimality(minimality, unifilarity, causal_sufficiency, prediction_optimality)
        
        if overall_score >= 0.9:
            assessment_level = "excellent"
            description = "Discovered machine exhibits excellent ε-machine optimality properties"
        elif overall_score >= 0.75:
            assessment_level = "good"
            description = "Discovered machine shows good ε-machine optimality properties"
        elif overall_score >= 0.6:
            assessment_level = "fair"
            description = "Discovered machine has fair ε-machine optimality properties"
        elif overall_score >= 0.4:
            assessment_level = "poor"
            description = "Discovered machine shows poor ε-machine optimality properties"
        else:
            assessment_level = "very_poor"
            description = "Discovered machine exhibits very poor ε-machine optimality properties"
        
        # Specific recommendations
        recommendations = []
        if minimality < 0.7:
            recommendations.append("Consider higher significance levels to reduce over-splitting")
        if unifilarity < 0.7:
            recommendations.append("State transitions may not be sufficiently deterministic")
        if causal_sufficiency < 0.7:
            recommendations.append("Consider longer history lengths to capture more predictive information")
        if prediction_optimality < 0.7:
            recommendations.append("Prediction performance may be suboptimal - check CSSR parameters")
        
        return {
            'overall_score': overall_score,
            'assessment_level': assessment_level,
            'description': description,
            'component_scores': {
                'minimality': minimality,
                'unifilarity': unifilarity,
                'causal_sufficiency': causal_sufficiency,
                'prediction_optimality': prediction_optimality
            },
            'recommendations': recommendations
        }
    
    def _compute_entropy(self, distribution: Dict[str, float]) -> float:
        """Compute entropy of a probability distribution."""
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def _compute_statistical_complexity(self, states: Dict[str, Any], total_observations: int) -> float:
        """Compute statistical complexity C_μ = H[S]."""
        if total_observations == 0:
            return 0.0
        
        complexity = 0.0
        for state_info in states.values():
            state_prob = state_info['observations'] / total_observations
            if state_prob > 0:
                complexity -= state_prob * math.log2(state_prob)
        
        return complexity
    
    def _compute_entropy_rate(self, states: Dict[str, Any], total_observations: int) -> float:
        """Compute entropy rate h_μ = H[Y|S]."""
        if total_observations == 0:
            return 0.0
        
        entropy_rate = 0.0
        for state_info in states.values():
            state_prob = state_info['observations'] / total_observations
            state_entropy = state_info['prediction_entropy']
            entropy_rate += state_prob * state_entropy
        
        return entropy_rate
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'minimality_score': 0.0,
            'unifilarity_score': 0.0,
            'causal_sufficiency_score': 0.0,
            'prediction_optimality': 0.0,
            'theoretical_bounds': {
                'minimum_possible_states': 1,
                'optimal_statistical_complexity': 0.0,
                'optimal_entropy_rate': 0.0,
                'complexity_range': [0.0, 0.0],
                'entropy_rate_range': [0.0, 0.0]
            },
            'discovered_machine_info': {},
            'best_ground_truth_match': {},
            'overall_optimality_score': 0.0,
            'optimality_assessment': {
                'overall_score': 0.0,
                'assessment_level': 'no_data',
                'description': 'No data available for optimality analysis',
                'component_scores': {
                    'minimality': 0.0,
                    'unifilarity': 0.0,
                    'causal_sufficiency': 0.0,
                    'prediction_optimality': 0.0
                },
                'recommendations': ['No data available for analysis']
            }
        }