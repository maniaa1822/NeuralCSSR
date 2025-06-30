"""State mapping distance using Hungarian algorithm and Jensen-Shannon divergence."""

from typing import Dict, List, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon


class StateMappingDistance:
    """Compute optimal state mapping between discovered and ground truth machines."""
    
    def compute(self, discovered_machine: Dict[str, Any], ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find optimal assignment of discovered states to ground truth states using Hungarian algorithm.
        
        Args:
            discovered_machine: CSSR discovered machine with states and symbol distributions
            ground_truth_machines: List of ground truth machines from dataset
            
        Returns:
            Dictionary containing optimal assignment, costs, and unmatched states
        """
        # Extract discovered states and their distributions
        discovered_states = self._extract_discovered_states(discovered_machine)
        
        # Extract all ground truth states from all machines
        ground_truth_states = self._extract_ground_truth_states(ground_truth_machines)
        
        if not discovered_states or not ground_truth_states:
            return self._empty_result()
        
        # Create cost matrix using Jensen-Shannon divergence
        cost_matrix = self._create_cost_matrix(discovered_states, ground_truth_states)
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract results
        return self._extract_assignment_results(
            discovered_states, ground_truth_states, cost_matrix, row_indices, col_indices
        )
    
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
                    'full_state_name': f"{machine_id}_{state_id}"
                })
        
        return states
    
    def _create_cost_matrix(self, discovered_states: List[Dict[str, Any]], 
                           ground_truth_states: List[Dict[str, Any]]) -> np.ndarray:
        """Create cost matrix using Jensen-Shannon divergence between distributions."""
        n_discovered = len(discovered_states)
        n_ground_truth = len(ground_truth_states)
        
        # Create square matrix by padding with high-cost dummy states
        matrix_size = max(n_discovered, n_ground_truth)
        cost_matrix = np.full((matrix_size, matrix_size), 1.0)  # High cost for dummy assignments
        
        # Fill actual costs
        for i, disc_state in enumerate(discovered_states):
            for j, gt_state in enumerate(ground_truth_states):
                cost = self._jensen_shannon_divergence(
                    disc_state['distribution'], 
                    gt_state['distribution']
                )
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
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
    
    def _extract_assignment_results(self, discovered_states: List[Dict[str, Any]], 
                                   ground_truth_states: List[Dict[str, Any]],
                                   cost_matrix: np.ndarray,
                                   row_indices: np.ndarray, 
                                   col_indices: np.ndarray) -> Dict[str, Any]:
        """Extract detailed results from Hungarian algorithm assignment."""
        n_discovered = len(discovered_states)
        n_ground_truth = len(ground_truth_states)
        
        assignments = []
        total_cost = 0.0
        per_state_costs = []
        
        for row, col in zip(row_indices, col_indices):
            cost = cost_matrix[row, col]
            
            # Only include real assignments (not dummy states)
            if row < n_discovered and col < n_ground_truth:
                assignments.append({
                    'discovered_state': discovered_states[row]['state_name'],
                    'ground_truth_state': ground_truth_states[col]['full_state_name'],
                    'machine_id': ground_truth_states[col]['machine_id'],
                    'state_id': ground_truth_states[col]['state_id'],
                    'cost': cost,
                    'discovered_distribution': discovered_states[row]['distribution'],
                    'ground_truth_distribution': ground_truth_states[col]['distribution']
                })
                per_state_costs.append(cost)
                total_cost += cost
        
        # Find unmatched states
        matched_discovered = set(row_indices[row_indices < n_discovered])
        matched_ground_truth = set(col_indices[col_indices < n_ground_truth])
        
        unmatched_discovered = [
            discovered_states[i]['state_name'] 
            for i in range(n_discovered) 
            if i not in matched_discovered
        ]
        
        unmatched_ground_truth = [
            ground_truth_states[i]['full_state_name']
            for i in range(n_ground_truth)
            if i not in matched_ground_truth
        ]
        
        return {
            'optimal_assignment': assignments,
            'total_cost': total_cost,
            'average_cost': total_cost / len(assignments) if assignments else 0.0,
            'per_state_costs': per_state_costs,
            'unmatched_discovered_states': unmatched_discovered,
            'unmatched_true_states': unmatched_ground_truth,
            'assignment_quality': self._compute_assignment_quality(per_state_costs)
        }
    
    def _compute_assignment_quality(self, costs: List[float]) -> Dict[str, float]:
        """Compute quality metrics for the assignment."""
        if not costs:
            return {'quality_score': 0.0, 'confidence': 0.0}
        
        costs_array = np.array(costs)
        
        # Quality score: 1 - average_cost (since JS divergence is [0,1])
        quality_score = 1.0 - np.mean(costs_array)
        
        # Confidence: based on consistency of costs (lower std = higher confidence)
        confidence = 1.0 / (1.0 + np.std(costs_array))
        
        return {
            'quality_score': max(0.0, quality_score),
            'confidence': confidence
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'optimal_assignment': [],
            'total_cost': 0.0,
            'average_cost': 0.0,
            'per_state_costs': [],
            'unmatched_discovered_states': [],
            'unmatched_true_states': [],
            'assignment_quality': {'quality_score': 0.0, 'confidence': 0.0}
        }