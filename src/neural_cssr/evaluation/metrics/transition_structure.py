"""Transition structure distance using graph-based metrics."""

from typing import Dict, List, Any
import numpy as np
import networkx as nx
from collections import defaultdict


class TransitionStructureDistance:
    """Compare transition graph structure between discovered and ground truth machines."""
    
    def compute(self, discovered_machine: Dict[str, Any], ground_truth_machines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare transition graph connectivity patterns using graph metrics.
        
        Args:
            discovered_machine: CSSR discovered machine with states and transitions
            ground_truth_machines: List of ground truth machines from dataset
            
        Returns:
            Dictionary containing graph-based distance metrics
        """
        # Build transition graphs
        discovered_graph = self._build_discovered_graph(discovered_machine)
        ground_truth_graph = self._build_ground_truth_graph(ground_truth_machines)
        
        if discovered_graph.number_of_nodes() == 0 or ground_truth_graph.number_of_nodes() == 0:
            return self._empty_result()
        
        # Compute various graph distance metrics
        results = {}
        
        # Graph edit distance (approximate)
        results['graph_edit_distance'] = self._approximate_graph_edit_distance(
            discovered_graph, ground_truth_graph
        )
        
        # Spectral distance
        results['spectral_distance'] = self._compute_spectral_distance(
            discovered_graph, ground_truth_graph
        )
        
        # Connectivity analysis
        results['connectivity_similarity'] = self._compute_connectivity_similarity(
            discovered_graph, ground_truth_graph
        )
        
        # Transition coverage
        results['transition_coverage'] = self._compute_transition_coverage(
            discovered_graph, ground_truth_graph
        )
        
        # Structural properties comparison
        results['structural_comparison'] = self._compare_structural_properties(
            discovered_graph, ground_truth_graph
        )
        
        return results
    
    def _build_discovered_graph(self, discovered_machine: Dict[str, Any]) -> nx.DiGraph:
        """Build NetworkX graph from discovered machine structure."""
        graph = nx.DiGraph()
        
        # Handle different possible formats of CSSR results
        if 'cssr_results' in discovered_machine:
            # Navigate to actual states data
            for param_data in discovered_machine['cssr_results']['parameter_results'].values():
                if 'discovered_structure' in param_data and 'states' in param_data['discovered_structure']:
                    state_data = param_data['discovered_structure']['states']
                    
                    # Add nodes
                    for state_name, state_info in state_data.items():
                        graph.add_node(state_name, **{
                            'observations': state_info.get('total_observations', 0),
                            'entropy': state_info.get('entropy', 0.0)
                        })
                    
                    # Add edges based on transition patterns (inferred from histories)
                    self._infer_transitions_from_histories(graph, state_data)
                    break  # Use first parameter set found
        
        elif 'states' in discovered_machine and 'transitions' in discovered_machine:
            # Direct format with explicit transitions
            for state_name, state_info in discovered_machine['states'].items():
                graph.add_node(state_name, **state_info)
            
            for (from_state, symbol), to_state in discovered_machine['transitions'].items():
                graph.add_edge(from_state, to_state, symbol=symbol)
        
        return graph
    
    def _build_ground_truth_graph(self, ground_truth_machines: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build NetworkX graph from ground truth machines."""
        graph = nx.DiGraph()
        
        for machine in ground_truth_machines:
            machine_id = machine.get('machine_id', 'unknown')
            
            # Add nodes
            for state_id, state_info in machine.get('states', {}).items():
                node_name = f"{machine_id}_{state_id}"
                graph.add_node(node_name, **{
                    'machine_id': machine_id,
                    'state_id': state_id,
                    'distribution': state_info.get('distribution', {})
                })
            
            # Add transitions if available
            if 'transitions' in machine:
                for (from_state, symbol), to_state in machine['transitions'].items():
                    from_node = f"{machine_id}_{from_state}"
                    to_node = f"{machine_id}_{to_state}"
                    graph.add_edge(from_node, to_node, symbol=symbol)
            else:
                # Create fully connected graph for states (common in epsilon machines)
                states = list(machine.get('states', {}).keys())
                for state1 in states:
                    for state2 in states:
                        from_node = f"{machine_id}_{state1}"
                        to_node = f"{machine_id}_{state2}"
                        graph.add_edge(from_node, to_node)
        
        return graph
    
    def _infer_transitions_from_histories(self, graph: nx.DiGraph, state_data: Dict[str, Any]):
        """Infer transitions from history strings in CSSR results."""
        # Create connections between states that appear in similar contexts
        # This is a heuristic approach since CSSR doesn't always provide explicit transitions
        
        state_names = list(state_data.keys())
        
        # Create basic connectivity based on co-occurrence patterns
        for state1 in state_names:
            for state2 in state_names:
                if state1 != state2:
                    # Add edge with low weight (heuristic)
                    graph.add_edge(state1, state2, weight=0.1)
        
        # Add self-loops for all states
        for state_name in state_names:
            graph.add_edge(state_name, state_name, weight=1.0)
    
    def _approximate_graph_edit_distance(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """Compute approximate graph edit distance."""
        try:
            # Use NetworkX's graph edit distance (can be expensive for large graphs)
            # For efficiency, we'll use a simpler approximation
            return self._simple_graph_distance(graph1, graph2)
        except Exception:
            return self._simple_graph_distance(graph1, graph2)
    
    def _simple_graph_distance(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """Simple graph distance based on node and edge count differences."""
        nodes_diff = abs(graph1.number_of_nodes() - graph2.number_of_nodes())
        edges_diff = abs(graph1.number_of_edges() - graph2.number_of_edges())
        
        # Normalize by maximum possible difference
        max_nodes = max(graph1.number_of_nodes(), graph2.number_of_nodes(), 1)
        max_edges = max(graph1.number_of_edges(), graph2.number_of_edges(), 1)
        
        normalized_distance = (nodes_diff / max_nodes + edges_diff / max_edges) / 2.0
        return min(normalized_distance, 1.0)
    
    def _compute_spectral_distance(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """Compute distance based on graph Laplacian eigenvalues."""
        try:
            # Convert to undirected for Laplacian
            undirected1 = graph1.to_undirected()
            undirected2 = graph2.to_undirected()
            
            # Get Laplacian matrices
            laplacian1 = nx.laplacian_matrix(undirected1, weight=None).toarray()
            laplacian2 = nx.laplacian_matrix(undirected2, weight=None).toarray()
            
            # Compute eigenvalues
            eigenvals1 = np.linalg.eigvals(laplacian1)
            eigenvals2 = np.linalg.eigvals(laplacian2)
            
            # Sort eigenvalues
            eigenvals1 = np.sort(np.real(eigenvals1))
            eigenvals2 = np.sort(np.real(eigenvals2))
            
            # Pad shorter array with zeros
            max_len = max(len(eigenvals1), len(eigenvals2))
            if len(eigenvals1) < max_len:
                eigenvals1 = np.pad(eigenvals1, (0, max_len - len(eigenvals1)))
            if len(eigenvals2) < max_len:
                eigenvals2 = np.pad(eigenvals2, (0, max_len - len(eigenvals2)))
            
            # Compute L2 distance between eigenvalue spectra
            spectral_distance = np.linalg.norm(eigenvals1 - eigenvals2)
            
            # Normalize by sum of eigenvalues
            normalizer = np.sum(eigenvals1) + np.sum(eigenvals2)
            if normalizer > 0:
                spectral_distance /= normalizer
            
            return min(spectral_distance, 1.0)
            
        except Exception:
            # Fallback to simple distance
            return self._simple_graph_distance(graph1, graph2)
    
    def _compute_connectivity_similarity(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """Compare connectivity patterns between graphs."""
        # Compute degree distributions
        degrees1 = dict(graph1.degree())
        degrees2 = dict(graph2.degree())
        
        if not degrees1 or not degrees2:
            return 0.0
        
        # Compare degree statistics
        stats1 = self._compute_degree_statistics(degrees1)
        stats2 = self._compute_degree_statistics(degrees2)
        
        # Compute similarity score
        similarity = 0.0
        for key in stats1:
            if key in stats2:
                diff = abs(stats1[key] - stats2[key])
                max_val = max(stats1[key], stats2[key], 1)
                similarity += 1.0 - (diff / max_val)
        
        return similarity / len(stats1) if stats1 else 0.0
    
    def _compute_degree_statistics(self, degrees: Dict[str, int]) -> Dict[str, float]:
        """Compute statistical properties of degree distribution."""
        degree_values = list(degrees.values())
        
        if not degree_values:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(degree_values),
            'std': np.std(degree_values),
            'max': np.max(degree_values)
        }
    
    def _compute_transition_coverage(self, discovered_graph: nx.DiGraph, ground_truth_graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute how well discovered transitions cover ground truth transitions."""
        # This is a simplified coverage metric
        # In practice, would need proper state alignment
        
        discovered_edges = set(discovered_graph.edges())
        ground_truth_edges = set(ground_truth_graph.edges())
        
        if not ground_truth_edges:
            return {'coverage_ratio': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Simple edge overlap (note: this is oversimplified)
        overlap = len(discovered_edges & ground_truth_edges)
        
        precision = overlap / len(discovered_edges) if discovered_edges else 0.0
        recall = overlap / len(ground_truth_edges) if ground_truth_edges else 0.0
        coverage_ratio = recall
        
        return {
            'coverage_ratio': coverage_ratio,
            'precision': precision,
            'recall': recall,
            'discovered_edges': len(discovered_edges),
            'ground_truth_edges': len(ground_truth_edges),
            'overlapping_edges': overlap
        }
    
    def _compare_structural_properties(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> Dict[str, Any]:
        """Compare structural properties of the graphs."""
        properties1 = self._compute_graph_properties(graph1)
        properties2 = self._compute_graph_properties(graph2)
        
        comparison = {}
        for prop in properties1:
            if prop in properties2:
                val1, val2 = properties1[prop], properties2[prop]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    max_val = max(abs(val1), abs(val2), 1)
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    comparison[prop] = {
                        'discovered': val1,
                        'ground_truth': val2,
                        'similarity': similarity
                    }
        
        return comparison
    
    def _compute_graph_properties(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute basic structural properties of a graph."""
        properties = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_strongly_connected': nx.is_strongly_connected(graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(graph)
        }
        
        # Add average clustering coefficient for undirected version
        try:
            undirected = graph.to_undirected()
            properties['avg_clustering'] = nx.average_clustering(undirected)
        except:
            properties['avg_clustering'] = 0.0
        
        return properties
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'graph_edit_distance': 1.0,
            'spectral_distance': 1.0,
            'connectivity_similarity': 0.0,
            'transition_coverage': {
                'coverage_ratio': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'discovered_edges': 0,
                'ground_truth_edges': 0,
                'overlapping_edges': 0
            },
            'structural_comparison': {}
        }