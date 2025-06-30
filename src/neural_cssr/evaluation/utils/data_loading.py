"""Data loading utilities for CSSR results and ground truth machines."""

import json
import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_cssr_results(results_path: str) -> Dict[str, Any]:
    """
    Load CSSR analysis results from JSON file.
    
    Args:
        results_path: Path to classical_cssr_results.json file
        
    Returns:
        Dictionary containing CSSR results
        
    Raises:
        FileNotFoundError: If results file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"CSSR results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def load_ground_truth(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load ground truth machine definitions from dataset directory.
    
    Args:
        dataset_path: Path to dataset directory (e.g., datasets/biased_exp)
        
    Returns:
        List of ground truth machine dictionaries
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data format is invalid
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Load experiment configuration to understand machine specs
    config_path = dataset_path / 'experiment_config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load machine properties
    machine_props_path = dataset_path / 'ground_truth' / 'machine_properties.json'
    if not machine_props_path.exists():
        raise FileNotFoundError(f"Machine properties not found: {machine_props_path}")
    
    with open(machine_props_path, 'r') as f:
        machine_properties = json.load(f)
    
    # Build complete machine definitions
    machines = []
    
    # Get machine IDs from properties file 
    for machine_id, properties in machine_properties.items():
        machine = _build_machine_definition(machine_id, properties, config, dataset_path)
        if machine:
            machines.append(machine)
    
    return machines


def _build_machine_definition(machine_id: str, properties: Dict[str, Any], 
                            config: Dict[str, Any], dataset_path: Path) -> Optional[Dict[str, Any]]:
    """Build complete machine definition from available data."""
    
    # Find matching machine spec in config
    machine_spec = _find_matching_spec(machine_id, properties, config)
    if not machine_spec:
        return None
    
    # Build states based on machine properties and spec
    states = _build_machine_states(machine_id, properties, machine_spec)
    
    machine = {
        'machine_id': machine_id,
        'states': states,
        'properties': {
            'num_states': properties.get('num_states', len(states)),
            'statistical_complexity': properties.get('statistical_complexity'),
            'entropy_rate': properties.get('entropy_rate'),
            'is_topological': properties.get('is_topological', True),
            'alphabet_size': properties.get('alphabet_size', 2)
        },
        'samples_count': machine_spec.get('samples_per_machine', 0),
        'weight': machine_spec.get('weight', 1.0)
    }
    
    return machine


def _find_matching_spec(machine_id: str, properties: Dict[str, Any], 
                       config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find the machine spec that matches this machine ID."""
    
    num_states = properties.get('num_states', 0)
    is_topological = properties.get('is_topological', True)
    
    # Try to match based on properties
    for spec in config.get('machine_specs', []):
        spec_states = _get_states_from_complexity_class(spec.get('complexity_class', ''))
        spec_topological = spec.get('topological', True)
        
        if spec_states == num_states and spec_topological == is_topological:
            return spec
    
    # Fallback: create minimal spec
    return {
        'complexity_class': f'{num_states}-state-binary',
        'topological': is_topological,
        'samples_per_machine': 1000,
        'weight': 1.0
    }


def _get_states_from_complexity_class(complexity_class: str) -> int:
    """Extract number of states from complexity class string."""
    if '2-state' in complexity_class:
        return 2
    elif '3-state' in complexity_class:
        return 3
    elif '4-state' in complexity_class:
        return 4
    else:
        # Try to extract number
        parts = complexity_class.split('-')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 2  # Default


def _build_machine_states(machine_id: str, properties: Dict[str, Any], 
                         machine_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build state definitions for a machine."""
    
    num_states = properties.get('num_states', 2)
    is_topological = properties.get('is_topological', True)
    
    states = {}
    
    if is_topological:
        # Uniform probabilities for topological machines
        uniform_prob = 0.5  # For binary alphabet
        for i in range(num_states):
            state_id = f'S{i}'
            states[state_id] = {
                'distribution': {'0': uniform_prob, '1': uniform_prob}
            }
    else:
        # Check for custom probabilities in spec
        custom_probs = machine_spec.get('custom_probabilities')
        if custom_probs:
            for state_id, probs in custom_probs.items():
                states[state_id] = {
                    'distribution': probs
                }
        else:
            # Generate biased probabilities based on machine ID and spec
            bias_strength = machine_spec.get('bias_strength', 0.5)
            prob_seed = machine_spec.get('probability_seed', 42)
            
            # Simple deterministic bias generation
            import random
            random.seed(prob_seed + int(machine_id) * 100)
            
            for i in range(num_states):
                state_id = f'S{i}'
                # Generate biased probability
                bias = random.uniform(-bias_strength, bias_strength)
                prob_0 = max(0.1, min(0.9, 0.5 + bias))
                prob_1 = 1.0 - prob_0
                
                states[state_id] = {
                    'distribution': {'0': prob_0, '1': prob_1}
                }
    
    return states


def validate_data_format(cssr_results: Dict[str, Any], 
                        ground_truth_machines: List[Dict[str, Any]]) -> bool:
    """
    Validate that loaded data has expected format.
    
    Args:
        cssr_results: Loaded CSSR results
        ground_truth_machines: Loaded ground truth machines
        
    Returns:
        True if data format is valid
        
    Raises:
        ValueError: If data format is invalid
    """
    # Validate CSSR results
    if not isinstance(cssr_results, dict):
        raise ValueError("CSSR results must be a dictionary")
    
    if 'cssr_results' not in cssr_results and 'states' not in cssr_results:
        raise ValueError("CSSR results must contain either 'cssr_results' or 'states' key")
    
    # Validate ground truth machines
    if not isinstance(ground_truth_machines, list):
        raise ValueError("Ground truth machines must be a list")
    
    if not ground_truth_machines:
        raise ValueError("Ground truth machines list cannot be empty")
    
    for i, machine in enumerate(ground_truth_machines):
        if not isinstance(machine, dict):
            raise ValueError(f"Ground truth machine {i} must be a dictionary")
        
        if 'machine_id' not in machine:
            raise ValueError(f"Ground truth machine {i} missing 'machine_id'")
        
        if 'states' not in machine:
            raise ValueError(f"Ground truth machine {i} missing 'states'")
        
        states = machine['states']
        if not isinstance(states, dict) or not states:
            raise ValueError(f"Ground truth machine {i} must have non-empty states dictionary")
        
        # Validate state format
        for state_id, state_info in states.items():
            if 'distribution' not in state_info:
                raise ValueError(f"State {state_id} in machine {machine['machine_id']} missing 'distribution'")
    
    return True


def load_experiment_data(dataset_name: str, results_dir: str = 'results') -> Dict[str, Any]:
    """
    Convenience function to load both CSSR results and ground truth for an experiment.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'biased_exp')
        results_dir: Directory containing CSSR results (default: 'results')
        
    Returns:
        Dictionary with 'cssr_results' and 'ground_truth_machines' keys
    """
    # Load CSSR results
    cssr_path = os.path.join(results_dir, dataset_name, 'classical_cssr_results.json')
    cssr_results = load_cssr_results(cssr_path)
    
    # Load ground truth
    dataset_path = os.path.join('datasets', dataset_name)
    ground_truth_machines = load_ground_truth(dataset_path)
    
    # Validate data
    validate_data_format(cssr_results, ground_truth_machines)
    
    return {
        'cssr_results': cssr_results,
        'ground_truth_machines': ground_truth_machines,
        'dataset_name': dataset_name
    }