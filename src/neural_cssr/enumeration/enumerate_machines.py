from typing import List, Dict, Set, Tuple, Iterator, Optional
import itertools
from collections import defaultdict, deque
import numpy as np
from ..core.epsilon_machine import EpsilonMachine


class MachineEnumerator:
    """
    Enumerate all topological epsilon-machines up to specified constraints.
    
    Based on the systematic enumeration approach described in "Enumerating Finitary Processes"
    by Almeida et al. Generates all possible machine structures and filters for 
    epsilon-machine properties.
    """
    
    def __init__(self, max_states: int, alphabet: List[str]):
        """
        Initialize machine enumerator.
        
        Args:
            max_states: Maximum number of states to enumerate
            alphabet: Alphabet of symbols
        """
        self.max_states = max_states
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        
    def enumerate_all_machines(self) -> Iterator[EpsilonMachine]:
        """
        Enumerate all topological epsilon-machines up to max_states.
        
        Yields:
            EpsilonMachine instances representing all possible structures
        """
        for num_states in range(1, self.max_states + 1):
            yield from self.enumerate_machines_with_n_states(num_states)
            
    def enumerate_machines_with_n_states(self, n_states: int) -> Iterator[EpsilonMachine]:
        """
        Enumerate all topological epsilon-machines with exactly n states.
        
        Args:
            n_states: Number of states
            
        Yields:
            EpsilonMachine instances with n states
        """
        states = [f"S{i}" for i in range(n_states)]
        
        # Generate all possible transition structures
        for transitions in self._generate_transition_structures(states):
            machine = self._create_machine_from_transitions(states, transitions)
            
            # Filter for valid epsilon-machines
            if self._is_valid_epsilon_machine(machine):
                # Make it topological (uniform probabilities)
                machine.make_topological()
                yield machine
                
    def _generate_transition_structures(self, states: List[str]) -> Iterator[Dict]:
        """
        Generate all possible transition structures for given states.
        
        Args:
            states: List of state names
            
        Yields:
            Dictionaries representing transition structures
        """
        n_states = len(states)
        
        # For each state and symbol, generate all possible outgoing transitions
        # Each transition goes to one of the n_states
        
        # Total number of (state, symbol) pairs
        total_pairs = n_states * self.alphabet_size
        
        # For each (state, symbol) pair, it can transition to any of the n_states
        # Generate all combinations
        for transition_targets in itertools.product(range(n_states), repeat=total_pairs):
            transitions = {}
            idx = 0
            
            for state_idx, state in enumerate(states):
                for symbol in self.alphabet:
                    target_state_idx = transition_targets[idx]
                    target_state = states[target_state_idx]
                    
                    key = (state, symbol)
                    transitions[key] = target_state
                    idx += 1
                    
            yield transitions
            
    def _create_machine_from_transitions(self, states: List[str], transitions: Dict) -> EpsilonMachine:
        """
        Create epsilon machine from transition structure.
        
        Args:
            states: List of state names
            transitions: Dictionary mapping (state, symbol) -> target_state
            
        Returns:
            EpsilonMachine instance
        """
        machine = EpsilonMachine(self.alphabet)
        
        # Add all states
        for state in states:
            machine.add_state(state)
            
        machine.start_state = states[0]  # Use first state as start state
        
        # Add transitions with initial uniform probabilities
        for (from_state, symbol), to_state in transitions.items():
            machine.add_transition(from_state, symbol, to_state, 1.0)
            
        return machine
        
    def _is_valid_epsilon_machine(self, machine: EpsilonMachine) -> bool:
        """
        Check if a machine satisfies epsilon-machine properties.
        
        Args:
            machine: Machine to validate
            
        Returns:
            True if machine is a valid epsilon-machine
        """
        # Check if machine is strongly connected (all states reachable)
        if not self._is_strongly_connected(machine):
            return False
            
        # Check if machine is deterministic (each (state, symbol) has unique transition)
        if not self._is_deterministic(machine):
            return False
            
        return True
        
    def _is_strongly_connected(self, machine: EpsilonMachine) -> bool:
        """
        Check if machine is strongly connected (all states mutually reachable).
        
        Args:
            machine: Machine to check
            
        Returns:
            True if strongly connected
        """
        if not machine.states:
            return False
            
        # Use DFS to check reachability from each state to all others
        for start_state in machine.states:
            reachable = self._get_reachable_states(machine, start_state)
            if len(reachable) != len(machine.states):
                return False
                
        return True
        
    def _get_reachable_states(self, machine: EpsilonMachine, start_state: str) -> Set[str]:
        """
        Get all states reachable from a given start state.
        
        Args:
            machine: Machine to analyze
            start_state: Starting state
            
        Returns:
            Set of reachable states
        """
        reachable = set()
        queue = deque([start_state])
        
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
                
            reachable.add(current)
            
            # Add all states reachable in one step
            for symbol in machine.alphabet:
                key = (current, symbol)
                if key in machine.transitions:
                    for next_state, _ in machine.transitions[key]:
                        if next_state not in reachable:
                            queue.append(next_state)
                            
        return reachable
        
    def _is_deterministic(self, machine: EpsilonMachine) -> bool:
        """
        Check if machine is deterministic.
        
        Args:
            machine: Machine to check
            
        Returns:
            True if deterministic
        """
        # For each (state, symbol), there should be exactly one outgoing transition
        for state in machine.states:
            for symbol in machine.alphabet:
                key = (state, symbol)
                if key not in machine.transitions:
                    return False
                if len(machine.transitions[key]) != 1:
                    return False
                    
        return True
        
    def compute_machine_properties(self, machine: EpsilonMachine) -> Dict:
        """
        Compute various properties of an epsilon machine.
        
        Args:
            machine: Machine to analyze
            
        Returns:
            Dictionary of computed properties
        """
        properties = {
            'num_states': len(machine.states),
            'alphabet_size': len(machine.alphabet),
            'num_transitions': sum(len(transitions) for transitions in machine.transitions.values()),
            'statistical_complexity': machine.get_statistical_complexity(),
            'is_topological': machine.is_topological(),
            'entropy_rate': self._compute_entropy_rate(machine),
            'period': self._compute_period(machine),
        }
        
        return properties
        
    def _compute_entropy_rate(self, machine: EpsilonMachine) -> float:
        """
        Compute entropy rate of the machine.
        
        Args:
            machine: Machine to analyze
            
        Returns:
            Entropy rate in bits
        """
        # For topological machines, entropy rate equals log2(alphabet_size)
        # This is a simplification; full computation would require stationary distribution
        if machine.is_topological():
            return np.log2(len(machine.alphabet))
        
        # For non-topological machines, compute weighted average
        total_entropy = 0.0
        for state in machine.states:
            probs = machine.get_emission_probabilities(state)
            if probs:
                entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
                total_entropy += entropy
                
        return total_entropy / len(machine.states) if machine.states else 0.0
        
    def _compute_period(self, machine: EpsilonMachine) -> int:
        """
        Compute period of the machine (length of shortest cycle).
        
        Args:
            machine: Machine to analyze
            
        Returns:
            Period length
        """
        # Simple implementation: find shortest cycle from any state
        min_period = float('inf')
        
        for start_state in machine.states:
            period = self._find_shortest_cycle(machine, start_state)
            if period > 0:
                min_period = min(min_period, period)
                
        return int(min_period) if min_period != float('inf') else 1
        
    def _find_shortest_cycle(self, machine: EpsilonMachine, start_state: str) -> int:
        """
        Find shortest cycle from a given state.
        
        Args:
            machine: Machine to analyze
            start_state: Starting state
            
        Returns:
            Length of shortest cycle, or 0 if no cycle found
        """
        visited = {}
        current = start_state
        step = 0
        
        # Follow first available transition repeatedly
        while current not in visited:
            visited[current] = step
            
            # Find first available transition
            next_state = None
            for symbol in machine.alphabet:
                key = (current, symbol)
                if key in machine.transitions and machine.transitions[key]:
                    next_state, _ = machine.transitions[key][0]
                    break
                    
            if next_state is None:
                return 0  # No outgoing transitions
                
            current = next_state
            step += 1
            
        # Found cycle
        return step - visited[current]


def enumerate_machines_library(max_states: int, alphabet: List[str], 
                             max_machines_per_size: Optional[int] = None) -> List[Dict]:
    """
    Generate a library of enumerated epsilon-machines with their properties.
    
    Args:
        max_states: Maximum number of states to enumerate
        alphabet: Alphabet symbols
        max_machines_per_size: Maximum machines to generate per state count (None for all)
        
    Returns:
        List of dictionaries containing machine data and properties
    """
    enumerator = MachineEnumerator(max_states, alphabet)
    machine_library = []
    
    for num_states in range(1, max_states + 1):
        machines_with_n_states = []
        
        for machine in enumerator.enumerate_machines_with_n_states(num_states):
            properties = enumerator.compute_machine_properties(machine)
            
            machine_data = {
                'machine': machine,
                'machine_dict': machine.to_dict(),
                'properties': properties,
                'id': len(machine_library)
            }
            
            machines_with_n_states.append(machine_data)
            
            if max_machines_per_size and len(machines_with_n_states) >= max_machines_per_size:
                break
                
        machine_library.extend(machines_with_n_states)
        
    return machine_library


def save_machine_library(library: List[Dict], filepath: str) -> None:
    """
    Save machine library to file.
    
    Args:
        library: Machine library to save
        filepath: Path to save file
    """
    import json
    
    # Convert to JSON-serializable format
    serializable_library = []
    for machine_data in library:
        serializable_data = {
            'machine_dict': machine_data['machine_dict'],
            'properties': machine_data['properties'],
            'id': machine_data['id']
        }
        serializable_library.append(serializable_data)
        
    with open(filepath, 'w') as f:
        json.dump(serializable_library, f, indent=2)


def load_machine_library(filepath: str) -> List[Dict]:
    """
    Load machine library from file.
    
    Args:
        filepath: Path to library file
        
    Returns:
        List of machine data dictionaries
    """
    import json
    
    with open(filepath, 'r') as f:
        serializable_library = json.load(f)
        
    library = []
    for data in serializable_library:
        machine = EpsilonMachine.from_dict(data['machine_dict'])
        machine_data = {
            'machine': machine,
            'machine_dict': data['machine_dict'],
            'properties': data['properties'],
            'id': data['id']
        }
        library.append(machine_data)
        
    return library