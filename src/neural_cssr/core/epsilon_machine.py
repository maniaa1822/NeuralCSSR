from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict, deque
import random


class EpsilonMachine:
    """
    Core epsilon-machine implementation.
    
    An epsilon-machine is a finite state machine that represents the minimal
    sufficient statistics for predicting the future of a process given its past.
    Each state represents a causal state - a set of histories that have identical
    conditional future distributions.
    """
    
    def __init__(self, alphabet: List[str]):
        """
        Initialize epsilon machine.
        
        Args:
            alphabet: List of symbols that can be emitted
        """
        self.alphabet = alphabet
        self.states = set()
        self.transitions = {}  # (state, symbol) -> [(next_state, probability), ...]
        self.start_state = None
        self.current_state = None
        
    def add_state(self, state: str) -> None:
        """Add a state to the machine."""
        self.states.add(state)
        if self.start_state is None:
            self.start_state = state
            
    def add_transition(self, from_state: str, symbol: str, to_state: str, probability: float) -> None:
        """
        Add a transition to the machine.
        
        Args:
            from_state: Source state
            symbol: Emitted symbol
            to_state: Destination state  
            probability: Transition probability
        """
        if from_state not in self.states:
            self.add_state(from_state)
        if to_state not in self.states:
            self.add_state(to_state)
            
        key = (from_state, symbol)
        if key not in self.transitions:
            self.transitions[key] = []
        self.transitions[key].append((to_state, probability))
        
    def normalize_transitions(self) -> None:
        """Normalize transition probabilities to sum to 1 for each state."""
        state_symbol_totals = defaultdict(float)
        
        # Calculate totals for each (state, symbol) pair
        for (state, symbol), transitions in self.transitions.items():
            total = sum(prob for _, prob in transitions)
            state_symbol_totals[(state, symbol)] = total
            
        # Normalize
        for key, transitions in self.transitions.items():
            total = state_symbol_totals[key]
            if total > 0:
                self.transitions[key] = [(next_state, prob / total) 
                                       for next_state, prob in transitions]
                                       
    def get_emission_probabilities(self, state: str) -> Dict[str, float]:
        """
        Get emission probabilities for a given state.
        
        Args:
            state: State to get probabilities for
            
        Returns:
            Dictionary mapping symbols to emission probabilities
        """
        probs = defaultdict(float)
        
        for symbol in self.alphabet:
            key = (state, symbol)
            if key in self.transitions:
                for _, prob in self.transitions[key]:
                    probs[symbol] += prob
                    
        return dict(probs)
        
    def generate_sequence(self, length: int, start_state: Optional[str] = None) -> List[str]:
        """Generate a sequence from the epsilon machine."""
        if not self.states:
            raise ValueError("Machine has no states")
            
        current = start_state or self.start_state
        sequence = []
        
        for _ in range(length):
            # FIX: Get all outgoing transitions from current state
            outgoing_transitions = []
            
            for symbol in self.alphabet:
                key = (current, symbol)
                if key in self.transitions and self.transitions[key]:
                    next_state, prob = self.transitions[key][0]  # Should be single transition
                    outgoing_transitions.append((symbol, next_state, prob))
                    
            if not outgoing_transitions:
                break
                
            # Sample transition based on topological probabilities
            total_prob = sum(prob for _, _, prob in outgoing_transitions)
            if total_prob == 0:
                break
                
            r = random.random() * total_prob
            cumulative = 0
            
            for symbol, next_state, prob in outgoing_transitions:
                cumulative += prob
                if r <= cumulative:
                    sequence.append(symbol)
                    current = next_state
                    break
                    
        return sequence
        
    def compute_causal_state(self, history: List[str]) -> Optional[str]:
        """Compute the causal state for a given history."""
        if not history:
            return self.start_state
            
        current = self.start_state
        
        for symbol in history:
            key = (current, symbol)
            if key in self.transitions and self.transitions[key]:
                # FIX: Should be deterministic - single transition
                next_state, _ = self.transitions[key][0]
                current = next_state
            else:
                return None  # Invalid transition
                
        return current
        
    def get_conditional_distribution(self, state: str) -> Dict[str, float]:
        """
        Get the conditional distribution P(X_t+1 | causal_state).
        
        Args:
            state: Causal state
            
        Returns:
            Dictionary mapping next symbols to their probabilities
        """
        return self.get_emission_probabilities(state)
        
    def is_topological(self) -> bool:
        """
        Check if this is a topological epsilon-machine (uniform probabilities).
        
        Returns:
            True if all outgoing transitions from each state have equal probability
        """
        for state in self.states:
            # Get all outgoing transitions from this state
            outgoing = []
            for symbol in self.alphabet:
                key = (state, symbol)
                if key in self.transitions:
                    for next_state, prob in self.transitions[key]:
                        outgoing.append(prob)
                        
            if outgoing:
                # Check if all probabilities are approximately equal
                target_prob = 1.0 / len(outgoing)
                if not all(abs(prob - target_prob) < 1e-10 for prob in outgoing):
                    return False
                    
        return True
        
    def make_topological(self) -> None:
        """Convert to topological epsilon-machine by making all outgoing probabilities uniform."""
        for state in self.states:
            # Count outgoing transitions from this state
            outgoing_count = 0
            outgoing_transitions = []
            
            for symbol in self.alphabet:
                key = (state, symbol)
                if key in self.transitions:
                    outgoing_count += len(self.transitions[key])
                    outgoing_transitions.append(key)
                    
            if outgoing_count > 0:
                uniform_prob = 1.0 / outgoing_count
                
                # Update all transitions to have uniform probability
                for key in outgoing_transitions:
                    updated_transitions = []
                    for next_state, _ in self.transitions[key]:
                        updated_transitions.append((next_state, uniform_prob))
                    self.transitions[key] = updated_transitions
                    
    def randomize_probabilities(self, seed: Optional[int] = None, bias_strength: float = 0.5) -> None:
        """
        Randomize transition probabilities with specified bias strength.
        
        Args:
            seed: Random seed for reproducibility
            bias_strength: How much to bias away from uniform (0.0=uniform, 1.0=maximum bias)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # For each state, randomize the emission probabilities
        for state in self.states:
            # Get all outgoing transitions from this state
            outgoing_transitions = []
            for symbol in self.alphabet:
                key = (state, symbol)
                if key in self.transitions:
                    for next_state, _ in self.transitions[key]:
                        outgoing_transitions.append((key, next_state))
            
            if outgoing_transitions:
                # Generate random probabilities with bias
                num_transitions = len(outgoing_transitions)
                
                if bias_strength == 0.0:
                    # Uniform probabilities
                    probs = [1.0 / num_transitions] * num_transitions
                else:
                    # Generate biased probabilities
                    # Start with uniform, then apply bias
                    base_probs = np.ones(num_transitions) / num_transitions
                    
                    # Add random noise scaled by bias_strength
                    noise = np.random.exponential(scale=bias_strength, size=num_transitions)
                    biased_probs = base_probs + noise
                    
                    # Normalize to sum to 1
                    probs = biased_probs / np.sum(biased_probs)
                
                # Update transitions with new probabilities
                for i, (key, next_state) in enumerate(outgoing_transitions):
                    # Clear existing transitions for this key
                    self.transitions[key] = [(next_state, probs[i])]
    
    def set_transition_probabilities(self, probability_map: Dict[str, Dict[str, float]]) -> None:
        """
        Set custom transition probabilities for each state.
        
        Args:
            probability_map: Mapping from state_id to {symbol: probability}
                            e.g., {"S0": {"0": 0.8, "1": 0.2}, "S1": {"0": 0.3, "1": 0.7}}
        """
        # Clear existing probabilities and rebuild with custom ones
        for state_id, symbol_probs in probability_map.items():
            # Find the actual state name (might be different from state_id)
            actual_state = None
            for state in self.states:
                if state == state_id or state.endswith(f"_{state_id}") or state_id in state:
                    actual_state = state
                    break
            
            if actual_state is None:
                # If state not found by ID, try to match by index
                state_list = sorted(list(self.states))
                try:
                    state_idx = int(state_id.replace('S', ''))
                    if 0 <= state_idx < len(state_list):
                        actual_state = state_list[state_idx]
                except (ValueError, IndexError):
                    continue
            
            if actual_state is not None:
                # Update probabilities for each symbol from this state
                for symbol, prob in symbol_probs.items():
                    key = (actual_state, symbol)
                    if key in self.transitions:
                        # Keep the same next states, just update probabilities
                        existing_transitions = self.transitions[key]
                        if existing_transitions:
                            # Assume single next state per (state, symbol) pair for simplicity
                            next_state = existing_transitions[0][0]
                            self.transitions[key] = [(next_state, prob)]

    def get_statistical_complexity(self) -> float:
        """
        Compute statistical complexity (entropy of the stationary distribution).
        
        Returns:
            Statistical complexity in bits
        """
        # For now, return log2 of number of states as approximation
        # In full implementation, would compute stationary distribution
        if len(self.states) == 0:
            return 0.0
        return np.log2(len(self.states))
        
    def to_dict(self) -> Dict:
        """
        Serialize epsilon machine to dictionary.
        
        Returns:
            Dictionary representation of the machine
        """
        return {
            'alphabet': self.alphabet,
            'states': list(self.states),
            'transitions': {
                f"{state},{symbol}": [(next_state, prob) for next_state, prob in transitions]
                for (state, symbol), transitions in self.transitions.items()
            },
            'start_state': self.start_state
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'EpsilonMachine':
        """
        Create epsilon machine from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            EpsilonMachine instance
        """
        machine = cls(data['alphabet'])
        
        for state in data['states']:
            machine.add_state(state)
            
        machine.start_state = data['start_state']
        
        for key, transitions in data['transitions'].items():
            state, symbol = key.split(',')
            for next_state, prob in transitions:
                machine.add_transition(state, symbol, next_state, prob)
                
        return machine
        
    def __str__(self) -> str:
        """String representation of the machine."""
        return f"EpsilonMachine(states={len(self.states)}, alphabet={self.alphabet})"
        
    def __repr__(self) -> str:
        return self.__str__()