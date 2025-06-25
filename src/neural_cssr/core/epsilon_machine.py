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
        """
        Generate a sequence from the epsilon machine.
        
        Args:
            length: Length of sequence to generate
            start_state: Starting state (uses machine's start_state if None)
            
        Returns:
            Generated sequence of symbols
        """
        if not self.states:
            raise ValueError("Machine has no states")
            
        current = start_state or self.start_state
        sequence = []
        
        for _ in range(length):
            # Get all possible emissions from current state
            possible_emissions = []
            
            for symbol in self.alphabet:
                key = (current, symbol)
                if key in self.transitions:
                    for next_state, prob in self.transitions[key]:
                        possible_emissions.append((symbol, next_state, prob))
                        
            if not possible_emissions:
                break
                
            # Sample emission based on probabilities
            total_prob = sum(prob for _, _, prob in possible_emissions)
            if total_prob == 0:
                break
                
            r = random.random() * total_prob
            cumulative = 0
            
            for symbol, next_state, prob in possible_emissions:
                cumulative += prob
                if r <= cumulative:
                    sequence.append(symbol)
                    current = next_state
                    break
                    
        return sequence
        
    def compute_causal_state(self, history: List[str]) -> Optional[str]:
        """
        Compute the causal state for a given history by following transitions.
        
        Args:
            history: Sequence of symbols representing the history
            
        Returns:
            Causal state after processing the history, or None if invalid
        """
        if not history:
            return self.start_state
            
        current = self.start_state
        
        for symbol in history:
            # Find transition from current state with this symbol
            found_transition = False
            
            key = (current, symbol)
            if key in self.transitions:
                # Take the most likely transition (or first if probabilities are equal)
                next_state, _ = max(self.transitions[key], key=lambda x: x[1])
                current = next_state
                found_transition = True
                
            if not found_transition:
                return None
                
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