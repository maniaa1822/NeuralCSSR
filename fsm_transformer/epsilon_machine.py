import numpy as np
import random
from typing import List, Tuple, Optional, Dict


class EpsilonMachine:
    """
    3-state epsilon-machine with probabilistic transitions as defined in the plan:
    
    State A: After seeing "0" 
    State B: After seeing "1"
    State C: After seeing "00" or "11"
    
    Transition Probabilities:
    A -> 0 (0.7) -> C, 1 (0.3) -> B
    B -> 0 (0.4) -> A, 1 (0.6) -> C  
    C -> 0 (0.5) -> A, 1 (0.5) -> B
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.states = ['A', 'B', 'C']
        self.alphabet = ['0', '1']
        
        # Transition probabilities: state -> symbol -> (probability, next_state)
        self.transitions = {
            'A': {
                '0': (0.7, 'C'),
                '1': (0.3, 'B')
            },
            'B': {
                '0': (0.4, 'A'),
                '1': (0.6, 'C')
            },
            'C': {
                '0': (0.5, 'A'),
                '1': (0.5, 'B')
            }
        }
        
    def get_causal_state(self, history: List[str]) -> str:
        """
        Map sequence history to causal state based on sufficient statistics.
        
        State rules:
        - State A: After seeing "0" (but not "00" or "10")
        - State B: After seeing "1" (but not "11" or "01") 
        - State C: After seeing "00" or "11"
        """
        if len(history) == 0:
            # Start from random state
            return random.choice(self.states)
        
        if len(history) == 1:
            return 'A' if history[-1] == '0' else 'B'
            
        # Check last two symbols
        last_two = ''.join(history[-2:])
        if last_two in ['00', '11']:
            return 'C'
        elif history[-1] == '0':
            return 'A'
        else:
            return 'B'
    
    def generate_sequence(self, length: int, start_state: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Generate sequence using probabilistic transitions.
        Returns (sequence, state_trajectory)
        """
        if start_state is None:
            start_state = random.choice(self.states)
            
        sequence = []
        state_trajectory = [start_state]
        current_state = start_state
        
        for _ in range(length):
            # Sample next symbol based on current state's transition probabilities
            rand = random.random()
            cumulative_prob = 0
            
            for symbol in self.alphabet:
                prob, next_state = self.transitions[current_state][symbol]
                cumulative_prob += prob
                
                if rand <= cumulative_prob:
                    sequence.append(symbol)
                    current_state = next_state
                    state_trajectory.append(current_state)
                    break
                    
        return sequence, state_trajectory[:-1]  # Remove last state (after final symbol)
    
    def get_transition_probability(self, state: str, symbol: str) -> float:
        """Get probability of emitting symbol from given state."""
        return self.transitions[state][symbol][0]
    
    def get_next_state(self, state: str, symbol: str) -> str:
        """Get next state after emitting symbol from given state."""
        return self.transitions[state][symbol][1]
    
    def validate_sequence(self, sequence: List[str]) -> bool:
        """Validate that sequence follows epsilon-machine rules."""
        current_state = self.get_causal_state([])
        
        for i, symbol in enumerate(sequence):
            history = sequence[:i]
            expected_state = self.get_causal_state(history)
            
            if current_state != expected_state:
                return False
                
            current_state = self.get_next_state(current_state, symbol)
            
        return True
    
    def compute_stationary_distribution(self) -> Dict[str, float]:
        """Compute stationary distribution of states."""
        # Transition matrix
        P = np.zeros((3, 3))
        state_to_idx = {'A': 0, 'B': 1, 'C': 2}
        
        for from_state in self.states:
            from_idx = state_to_idx[from_state]
            for symbol in self.alphabet:
                prob, to_state = self.transitions[from_state][symbol]
                to_idx = state_to_idx[to_state]
                P[from_idx, to_idx] += prob
        
        # Find stationary distribution (eigenvector of P^T with eigenvalue 1)
        eigenvals, eigenvecs = np.linalg.eig(P.T)
        stationary_idx = np.argmin(np.abs(eigenvals - 1))
        stationary = np.real(eigenvecs[:, stationary_idx])
        stationary = stationary / stationary.sum()
        
        return {state: float(stationary[state_to_idx[state]]) for state in self.states}