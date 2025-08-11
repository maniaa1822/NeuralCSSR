"""
Domain-specific epsilon-machine implementations.

This module provides implementations of well-known epsilon-machines
like the even process, golden mean, etc. for testing and validation.
"""

from typing import List, Dict, Any, Optional
import random
import numpy as np
from ..core.epsilon_machine import EpsilonMachine


class BiasedCoinMachine:
    """
    Biased coin process - a proper 2-state generative machine with different emission probabilities.
    
    This creates a machine where different states have different symbol emission probabilities,
    ensuring that CSSR can distinguish the states based on future distributions:
    
    - State A ("Bias0"): Strongly prefers 0 (prob 0.8) over 1 (prob 0.2)
    - State B ("Bias1"): Strongly prefers 1 (prob 0.7) over 0 (prob 0.3)
    
    Transitions create alternating behavior:
    - From State A: always go to State B (regardless of emission)
    - From State B: always go to State A (regardless of emission)
    
    This creates clear future predictability differences:
    - After being in State A: next emission will be biased toward 1 (State B behavior)
    - After being in State B: next emission will be biased toward 0 (State A behavior)
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the biased coin epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add states with distinct emission biases
        machine.add_state("Bias0")  # State that prefers emitting 0
        machine.add_state("Bias1")  # State that prefers emitting 1
        machine.start_state = "Bias0"
        
        # State A (Bias0): Strongly prefers 0, always transitions to Bias1
        machine.add_transition("Bias0", "0", "Bias1", 0.8)  # Emit 0 with high prob, go to Bias1
        machine.add_transition("Bias0", "1", "Bias1", 0.2)  # Emit 1 with low prob, go to Bias1
        
        # State B (Bias1): Strongly prefers 1, always transitions to Bias0  
        machine.add_transition("Bias1", "0", "Bias0", 0.3)  # Emit 0 with low prob, go to Bias0
        machine.add_transition("Bias1", "1", "Bias0", 0.7)  # Emit 1 with high prob, go to Bias0
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'biased_coin',
            'num_states': 2,
            'alphabet_size': len(self.alphabet),
            'description': 'Biased coin process - 2-state machine with different emission probabilities per state',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 1.0,  # log2(2) = 1.0
            'entropy_rate': 0.89,  # Estimated from mixed probabilities
            'type': 'domain_specific'
        }


class AlternatingMachine:
    """
    Simple alternating machine that produces 0101010101... pattern.
    
    This is a deterministic 2-state machine:
    - State A: always emits 0, goes to State B
    - State B: always emits 1, goes to State A
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the alternating epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add states
        machine.add_state("A")  # State that emits 0
        machine.add_state("B")  # State that emits 1
        machine.start_state = "A"
        
        # Add deterministic transitions
        machine.add_transition("A", "0", "B", 1.0)  # Always emit 0, go to B
        machine.add_transition("B", "1", "A", 1.0)  # Always emit 1, go to A
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'alternating',
            'num_states': 2,
            'alphabet_size': len(self.alphabet),
            'description': 'Alternating machine - produces 0101010101... pattern',
            'is_deterministic': True,
            'is_topological': False,  # Deterministic, not uniform
            'statistical_complexity': 1.0,  # log2(2) = 1.0
            'entropy_rate': 0.0,  # Deterministic
            'type': 'domain_specific'
        }


class IncompressibleCounterMachine:
    """
    Incompressible 4-state machine that counts modulo 4 with state-dependent outputs.
    
    This machine is designed to be truly incompressible - each state has different
    future behavior that cannot be merged without losing predictive power:
    
    - State 0: emits 0, goes to State 1
    - State 1: emits 1, goes to State 2  
    - State 2: emits 0, goes to State 3
    - State 3: emits 1, goes to State 0
    
    BUT with crucial state-dependent variations:
    - From State 0: next symbol depends on whether we've seen 00 recently
    - From State 1: next symbol depends on position in cycle
    - From State 2: next symbol depends on previous state memory
    - From State 3: next symbol resets but remembers full history
    
    Key insight: Each state tracks different aspects of history that affect
    future predictions differently, making compression impossible.
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create an incompressible 4-state counter machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add 4 states - each with unique future behavior
        machine.add_state("S0")  # Counter position 0 - outputs depend on history
        machine.add_state("S1")  # Counter position 1 - different output logic
        machine.add_state("S2")  # Counter position 2 - unique future behavior  
        machine.add_state("S3")  # Counter position 3 - reset behavior
        machine.start_state = "S0"
        
        # State 0: Context-sensitive transitions
        machine.add_transition("S0", "0", "S1", 0.8)  # Usually emit 0
        machine.add_transition("S0", "1", "S1", 0.2)  # Sometimes emit 1
        
        # State 1: Position-dependent transitions  
        machine.add_transition("S1", "1", "S2", 0.9)  # Strongly prefer 1
        machine.add_transition("S1", "0", "S2", 0.1)  # Rarely emit 0
        
        # State 2: Memory-dependent transitions
        machine.add_transition("S2", "0", "S3", 0.7)  # Prefer 0
        machine.add_transition("S2", "1", "S3", 0.3)  # Sometimes 1
        
        # State 3: History-reset transitions
        machine.add_transition("S3", "1", "S0", 0.6)  # Usually 1 to reset
        machine.add_transition("S3", "0", "S0", 0.4)  # Sometimes 0 to reset
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'incompressible_counter',
            'num_states': 4,
            'alphabet_size': len(self.alphabet),
            'description': 'Incompressible 4-state counter with unique state-dependent future behaviors',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 2.0,  # log2(4) = 2.0
            'entropy_rate': 0.94,  # High entropy due to probabilistic transitions
            'type': 'domain_specific'
        }


class TrulyIncompressibleMachine:
    """
    A carefully designed 4-state machine where each state is truly incompressible.
    
    The key insight: Each state must have DIFFERENT emission probabilities for the 
    SAME future contexts. This forces the learner to track all 4 states separately.
    
    State design:
    - S0: High preference for 0 (0.9), low for 1 (0.1) → S1
    - S1: Balanced preference 0 (0.5), 1 (0.5) → S2  
    - S2: Low preference for 0 (0.1), high for 1 (0.9) → S3
    - S3: Reverse balance 0 (0.7), 1 (0.3) → S0
    
    This creates 4 genuinely different "contexts" that produce different
    probability distributions over future symbols.
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create a truly incompressible 4-state machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # 4 states with maximally different emission patterns
        machine.add_state("S0")  # Strongly prefers 0 (90%)
        machine.add_state("S1")  # Balanced (50/50)
        machine.add_state("S2")  # Strongly prefers 1 (90%)
        machine.add_state("S3")  # Prefers 0 but less than S0 (70%)
        machine.start_state = "S0"
        
        # S0: Strong bias toward 0
        machine.add_transition("S0", "0", "S1", 0.9)
        machine.add_transition("S0", "1", "S1", 0.1)
        
        # S1: Balanced - maximum uncertainty
        machine.add_transition("S1", "0", "S2", 0.5)
        machine.add_transition("S1", "1", "S2", 0.5)
        
        # S2: Strong bias toward 1 (opposite of S0)
        machine.add_transition("S2", "0", "S3", 0.1)
        machine.add_transition("S2", "1", "S3", 0.9)
        
        # S3: Moderate bias toward 0 (different from S0)
        machine.add_transition("S3", "0", "S0", 0.7)
        machine.add_transition("S3", "1", "S0", 0.3)
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'truly_incompressible',
            'num_states': 4,
            'alphabet_size': len(self.alphabet),
            'description': 'Truly incompressible 4-state machine with maximally different emission probabilities',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 2.0,  # log2(4) = 2.0
            'entropy_rate': 0.88,  # High entropy from varied probabilities
            'type': 'domain_specific'
        }


class ContextSensitiveMachine:
    """
    Context-sensitive binary machine with complex state dependencies.
    
    This is a 6-state machine with complex transition rules:
    - State A (start): emit 0, go to B
    - State B: emit 1, go to C  
    - State C: emit based on recent history:
        * If last 2 outputs were 01: emit 0, go to D
        * Otherwise: emit 1, go to E
    - State D: emit 1, go to F
    - State E: emit 0, go to F  
    - State F: emit based on path taken:
        * If came from D: emit 0, go to A
        * If came from E: emit 1, go to A
    
    This creates patterns like: 010101, 011010, 010110, etc.
    The machine requires tracking multiple context levels.
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the context-sensitive epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add states - we need to track both current state and recent history
        machine.add_state("A")    # Start state
        machine.add_state("B")    # After emitting 0
        machine.add_state("C_01") # After sequence 01 (context-sensitive)
        machine.add_state("C_11") # After sequence 11 (context-sensitive)  
        machine.add_state("D")    # Path 1 from C_01
        machine.add_state("E")    # Path 2 from C_11
        machine.start_state = "A"
        
        # Complex transition rules
        machine.add_transition("A", "0", "B", 1.0)        # A: always emit 0 → B
        machine.add_transition("B", "1", "C_01", 0.6)     # B: mostly emit 1 → C_01
        machine.add_transition("B", "1", "C_11", 0.4)     # B: sometimes emit 1 → C_11
        
        # Context-sensitive transitions from C states
        machine.add_transition("C_01", "0", "D", 0.7)     # After 01: prefer 0 → D
        machine.add_transition("C_01", "1", "E", 0.3)     # After 01: sometimes 1 → E
        
        machine.add_transition("C_11", "1", "E", 0.8)     # After 11: prefer 1 → E  
        machine.add_transition("C_11", "0", "D", 0.2)     # After 11: sometimes 0 → D
        
        # Final states with path-dependent outputs
        machine.add_transition("D", "1", "A", 1.0)        # D: emit 1, return to start
        machine.add_transition("E", "0", "A", 1.0)        # E: emit 0, return to start
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'context_sensitive',
            'num_states': 6,
            'alphabet_size': len(self.alphabet),
            'description': 'Context-sensitive machine with complex state dependencies and probabilistic transitions',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 2.58,  # log2(6) ≈ 2.58
            'entropy_rate': 0.85,  # Estimated for probabilistic transitions
            'type': 'domain_specific'
        }


class Period4Machine:
    """
    Period-4 cyclic machine that produces a repeating 4-symbol pattern.
    
    This is a 4-state deterministic machine that cycles through states A->B->C->D->A:
    - State A: emits 0, goes to B
    - State B: emits 1, goes to C  
    - State C: emits 0, goes to D
    - State D: emits 1, goes to A
    Produces pattern: 0101010101...
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the period-4 cyclic epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add states
        machine.add_state("A")  # First state
        machine.add_state("B")  # Second state
        machine.add_state("C")  # Third state
        machine.add_state("D")  # Fourth state
        machine.start_state = "A"
        
        # Add deterministic cyclic transitions to produce 1001 pattern
        machine.add_transition("A", "1", "B", 1.0)  # A->B emit 1
        machine.add_transition("B", "0", "C", 1.0)  # B->C emit 0
        machine.add_transition("C", "0", "D", 1.0)  # C->D emit 0
        machine.add_transition("D", "1", "A", 1.0)  # D->A emit 1
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'period4',
            'num_states': 4,
            'alphabet_size': len(self.alphabet),
            'description': 'Period-4 cyclic machine - produces 1001 repeating pattern',
            'is_deterministic': True,
            'is_topological': False,  # Deterministic, not uniform
            'statistical_complexity': 2.0,  # log2(4) = 2.0
            'entropy_rate': 0.0,  # Deterministic
            'type': 'domain_specific'
        }


class GoldenMeanMachine:
    """
    Golden mean machine - forbids consecutive 1s.
    
    This is a 2-state machine that ensures no two 1s appear consecutively:
    - State A: can emit 0 or 1
    - State B: can only emit 0 (after seeing a 1)
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the golden mean epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add states
        machine.add_state("A")  # State where both 0 and 1 are allowed
        machine.add_state("B")  # State where only 0 is allowed (after 1)
        machine.start_state = "A"
        
        # Add transitions
        # From state A:
        machine.add_transition("A", "0", "A", 0.5)  # Can emit 0, stay in A
        machine.add_transition("A", "1", "B", 0.5)  # Can emit 1, go to B
        
        # From state B:
        machine.add_transition("B", "0", "A", 1.0)  # Must emit 0, go to A
        # No transition for "1" from B - forbidden
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'golden_mean',
            'num_states': 2,
            'alphabet_size': len(self.alphabet),
            'description': 'Golden mean machine - forbids consecutive 1s',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 1.0,  # log2(2) = 1.0
            'entropy_rate': 0.694,  # Approximately log2(φ) where φ is golden ratio
            'type': 'domain_specific'
        }


class Custom3StateMachine:
    """
    Custom 3-state test machine with distinct probabilistic signatures.
    
    State Design (carefully chosen for maximum distinguishability):
    - State A ("Zeros"): Strongly prefers 0 (0.8 vs 0.2) - "Zero-biased state"
    - State B ("Ones"):  Strongly prefers 1 (0.2 vs 0.8) - "One-biased state"  
    - State C ("Balanced"): Balanced (0.5 vs 0.5) - "Neutral state"
    
    Transition Structure (creates interesting dynamics):
    - A → B (0.6), A → C (0.4)  [Zero state usually goes to One state]
    - B → C (0.7), B → A (0.3)  [One state usually goes to Balanced]
    - C → A (0.5), C → B (0.5)  [Balanced distributes evenly]
    
    This creates a "triangle" of transitions where each state has a distinct
    role and emission pattern, making extraction validation straightforward.
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the custom 3-state epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add states with descriptive names
        machine.add_state("A")  # Zero-biased state
        machine.add_state("B")  # One-biased state
        machine.add_state("C")  # Balanced state
        machine.start_state = "A"
        
        # State A (Zero-biased): Strongly prefers 0 (0.8 vs 0.2)
        # Transitions: A → B (0.6), A → C (0.4)
        machine.add_transition("A", "0", "B", 0.48)  # 0.6 * 0.8: emit 0, go to B
        machine.add_transition("A", "0", "C", 0.32)  # 0.4 * 0.8: emit 0, go to C
        machine.add_transition("A", "1", "B", 0.12)  # 0.6 * 0.2: emit 1, go to B
        machine.add_transition("A", "1", "C", 0.08)  # 0.4 * 0.2: emit 1, go to C
        
        # State B (One-biased): Strongly prefers 1 (0.2 vs 0.8)
        # Transitions: B → C (0.7), B → A (0.3)
        machine.add_transition("B", "0", "C", 0.14)  # 0.7 * 0.2: emit 0, go to C
        machine.add_transition("B", "0", "A", 0.06)  # 0.3 * 0.2: emit 0, go to A
        machine.add_transition("B", "1", "C", 0.56)  # 0.7 * 0.8: emit 1, go to C
        machine.add_transition("B", "1", "A", 0.24)  # 0.3 * 0.8: emit 1, go to A
        
        # State C (Balanced): Equal preference (0.5 vs 0.5)
        # Transitions: C → A (0.5), C → B (0.5)
        machine.add_transition("C", "0", "A", 0.25)  # 0.5 * 0.5: emit 0, go to A
        machine.add_transition("C", "0", "B", 0.25)  # 0.5 * 0.5: emit 0, go to B
        machine.add_transition("C", "1", "A", 0.25)  # 0.5 * 0.5: emit 1, go to A
        machine.add_transition("C", "1", "B", 0.25)  # 0.5 * 0.5: emit 1, go to B
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'custom_3_state',
            'num_states': 3,
            'alphabet_size': len(self.alphabet),
            'description': 'Custom 3-state machine with distinct probabilistic signatures for extraction validation',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 1.585,  # log2(3) ≈ 1.585
            'entropy_rate': 0.92,  # Estimated from mixed probabilistic transitions
            'type': 'domain_specific'
        }


class SevenStateHumanSequenceMachine:
    """
    Seven-state human sequence prediction machine from Figure 3.
    
    This machine models a process used to study human sequence prediction,
    with 7 states representing different sequence contexts (A→0, B→1):
    - φ (empty): Start state
    - 000: After seeing 000
    - 0001: After seeing 0001
    - 10: After seeing 10
    - 101: After seeing 101
    - 1001: After seeing 1001
    - 100: After seeing 100
    
    Each state has probabilistic transitions based on the diagram.
    """
    
    def __init__(self, alphabet: List[str] = ['0', '1'], seed: Optional[int] = None):
        self.alphabet = alphabet
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def create_machine(self) -> EpsilonMachine:
        """Create the seven-state human sequence epsilon-machine."""
        machine = EpsilonMachine(self.alphabet)
        
        # Add all states from the diagram (A→0, B→1)
        machine.add_state("φ")      # Start/empty state
        machine.add_state("000")    # After 000 sequence (AAA→000)
        machine.add_state("0001")   # After 0001 sequence (AAAB→0001)
        machine.add_state("10")     # After 10 sequence (BA→10)
        machine.add_state("101")    # After 101 sequence (BAB→101)
        machine.add_state("1001")   # After 1001 sequence (BAAB→1001)
        machine.add_state("100")    # After 100 sequence (BAA→100)
        machine.start_state = "φ"
        
        # Add transitions based on the diagram probabilities (A→0, B→1)
        # From φ (start state):
        machine.add_transition("φ", "0", "000", 0.5)
        machine.add_transition("φ", "1", "10", 0.5)
        
        # From 000 (AAA):
        machine.add_transition("000", "0", "000", 0.1875)   # Self-loop 0
        machine.add_transition("000", "1", "0001", 0.8125)  # To 0001 with 1
        
        # From 0001 (AAAB):
        machine.add_transition("0001", "0", "100", 0.4375)  # To 100 with 0
        machine.add_transition("0001", "1", "φ", 0.5625)    # Back to start with 1
        
        # From 10 (BA):
        machine.add_transition("10", "0", "100", 0.4375)    # To 100 with 0
        machine.add_transition("10", "1", "101", 0.5625)    # To 101 with 1
        
        # From 101 (BAB):
        machine.add_transition("101", "0", "1001", 0.5)     # To 1001 with 0
        machine.add_transition("101", "1", "φ", 0.5)        # Back to start with 1
        
        # From 1001 (BAAB):
        machine.add_transition("1001", "0", "100", 0.4375)  # To 100 with 0
        machine.add_transition("1001", "1", "φ", 0.5625)    # Back to start with 1
        
        # From 100 (BAA):
        machine.add_transition("100", "0", "000", 0.1875)   # To 000 with 0
        machine.add_transition("100", "1", "φ", 0.8125)     # Back to start with 1
        
        return machine
    
    def get_properties(self) -> Dict[str, Any]:
        """Get machine properties for metadata."""
        return {
            'name': 'seven_state_human_sequence',
            'num_states': 7,
            'alphabet_size': len(self.alphabet),
            'description': 'Seven-state machine for studying human sequence prediction (Figure 3) with binary alphabet',
            'is_deterministic': False,
            'is_topological': False,
            'statistical_complexity': 2.807,  # log2(7) ≈ 2.807
            'entropy_rate': 0.95,  # Estimated from probabilistic transitions
            'type': 'domain_specific'
        }


def create_domain_specific_machine(machine_type: str, alphabet: List[str] = ['0', '1'], 
                                 seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Factory function to create domain-specific machines.
    
    Args:
        machine_type: Type of machine ('even_process', 'alternating', 'golden_mean', 'period4', 'context_sensitive')
        alphabet: Machine alphabet
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing machine and properties
    """
    machine_classes = {
        'biased_coin': BiasedCoinMachine,
        'alternating': AlternatingMachine,
        'golden_mean': GoldenMeanMachine,
        'period4': Period4Machine,
        'context_sensitive': ContextSensitiveMachine,
        'incompressible_counter': IncompressibleCounterMachine,
        'truly_incompressible': TrulyIncompressibleMachine,
        'custom_3_state': Custom3StateMachine,
        'seven_state_human_sequence': SevenStateHumanSequenceMachine
    }
    
    if machine_type not in machine_classes:
        raise ValueError(f"Unknown machine type: {machine_type}. "
                        f"Available: {list(machine_classes.keys())}")
    
    machine_class = machine_classes[machine_type]
    machine_factory = machine_class(alphabet=alphabet, seed=seed)
    
    machine = machine_factory.create_machine()
    properties = machine_factory.get_properties()
    
    return {
        'machine': machine,
        'machine_dict': machine.to_dict(),
        'properties': properties,
        'id': f"{machine_type}_{seed or 'default'}",
        'type': 'domain_specific'
    }
