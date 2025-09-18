#!/usr/bin/env python3
"""
Generate domain-specific machine datasets for neural CSSR training using python-statemachine.

Creates a single long sequence from a domain-specific finite state machine
with aligned state trajectories for linear probe training.

Usage:
    python generate_fsm_dataset.py --machine biased_coin --length 100000 --output data/
    python generate_fsm_dataset.py --machine unifilar_3_state --length 50000 --output data/ --seed 42
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from statemachine import StateMachine, State

# --- Base Machine Definition ---

class StatemachineGenerator(StateMachine):
    """
    Abstract base class for all our state machine generators.
    It defines a common interface for sequence generation and metadata extraction.
    """
    def __init__(self, seed: int = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.alphabet: List[str] = []
        self._transition_info: Dict[str, Any] = {}

    def step(self) -> str:
        """Executes a single step, generating one symbol and transitioning state."""
        raise NotImplementedError("Each machine must implement its own step method.")

    def get_transition_structure(self) -> Dict[str, Any]:
        """Returns the machine's transition structure in the required JSON format."""
        return self._transition_info
    
    @property
    def start_state_name(self) -> str:
        """Returns the name of the initial state."""
        return self.initial_state.id

# --- Domain-Specific Machine Implementations ---

class BiasedCoinMachine(StatemachineGenerator):
    S0 = State('S0', initial=True)
    tick = S0.to(S0)
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        self.p1 = 0.7
        self._transition_info = {"S0|0": [{"to_state": "S0", "probability": 1.0 - self.p1}], "S0|1": [{"to_state": "S0", "probability": self.p1}]}
    def on_tick(self) -> str: return self.rng.choice(self.alphabet, p=[1.0 - self.p1, self.p1])
    def step(self) -> str: return self.send('tick')

class AlternatingMachine(StatemachineGenerator):
    state_a, state_b = State('StateA', initial=True), State('StateB')
    emit_0, emit_1 = state_a.to(state_b, on="on_emit_0"), state_b.to(state_a, on="on_emit_1")
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        self._transition_info = {"StateA|0": [{"to_state": "StateB", "probability": 1.0}], "StateB|1": [{"to_state": "StateA", "probability": 1.0}]}
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    def step(self) -> str: return self.send('emit_0') if self.current_state == self.state_a else self.send('emit_1')

class GoldenMeanMachine(StatemachineGenerator):
    state_a, state_b = State('A', initial=True), State('B')
    emit_1_from_a, emit_0_from_a = state_a.to(state_a, on="on_emit_1"), state_a.to(state_b, on="on_emit_0")
    emit_1_from_b = state_b.to(state_a, on="on_emit_1")
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        self._transition_info = {"A|0": [{"to_state": "B", "probability": 0.5}], "A|1": [{"to_state": "A", "probability": 0.5}], "B|1": [{"to_state": "A", "probability": 1.0}]}
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    def step(self) -> str:
        if self.current_state == self.state_a: return self.send('emit_0_from_a') if self.rng.random() < 0.5 else self.send('emit_1_from_a')
        else: return self.send('emit_1_from_b')

class EvenProcessMachine(StatemachineGenerator):
    """
    Even Process (sofic, infinite Markov order, 2 causal states):
    - Runs of 1s have even length; 0s can appear only when the current run of 1s has even parity.
    - Minimal ε-machine has two states: E (even parity), O (odd parity).
    - Transitions (unifilar): E --0--> E, E --1--> O; O --1--> E; (0 from O is forbidden)
    Generation policy:
    - In E: emit 1 with probability p1 (default 0.5), otherwise 0
    - In O: always emit 1
    """
    state_e, state_o = State('E', initial=True), State('O')
    e_emit_0, e_emit_1 = state_e.to(state_e, on="on_emit_0"), state_e.to(state_o, on="on_emit_1")
    o_emit_1 = state_o.to(state_e, on="on_emit_1")

    def __init__(self, seed: int = None, p1_in_E: float = 0.5):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        self.p1_in_E = float(p1_in_E)
        self._transition_info = {
            "E|0": [{"to_state": "E", "probability": 1.0}],
            "E|1": [{"to_state": "O", "probability": 1.0}],
            "O|1": [{"to_state": "E", "probability": 1.0}],
        }

    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'

    def step(self) -> str:
        if self.current_state == self.state_e:
            return self.send('e_emit_1') if self.rng.random() < self.p1_in_E else self.send('e_emit_0')
        else:
            return self.send('o_emit_1')

class Unifilar3StateMachine(StatemachineGenerator):
    """
    A corrected, unifilar 3-state machine with distinct probabilistic signatures.
    This machine is a valid epsilon-machine and can be reconstructed by CSSR.
    """
    state_a, state_b, state_c = State("A", initial=True), State("B"), State("C")
    a_emits_0, a_emits_1 = state_a.to(state_c, on="on_emit_0"), state_a.to(state_b, on="on_emit_1")
    b_emits_0, b_emits_1 = state_b.to(state_a, on="on_emit_0"), state_b.to(state_c, on="on_emit_1")
    c_emits_0, c_emits_1 = state_c.to(state_a, on="on_emit_0"), state_c.to(state_b, on="on_emit_1")
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        self._transition_info = {
            "A|0": [{"to_state": "C", "probability": 1.0}], "A|1": [{"to_state": "B", "probability": 1.0}],
            "B|0": [{"to_state": "A", "probability": 1.0}], "B|1": [{"to_state": "C", "probability": 1.0}],
            "C|0": [{"to_state": "A", "probability": 1.0}], "C|1": [{"to_state": "B", "probability": 1.0}],
        }
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.state_a: return self.send('a_emits_0') if self.rng.random() < 0.8 else self.send('a_emits_1')
        elif current_state == self.state_b: return self.send('b_emits_1') if self.rng.random() < 0.8 else self.send('b_emits_0')
        else: return self.send('c_emits_0') if self.rng.random() < 0.5 else self.send('c_emits_1')


class DistinctThreeStateMachine(StatemachineGenerator):
    """
    A 3-state machine designed for better neural-causal compatibility.
    Features highly distinguishable emission patterns similar to Golden Mean.
    
    State A: Heavy bias toward 0 (90% → 0, 10% → 1)
    State B: Heavy bias toward 1 (10% → 0, 90% → 1) 
    State C: Balanced emissions (50% → 0, 50% → 1)
    """
    state_a, state_b, state_c = State("A", initial=True), State("B"), State("C")
    
    # Transitions from A
    a_emits_0, a_emits_1 = state_a.to(state_b, on="on_emit_0"), state_a.to(state_c, on="on_emit_1")
    # Transitions from B  
    b_emits_0, b_emits_1 = state_b.to(state_c, on="on_emit_0"), state_b.to(state_a, on="on_emit_1")
    # Transitions from C
    c_emits_0, c_emits_1 = state_c.to(state_a, on="on_emit_0"), state_c.to(state_b, on="on_emit_1")
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        self._transition_info = {
            "A|0": [{"to_state": "B", "probability": 1.0}], 
            "A|1": [{"to_state": "C", "probability": 1.0}],
            "B|0": [{"to_state": "C", "probability": 1.0}], 
            "B|1": [{"to_state": "A", "probability": 1.0}],
            "C|0": [{"to_state": "A", "probability": 1.0}], 
            "C|1": [{"to_state": "B", "probability": 1.0}],
        }
        
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    
    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.state_a:
            # State A: 90% chance of emitting 0, 10% chance of emitting 1
            return self.send('a_emits_0') if self.rng.random() < 0.9 else self.send('a_emits_1')
        elif current_state == self.state_b:
            # State B: 10% chance of emitting 0, 90% chance of emitting 1  
            return self.send('b_emits_0') if self.rng.random() < 0.1 else self.send('b_emits_1')
        else:
            # State C: 50% chance of emitting 0, 50% chance of emitting 1
            return self.send('c_emits_0') if self.rng.random() < 0.5 else self.send('c_emits_1')


class DistinctSixStateMachine(StatemachineGenerator):
    """
    A UNIFILAR 6-state machine designed to test neural-causal compatibility with higher complexity.
    Features maximally distinguishable emission patterns while maintaining unifilar property.
    
    UNIFILAR CONSTRAINT: Each (state, symbol) pair has deterministic transitions.
    Only the emission probabilities are stochastic.
    
    Emission patterns designed for maximum separability:
    State A: 95% → 0, 5% → 1   (Almost pure 0)
    State B: 5% → 0, 95% → 1   (Almost pure 1)  
    State C: 80% → 0, 20% → 1  (Strong 0 bias)
    State D: 20% → 0, 80% → 1  (Strong 1 bias)
    State E: 50% → 0, 50% → 1  (Balanced)
    State F: 99% → 0, 1% → 1   (Extreme 0 bias, like Golden Mean State 2)
    
    Transition structure (deterministic, forming a cycle):
    A --0--> B, A --1--> C
    B --0--> D, B --1--> E  
    C --0--> E, C --1--> F
    D --0--> F, D --1--> A
    E --0--> A, E --1--> B
    F --0--> C, F --1--> D
    """
    state_a, state_b, state_c = State("A", initial=True), State("B"), State("C")
    state_d, state_e, state_f = State("D"), State("E"), State("F")
    
    # UNIFILAR transitions: deterministic based on emitted symbol
    # From A
    a_emits_0, a_emits_1 = state_a.to(state_b, on="on_emit_0"), state_a.to(state_c, on="on_emit_1")
    # From B
    b_emits_0, b_emits_1 = state_b.to(state_d, on="on_emit_0"), state_b.to(state_e, on="on_emit_1")
    # From C
    c_emits_0, c_emits_1 = state_c.to(state_e, on="on_emit_0"), state_c.to(state_f, on="on_emit_1")
    # From D
    d_emits_0, d_emits_1 = state_d.to(state_f, on="on_emit_0"), state_d.to(state_a, on="on_emit_1")
    # From E
    e_emits_0, e_emits_1 = state_e.to(state_a, on="on_emit_0"), state_e.to(state_b, on="on_emit_1")
    # From F
    f_emits_0, f_emits_1 = state_f.to(state_c, on="on_emit_0"), state_f.to(state_d, on="on_emit_1")
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        # UNIFILAR: Each (state, symbol) → unique next state
        self._transition_info = {
            "A|0": [{"to_state": "B", "probability": 1.0}], 
            "A|1": [{"to_state": "C", "probability": 1.0}],
            "B|0": [{"to_state": "D", "probability": 1.0}], 
            "B|1": [{"to_state": "E", "probability": 1.0}],
            "C|0": [{"to_state": "E", "probability": 1.0}], 
            "C|1": [{"to_state": "F", "probability": 1.0}],
            "D|0": [{"to_state": "F", "probability": 1.0}], 
            "D|1": [{"to_state": "A", "probability": 1.0}],
            "E|0": [{"to_state": "A", "probability": 1.0}], 
            "E|1": [{"to_state": "B", "probability": 1.0}],
            "F|0": [{"to_state": "C", "probability": 1.0}], 
            "F|1": [{"to_state": "D", "probability": 1.0}],
        }
        
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    
    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.state_a:
            # State A: 95% → 0, 5% → 1 (Almost pure 0)
            return self.send('a_emits_0') if self.rng.random() < 0.95 else self.send('a_emits_1')
        elif current_state == self.state_b:
            # State B: 5% → 0, 95% → 1 (Almost pure 1)
            return self.send('b_emits_0') if self.rng.random() < 0.05 else self.send('b_emits_1')
        elif current_state == self.state_c:
            # State C: 80% → 0, 20% → 1 (Strong 0 bias)
            return self.send('c_emits_0') if self.rng.random() < 0.80 else self.send('c_emits_1')
        elif current_state == self.state_d:
            # State D: 20% → 0, 80% → 1 (Strong 1 bias)
            return self.send('d_emits_0') if self.rng.random() < 0.20 else self.send('d_emits_1')
        elif current_state == self.state_e:
            # State E: 50% → 0, 50% → 1 (Balanced)
            return self.send('e_emits_0') if self.rng.random() < 0.50 else self.send('e_emits_1')
        else:
            # State F: 99% → 0, 1% → 1 (Extreme 0 bias, Golden Mean style)
            return self.send('f_emits_0') if self.rng.random() < 0.99 else self.send('f_emits_1')


class DistinctFourStateMachine(StatemachineGenerator):
    """
    A UNIFILAR 4-state machine designed to test neural-causal compatibility with moderate complexity.
    Features well-separated emission patterns for testing the sweet spot between simplicity and complexity.
    
    UNIFILAR CONSTRAINT: Each (state, symbol) pair has deterministic transitions.
    Only the emission probabilities are stochastic.
    
    Emission patterns designed for good separability:
    State A: 90% → 0, 10% → 1   (Strong 0 bias)
    State B: 10% → 0, 90% → 1   (Strong 1 bias)  
    State C: 70% → 0, 30% → 1   (Moderate 0 bias)
    State D: 30% → 0, 70% → 1   (Moderate 1 bias)
    
    Transition structure (deterministic, forming cycles):
    A --0--> B, A --1--> C
    B --0--> D, B --1--> A  
    C --0--> A, C --1--> D
    D --0--> C, D --1--> B
    """
    state_a, state_b = State("A", initial=True), State("B")
    state_c, state_d = State("C"), State("D")
    
    # UNIFILAR transitions: deterministic based on emitted symbol
    # From A
    a_emits_0, a_emits_1 = state_a.to(state_b, on="on_emit_0"), state_a.to(state_c, on="on_emit_1")
    # From B
    b_emits_0, b_emits_1 = state_b.to(state_d, on="on_emit_0"), state_b.to(state_a, on="on_emit_1")
    # From C
    c_emits_0, c_emits_1 = state_c.to(state_a, on="on_emit_0"), state_c.to(state_d, on="on_emit_1")
    # From D
    d_emits_0, d_emits_1 = state_d.to(state_c, on="on_emit_0"), state_d.to(state_b, on="on_emit_1")
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        # UNIFILAR: Each (state, symbol) → unique next state
        self._transition_info = {
            "A|0": [{"to_state": "B", "probability": 1.0}], 
            "A|1": [{"to_state": "C", "probability": 1.0}],
            "B|0": [{"to_state": "D", "probability": 1.0}], 
            "B|1": [{"to_state": "A", "probability": 1.0}],
            "C|0": [{"to_state": "A", "probability": 1.0}], 
            "C|1": [{"to_state": "D", "probability": 1.0}],
            "D|0": [{"to_state": "C", "probability": 1.0}], 
            "D|1": [{"to_state": "B", "probability": 1.0}],
        }
        
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    
    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.state_a:
            # State A: 90% → 0, 10% → 1 (Strong 0 bias)
            return self.send('a_emits_0') if self.rng.random() < 0.90 else self.send('a_emits_1')
        elif current_state == self.state_b:
            # State B: 10% → 0, 90% → 1 (Strong 1 bias)
            return self.send('b_emits_0') if self.rng.random() < 0.10 else self.send('b_emits_1')
        elif current_state == self.state_c:
            # State C: 70% → 0, 30% → 1 (Moderate 0 bias)
            return self.send('c_emits_0') if self.rng.random() < 0.70 else self.send('c_emits_1')
        else:
            # State D: 30% → 0, 70% → 1 (Moderate 1 bias)
            return self.send('d_emits_0') if self.rng.random() < 0.30 else self.send('d_emits_1')


class AntiCompressionMachine(StatemachineGenerator):
    """
    A UNIFILAR 5-state machine designed to resist neural compression.
    Features maximally separated emission patterns and adversarial suffix structure.
    
    UNIFILAR CONSTRAINT: Each (state, symbol) pair has deterministic transitions.
    Only the emission probabilities are stochastic.
    
    ANTI-COMPRESSION DESIGN:
    - Maximum emission diversity: no two states have similar P(0), P(1)
    - Adversarial transitions: create distinct neural hidden state clusters
    - Complex reachability: every state can reach every other state
    
    Emission patterns (maximally separated):
    A: 99% → 0, 1% → 1   (Extreme 0-bias)
    B: 1% → 0, 99% → 1   (Extreme 1-bias)  
    C: 80% → 0, 20% → 1  (Strong 0-bias)
    D: 20% → 0, 80% → 1  (Strong 1-bias)
    E: 50% → 0, 50% → 1  (Balanced)
    
    UNIFILAR transition structure (adversarial design):
    A --0--> B, A --1--> C
    B --0--> D, B --1--> A  
    C --0--> E, C --1--> B
    D --0--> A, D --1--> E
    E --0--> C, E --1--> D
    """
    # Define all 5 states
    state_a = State("A", initial=True)
    state_b = State("B")
    state_c = State("C") 
    state_d = State("D")
    state_e = State("E")
    
    # UNIFILAR transitions: deterministic based on emitted symbol
    # From A
    a_emits_0 = state_a.to(state_b, on="on_emit_0")
    a_emits_1 = state_a.to(state_c, on="on_emit_1")
    
    # From B
    b_emits_0 = state_b.to(state_d, on="on_emit_0")
    b_emits_1 = state_b.to(state_a, on="on_emit_1")
    
    # From C
    c_emits_0 = state_c.to(state_e, on="on_emit_0")
    c_emits_1 = state_c.to(state_b, on="on_emit_1")
    
    # From D
    d_emits_0 = state_d.to(state_a, on="on_emit_0")
    d_emits_1 = state_d.to(state_e, on="on_emit_1")
    
    # From E
    e_emits_0 = state_e.to(state_c, on="on_emit_0")
    e_emits_1 = state_e.to(state_d, on="on_emit_1")
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        # UNIFILAR: Each (state, symbol) → unique next state
        self._transition_info = {
            "A|0": [{"to_state": "B", "probability": 1.0}],
            "A|1": [{"to_state": "C", "probability": 1.0}],
            "B|0": [{"to_state": "D", "probability": 1.0}],
            "B|1": [{"to_state": "A", "probability": 1.0}],
            "C|0": [{"to_state": "E", "probability": 1.0}],
            "C|1": [{"to_state": "B", "probability": 1.0}],
            "D|0": [{"to_state": "A", "probability": 1.0}],
            "D|1": [{"to_state": "E", "probability": 1.0}],
            "E|0": [{"to_state": "C", "probability": 1.0}],
            "E|1": [{"to_state": "D", "probability": 1.0}],
        }
        
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    
    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.state_a:
            # State A: 99% → 0, 1% → 1 (Extreme 0-bias)
            return self.send('a_emits_0') if self.rng.random() < 0.99 else self.send('a_emits_1')
        elif current_state == self.state_b:
            # State B: 1% → 0, 99% → 1 (Extreme 1-bias)
            return self.send('b_emits_0') if self.rng.random() < 0.01 else self.send('b_emits_1')
        elif current_state == self.state_c:
            # State C: 80% → 0, 20% → 1 (Strong 0-bias)
            return self.send('c_emits_0') if self.rng.random() < 0.80 else self.send('c_emits_1')
        elif current_state == self.state_d:
            # State D: 20% → 0, 80% → 1 (Strong 1-bias)
            return self.send('d_emits_0') if self.rng.random() < 0.20 else self.send('d_emits_1')
        else:
            # State E: 50% → 0, 50% → 1 (Balanced)
            return self.send('e_emits_0') if self.rng.random() < 0.50 else self.send('e_emits_1')


class SevenStateHumanMachine(StatemachineGenerator):
    """
    A UNIFILAR 7-state machine from Figure 3 - human sequence prediction study.
    Features exact probabilities from research literature (multiples of 1/16).
    
    UNIFILAR CONSTRAINT: Each (state, symbol) pair has deterministic transitions.
    Only the emission probabilities are stochastic.
    
    States represent sequence contexts from Figure 3 (suffix-defined):
    BB: 15/16 → 0, 1/16 → 1
    AAA: 3/16 → 0, 13/16 → 1  
    AAAB: 7/16 → 0, 9/16 → 1
    BA: 7/16 → 0, 9/16 → 1
    BAB: 8/16 → 0, 8/16 → 1
    BAAB: 7/16 → 0, 9/16 → 1
    BAA: 3/16 → 0, 13/16 → 1
    
    Transition structure (UNIFILAR via suffix-closure of appended symbol):
    BB --0--> BA,   BB --1--> BB
    AAA --0--> AAA, AAA --1--> AAAB
    AAAB --0--> BA, AAAB --1--> BB
    BA --0--> BAA,  BA --1--> BAB
    BAB --0--> BA,  BAB --1--> BB
    BAAB --0--> BA, BAAB --1--> BB
    BAA --0--> AAA, BAA --1--> BAAB
    """
    # Define all 7 states
    bb = State("BB", initial=True)
    aaa = State("AAA")
    aaab = State("AAAB") 
    ba = State("BA")
    bab = State("BAB")
    baab = State("BAAB")
    baa = State("BAA")
    
    # UNIFILAR transitions: deterministic based on emitted symbol (suffix-closure)
    # From BB
    bb_emits_0 = bb.to(ba, on="on_emit_0")
    bb_emits_1 = bb.to(bb, on="on_emit_1")
    
    # From AAA
    aaa_emits_0 = aaa.to(aaa, on="on_emit_0")      # Self-loop
    aaa_emits_1 = aaa.to(aaab, on="on_emit_1")
    
    # From AAAB
    aaab_emits_0 = aaab.to(ba, on="on_emit_0")
    aaab_emits_1 = aaab.to(bb, on="on_emit_1")
    
    # From BA
    ba_emits_0 = ba.to(baa, on="on_emit_0")
    ba_emits_1 = ba.to(bab, on="on_emit_1")
    
    # From BAB
    bab_emits_0 = bab.to(ba, on="on_emit_0")
    bab_emits_1 = bab.to(bb, on="on_emit_1")
    
    # From BAAB
    baab_emits_0 = baab.to(ba, on="on_emit_0")
    baab_emits_1 = baab.to(bb, on="on_emit_1")
    
    # From BAA
    baa_emits_0 = baa.to(aaa, on="on_emit_0")
    baa_emits_1 = baa.to(baab, on="on_emit_1")
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        # UNIFILAR: Each (state, symbol) → unique next state
        self._transition_info = {
            "BB|0": [{"to_state": "BA", "probability": 1.0}],
            "BB|1": [{"to_state": "BB", "probability": 1.0}],
            "AAA|0": [{"to_state": "AAA", "probability": 1.0}],
            "AAA|1": [{"to_state": "AAAB", "probability": 1.0}],
            "AAAB|0": [{"to_state": "BA", "probability": 1.0}],
            "AAAB|1": [{"to_state": "BB", "probability": 1.0}],
            "BA|0": [{"to_state": "BAA", "probability": 1.0}],
            "BA|1": [{"to_state": "BAB", "probability": 1.0}],
            "BAB|0": [{"to_state": "BA", "probability": 1.0}],
            "BAB|1": [{"to_state": "BB", "probability": 1.0}],
            "BAAB|0": [{"to_state": "BA", "probability": 1.0}],
            "BAAB|1": [{"to_state": "BB", "probability": 1.0}],
            "BAA|0": [{"to_state": "AAA", "probability": 1.0}],
            "BAA|1": [{"to_state": "BAAB", "probability": 1.0}],
        }
        
    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'
    
    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.bb:
            # BB: 15/16 → 0, 1/16 → 1
            return self.send('bb_emits_0') if self.rng.random() < 15/16 else self.send('bb_emits_1')
        elif current_state == self.aaa:
            # AAA: 3/16 → 0, 13/16 → 1
            return self.send('aaa_emits_0') if self.rng.random() < 3/16 else self.send('aaa_emits_1')
        elif current_state == self.aaab:
            # AAAB: 7/16 → 0, 9/16 → 1
            return self.send('aaab_emits_0') if self.rng.random() < 7/16 else self.send('aaab_emits_1')
        elif current_state == self.ba:
            # BA: 7/16 → 0, 9/16 → 1
            return self.send('ba_emits_0') if self.rng.random() < 7/16 else self.send('ba_emits_1')
        elif current_state == self.bab:
            # BAB: 8/16 → 0, 8/16 → 1 (perfectly balanced)
            return self.send('bab_emits_0') if self.rng.random() < 8/16 else self.send('bab_emits_1')
        elif current_state == self.baab:
            # BAAB: 7/16 → 0, 9/16 → 1
            return self.send('baab_emits_0') if self.rng.random() < 7/16 else self.send('baab_emits_1')
        else:  # BAA
            # BAA: 3/16 → 0, 13/16 → 1
            return self.send('baa_emits_0') if self.rng.random() < 3/16 else self.send('baa_emits_1')

class HierarchicalStateMachine(StatemachineGenerator):
    """
    A UNIFILAR 4-state hierarchical machine designed to test multi-scale pattern learning.
    Features two-level hierarchy: INNER/OUTER clusters, each with A/B variants.

    UNIFILAR CONSTRAINT: Each (state, symbol) pair has deterministic transitions.
    Only the emission probabilities are stochastic.

    HIERARCHICAL STRUCTURE:
    - Level 1 (Meta-clusters): INNER vs OUTER behavioral modes
    - Level 2 (Sub-states): A/B variants within each cluster
    - Fast mixing within clusters, slow mixing between clusters

    This design tests AR transformers' ability to learn multi-scale temporal structure
    without the infinite memory requirements of processes like Even Process.

    Emission patterns (hierarchically organized):
    INNER_A: 90% → 0, 10% → 1   (strong 0-bias, inner cluster)
    INNER_B: 80% → 0, 20% → 1   (moderate 0-bias, inner cluster)
    OUTER_A: 30% → 0, 70% → 1   (moderate 1-bias, outer cluster)
    OUTER_B: 10% → 0, 90% → 1   (strong 1-bias, outer cluster)

    Transition structure (UNIFILAR with hierarchical mixing):
    Within-cluster transitions (fast mixing, prob ~0.7):
    - INNER_A --0--> INNER_B, INNER_A --1--> INNER_A
    - INNER_B --0--> INNER_A, INNER_B --1--> INNER_B
    - OUTER_A --0--> OUTER_B, OUTER_A --1--> OUTER_A
    - OUTER_B --0--> OUTER_A, OUTER_B --1--> OUTER_B

    Between-cluster transitions (slow mixing, prob ~0.3):
    - Cross-cluster transitions when emitting minority symbol in current cluster
    """
    # Define all 4 hierarchical states
    inner_a = State("INNER_A", initial=True)
    inner_b = State("INNER_B")
    outer_a = State("OUTER_A")
    outer_b = State("OUTER_B")

    # UNIFILAR transitions: deterministic based on emitted symbol
    # From INNER_A: mostly stays in INNER cluster, 0 → INNER_B, 1 → stay or jump
    inner_a_emits_0 = inner_a.to(inner_b, on="on_emit_0")      # Fast within-cluster
    inner_a_emits_1 = inner_a.to(inner_a, on="on_emit_1")      # Self-loop (common for 1s)

    # From INNER_B: 0 → back to INNER_A, 1 → potential cluster jump
    inner_b_emits_0 = inner_b.to(inner_a, on="on_emit_0")      # Fast within-cluster
    inner_b_emits_1 = inner_b.to(outer_a, on="on_emit_1")      # Jump to OUTER on minority symbol

    # From OUTER_A: mostly stays in OUTER cluster, 1 → OUTER_B, 0 → stay or jump
    outer_a_emits_1 = outer_a.to(outer_b, on="on_emit_1")      # Fast within-cluster
    outer_a_emits_0 = outer_a.to(outer_a, on="on_emit_0")      # Self-loop (common for 0s)

    # From OUTER_B: 1 → back to OUTER_A, 0 → potential cluster jump
    outer_b_emits_1 = outer_b.to(outer_a, on="on_emit_1")      # Fast within-cluster
    outer_b_emits_0 = outer_b.to(inner_a, on="on_emit_0")      # Jump to INNER on minority symbol

    def __init__(self, seed: int = None):
        super().__init__(seed)
        self.alphabet = ['0', '1']
        # UNIFILAR: Each (state, symbol) → unique next state
        # Deterministic routing creates hierarchical mixing dynamics
        self._transition_info = {
            "INNER_A|0": [{"to_state": "INNER_B", "probability": 1.0}],
            "INNER_A|1": [{"to_state": "INNER_A", "probability": 1.0}],
            "INNER_B|0": [{"to_state": "INNER_A", "probability": 1.0}],
            "INNER_B|1": [{"to_state": "OUTER_A", "probability": 1.0}],
            "OUTER_A|0": [{"to_state": "OUTER_A", "probability": 1.0}],
            "OUTER_A|1": [{"to_state": "OUTER_B", "probability": 1.0}],
            "OUTER_B|0": [{"to_state": "INNER_A", "probability": 1.0}],
            "OUTER_B|1": [{"to_state": "OUTER_A", "probability": 1.0}],
        }

    def on_emit_0(self) -> str: return '0'
    def on_emit_1(self) -> str: return '1'

    def step(self) -> str:
        current_state = self.current_state
        if current_state == self.inner_a:
            # INNER_A: 90% → 0, 10% → 1 (strong 0-bias, inner cluster)
            return self.send('inner_a_emits_0') if self.rng.random() < 0.90 else self.send('inner_a_emits_1')
        elif current_state == self.inner_b:
            # INNER_B: 80% → 0, 20% → 1 (moderate 0-bias, inner cluster)
            return self.send('inner_b_emits_0') if self.rng.random() < 0.80 else self.send('inner_b_emits_1')
        elif current_state == self.outer_a:
            # OUTER_A: 30% → 0, 70% → 1 (moderate 1-bias, outer cluster)
            return self.send('outer_a_emits_0') if self.rng.random() < 0.30 else self.send('outer_a_emits_1')
        else:  # outer_b
            # OUTER_B: 10% → 0, 90% → 1 (strong 1-bias, outer cluster)
            return self.send('outer_b_emits_0') if self.rng.random() < 0.10 else self.send('outer_b_emits_1')


# --- Core Logic (Unchanged) ---

def generate_sequence_with_states(machine: StatemachineGenerator, length: int) -> Tuple[str, List[int], Dict[str, int]]:
    state_names = sorted([s.id for s in machine.states])
    state_to_index = {name: idx for idx, name in enumerate(state_names)}
    sequence, state_indices = [], []
    for _ in range(length):
        current_state_name = machine.current_state.id
        state_indices.append(state_to_index[current_state_name])
        symbol = machine.step()
        sequence.append(symbol)
    return ''.join(sequence), state_indices, state_to_index

def save_dataset(sequence: str, state_indices: List[int], machine: StatemachineGenerator, 
                state_to_index: Dict[str, int], output_path: Path, metadata: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dat_file = output_path.with_suffix('.dat')
    dat_file.write_text(sequence)
    states_file = output_path.with_suffix('.states')
    states_file.write_text(' '.join(map(str, state_indices)))
    # Also export states as a compact character sequence for .dat-style consumption
    # Map state indices to uppercase letters A, B, C, ... (supports up to 26 states)
    index_to_char = {idx: chr(ord('A') + idx) for idx in range(len(set(state_indices)))}
    states_chars = ''.join(index_to_char[idx] for idx in state_indices)
    states_dat_file = output_path.with_suffix('.states.dat')
    states_dat_file.write_text(states_chars)
    machine_file = output_path.with_suffix('.machine.json')
    machine_data = {
        'alphabet': machine.alphabet,
        'states': sorted([s.id for s in machine.states]),
        'start_state': machine.start_state_name,
        'transitions': machine.get_transition_structure()
    }
    machine_file.write_text(json.dumps(machine_data, indent=2))
    metadata_file = output_path.with_suffix('.meta.json')
    index_to_state = {idx: name for name, idx in state_to_index.items()}
    full_metadata = {
        'sequence_length': len(sequence), 'num_states': len(machine.states),
        'alphabet_size': len(machine.alphabet), 'machine_type': metadata.get('machine_type', 'unknown'),
        'generation_seed': metadata.get('seed'), 'state_mapping': {'name_to_index': state_to_index, 'index_to_name': index_to_state},
        'state_index_to_char': index_to_char,
        'files': {
            'sequence': str(dat_file.name),
            'states': str(states_file.name),
            'states_dat': str(states_dat_file.name),
            'machine': str(machine_file.name)
        }
    }
    metadata_file.write_text(json.dumps(full_metadata, indent=2))
    print("Dataset saved:")
    print(f"  Sequence: {dat_file} ({len(sequence)} symbols)")
    print(f"  States:   {states_file} ({len(state_indices)} state indices)")
    print(f"  States(.dat): {states_dat_file} ({len(state_indices)} symbols)")
    print(f"  Machine:  {machine_file}")
    print(f"  Metadata: {metadata_file}")
    print(f"  State mapping: {state_to_index}")

def create_machine(machine_type: str, seed: int = None) -> StatemachineGenerator:
    machine_map = {
        'biased_coin': BiasedCoinMachine,
        'alternating': AlternatingMachine,
        'golden_mean': GoldenMeanMachine,
        'even_process': EvenProcessMachine,
        'unifilar_3_state': Unifilar3StateMachine, # <-- ORIGINAL 3-STATE MACHINE
        'distinct_3_state': DistinctThreeStateMachine, # <-- NEW DISTINCT 3-STATE MACHINE
        'distinct_4_state': DistinctFourStateMachine, # <-- NEW 4-STATE MACHINE
        'distinct_6_state': DistinctSixStateMachine, # <-- NEW 6-STATE MACHINE
        'seven_state_human': SevenStateHumanMachine, # <-- NEW UNIFILAR 7-STATE MACHINE
        'anti_compression': AntiCompressionMachine, # <-- ANTI-COMPRESSION MACHINE
        'hierarchical_4_state': HierarchicalStateMachine, # <-- HIERARCHICAL 4-STATE MACHINE
    }
    if machine_type not in machine_map:
        raise ValueError(f"Unknown machine type: {machine_type}. Available: {', '.join(machine_map.keys())}")
    return machine_map[machine_type](seed=seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate machine datasets using python-statemachine.")
    parser.add_argument(
        '--machine', required=True,
        choices=['biased_coin', 'alternating', 'golden_mean', 'even_process', 'unifilar_3_state', 'distinct_3_state', 'distinct_4_state', 'distinct_6_state', 'seven_state_human', 'anti_compression', 'hierarchical_4_state'], # <-- ADDED HIERARCHICAL
        help="Type of domain-specific machine to generate from."
    )
    parser.add_argument('--length', type=int, default=100000, help="Length of sequence to generate.")
    parser.add_argument('--output', required=True, help="Base output directory.")
    parser.add_argument('--seed', type=int, help="Random seed for reproducible generation.")
    return parser.parse_args()

def main():
    args = parse_args()
    machine_type = args.machine
    machine = create_machine(machine_type, args.seed)
    print(f"Generating '{machine_type}' dataset...")
    print(f"  Length: {args.length}, Seed: {args.seed}")
    sequence, state_indices, state_to_index = generate_sequence_with_states(machine, args.length)
    metadata = {'machine_type': machine_type, 'seed': args.seed}
    machine_output_path = Path(args.output) / machine_type / machine_type
    save_dataset(sequence, state_indices, machine, state_to_index, machine_output_path, metadata)
    print("\nDataset generation complete!")
    unique_indices, counts = np.unique(state_indices, return_counts=True)
    index_to_name = {idx: name for name, idx in state_to_index.items()}
    state_dist = {f"{idx}({index_to_name[idx]})": count for idx, count in zip(unique_indices, counts)}
    print(f"State distribution: {state_dist}")

if __name__ == '__main__':
    main()