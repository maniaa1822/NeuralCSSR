"""Utility functions for theoretical limits of machine-generated data."""

from typing import Dict, Tuple

import numpy as np


def compute_stationary_distribution(machine, tol: float = 1e-12,
                                    max_iter: int = 100000) -> Dict[str, float]:
    """Compute stationary distribution over states for a unifilar machine."""

    states = list(machine.states)
    state_to_idx = {state: i for i, state in enumerate(states)}

    # Transition matrix weighted by emission probabilities
    n_states = len(states)
    transition = np.zeros((n_states, n_states))

    for (state, symbol), next_state in machine.transitions.items():
        i = state_to_idx[state]
        j = state_to_idx[next_state]
        prob = machine.emissions[state][symbol]
        transition[i, j] += prob

    pi = np.ones(n_states) / n_states

    for _ in range(max_iter):
        new_pi = transition.T @ pi
        diff = np.max(np.abs(new_pi - pi))
        pi = new_pi
        if diff < tol:
            break

    pi = pi / pi.sum()
    return {states[i]: float(pi[i]) for i in range(n_states)}


def compute_next_token_bayes_accuracy(machine) -> float:
    """Compute Bayes-optimal accuracy for next-token prediction."""

    stationary = compute_stationary_distribution(machine)
    accuracy = 0.0

    for state, pi in stationary.items():
        emissions = machine.emissions[state]
        accuracy += pi * max(emissions.values())

    return float(accuracy)
