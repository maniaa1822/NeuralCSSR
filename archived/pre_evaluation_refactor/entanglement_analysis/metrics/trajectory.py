"""
Trajectory analysis for successor alignment and geometric encoding.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math


class TrajectoryAnalyzer:
    """
    Analyzes trajectory geometry and successor state alignments.
    """

    @staticmethod
    def compute_successor_alignment(activations: np.ndarray,
                                  epsilon_states: np.ndarray,
                                  transitions: Dict[Tuple[str, str], str],
                                  symbol_sequences: List[np.ndarray],
                                  state_to_idx: Dict[str, int]) -> Dict[str, float]:
        """
        Compute successor state alignment in latent space.

        Measures how well the geometric direction from state A to state B
        aligns with the actual transitions.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state indices
            transitions: (state, symbol) -> next_state mapping
            symbol_sequences: List of symbol sequences
            state_to_idx: state_name -> state_index mapping

        Returns:
            Alignment metrics
        """
        # Compute state centroids
        state_centroids = {}
        state_counts = defaultdict(int)

        for act, state_idx in zip(activations, epsilon_states):
            if state_idx not in state_centroids:
                state_centroids[state_idx] = np.zeros(act.shape)
            state_centroids[state_idx] += act
            state_counts[state_idx] += 1

        # Normalize centroids
        for state_idx in state_centroids:
            state_centroids[state_idx] /= state_counts[state_idx]

        # Compute transition directions
        transition_directions = defaultdict(list)
        idx_to_state = {v: k for k, v in state_to_idx.items()}

        for seq, seq_acts, seq_states in zip(symbol_sequences, 
                                           np.array_split(activations, len(symbol_sequences)),
                                           np.array_split(epsilon_states, len(symbol_sequences))):
            for i in range(len(seq) - 1):
                current_state = idx_to_state[seq_states[i]]
                next_state = idx_to_state[seq_states[i + 1]]
                symbol = str(seq[i])

                # Check if this is a valid transition
                expected_next = transitions.get((current_state, symbol))
                if expected_next == next_state:
                    # Valid transition: compute direction
                    direction = seq_acts[i + 1] - seq_acts[i]
                    transition_directions[(seq_states[i], seq_states[i + 1])].append(direction)

        # Compute mean transition directions
        mean_directions = {}
        for (from_state, to_state), directions in transition_directions.items():
            if len(directions) > 0:
                mean_directions[(from_state, to_state)] = np.mean(directions, axis=0)

        # Compute alignment with centroid differences
        alignments = []
        for (from_state, to_state), mean_dir in mean_directions.items():
            if from_state in state_centroids and to_state in state_centroids:
                centroid_diff = state_centroids[to_state] - state_centroids[from_state]

                # Cosine similarity
                cos_sim = np.dot(mean_dir, centroid_diff) / (
                    np.linalg.norm(mean_dir) * np.linalg.norm(centroid_diff)
                )
                alignments.append(cos_sim)

        if alignments:
            return {
                'mean_successor_alignment': np.mean(alignments),
                'median_successor_alignment': np.median(alignments),
                'std_successor_alignment': np.std(alignments),
                'n_transitions_analyzed': len(alignments)
            }
        else:
            return {
                'mean_successor_alignment': 0.0,
                'median_successor_alignment': 0.0,
                'std_successor_alignment': 0.0,
                'n_transitions_analyzed': 0
            }

    @staticmethod
    def compute_trajectory_curvature(activations: np.ndarray,
                                   epsilon_states: np.ndarray,
                                   symbol_sequences: List[np.ndarray],
                                   window_size: int = 3) -> Dict[str, float]:
        """
        Compute curvature along state trajectories.

        Measures how much the trajectory bends, indicating geometric
        vs tabular encoding.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state indices
            symbol_sequences: List of symbol sequences
            window_size: Window size for curvature computation

        Returns:
            Curvature metrics
        """
        curvatures = []

        for seq_acts in np.array_split(activations, len(symbol_sequences)):
            if len(seq_acts) < window_size:
                continue

            # Compute angles between consecutive segments
            for i in range(len(seq_acts) - window_size + 1):
                window_acts = seq_acts[i:i + window_size]

                # Compute vectors between consecutive points
                vectors = []
                for j in range(len(window_acts) - 1):
                    vec = window_acts[j + 1] - window_acts[j]
                    vectors.append(vec)

                # Compute angles between consecutive vectors
                for j in range(len(vectors) - 1):
                    v1 = vectors[j]
                    v2 = vectors[j + 1]

                    # Cosine similarity
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical issues

                    angle = math.acos(cos_angle)
                    curvatures.append(angle)

        if curvatures:
            # Convert to curvature (lower angle = less curvature)
            curvature_scores = np.array(curvatures)
            return {
                'mean_curvature': np.mean(curvature_scores),
                'median_curvature': np.median(curvature_scores),
                'std_curvature': np.std(curvature_scores),
                'max_curvature': np.max(curvature_scores),
                'n_windows_analyzed': len(curvatures)
            }
        else:
            return {
                'mean_curvature': 0.0,
                'median_curvature': 0.0,
                'std_curvature': 0.0,
                'max_curvature': 0.0,
                'n_windows_analyzed': 0
            }

    @staticmethod
    def analyze_state_space_geometry(activations: np.ndarray,
                                   epsilon_states: np.ndarray,
                                   transitions: Dict[Tuple[str, str], str],
                                   state_to_idx: Dict[str, int]) -> Dict[str, Any]:
        """
        Comprehensive geometric analysis of state space.

        Args:
            activations: (N, hidden_dim) representations
            epsilon_states: (N,) state indices
            transitions: State transition mapping
            state_to_idx: state_name -> state_index mapping

        Returns:
            Geometric analysis results
        """
        # Compute state centroids
        state_centroids = {}
        state_counts = defaultdict(int)

        for act, state_idx in zip(activations, epsilon_states):
            if state_idx not in state_centroids:
                state_centroids[state_idx] = np.zeros(act.shape)
            state_centroids[state_idx] += act
            state_counts[state_idx] += 1

        for state_idx in state_centroids:
            state_centroids[state_idx] /= state_counts[state_idx]

        # Analyze transition structure
        transition_lengths = []
        transition_angles = []

        idx_to_state = {v: k for k, v in state_to_idx.items()}

        for (from_state, symbol), to_state in transitions.items():
            from_idx = state_to_idx[from_state]
            to_idx = state_to_idx[to_state]

            if from_idx in state_centroids and to_idx in state_centroids:
                transition_vec = state_centroids[to_idx] - state_centroids[from_idx]
                transition_lengths.append(np.linalg.norm(transition_vec))

                # Find other transitions from this state to compute angles
                from_transitions = [
                    state_to_idx[transitions[(from_state, sym)]] - from_idx
                    for sym in ['0', '1']
                    if (from_state, sym) in transitions
                ]

                if len(from_transitions) == 2:
                    v1 = np.array(from_transitions[0])
                    v2 = np.array(from_transitions[1])

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = math.acos(cos_angle)
                    transition_angles.append(angle)

        geometry_results = {
            'mean_transition_length': np.mean(transition_lengths) if transition_lengths else 0.0,
            'std_transition_length': np.std(transition_lengths) if transition_lengths else 0.0,
            'mean_transition_angle': np.mean(transition_angles) if transition_angles else 0.0,
            'std_transition_angle': np.std(transition_angles) if transition_angles else 0.0,
            'n_transitions_analyzed': len(transition_lengths)
        }

        # State space regularity
        centroid_positions = np.array(list(state_centroids.values()))
        if len(centroid_positions) > 1:
            # Pairwise distances between centroids
            distances = []
            for i in range(len(centroid_positions)):
                for j in range(i + 1, len(centroid_positions)):
                    dist = np.linalg.norm(centroid_positions[i] - centroid_positions[j])
                    distances.append(dist)

            geometry_results.update({
                'mean_centroid_distance': np.mean(distances),
                'std_centroid_distance': np.std(distances),
                'min_centroid_distance': np.min(distances),
                'max_centroid_distance': np.max(distances)
            })

        return geometry_results
