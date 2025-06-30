"""
Neural dataset formatter for Neural CSSR.

This module converts raw sequences to PyTorch-compatible datasets
for transformer training with rich ground truth annotations.
"""

from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class NeuralDatasetFormatter:
    """
    Formats raw sequences into PyTorch datasets for neural network training.
    
    Creates autoregressive datasets with attention masks, ground truth labels,
    and comprehensive metadata for transformer-based causal state discovery.
    """
    
    def __init__(self):
        """Initialize neural dataset formatter."""
        pass
        
    def create_pytorch_datasets(
        self, 
        sequences_data: Dict[str, Any],
        vocab_config: Dict[str, Any],
        context_length: int = 512
    ) -> Dataset:
        """
        Convert raw sequences to PyTorch datasets for transformer training.
        
        Args:
            sequences_data: Dictionary containing sequences and metadata
            vocab_config: Vocabulary configuration with alphabet and special tokens
            context_length: Maximum sequence length for transformers
            
        Returns:
            PyTorch Dataset instance
        """
        # Build vocabulary
        vocab_info = self._build_vocabulary(vocab_config)
        
        # Create training examples
        examples = self._create_training_examples(
            sequences_data, vocab_info, context_length
        )
        
        # Create PyTorch dataset
        dataset = NeuralCSSRDataset(examples, vocab_info)
        
        return dataset
        
    def _build_vocabulary(self, vocab_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build vocabulary from configuration.
        
        Args:
            vocab_config: Configuration containing alphabet and special tokens
            
        Returns:
            Dictionary with vocabulary mappings and metadata
        """
        alphabet = vocab_config['alphabet']
        special_tokens = vocab_config.get('special_tokens', ['<PAD>', '<UNK>'])
        
        # Create token list with special tokens first
        all_tokens = special_tokens + alphabet
        
        # Create mappings
        token_to_id = {token: i for i, token in enumerate(all_tokens)}
        id_to_token = {i: token for token, i in token_to_id.items()}
        
        # Special token IDs
        pad_token_id = token_to_id.get('<PAD>', 0)
        unk_token_id = token_to_id.get('<UNK>', len(special_tokens) - 1)
        
        return {
            'all_tokens': all_tokens,
            'alphabet': alphabet,
            'special_tokens': special_tokens,
            'token_to_id': token_to_id,
            'id_to_token': id_to_token,
            'vocab_size': len(all_tokens),
            'pad_token_id': pad_token_id,
            'unk_token_id': unk_token_id
        }
        
    def _create_training_examples(
        self, 
        sequences_data: Dict[str, Any], 
        vocab_info: Dict[str, Any],
        context_length: int
    ) -> List[Dict[str, Any]]:
        """
        Create autoregressive training examples from sequences.
        
        Args:
            sequences_data: Raw sequences with metadata
            vocab_info: Vocabulary information
            context_length: Maximum context length
            
        Returns:
            List of training examples
        """
        sequences = sequences_data['sequences']
        metadata = sequences_data['metadata']
        
        examples = []
        
        for seq_idx, (sequence, seq_metadata) in enumerate(zip(sequences, metadata)):
            # Create autoregressive examples for this sequence
            seq_examples = self._create_autoregressive_examples(
                sequence, seq_metadata, vocab_info, context_length, seq_idx
            )
            examples.extend(seq_examples)
            
        return examples
        
    def _create_autoregressive_examples(
        self, 
        sequence: List[str], 
        seq_metadata: Dict[str, Any],
        vocab_info: Dict[str, Any],
        context_length: int,
        seq_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Create autoregressive training examples from a single sequence.
        
        Args:
            sequence: List of symbols
            seq_metadata: Metadata for this sequence
            vocab_info: Vocabulary information
            context_length: Maximum context length
            seq_idx: Sequence index
            
        Returns:
            List of training examples
        """
        examples = []
        state_trajectory = seq_metadata.get('state_trajectory', [])
        transition_log = seq_metadata.get('transition_log', [])
        
        # Convert sequence to token IDs
        token_ids = []
        for symbol in sequence:
            token_id = vocab_info['token_to_id'].get(
                symbol, vocab_info['unk_token_id']
            )
            token_ids.append(token_id)
            
        # Create autoregressive examples
        for pos in range(1, len(token_ids)):
            # Determine context window
            start_pos = max(0, pos - context_length + 1)
            context_ids = token_ids[start_pos:pos]
            target_id = token_ids[pos]
            
            # Pad context if necessary
            if len(context_ids) < context_length:
                padding_length = context_length - len(context_ids)
                padded_context = [vocab_info['pad_token_id']] * padding_length + context_ids
                attention_mask = [0] * padding_length + [1] * len(context_ids)
            else:
                padded_context = context_ids
                attention_mask = [1] * len(context_ids)
                
            # Get ground truth information
            ground_truth = self._extract_ground_truth(
                pos, state_trajectory, transition_log, seq_metadata
            )
            
            example = {
                # Standard transformer inputs
                'input_ids': padded_context,
                'attention_mask': attention_mask,
                'target_ids': target_id,
                
                # Ground truth labels
                'causal_states': ground_truth.get('causal_state', 'UNKNOWN'),
                'machine_id': seq_metadata.get('machine_id', -1),
                'complexity_class': seq_metadata.get('complexity_class', 'unknown'),
                
                # Position-wise metadata
                'true_probs': ground_truth.get('transition_prob', 0.5),
                'transition_types': ground_truth.get('transition_type', 0),
                
                # Sequence metadata
                'original_length': len(sequence),
                'sequence_id': seq_idx,
                'position_in_sequence': pos,
                'context_length_used': len(context_ids),
                
                # Machine properties
                'num_states': seq_metadata.get('machine_properties', {}).get('num_states', 0),
                'statistical_complexity': seq_metadata.get('machine_properties', {}).get('statistical_complexity', 0.0),
                'entropy_rate': seq_metadata.get('machine_properties', {}).get('entropy_rate', 0.0)
            }
            
            examples.append(example)
            
        return examples
        
    def _extract_ground_truth(
        self, 
        position: int, 
        state_trajectory: List[str], 
        transition_log: List[Dict],
        seq_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract ground truth information for a specific position.
        
        Args:
            position: Position in sequence
            state_trajectory: State at each position
            transition_log: Transition details
            seq_metadata: Sequence metadata
            
        Returns:
            Dictionary with ground truth information
        """
        ground_truth = {}
        
        # Causal state at this position
        if state_trajectory and position < len(state_trajectory):
            ground_truth['causal_state'] = state_trajectory[position]
        else:
            ground_truth['causal_state'] = 'UNKNOWN'
            
        # Transition information
        if transition_log and position - 1 < len(transition_log):
            transition = transition_log[position - 1]
            ground_truth['transition_prob'] = transition.get('probability', 0.5)
            ground_truth['from_state'] = transition.get('from_state', 'UNKNOWN')
            ground_truth['to_state'] = transition.get('to_state', 'UNKNOWN')
            
            # Encode transition type as integer
            transition_key = f"{transition.get('from_state', '')}-{transition.get('symbol', '')}->{transition.get('to_state', '')}"
            ground_truth['transition_type'] = hash(transition_key) % 1000  # Simple encoding
        else:
            ground_truth['transition_prob'] = 0.5
            ground_truth['transition_type'] = 0
            
        return ground_truth
        
    def create_state_mapping(self, sequences_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Create mapping from causal state names to integers.
        
        Args:
            sequences_data: Sequences with metadata
            
        Returns:
            Dictionary mapping state names to IDs
        """
        all_states = set()
        
        for seq_metadata in sequences_data['metadata']:
            state_trajectory = seq_metadata.get('state_trajectory', [])
            all_states.update(state_trajectory)
            
        # Remove empty states
        all_states.discard('')
        all_states.discard('UNKNOWN')
        
        # Create mapping
        state_to_id = {'UNKNOWN': 0}
        for i, state in enumerate(sorted(all_states), 1):
            state_to_id[state] = i
            
        return state_to_id


class NeuralCSSRDataset(Dataset):
    """
    PyTorch Dataset for Neural CSSR training.
    
    Provides autoregressive examples with ground truth causal states
    and comprehensive metadata for transformer-based learning.
    """
    
    def __init__(self, examples: List[Dict[str, Any]], vocab_info: Dict[str, Any]):
        """
        Initialize dataset.
        
        Args:
            examples: List of training examples
            vocab_info: Vocabulary information
        """
        self.examples = examples
        self.vocab_info = vocab_info
        
        # Create state mapping if needed
        self.state_mapping = self._create_state_mapping()
        
    def _create_state_mapping(self) -> Dict[str, int]:
        """Create mapping from state names to IDs."""
        all_states = set()
        for example in self.examples:
            all_states.add(example['causal_states'])
            
        state_to_id = {}
        for i, state in enumerate(sorted(all_states)):
            state_to_id[state] = i
            
        return state_to_id
        
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example as tensors.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary of tensors for training
        """
        example = self.examples[idx]
        
        # Convert to tensors
        tensor_example = {
            # Standard transformer inputs
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.bool),
            'target_ids': torch.tensor(example['target_ids'], dtype=torch.long),
            
            # Ground truth labels
            'causal_states': torch.tensor(
                self.state_mapping.get(example['causal_states'], 0), 
                dtype=torch.long
            ),
            'machine_id': torch.tensor(example['machine_id'], dtype=torch.long),
            'complexity_class': example['complexity_class'],  # Keep as string
            
            # Position-wise metadata
            'true_probs': torch.tensor(example['true_probs'], dtype=torch.float),
            'transition_types': torch.tensor(example['transition_types'], dtype=torch.long),
            
            # Sequence metadata
            'original_length': torch.tensor(example['original_length'], dtype=torch.long),
            'sequence_id': torch.tensor(example['sequence_id'], dtype=torch.long),
            'position_in_sequence': torch.tensor(example['position_in_sequence'], dtype=torch.long),
            'context_length_used': torch.tensor(example['context_length_used'], dtype=torch.long),
            
            # Machine properties
            'num_states': torch.tensor(example['num_states'], dtype=torch.long),
            'statistical_complexity': torch.tensor(example['statistical_complexity'], dtype=torch.float),
            'entropy_rate': torch.tensor(example['entropy_rate'], dtype=torch.float)
        }
        
        return tensor_example
        
    def get_vocab_info(self) -> Dict[str, Any]:
        """Return vocabulary information."""
        return self.vocab_info
        
    def get_state_mapping(self) -> Dict[str, int]:
        """Return state name to ID mapping."""
        return self.state_mapping
        
    def get_example_by_sequence(self, sequence_id: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get all examples from a specific sequence.
        
        Args:
            sequence_id: Sequence identifier
            
        Returns:
            List of examples from that sequence
        """
        sequence_examples = []
        for i, example in enumerate(self.examples):
            if example['sequence_id'] == sequence_id:
                sequence_examples.append(self.__getitem__(i))
                
        return sequence_examples
        
    def compute_dataset_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the dataset.
        
        Returns:
            Dictionary of dataset statistics
        """
        # Length statistics
        lengths = [ex['original_length'] for ex in self.examples]
        context_lengths = [ex['context_length_used'] for ex in self.examples]
        
        # State distribution
        state_counts = {}
        for example in self.examples:
            state = example['causal_states']
            state_counts[state] = state_counts.get(state, 0) + 1
            
        # Machine distribution
        machine_counts = {}
        for example in self.examples:
            machine_id = example['machine_id']
            machine_counts[machine_id] = machine_counts.get(machine_id, 0) + 1
            
        # Complexity distribution
        complexity_counts = {}
        for example in self.examples:
            complexity = example['complexity_class']
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
        return {
            'total_examples': len(self.examples),
            'unique_sequences': len(set(ex['sequence_id'] for ex in self.examples)),
            'length_statistics': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'std': np.std(lengths)
            },
            'context_length_statistics': {
                'min': min(context_lengths),
                'max': max(context_lengths),
                'mean': np.mean(context_lengths),
                'std': np.std(context_lengths)
            },
            'state_distribution': state_counts,
            'machine_distribution': machine_counts,
            'complexity_distribution': complexity_counts,
            'vocab_size': self.vocab_info['vocab_size'],
            'num_states': len(self.state_mapping)
        }