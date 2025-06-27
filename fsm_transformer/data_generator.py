import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import json
import os
from pathlib import Path

from .epsilon_machine import EpsilonMachine


class SequenceDataset(Dataset):
    """PyTorch dataset for autoregressive training on epsilon-machine sequences."""
    
    def __init__(self, sequences: List[List[str]], max_length: int = 512):
        self.sequences = sequences
        self.max_length = max_length
        
        # Vocabulary mapping
        self.token_to_id = {'0': 0, '1': 1, '<PAD>': 2}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.pad_token_id = self.token_to_id['<PAD>']
        
        # Convert sequences to tensor format
        self.processed_sequences = self._process_sequences()
        
    def _process_sequences(self) -> List[Dict]:
        """Convert sequences to autoregressive training format."""
        processed = []
        
        for seq in self.sequences:
            seq_len = len(seq)
            
            # Create autoregressive pairs: for each position, predict next token
            for i in range(seq_len - 1):
                context = seq[:i+1]  # Input context
                target = seq[i+1]    # Next token to predict
                
                # Convert to token IDs
                input_ids = [self.token_to_id[token] for token in context]
                target_id = self.token_to_id[target]
                
                # Pad to max_length
                attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
                input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))
                
                processed.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'target_id': target_id,
                    'context_length': len(context)
                })
                
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_sequences[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'target_id': torch.tensor(item['target_id'], dtype=torch.long),
            'context_length': torch.tensor(item['context_length'], dtype=torch.long)
        }


class EpsilonMachineDataGenerator:
    """Generates datasets from epsilon-machine for transformer training."""
    
    def __init__(self, seed: Optional[int] = None):
        self.epsilon_machine = EpsilonMachine(seed=seed)
        self.seed = seed
        
    def generate_training_data(
        self,
        num_sequences: int = 50000,
        min_length: int = 100,
        max_length: int = 200,
        output_dir: str = "data/fsm_transformer"
    ) -> Tuple[SequenceDataset, Dict]:
        """
        Generate training dataset with sequences from epsilon-machine.
        
        Args:
            num_sequences: Number of sequences to generate
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            output_dir: Directory to save data
            
        Returns:
            (dataset, metadata)
        """
        sequences = []
        state_trajectories = []
        sequence_metadata = []
        
        print(f"Generating {num_sequences} sequences...")
        
        for i in range(num_sequences):
            if i % 10000 == 0:
                print(f"Generated {i} sequences")
                
            # Random sequence length
            length = random.randint(min_length, max_length)
            
            # Generate sequence and state trajectory
            sequence, states = self.epsilon_machine.generate_sequence(length)
            
            sequences.append(sequence)
            state_trajectories.append(states)
            sequence_metadata.append({
                'length': length,
                'start_state': states[0] if states else None,
                'end_state': states[-1] if states else None
            })
        
        # Create dataset
        dataset = SequenceDataset(sequences, max_length=max_length + 50)  # Extra padding
        
        # Compute dataset statistics
        metadata = self._compute_metadata(sequences, state_trajectories, sequence_metadata)
        
        # Save data
        self._save_data(dataset, metadata, sequences, state_trajectories, output_dir)
        
        return dataset, metadata
    
    def _compute_metadata(
        self,
        sequences: List[List[str]],
        state_trajectories: List[List[str]],
        sequence_metadata: List[Dict]
    ) -> Dict:
        """Compute dataset statistics and metadata."""
        
        # Sequence length statistics
        lengths = [len(seq) for seq in sequences]
        
        # State distribution
        all_states = [state for trajectory in state_trajectories for state in trajectory]
        state_counts = {state: all_states.count(state) for state in self.epsilon_machine.states}
        
        # Symbol distribution
        all_symbols = [symbol for seq in sequences for symbol in seq]
        symbol_counts = {'0': all_symbols.count('0'), '1': all_symbols.count('1')}
        
        # Transition statistics
        transition_counts = {}
        for trajectory, sequence in zip(state_trajectories, sequences):
            for i, (state, symbol) in enumerate(zip(trajectory, sequence)):
                key = f"{state}->{symbol}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
        
        metadata = {
            'generation_params': {
                'num_sequences': len(sequences),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'mean_length': np.mean(lengths),
                'seed': self.seed
            },
            'sequence_stats': {
                'total_sequences': len(sequences),
                'total_tokens': sum(lengths),
                'length_distribution': {
                    'min': min(lengths),
                    'max': max(lengths),
                    'mean': float(np.mean(lengths)),
                    'std': float(np.std(lengths))
                }
            },
            'state_distribution': {
                state: count / len(all_states) for state, count in state_counts.items()
            },
            'symbol_distribution': {
                symbol: count / len(all_symbols) for symbol, count in symbol_counts.items()
            },
            'transition_counts': transition_counts,
            'epsilon_machine': {
                'states': self.epsilon_machine.states,
                'transitions': self.epsilon_machine.transitions,
                'stationary_distribution': self.epsilon_machine.compute_stationary_distribution()
            },
            'dataset_info': {
                'vocab_size': len(SequenceDataset(sequences).token_to_id),
                'num_training_examples': len(SequenceDataset(sequences)),
                'autoregressive_pairs': sum(len(seq) - 1 for seq in sequences)
            }
        }
        
        return metadata
    
    def _save_data(
        self,
        dataset: SequenceDataset,
        metadata: Dict,
        sequences: List[List[str]],
        state_trajectories: List[List[str]],
        output_dir: str
    ):
        """Save dataset and metadata to disk."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw sequences and trajectories
        raw_data = {
            'sequences': sequences,
            'state_trajectories': state_trajectories,
            'metadata': metadata
        }
        
        with open(output_path / 'raw_data.json', 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        # Save processed dataset
        torch.save({
            'dataset': dataset.processed_sequences,
            'vocab': {
                'token_to_id': dataset.token_to_id,
                'id_to_token': dataset.id_to_token,
                'vocab_size': dataset.vocab_size
            }
        }, output_path / 'processed_dataset.pt')
        
        # Save metadata separately
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Data saved to {output_path}")
        print(f"Generated {len(sequences)} sequences")
        print(f"Total training examples: {len(dataset)}")
        print(f"Vocabulary size: {dataset.vocab_size}")


def load_dataset(data_dir: str) -> Tuple[SequenceDataset, Dict]:
    """Load saved dataset and metadata."""
    data_path = Path(data_dir)
    
    # Load processed dataset
    data = torch.load(data_path / 'processed_dataset.pt')
    
    # Reconstruct dataset object
    dataset = SequenceDataset([])  # Empty init
    dataset.processed_sequences = data['dataset']
    dataset.token_to_id = data['vocab']['token_to_id']
    dataset.id_to_token = data['vocab']['id_to_token']
    dataset.vocab_size = data['vocab']['vocab_size']
    dataset.pad_token_id = dataset.token_to_id['<PAD>']
    
    # Load metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return dataset, metadata