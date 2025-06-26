from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from ..core.epsilon_machine import EpsilonMachine
from ..enumeration.enumerate_machines import enumerate_machines_library


class EpsilonMachineDataset(Dataset):
    """
    PyTorch Dataset for epsilon-machine sequences with complete ground truth annotations.
    
    Each example contains:
    - Input sequence (history)
    - Target next token
    - Ground truth causal state
    - Machine properties
    """
    
    def __init__(self, machine_library: List[Dict], 
                 sequences_per_machine: int = 100,
                 sequence_length: int = 50,
                 max_history_length: int = 20,
                 complexity_range: Optional[Tuple[int, int]] = None,
                 use_variable_length: bool = False):
        """
        Initialize dataset from machine library.
        
        Args:
            machine_library: List of machine dictionaries from enumerate_machines
            sequences_per_machine: Number of sequences per machine
            sequence_length: Length of each generated sequence
            max_history_length: Maximum history length for training examples
            complexity_range: (min_states, max_states) to include
            use_variable_length: If True, __getitem__ returns variable-length tensors
        """
        self.machine_library = machine_library
        self.sequences_per_machine = sequences_per_machine
        self.sequence_length = sequence_length
        self.max_history_length = max_history_length
        self.use_variable_length = use_variable_length
        
        # Filter machines by complexity if specified
        if complexity_range:
            min_complexity, max_complexity = complexity_range
            self.machine_library = [
                m for m in machine_library 
                if min_complexity <= m['properties']['num_states'] <= max_complexity
            ]
        
        # Build vocabulary
        self.alphabet = self._build_alphabet()
        self.vocab_size = len(self.alphabet)
        self.token_to_id = {token: i for i, token in enumerate(self.alphabet)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        # Generate all training examples
        self.examples = self._generate_examples()
        
    def _build_alphabet(self) -> List[str]:
        """Extract alphabet from machine library."""
        alphabet_set = set()
        for machine_data in self.machine_library:
            alphabet_set.update(machine_data['machine'].alphabet)
        return sorted(list(alphabet_set))
        
    def _generate_examples(self) -> List[Dict]:
        """Generate all training examples from machine library."""
        examples = []
        
        for machine_data in self.machine_library:
            machine = machine_data['machine']
            machine_id = machine_data['id']
            properties = machine_data['properties']
            
            # Generate sequences from this machine
            for seq_idx in range(self.sequences_per_machine):
                sequence = machine.generate_sequence(self.sequence_length)
                
                # Create training examples from sequence
                seq_examples = self._create_examples_from_sequence(
                    sequence, machine, machine_id, properties, seq_idx
                )
                examples.extend(seq_examples)
                
        return examples
        
    def _create_examples_from_sequence(self, sequence: List[str], machine: EpsilonMachine,
                                     machine_id: int, properties: Dict, 
                                     sequence_id: int) -> List[Dict]:
        """Create training examples from a single sequence."""
        examples = []
        
        for pos in range(1, len(sequence)):
            # Limit history length
            start_pos = max(0, pos - self.max_history_length)
            history = sequence[start_pos:pos]
            target = sequence[pos]
            
            # FIX: Compute causal state from SAME history as transformer sees
            # Instead of: causal_state = machine.compute_causal_state(sequence[:pos])
            causal_state = machine.compute_causal_state(history)
            
            # Get emission probabilities
            if causal_state:
                emission_probs = machine.get_conditional_distribution(causal_state)
                target_prob = emission_probs.get(target, 0.0)
            else:
                emission_probs = {}
                target_prob = 0.0
                
            example = {
                'history': history,
                'target': target,
                'history_length': len(history),
                'causal_state': causal_state or 'UNKNOWN',
                'machine_id': machine_id,
                'target_prob': target_prob,
                'emission_probs': emission_probs,
                'num_states': properties['num_states'],
                'statistical_complexity': properties['statistical_complexity'],
                'entropy_rate': properties['entropy_rate'],
                'sequence_id': sequence_id,
                'position': pos
            }
            
            examples.append(example)
            
        return examples

    def _create_examples_from_sequence_alternative(self, sequence: List[str], machine: EpsilonMachine,
                                                 machine_id: int, properties: Dict, 
                                                 sequence_id: int) -> List[Dict]:
        """Alternative: Keep full causal state but add position info."""
        examples = []
        
        for pos in range(1, len(sequence)):
            start_pos = max(0, pos - self.max_history_length)
            history = sequence[start_pos:pos]
            target = sequence[pos]
            
            # Compute both full and truncated causal states
            full_causal_state = machine.compute_causal_state(sequence[:pos])
            truncated_causal_state = machine.compute_causal_state(history)
            
            example = {
                'history': history,
                'target': target,
                'full_causal_state': full_causal_state or 'UNKNOWN',
                'truncated_causal_state': truncated_causal_state or 'UNKNOWN',
                'causal_state': truncated_causal_state or 'UNKNOWN',  # Use truncated for training
                'machine_id': machine_id,
                'sequence_id': sequence_id,
                'position': pos,
                'truncation_occurred': pos > self.max_history_length
            }
            
            examples.append(example)
            
        return examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example as tensors."""
        example = self.examples[idx]
        
        # Convert history to token IDs
        history_ids = [self.token_to_id[token] for token in example['history']]
        target_id = self.token_to_id[example['target']]
        
        if self.use_variable_length:
            # Return variable-length tensors (no padding)
            return {
                'input_ids': torch.tensor(history_ids, dtype=torch.long),
                'target_id': torch.tensor(target_id, dtype=torch.long),
                'history_length': torch.tensor(len(history_ids), dtype=torch.long),
                # Include other fields for compatibility
                'target_prob': torch.tensor(example['target_prob'], dtype=torch.float),
                'machine_id': torch.tensor(example['machine_id'], dtype=torch.long),
                'num_states': torch.tensor(example['num_states'], dtype=torch.long),
                'statistical_complexity': torch.tensor(example['statistical_complexity'], dtype=torch.float),
                'entropy_rate': torch.tensor(example['entropy_rate'], dtype=torch.float),
                'causal_state': example['causal_state'],
            }
        else:
            # Original behavior: pad history to max length
            if len(history_ids) < self.max_history_length:
                # Pad with zeros (assuming 0 is padding token)
                padding = [0] * (self.max_history_length - len(history_ids))
                history_ids = padding + history_ids
                attention_mask = [0] * len(padding) + [1] * len(history_ids[len(padding):])
            else:
                attention_mask = [1] * len(history_ids)
                
            # Convert emission probabilities to tensor
            emission_tensor = torch.zeros(self.vocab_size)
            for token, prob in example['emission_probs'].items():
                if token in self.token_to_id:
                    emission_tensor[self.token_to_id[token]] = prob
                    
            return {
                'input_ids': torch.tensor(history_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'target_id': torch.tensor(target_id, dtype=torch.long),
                'target_prob': torch.tensor(example['target_prob'], dtype=torch.float),
            'emission_probs': emission_tensor,
            'machine_id': torch.tensor(example['machine_id'], dtype=torch.long),
            'num_states': torch.tensor(example['num_states'], dtype=torch.long),
            'statistical_complexity': torch.tensor(example['statistical_complexity'], dtype=torch.float),
            'entropy_rate': torch.tensor(example['entropy_rate'], dtype=torch.float),
            'history_length': torch.tensor(example['history_length'], dtype=torch.long),
            'causal_state': example['causal_state'],  # Keep as string for now
        }
        
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                      num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Stack tensor fields
        tensor_fields = ['input_ids', 'attention_mask', 'target_id', 'target_prob',
                        'emission_probs', 'machine_id', 'num_states', 
                        'statistical_complexity', 'entropy_rate', 'history_length']
        
        collated = {}
        for field in tensor_fields:
            collated[field] = torch.stack([item[field] for item in batch])
            
        # Keep causal states as list of strings
        collated['causal_state'] = [item['causal_state'] for item in batch]
        
        return collated


class CurriculumDatasetManager:
    """
    Manages curriculum learning progression through machine complexities.
    """
    
    def __init__(self, machine_library: List[Dict], 
                 sequences_per_machine: int = 100,
                 sequence_length: int = 50,
                 max_history_length: int = 20):
        """
        Initialize curriculum manager.
        
        Args:
            machine_library: Complete machine library
            sequences_per_machine: Sequences per machine
            sequence_length: Length of sequences
            max_history_length: Max history length
        """
        self.machine_library = machine_library
        self.sequences_per_machine = sequences_per_machine
        self.sequence_length = sequence_length
        self.max_history_length = max_history_length
        
        # Group machines by complexity
        self.machines_by_complexity = defaultdict(list)
        for machine_data in machine_library:
            complexity = machine_data['properties']['num_states']
            self.machines_by_complexity[complexity].append(machine_data)
            
        self.complexities = sorted(self.machines_by_complexity.keys())
        
    def get_stage_dataset(self, max_complexity: int) -> EpsilonMachineDataset:
        """
        Get dataset for curriculum stage up to specified complexity.
        
        Args:
            max_complexity: Maximum number of states to include
            
        Returns:
            Dataset containing machines up to max_complexity
        """
        stage_machines = []
        for complexity in self.complexities:
            if complexity <= max_complexity:
                stage_machines.extend(self.machines_by_complexity[complexity])
                
        return EpsilonMachineDataset(
            machine_library=stage_machines,
            sequences_per_machine=self.sequences_per_machine,
            sequence_length=self.sequence_length,
            max_history_length=self.max_history_length
        )
        
    def get_all_stages(self) -> Dict[int, EpsilonMachineDataset]:
        """
        Get datasets for all curriculum stages.
        
        Returns:
            Dictionary mapping max_complexity -> dataset
        """
        stages = {}
        for max_complexity in self.complexities:
            stages[max_complexity] = self.get_stage_dataset(max_complexity)
        return stages


def create_train_val_datasets(max_states: int = 4,
                            alphabet: List[str] = ['0', '1'],
                            sequences_per_machine: int = 100,
                            sequence_length: int = 50,
                            max_history_length: int = 20,
                            val_split: float = 0.2,
                            max_machines_per_size: int = 50) -> Tuple[EpsilonMachineDataset, EpsilonMachineDataset]:
    """
    Create training and validation datasets.
    
    Args:
        max_states: Maximum number of states to enumerate
        alphabet: Alphabet symbols
        sequences_per_machine: Sequences per machine
        sequence_length: Length of sequences
        max_history_length: Maximum history length
        val_split: Fraction for validation
        max_machines_per_size: Max machines per state count
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print(f"Enumerating machines up to {max_states} states...")
    machine_library = enumerate_machines_library(
        max_states=max_states,
        alphabet=alphabet,
        max_machines_per_size=max_machines_per_size
    )
    print(f"Generated {len(machine_library)} machines")
    
    # Split machines for train/val
    random.shuffle(machine_library)
    val_size = int(len(machine_library) * val_split)
    
    val_machines = machine_library[:val_size]
    train_machines = machine_library[val_size:]
    
    print(f"Train machines: {len(train_machines)}, Val machines: {len(val_machines)}")
    
    # Create datasets
    train_dataset = EpsilonMachineDataset(
        machine_library=train_machines,
        sequences_per_machine=sequences_per_machine,
        sequence_length=sequence_length,
        max_history_length=max_history_length
    )
    
    val_dataset = EpsilonMachineDataset(
        machine_library=val_machines,
        sequences_per_machine=sequences_per_machine // 2,  # Fewer val sequences
        sequence_length=sequence_length,
        max_history_length=max_history_length
    )
    
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_curriculum_manager(max_states: int = 4,
                            alphabet: List[str] = ['0', '1'],
                            sequences_per_machine: int = 100,
                            sequence_length: int = 50,
                            max_history_length: int = 20,
                            max_machines_per_size: int = 50) -> CurriculumDatasetManager:
    """
    Create curriculum learning manager.
    
    Args:
        max_states: Maximum number of states
        alphabet: Alphabet symbols  
        sequences_per_machine: Sequences per machine
        sequence_length: Length of sequences
        max_history_length: Max history length
        max_machines_per_size: Max machines per state count
        
    Returns:
        CurriculumDatasetManager instance
    """
    print(f"Creating curriculum manager for machines up to {max_states} states...")
    machine_library = enumerate_machines_library(
        max_states=max_states,
        alphabet=alphabet,
        max_machines_per_size=max_machines_per_size
    )
    
    return CurriculumDatasetManager(
        machine_library=machine_library,
        sequences_per_machine=sequences_per_machine,
        sequence_length=sequence_length,
        max_history_length=max_history_length
    )


if __name__ == "__main__":
    # Example usage
    print("Creating sample dataset...")
    
    train_dataset, val_dataset = create_train_val_datasets(
        max_states=3,
        sequences_per_machine=50,
        sequence_length=30,
        max_machines_per_size=20
    )
    
    # Test DataLoader
    train_loader = train_dataset.get_dataloader(batch_size=8, shuffle=True)
    
    print("Sample batch:")
    for batch in train_loader:
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Target shape: {batch['target_id'].shape}")
        print(f"Emission probs shape: {batch['emission_probs'].shape}")
        print(f"Batch causal states: {batch['causal_state'][:3]}")
        break