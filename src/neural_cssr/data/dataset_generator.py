"""
Main dataset generation orchestrator for Neural CSSR.

This module implements the UnifiedDatasetGenerator class that coordinates
the entire dataset generation process according to the framework specification.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import json
import yaml
from pathlib import Path
import random
import numpy as np
from dataclasses import asdict

from ..core.epsilon_machine import EpsilonMachine
from ..enumeration.enumerate_machines import enumerate_machines_library
from ..config.generation_schemas import DatasetConfig, MachineSpec, SequenceSpec, QualitySpec, load_config_from_dict
from .sequence_processor import SequenceProcessor
from .neural_formatter import NeuralDatasetFormatter
from .metadata_computer import StatisticalMetadataComputer
from .quality_validator import DatasetQualityValidator
from .evaluation_baselines import BaselineComputer


class UnifiedDatasetGenerator:
    """
    Main orchestrator for unified dataset generation.
    
    Coordinates the entire dataset creation process from machine enumeration
    to final output with rich metadata and quality validation.
    """
    
    def __init__(self, config_path: str, output_dir: str, seed: Optional[int] = None):
        """
        Initialize dataset generator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            output_dir: Base directory for dataset output
            seed: Random seed for reproducibility
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Load and validate configuration
        self.config = self._load_config()
        
        # Set random seeds
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Initialize components
        self.sequence_processor = SequenceProcessor()
        self.neural_formatter = NeuralDatasetFormatter()
        self.metadata_computer = StatisticalMetadataComputer()
        self.quality_validator = DatasetQualityValidator()
        self.baseline_computer = BaselineComputer()
        
        # Create output directory structure
        self._create_output_structure()
        
    def _load_config(self) -> DatasetConfig:
        """Load and validate configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Use the proper config loading function
        return load_config_from_dict(config_dict)
        
    def _create_output_structure(self) -> None:
        """Create output directory structure."""
        directories = [
            'raw_sequences',
            'neural_format', 
            'ground_truth',
            'statistical_analysis',
            'quality_reports'
        ]
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in directories:
            (self.output_dir / subdir).mkdir(exist_ok=True)
            
    def generate_dataset(self) -> Dict[str, Any]:
        """
        Main orchestration method for dataset generation.
        
        Returns:
            generation_report: Summary of dataset creation process
        """
        print(f"Starting dataset generation: {self.config.experiment_name}")
        
        # Step 1: Load machine specifications
        machine_library = self._load_machine_library()
        print(f"Loaded {len(machine_library)} machines")
        
        # Step 2: Plan generation strategy
        generation_plan = self._plan_generation(machine_library)
        print(f"Generation plan: {generation_plan['total_sequences']} total sequences")
        
        # Step 3: Generate raw sequences with metadata
        sequences_data = self._generate_sequences(machine_library, generation_plan)
        print(f"Generated {len(sequences_data['sequences'])} sequences")
        
        # Step 4: Create train/val/test splits
        split_data = self._create_splits(sequences_data)
        print(f"Created splits: {split_data['split_sizes']}")
        
        # Step 5: Process for neural format
        neural_datasets = self._process_for_neural(split_data)
        print(f"Created neural format datasets")
        
        # Step 6: Compute comprehensive metadata
        metadata = self._compute_metadata(sequences_data, machine_library)
        print(f"Computed statistical metadata")
        
        # Step 7: Validate dataset quality
        quality_report = self._validate_quality(sequences_data, metadata)
        print(f"Quality validation complete: {quality_report['overall_score']:.3f}")
        
        # Step 8: Save all outputs
        self._save_dataset(sequences_data, split_data, neural_datasets, metadata, quality_report)
        print(f"Dataset saved to {self.output_dir}")
        
        # Generate final report
        generation_report = {
            'experiment_name': self.config.experiment_name,
            'total_sequences': len(sequences_data['sequences']),
            'total_machines': len(machine_library),
            'quality_score': quality_report['overall_score'],
            'split_sizes': split_data['split_sizes'],
            'output_directory': str(self.output_dir),
            'config_used': asdict(self.config)
        }
        
        return generation_report
        
    def _load_machine_library(self) -> List[Dict[str, Any]]:
        """Load enumerated epsilon-machines based on config specifications."""
        all_machines = []
        
        # Determine alphabet from first machine spec (simplified)
        alphabet = ['0', '1']  # Default binary alphabet
        
        for machine_spec in self.config.machine_specs:
            # Parse complexity class (e.g., "2-state-binary" -> 2 states)
            num_states = int(machine_spec.complexity_class.split('-')[0])
            
            # Generate all possible machines with this number of states
            all_possible_machines = enumerate_machines_library(
                max_states=num_states,
                alphabet=alphabet,
                max_machines_per_size=None  # Get all possible machines
            )
            
            # Filter to machines with exactly the requested number of states
            machines_with_n_states = [
                m for m in all_possible_machines 
                if m['properties']['num_states'] == num_states
            ]
            
            # Select the requested number of distinct machines
            if len(machines_with_n_states) >= machine_spec.machine_count:
                selected_machines = machines_with_n_states[:machine_spec.machine_count]
            else:
                # If not enough distinct machines, repeat the available ones
                selected_machines = []
                for i in range(machine_spec.machine_count):
                    machine_idx = i % len(machines_with_n_states)
                    selected_machines.append(machines_with_n_states[machine_idx])
                
            # Configure machine probabilities and add metadata
            for i, machine in enumerate(selected_machines):
                machine_obj = machine['machine']
                
                # Configure transition probabilities
                if not machine_spec.topological:
                    if machine_spec.custom_probabilities:
                        # Use custom probabilities
                        machine_obj.set_transition_probabilities(machine_spec.custom_probabilities)
                    else:
                        # Use random probabilities with specified bias
                        machine_obj.randomize_probabilities(
                            seed=machine_spec.probability_seed + i if machine_spec.probability_seed else None,
                            bias_strength=machine_spec.bias_strength
                        )
                
                # Recompute properties after bias application
                from ..enumeration.enumerate_machines import MachineEnumerator
                enumerator = MachineEnumerator(max_states=num_states, alphabet=alphabet)
                updated_properties = enumerator.compute_machine_properties(machine_obj)
                
                # Add metadata - each machine will generate samples_per_machine sequences
                machine['sampling_weight'] = machine_spec.weight
                machine['complexity_class'] = machine_spec.complexity_class
                machine['samples_per_machine'] = machine_spec.samples_per_machine
                machine['is_topological'] = machine_obj.is_topological()  # Check actual machine state
                machine['bias_strength'] = machine_spec.bias_strength
                machine['machine_spec_id'] = len(all_machines)  # Unique ID for this spec instance
                machine['properties'] = updated_properties  # Update with post-bias properties
                
            all_machines.extend(selected_machines)
            
        return all_machines
        
    def _plan_generation(self, machine_library: List[Dict]) -> Dict[str, Any]:
        """Create sampling strategy across machines."""
        total_sequences = 0
        machine_allocations = []
        
        for machine in machine_library:
            num_sequences = machine['samples_per_machine']
            total_sequences += num_sequences
            
            machine_allocations.append({
                'machine_id': machine['id'],
                'complexity_class': machine['complexity_class'],
                'num_sequences': num_sequences,
                'weight': machine['sampling_weight']
            })
            
        return {
            'total_sequences': total_sequences,
            'machine_allocations': machine_allocations,
            'length_distribution': self.config.sequence_spec.length_distribution,
            'split_ratios': {
                'train': self.config.sequence_spec.train_ratio,
                'val': self.config.sequence_spec.val_ratio,
                'test': self.config.sequence_spec.test_ratio
            }
        }
        
    def _generate_sequences(self, machine_library: List[Dict], 
                          generation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate raw sequences with complete metadata."""
        all_sequences = []
        all_metadata = []
        
        for allocation in generation_plan['machine_allocations']:
            machine_id = allocation['machine_id']
            machine = next(m for m in machine_library if m['id'] == machine_id)
            num_sequences = allocation['num_sequences']
            
            # Generate sequences from this machine
            sequences_data = self.sequence_processor.process_machine_sequences(
                machine=machine['machine'],
                num_sequences=num_sequences,
                length_distribution=tuple(generation_plan['length_distribution']),
                track_states=True
            )
            
            # Add machine metadata to each sequence
            for i, seq in enumerate(sequences_data['sequences']):
                sequence_metadata = {
                    'sequence_id': len(all_sequences),
                    'machine_id': machine_id,
                    'complexity_class': allocation['complexity_class'],
                    'sequence': seq,
                    'state_trajectory': sequences_data['state_trajectories'][i],
                    'transition_log': sequences_data['transition_log'][i],
                    'machine_properties': machine['properties']
                }
                
                all_sequences.append(seq)
                all_metadata.append(sequence_metadata)
                
        return {
            'sequences': all_sequences,
            'metadata': all_metadata,
            'generation_plan': generation_plan,
            'alphabet': machine_library[0]['machine'].alphabet
        }
        
    def _create_splits(self, sequences_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create train/val/test splits preserving distributions."""
        sequences = sequences_data['sequences']
        metadata = sequences_data['metadata']
        
        # Shuffle with seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            
        indices = list(range(len(sequences)))
        random.shuffle(indices)
        
        # Calculate split sizes
        total_size = len(sequences)
        train_size = int(total_size * self.config.sequence_spec.train_ratio)
        val_size = int(total_size * self.config.sequence_spec.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        splits = {
            'train': {
                'sequences': [sequences[i] for i in train_indices],
                'metadata': [metadata[i] for i in train_indices]
            },
            'val': {
                'sequences': [sequences[i] for i in val_indices],
                'metadata': [metadata[i] for i in val_indices]
            },
            'test': {
                'sequences': [sequences[i] for i in test_indices],
                'metadata': [metadata[i] for i in test_indices]
            }
        }
        
        return {
            'splits': splits,
            'split_sizes': {
                'train': len(train_indices),
                'val': len(val_indices),
                'test': len(test_indices)
            },
            'alphabet': sequences_data['alphabet']
        }
        
    def _process_for_neural(self, split_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to PyTorch format datasets."""
        neural_datasets = {}
        
        for split_name, split_info in split_data['splits'].items():
            dataset = self.neural_formatter.create_pytorch_datasets(
                sequences_data=split_info,
                vocab_config={
                    'alphabet': split_data['alphabet'],
                    'special_tokens': getattr(self.config.neural_format_config, 'vocab_special_tokens', ['<PAD>', '<UNK>'])
                },
                context_length=getattr(self.config.neural_format_config, 'context_length', 512)
            )
            neural_datasets[split_name] = dataset
            
        return neural_datasets
        
    def _compute_metadata(self, sequences_data: Dict[str, Any], 
                         machine_library: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive statistical metadata."""
        metadata = {}
        
        # Information-theoretic measures
        metadata['information_measures'] = self.metadata_computer.compute_information_measures(
            sequences_data
        )
        
        # Complexity metrics
        metadata['complexity_metrics'] = self.metadata_computer.compute_complexity_metrics(
            [m['machine'] for m in machine_library]
        )
        
        # Sequence statistics
        metadata['sequence_statistics'] = self.metadata_computer.compute_sequence_statistics(
            sequences_data['sequences']
        )
        
        # Baseline computations
        alphabet_size = len(sequences_data['alphabet'])
        max_states = max(m['properties']['num_states'] for m in machine_library)
        
        # Extract epsilon machines for theoretical analysis
        machines = [m['machine'] for m in machine_library]
        
        metadata['baseline_metrics'] = {
            'random_baselines': self.baseline_computer.compute_random_baselines(
                alphabet_size, max_states
            ),
            'empirical_baselines': self.baseline_computer.compute_empirical_baselines(
                sequences_data['sequences']
            ),
            'optimal_baselines': self.baseline_computer.estimate_classical_performance(
                machines
            )
        }
        
        return metadata
        
    def _validate_quality(self, sequences_data: Dict[str, Any], 
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive quality validation."""
        quality_report = {}
        
        # Coverage validation
        quality_report['coverage'] = self.quality_validator.validate_coverage(
            sequences_data, self.config.quality_spec
        )
        
        # Distribution validation
        quality_report['distributions'] = self.quality_validator.validate_distributions(
            sequences_data, metadata
        )
        
        # Compute overall quality score
        coverage_score = quality_report['coverage']['overall_score']
        distribution_score = quality_report['distributions']['overall_score']
        quality_report['overall_score'] = (coverage_score + distribution_score) / 2
        
        return quality_report
        
    def _save_dataset(self, sequences_data: Dict[str, Any], split_data: Dict[str, Any],
                     neural_datasets: Dict[str, Any], metadata: Dict[str, Any],
                     quality_report: Dict[str, Any]) -> None:
        """Save all outputs in structured format."""
        
        # Save raw sequences
        for split_name, split_info in split_data['splits'].items():
            # Save sequences as text
            with open(self.output_dir / 'raw_sequences' / f'{split_name}_sequences.txt', 'w') as f:
                for seq in split_info['sequences']:
                    f.write(''.join(seq) + '\n')
                    
        # Save sequence metadata
        with open(self.output_dir / 'raw_sequences' / 'sequence_metadata.json', 'w') as f:
            json.dump({
                'train_metadata': split_data['splits']['train']['metadata'],
                'val_metadata': split_data['splits']['val']['metadata'],
                'test_metadata': split_data['splits']['test']['metadata'],
                'alphabet': split_data['alphabet']
            }, f, indent=2)
            
        # Save neural format datasets
        import torch
        for split_name, dataset in neural_datasets.items():
            torch.save(dataset, self.output_dir / 'neural_format' / f'{split_name}_dataset.pt')
            
        # Save vocabulary metadata
        vocab_metadata = {
            'alphabet': split_data['alphabet'],
            'vocab_size': len(split_data['alphabet']),
            'token_to_id': {token: i for i, token in enumerate(split_data['alphabet'])},
            'id_to_token': {i: token for i, token in enumerate(split_data['alphabet'])}
        }
        with open(self.output_dir / 'neural_format' / 'vocab_metadata.json', 'w') as f:
            json.dump(vocab_metadata, f, indent=2)
            
        # Save ground truth information
        # Group machine properties by unique machine ID
        unique_machines = {}
        for metadata in sequences_data['metadata']:
            machine_id = metadata['machine_id']
            if machine_id not in unique_machines:
                unique_machines[machine_id] = metadata.get('machine_properties', {})
        
        ground_truth = {
            'causal_state_labels': {i: m.get('state_trajectory', []) for i, m in enumerate(sequences_data['metadata'])},
            'machine_properties': unique_machines,  # One entry per unique machine
            'sequence_metadata': sequences_data['metadata']
        }
        
        for key, data in ground_truth.items():
            with open(self.output_dir / 'ground_truth' / f'{key}.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        # Save statistical analysis
        for key, data in metadata.items():
            with open(self.output_dir / 'statistical_analysis' / f'{key}.json', 'w') as f:
                json.dump(self._make_json_serializable(data), f, indent=2)
                
        # Save quality reports
        for key, report in quality_report.items():
            with open(self.output_dir / 'quality_reports' / f'{key}.json', 'w') as f:
                json.dump(self._make_json_serializable(report), f, indent=2)
                
        # Save experiment configuration
        with open(self.output_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(asdict(self.config), f, indent=2)
            
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            # Convert tuple keys to strings
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    new_key = str(k)
                else:
                    new_key = k
                new_dict[new_key] = self._make_json_serializable(v)
            return new_dict
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj