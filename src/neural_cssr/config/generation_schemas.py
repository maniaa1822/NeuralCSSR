"""
Configuration schemas for Neural CSSR dataset generation.

This module defines dataclasses and validation schemas for
dataset generation configuration.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class MachineSpec:
    """Specification for machine family selection."""
    complexity_class: str  # "2-state-binary", "3-state-binary", etc.
    machine_count: int     # Number of machines from this class
    samples_per_machine: int  # Sequences per machine
    weight: float = 1.0    # Sampling weight for this machine class
    
    # Probability distribution settings
    topological: bool = True  # If True, use uniform probabilities
    bias_strength: float = 0.0  # For random probs: 0.0=uniform, 1.0=max bias
    custom_probabilities: Optional[Dict[str, Dict[str, float]]] = None  # Custom transition probs
    probability_seed: Optional[int] = None  # Seed for random probability generation


@dataclass
class SequenceSpec:
    """Sequence generation parameters."""
    length_distribution: Tuple[int, int]  # (min_length, max_length)
    total_sequences: int
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def __post_init__(self):
        """Validate that ratios sum to 1.0."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")


@dataclass
class QualitySpec:
    """Quality assurance thresholds."""
    min_state_coverage: int = 50
    min_transition_coverage: int = 20
    entropy_tolerance: float = 0.05
    length_diversity_threshold: float = 0.3


@dataclass
class NeuralFormatConfig:
    """Neural network format configuration."""
    context_length: int = 512
    vocab_special_tokens: List[str] = field(default_factory=lambda: ["<PAD>", "<UNK>"])
    include_position_metadata: bool = True
    batch_friendly_padding: bool = True


@dataclass
class OutputConfig:
    """Output format configuration."""
    save_raw_sequences: bool = True
    save_neural_format: bool = True
    save_statistical_analysis: bool = True
    save_quality_reports: bool = True
    compress_outputs: bool = False


@dataclass
class DatasetConfig:
    """Complete dataset generation configuration."""
    experiment_name: str
    machine_specs: List[MachineSpec]
    sequence_spec: SequenceSpec
    quality_spec: QualitySpec
    neural_format_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    
    def __post_init__(self):
        """Convert dict configs to proper objects if needed."""
        # Convert neural_format_config if it's a dict
        if isinstance(self.neural_format_config, dict):
            neural_config = NeuralFormatConfig()
            for key, value in self.neural_format_config.items():
                if hasattr(neural_config, key):
                    setattr(neural_config, key, value)
            self.neural_format_config = neural_config
            
        # Convert output_config if it's a dict
        if isinstance(self.output_config, dict):
            output_config = OutputConfig()
            for key, value in self.output_config.items():
                if hasattr(output_config, key):
                    setattr(output_config, key, value)
            self.output_config = output_config


class ConfigurationValidator:
    """Validates dataset generation configurations."""
    
    @staticmethod
    def validate_machine_specs(machine_specs: List[MachineSpec]) -> List[str]:
        """
        Validate machine specifications.
        
        Args:
            machine_specs: List of machine specifications
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not machine_specs:
            errors.append("At least one machine specification is required")
            return errors
            
        for i, spec in enumerate(machine_specs):
            # Validate complexity class format
            if not spec.complexity_class:
                errors.append(f"Machine spec {i}: complexity_class cannot be empty")
            elif not spec.complexity_class.endswith('-binary') and not spec.complexity_class.endswith('-ternary'):
                errors.append(f"Machine spec {i}: complexity_class must end with '-binary' or '-ternary'")
                
            # Validate machine count
            if spec.machine_count <= 0:
                errors.append(f"Machine spec {i}: machine_count must be positive")
                
            # Validate samples per machine
            if spec.samples_per_machine <= 0:
                errors.append(f"Machine spec {i}: samples_per_machine must be positive")
                
            # Validate weight
            if spec.weight <= 0:
                errors.append(f"Machine spec {i}: weight must be positive")
                
        return errors
        
    @staticmethod
    def validate_sequence_spec(sequence_spec: SequenceSpec) -> List[str]:
        """
        Validate sequence specification.
        
        Args:
            sequence_spec: Sequence specification
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate length distribution
        min_len, max_len = sequence_spec.length_distribution
        if min_len <= 0:
            errors.append("Minimum sequence length must be positive")
        if max_len <= min_len:
            errors.append("Maximum sequence length must be greater than minimum")
            
        # Validate total sequences
        if sequence_spec.total_sequences <= 0:
            errors.append("Total sequences must be positive")
            
        # Validate ratios
        if not (0 < sequence_spec.train_ratio < 1):
            errors.append("Train ratio must be between 0 and 1")
        if not (0 < sequence_spec.val_ratio < 1):
            errors.append("Validation ratio must be between 0 and 1")
        if not (0 < sequence_spec.test_ratio < 1):
            errors.append("Test ratio must be between 0 and 1")
            
        return errors
        
    @staticmethod
    def validate_quality_spec(quality_spec: QualitySpec) -> List[str]:
        """
        Validate quality specification.
        
        Args:
            quality_spec: Quality specification
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if quality_spec.min_state_coverage <= 0:
            errors.append("Minimum state coverage must be positive")
            
        if quality_spec.min_transition_coverage <= 0:
            errors.append("Minimum transition coverage must be positive")
            
        if not (0 < quality_spec.entropy_tolerance < 1):
            errors.append("Entropy tolerance must be between 0 and 1")
            
        if not (0 < quality_spec.length_diversity_threshold < 1):
            errors.append("Length diversity threshold must be between 0 and 1")
            
        return errors
        
    @staticmethod
    def validate_complete_config(config: DatasetConfig) -> List[str]:
        """
        Validate complete dataset configuration.
        
        Args:
            config: Complete dataset configuration
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate experiment name
        if not config.experiment_name:
            errors.append("Experiment name cannot be empty")
            
        # Validate machine specs
        errors.extend(ConfigurationValidator.validate_machine_specs(config.machine_specs))
        
        # Validate sequence spec
        errors.extend(ConfigurationValidator.validate_sequence_spec(config.sequence_spec))
        
        # Validate quality spec
        errors.extend(ConfigurationValidator.validate_quality_spec(config.quality_spec))
        
        # Cross-validation: check that total sequences is consistent
        total_from_machines = sum(
            spec.machine_count * spec.samples_per_machine 
            for spec in config.machine_specs
        )
        
        if total_from_machines != config.sequence_spec.total_sequences:
            errors.append(
                f"Total sequences mismatch: machine specs generate {total_from_machines} "
                f"sequences, but sequence_spec.total_sequences is {config.sequence_spec.total_sequences}"
            )
            
        return errors


def load_config_from_dict(config_dict: Dict[str, Any]) -> DatasetConfig:
    """
    Load configuration from dictionary with validation.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated DatasetConfig instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Convert machine specs
    machine_specs = []
    for spec_dict in config_dict.get('machine_specs', []):
        machine_spec = MachineSpec(**spec_dict)
        machine_specs.append(machine_spec)
        
    # Convert sequence spec
    sequence_spec = SequenceSpec(**config_dict.get('sequence_spec', {}))
    
    # Convert quality spec
    quality_spec = QualitySpec(**config_dict.get('quality_spec', {}))
    
    # Create config
    config = DatasetConfig(
        experiment_name=config_dict.get('experiment_name', ''),
        machine_specs=machine_specs,
        sequence_spec=sequence_spec,
        quality_spec=quality_spec,
        neural_format_config=config_dict.get('neural_format_config', {}),
        output_config=config_dict.get('output_config', {}),
        random_seed=config_dict.get('random_seed', 42)
    )
    
    # Validate
    errors = ConfigurationValidator.validate_complete_config(config)
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
    return config


def create_example_configs() -> Dict[str, DatasetConfig]:
    """
    Create example configurations for common use cases.
    
    Returns:
        Dictionary of example configurations
    """
    examples = {}
    
    # Small experiment configuration
    examples['small_experiment'] = DatasetConfig(
        experiment_name="small_transfer_experiment",
        machine_specs=[
            MachineSpec(
                complexity_class="2-state-binary",
                machine_count=3,
                samples_per_machine=500,
                weight=1.0
            ),
            MachineSpec(
                complexity_class="3-state-binary", 
                machine_count=2,
                samples_per_machine=500,
                weight=1.0
            )
        ],
        sequence_spec=SequenceSpec(
            length_distribution=(20, 100),
            total_sequences=2500,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        ),
        quality_spec=QualitySpec(
            min_state_coverage=30,
            min_transition_coverage=15,
            entropy_tolerance=0.05,
            length_diversity_threshold=0.3
        ),
        neural_format_config={
            'context_length': 256,
            'vocab_special_tokens': ['<PAD>', '<UNK>'],
            'include_position_metadata': True
        },
        random_seed=42
    )
    
    # Medium experiment configuration
    examples['medium_experiment'] = DatasetConfig(
        experiment_name="medium_transfer_experiment",
        machine_specs=[
            MachineSpec(
                complexity_class="2-state-binary",
                machine_count=5,
                samples_per_machine=2000,
                weight=1.0
            ),
            MachineSpec(
                complexity_class="3-state-binary",
                machine_count=8,
                samples_per_machine=1500,
                weight=1.2
            ),
            MachineSpec(
                complexity_class="2-state-ternary",
                machine_count=3,
                samples_per_machine=1000,
                weight=0.8
            )
        ],
        sequence_spec=SequenceSpec(
            length_distribution=(50, 500),
            total_sequences=25000,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        ),
        quality_spec=QualitySpec(
            min_state_coverage=100,
            min_transition_coverage=50,
            entropy_tolerance=0.03,
            length_diversity_threshold=0.4
        ),
        neural_format_config={
            'context_length': 512,
            'vocab_special_tokens': ['<PAD>', '<UNK>'],
            'include_position_metadata': True
        },
        random_seed=42
    )
    
    return examples