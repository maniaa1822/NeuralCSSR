# Neural CSSR Dataset Generation Framework
## Complete Implementation Specification for Claude Code

### Overview
This specification defines a comprehensive dataset generation framework for studying transfer learning scaling laws between synthetic finite state machine datasets using autoregressive transformers. The framework generates datasets compatible with both neural methods (transformers) and classical CSSR, with rich statistical metadata for rigorous evaluation.

## Architecture

### File Structure
```
src/neural_cssr/
├── data/
│   ├── __init__.py
│   ├── dataset_generator.py        # Main generation orchestrator
│   ├── sequence_processor.py       # Raw sequence processing
│   ├── neural_formatter.py         # PyTorch dataset creation
│   ├── metadata_computer.py        # Statistical analysis computation
│   ├── quality_validator.py        # Dataset quality checks
│   └── evaluation_baselines.py     # Baseline metrics computation
├── analysis/
│   ├── __init__.py
│   ├── information_theory.py       # Entropy, complexity measures
│   ├── sequence_analysis.py        # Temporal pattern analysis
│   └── visualization.py            # Dataset visualization tools
└── config/
    ├── dataset_configs/
    │   ├── small_experiment.yaml
    │   ├── medium_experiment.yaml
    │   └── full_experiment.yaml
    └── generation_schemas.py        # Config validation schemas
```

### Output Structure
```
datasets/experiment_name/
├── raw_sequences/
│   ├── train_sequences.txt
│   ├── val_sequences.txt
│   ├── test_sequences.txt
│   └── sequence_metadata.json
├── neural_format/
│   ├── train_dataset.pt
│   ├── val_dataset.pt
│   ├── test_dataset.pt
│   └── vocab_metadata.json
├── ground_truth/
│   ├── machine_definitions.json
│   ├── causal_state_labels.json
│   ├── transition_matrices.json
│   └── optimal_predictions.json
├── statistical_analysis/
│   ├── information_measures.json
│   ├── complexity_metrics.json
│   ├── sequence_statistics.json
│   └── comparative_baselines.json
├── quality_reports/
│   ├── coverage_analysis.json
│   ├── distribution_validation.json
│   └── generation_diagnostics.json
└── experiment_config.yaml
```

## Core Components Specification

### 1. Main Dataset Generator (`dataset_generator.py`)

**Class: `UnifiedDatasetGenerator`**

**Constructor:**
```python
def __init__(self, config_path: str, output_dir: str, seed: Optional[int] = None):
    """
    Initialize dataset generator with configuration.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Base directory for dataset output
        seed: Random seed for reproducibility
    """
```

**Main Generation Method:**
```python
def generate_dataset(self) -> Dict[str, Any]:
    """
    Main orchestration method that:
    1. Loads machine specifications from enumeration
    2. Generates raw sequences with metadata
    3. Creates neural format datasets
    4. Computes statistical analysis
    5. Validates dataset quality
    6. Saves all outputs in structured format
    
    Returns:
        generation_report: Summary of dataset creation process
    """
```

**Key Methods:**
- `_load_machine_library()`: Load enumerated epsilon-machines
- `_plan_generation()`: Create sampling strategy across machines
- `_generate_sequences()`: Generate raw sequences with state tracking
- `_create_splits()`: Create train/val/test splits
- `_process_for_neural()`: Convert to PyTorch format
- `_compute_metadata()`: Generate all statistical metadata
- `_validate_quality()`: Run quality assurance checks
- `_save_dataset()`: Write all outputs to disk

### 2. Sequence Processor (`sequence_processor.py`)

**Class: `SequenceProcessor`**

**Core Functionality:**
```python
def process_machine_sequences(
    self, 
    machine: EpsilonMachine, 
    num_sequences: int,
    length_distribution: Tuple[int, int],
    track_states: bool = True
) -> Dict[str, Any]:
    """
    Generate sequences from a single machine with complete metadata.
    
    Returns:
        {
            'sequences': List[str],           # Raw symbol sequences
            'state_trajectories': List[List[str]],  # State at each step
            'transition_log': List[Dict],     # Detailed transition info
            'generation_metadata': Dict      # Statistics and parameters
        }
    """
```

**Output Formats:**
- **Classical CSSR format**: Plain text sequences, one per line
- **Sequence metadata**: JSON with state trajectories and transition details
- **Quality metrics**: Coverage statistics, length distributions

### 3. Neural Formatter (`neural_formatter.py`)

**Class: `NeuralDatasetFormatter`**

**PyTorch Dataset Creation:**
```python
def create_pytorch_datasets(
    self, 
    sequences_data: Dict[str, Any],
    vocab_config: Dict[str, Any],
    context_length: int = 512
) -> Dict[str, torch.utils.data.Dataset]:
    """
    Convert raw sequences to PyTorch datasets for transformer training.
    
    Features:
    - Autoregressive input/target pairs
    - Attention masks for variable lengths
    - Ground truth causal state labels
    - Position-wise transition metadata
    - Machine ID and complexity labels
    """
```

**Dataset Item Structure:**
```python
{
    # Standard transformer inputs
    'input_ids': torch.LongTensor,      # [seq_len] - token IDs
    'attention_mask': torch.BoolTensor, # [seq_len] - valid positions
    'target_ids': torch.LongTensor,     # [seq_len] - next token targets
    
    # Ground truth labels
    'causal_states': torch.LongTensor,  # [seq_len] - true causal states
    'machine_id': torch.LongTensor,     # Scalar - source machine
    'complexity_class': torch.LongTensor, # Scalar - complexity category
    
    # Position-wise metadata
    'true_probs': torch.FloatTensor,    # [seq_len] - P(next|history)
    'transition_types': torch.LongTensor, # [seq_len] - transition IDs
    
    # Sequence metadata
    'original_length': torch.LongTensor, # Scalar - pre-padding length
    'sequence_id': torch.LongTensor,     # Scalar - unique sequence ID
}
```

### 4. Metadata Computer (`metadata_computer.py`)

**Class: `StatisticalMetadataComputer`**

**Information-Theoretic Measures:**
```python
def compute_information_measures(self, sequences_data: Dict) -> Dict[str, Any]:
    """
    Compute comprehensive information-theoretic measures:
    
    Machine-level:
    - Entropy rate (theoretical and empirical)
    - Statistical complexity 
    - Excess entropy
    - Predictive information
    - Thermodynamic depth
    
    Dataset-level:
    - Total entropy
    - Compression ratios
    - Effective alphabet sizes
    - Sequence diversity metrics
    """
```

**Complexity Metrics:**
```python
def compute_complexity_metrics(self, machines: List[EpsilonMachine]) -> Dict[str, Any]:
    """
    Analyze structural and learning complexity:
    
    Structural:
    - State count distributions
    - Transition density patterns
    - Cycle analysis in state graphs
    - Mixing time calculations
    
    Learning difficulty:
    - Estimated sample complexity (classical CSSR)
    - Memory requirements (max history length)
    - State ambiguity scores
    - Separation margins between states
    """
```

**Sequence Statistics:**
```python
def compute_sequence_statistics(self, sequences: List[str]) -> Dict[str, Any]:
    """
    Detailed sequence-level analysis:
    
    Length patterns:
    - Distribution moments and histograms
    - Coverage across length ranges
    
    Symbol patterns:
    - N-gram frequency analysis (up to order 6)
    - Run length distributions
    - Autocorrelation functions
    - Periodicity detection
    
    Temporal structure:
    - Motif discovery
    - Complexity evolution
    - Predictability measures
    """
```

### 5. Quality Validator (`quality_validator.py`)

**Class: `DatasetQualityValidator`**

**Coverage Validation:**
```python
def validate_coverage(self, dataset: Dict) -> Dict[str, Any]:
    """
    Ensure adequate statistical coverage:
    
    State coverage:
    - All causal states appear >= min_threshold times
    - Balanced representation across machines
    
    Transition coverage:  
    - All transitions appear >= min_threshold times
    - Uniform sampling of transition types
    
    Length coverage:
    - Adequate representation across length ranges
    - Sufficient long sequences for analysis
    """
```

**Distribution Validation:**
```python
def validate_distributions(self, dataset: Dict) -> Dict[str, Any]:
    """
    Check statistical consistency:
    
    Theoretical vs empirical:
    - Entropy rates within tolerance
    - Stationary distributions match theory
    - Transition probabilities accurate
    
    Dataset balance:
    - Train/val/test splits preserve distributions
    - No systematic biases in generation
    - Adequate diversity measures
    """
```

### 6. Evaluation Baselines (`evaluation_baselines.py`)

**Class: `BaselineComputer`**

**Random Baselines:**
```python
def compute_random_baselines(self, alphabet_size: int, num_states: int) -> Dict[str, float]:
    """
    Theoretical random performance:
    - Random prediction cross-entropy: log2(alphabet_size)
    - Random state prediction: 1/num_states accuracy
    - Random perplexity: alphabet_size
    """
```

**Empirical Baselines:**
```python
def compute_empirical_baselines(self, sequences: List[str]) -> Dict[str, Any]:
    """
    N-gram model baselines:
    - Unigram, bigram, trigram models
    - Cross-entropy and perplexity
    - Memory vs performance tradeoffs
    """
```

**Classical CSSR Oracle:**
```python
def estimate_classical_performance(self, machines: List[EpsilonMachine]) -> Dict[str, Any]:
    """
    Theoretical optimal performance:
    - Perfect causal state recovery accuracy: 1.0
    - Optimal cross-entropy: entropy_rate
    - Estimated sample complexity for convergence
    - Computational complexity estimates
    """
```

## Configuration System

### Configuration Schema (`generation_schemas.py`)

```python
@dataclass
class MachineSpec:
    """Specification for machine family selection."""
    complexity_class: str  # "2-state-binary", "3-state-binary", etc.
    machine_count: int     # Number of machines from this class
    samples_per_machine: int  # Sequences per machine
    weight: float = 1.0    # Sampling weight

@dataclass  
class SequenceSpec:
    """Sequence generation parameters."""
    length_distribution: Tuple[int, int]  # (min_length, max_length)
    total_sequences: int
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass
class QualitySpec:
    """Quality assurance thresholds."""
    min_state_coverage: int = 50
    min_transition_coverage: int = 20
    entropy_tolerance: float = 0.05
    length_diversity_threshold: float = 0.3

@dataclass
class DatasetConfig:
    """Complete dataset generation configuration."""
    experiment_name: str
    machine_specs: List[MachineSpec]
    sequence_spec: SequenceSpec
    quality_spec: QualitySpec
    neural_format_config: Dict[str, Any]
    random_seed: int = 42
```

### Example Configuration (`config/dataset_configs/medium_experiment.yaml`)

```yaml
experiment_name: "medium_transfer_experiment"
random_seed: 42

machine_specs:
  - complexity_class: "2-state-binary"
    machine_count: 5
    samples_per_machine: 2000
    weight: 1.0
  - complexity_class: "3-state-binary" 
    machine_count: 8
    samples_per_machine: 1500
    weight: 1.2
  - complexity_class: "2-state-ternary"
    machine_count: 3
    samples_per_machine: 1000
    weight: 0.8

sequence_spec:
  length_distribution: [50, 500]
  total_sequences: 50000
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

quality_spec:
  min_state_coverage: 100
  min_transition_coverage: 50
  entropy_tolerance: 0.03
  length_diversity_threshold: 0.4

neural_format_config:
  context_length: 512
  vocab_special_tokens: ["<PAD>", "<UNK>"]
  include_position_metadata: true
  batch_friendly_padding: true

output_config:
  save_raw_sequences: true
  save_neural_format: true
  save_statistical_analysis: true
  save_quality_reports: true
  compress_outputs: false
```

## Integration with Existing Code

### Machine Library Integration
```python
# Use existing enumeration system
from neural_cssr.enumeration.enumerate_machines import MachineEnumerator

def load_machine_library(config: Dict) -> List[EpsilonMachine]:
    """Load machines from existing enumeration system."""
    enumerator = MachineEnumerator()
    machines = enumerator.load_enumerated_machines()
    
    # Filter and select based on config
    selected_machines = filter_machines_by_spec(machines, config.machine_specs)
    return selected_machines
```

### Existing Dataset Generator Enhancement
```python
# Extend existing fsm_transformer/data_generator.py functionality
from fsm_transformer.data_generator import DataGenerator

class UnifiedDatasetGenerator(DataGenerator):
    """Enhanced version with full metadata support."""
    
    def __init__(self, config_path: str, **kwargs):
        super().__init__(**kwargs)
        self.config = load_config(config_path)
        self.metadata_computer = StatisticalMetadataComputer()
        self.quality_validator = DatasetQualityValidator()
```

## Implementation Priorities

### Phase 1: Core Infrastructure
1. **Dataset Generator**: Main orchestration class
2. **Configuration System**: YAML configs and validation
3. **Sequence Processor**: Enhanced sequence generation
4. **Basic Metadata**: Essential statistical measures

### Phase 2: Neural Integration  
1. **Neural Formatter**: PyTorch dataset creation
2. **Quality Validator**: Coverage and distribution checks
3. **Integration Testing**: End-to-end dataset creation

### Phase 3: Advanced Analytics
1. **Statistical Analysis**: Full information-theoretic measures
2. **Baseline Computer**: Comprehensive baseline metrics
3. **Visualization Tools**: Dataset analysis and exploration

### Phase 4: Optimization
1. **Performance Optimization**: Large-scale generation
2. **Memory Management**: Efficient processing pipelines
3. **Parallel Generation**: Multi-process/GPU acceleration

## Usage Examples

### Basic Dataset Generation
```python
# Generate medium-scale dataset
generator = UnifiedDatasetGenerator(
    config_path="config/dataset_configs/medium_experiment.yaml",
    output_dir="datasets/medium_experiment",
    seed=42
)

report = generator.generate_dataset()
print(f"Generated {report['total_sequences']} sequences")
print(f"Dataset quality score: {report['quality_score']}")
```

### Loading for Classical CSSR
```python
# Load sequences for classical CSSR
sequences = load_sequences("datasets/medium_experiment/raw_sequences/train_sequences.txt")
metadata = load_json("datasets/medium_experiment/raw_sequences/sequence_metadata.json")

# Run classical CSSR
classical_results = run_classical_cssr(sequences, max_length=10)
```

### Loading for Neural Training
```python
# Load PyTorch dataset for transformer training
train_dataset = torch.load("datasets/medium_experiment/neural_format/train_dataset.pt")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Access ground truth for evaluation
ground_truth = load_json("datasets/medium_experiment/ground_truth/causal_state_labels.json")
```

## Testing Strategy

### Unit Tests
- Each component tested independently
- Mocked dependencies for isolation
- Property-based testing for statistical measures

### Integration Tests  
- End-to-end dataset generation
- Format compatibility validation
- Statistical consistency checks

### Performance Tests
- Large-scale generation benchmarks
- Memory usage profiling
- Scaling behavior analysis

## Expected Outputs

After implementation, this framework will produce:

1. **Datasets ready for both neural and classical CSSR**
2. **Rich metadata enabling rigorous evaluation**  
3. **Quality-assured data with coverage guarantees**
4. **Baseline metrics for performance comparison**
5. **Reproducible experimental configurations**
6. **Comprehensive documentation and examples**

This framework serves as the foundation for systematic study of transfer learning scaling laws in neural CSSR, providing the high-quality, well-characterized datasets essential for rigorous scientific investigation.