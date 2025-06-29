# Neural CSSR Project Memory

## Project Overview
Neural CSSR (Causal State Splitting Reconstruction) implementation with a unified dataset generation framework for studying transfer learning scaling laws between synthetic finite state machine datasets. The project combines epsilon-machine enumeration, classical CSSR algorithms, and comprehensive dataset generation for neural network training.

## Current Status
✅ **Unified Framework Complete**: Production-ready dataset generation system
- Implemented comprehensive unified dataset generation framework
- Configuration-driven experiments with quality validation
- Multi-format outputs (raw sequences + PyTorch datasets)
- Rich metadata with information-theoretic measures
- Automated quality assurance and baseline computations

✅ **Core Infrastructure Complete**: 
- Epsilon-machine enumeration system
- Classical CSSR baseline implementation
- Statistical analysis and metadata computation
- Neural dataset formatting for transformers

## Unified Dataset Generation Framework

### Core Architecture
The new framework provides a complete pipeline from machine enumeration to ready-to-use datasets:

```
Configuration → Machine Library → Sequence Generation → 
Neural Formatting → Quality Validation → Structured Output
```

### Key Components
- **Dataset Generator**: Main orchestration (`dataset_generator.py`)
- **Sequence Processor**: Raw sequence generation with state tracking (`sequence_processor.py`)
- **Neural Formatter**: PyTorch dataset creation (`neural_formatter.py`)
- **Metadata Computer**: Information-theoretic analysis (`metadata_computer.py`)
- **Quality Validator**: Coverage and distribution validation (`quality_validator.py`)
- **Configuration System**: YAML-based experiment configuration (`generation_schemas.py`)

## Current Codebase Structure
```
/home/matteo/NeuralCSSR/
├── CLAUDE.md                          # Project memory/instructions
├── generate_unified_dataset.py        # Main dataset generation script
├── neural_cssr_dataset_framework.md   # Framework specification
├── pyproject.toml                     # Python project configuration
├── uv.lock                           # UV package manager lock file
├── datasets/                         # Generated datasets directory
│   └── small_exp/                    # Example: small experiment
│       ├── raw_sequences/            # Plain text sequences for classical CSSR
│       ├── neural_format/            # PyTorch datasets (.pt files)
│       ├── ground_truth/             # Machine definitions & state labels
│       ├── statistical_analysis/    # Information theory metrics
│       ├── quality_reports/          # Coverage & validation analysis
│       ├── experiment_config.yaml    # Configuration used
│       └── generation_info.yaml      # Reproducibility information
├── src/neural_cssr/                  # Source code directory
│   ├── __init__.py                   # Package initialization
│   ├── classical/                    # Classical CSSR implementation
│   │   ├── __init__.py
│   │   ├── cssr.py                   # Main classical CSSR algorithm
│   │   └── statistical_tests.py     # Statistical testing framework
│   ├── config/                       # Configuration system
│   │   ├── __init__.py
│   │   ├── generation_schemas.py     # Configuration dataclasses
│   │   └── dataset_configs/          # Example configurations
│   │       ├── small_experiment.yaml
│   │       ├── medium_experiment.yaml
│   │       └── full_experiment.yaml
│   ├── core/                         # Core epsilon-machine implementation
│   │   ├── __init__.py
│   │   └── epsilon_machine.py        # Core epsilon-machine classes
│   ├── data/                         # Unified dataset generation
│   │   ├── __init__.py
│   │   ├── dataset_generator.py      # Main orchestrator
│   │   ├── sequence_processor.py     # Raw sequence processing
│   │   ├── neural_formatter.py       # PyTorch dataset creation
│   │   ├── metadata_computer.py      # Statistical analysis
│   │   ├── quality_validator.py      # Quality validation
│   │   ├── evaluation_baselines.py   # Baseline computations
│   │   └── dataset_generation.py     # Legacy (kept for compatibility)
│   ├── enumeration/                  # Machine enumeration system
│   │   ├── __init__.py
│   │   └── enumerate_machines.py     # Machine enumeration logic
│   ├── analysis/                     # Dataset analysis tools
│   │   └── __init__.py
│   └── neural/                       # Neural network implementations
│       └── __init__.py
└── tests/                            # Test files directory
    └── classical_cssr/               # Classical CSSR tests
        ├── README.md                 # Test documentation
        └── test_classical_cssr.py    # Main CSSR test script
```

## Key Implementation Files
- **Main Script**: `generate_unified_dataset.py` - Complete dataset generation with CLI
- **Core Framework**: `src/neural_cssr/data/dataset_generator.py` - Main orchestrator
- **Machine Enumeration**: `src/neural_cssr/enumeration/enumerate_machines.py` - Machine library
- **Configuration**: `src/neural_cssr/config/generation_schemas.py` - Experiment configs
- **Classical CSSR**: `src/neural_cssr/classical/cssr.py` - Baseline algorithm

## How to Use the Framework

### Quick Start - Generate Datasets
```bash
# Small experiment (2,500 sequences, 5 machines) - uniform probabilities
python generate_unified_dataset.py --preset small --output datasets/small_exp

# Medium experiment (25,000 sequences, 16 machines) - uniform probabilities
python generate_unified_dataset.py --preset medium --output datasets/medium_exp

# Large experiment (122,000 sequences, 33 machines) - uniform probabilities
python generate_unified_dataset.py --preset large --output datasets/large_exp

# Biased experiment (6,500 sequences, 6 machines) - mixed uniform/biased probabilities
python generate_unified_dataset.py --preset biased --output datasets/biased_exp

# Custom configuration
python generate_unified_dataset.py --config my_config.yaml --output datasets/custom_exp

# Preview configuration without generating
python generate_unified_dataset.py --preset medium --output /tmp --dry-run
```

### Configuration Examples
The framework supports four built-in presets and custom YAML configurations:

**Small Preset**: 5 machines, 2,500 sequences, uniform probabilities
**Medium Preset**: 16 machines, 25,000 sequences, uniform probabilities  
**Large Preset**: 33 machines, 122,000 sequences, uniform probabilities
**Biased Preset**: 6 machines, 6,500 sequences, mixed uniform/biased probabilities

### Custom Configuration
Create YAML files following the schema in `src/neural_cssr/config/dataset_configs/`:

#### Uniform (Topological) Machines:
```yaml
experiment_name: "uniform_experiment"
machine_specs:
  - complexity_class: "2-state-binary"
    machine_count: 3
    samples_per_machine: 1000
    weight: 1.0
    topological: true  # Uniform probabilities (default)
```

#### Biased (Non-Topological) Machines:
```yaml
experiment_name: "biased_experiment"
machine_specs:
  # Random bias with specified strength
  - complexity_class: "2-state-binary"
    machine_count: 3
    samples_per_machine: 1000
    topological: false
    bias_strength: 0.7        # 0.0=uniform, 1.0=maximum bias
    probability_seed: 123     # Reproducible random bias
    
  # Custom transition probabilities
  - complexity_class: "3-state-binary"
    machine_count: 1
    samples_per_machine: 1500
    topological: false
    custom_probabilities:
      "S0": {"0": 0.8, "1": 0.2}  # Strong bias toward '0'
      "S1": {"0": 0.3, "1": 0.7}  # Strong bias toward '1'
      "S2": {"0": 0.5, "1": 0.5}  # Uniform for comparison
```

## Output Format
Each generated dataset includes:

### Raw Sequences (`raw_sequences/`)
- Plain text format for classical CSSR
- Train/val/test splits as separate files
- Complete sequence metadata in JSON

### Neural Format (`neural_format/`)
- PyTorch datasets (.pt files) ready for training
- Autoregressive format with attention masks
- Ground truth causal state labels
- Vocabulary and tokenization metadata

### Ground Truth (`ground_truth/`)
- Complete causal state trajectories
- Machine properties and definitions
- Transition logs and metadata

### Statistical Analysis (`statistical_analysis/`)
- Information-theoretic measures (entropy, complexity)
- N-gram analysis and sequence statistics
- Baseline performance metrics

### Quality Reports (`quality_reports/`)
- Coverage validation (states, transitions)
- Distribution consistency checks
- Quality scores and recommendations

## Technical Implementation Details
- **Epsilon-machines**: Both topological (uniform) and non-topological (biased) machines supported
- **Probability control**: Custom transition probabilities or random bias generation
- **Enumeration**: Systematic generation of all valid finite-state machines
- **Dataset format**: PyTorch tensors with rich metadata annotations
- **Quality assurance**: Automated coverage and distribution validation (adapted for biased machines)
- **Reproducibility**: Complete configuration and generation tracking with bias seeds

## Framework Features
- **Configuration-driven**: YAML-based experiment specification
- **Multi-format output**: Raw sequences + PyTorch datasets
- **Rich metadata**: Information theory, complexity metrics, quality validation
- **Quality assurance**: Automated coverage analysis and validation
- **Baseline computation**: Random, n-gram, and theoretical optimal baselines
- **Reproducibility**: Full experiment tracking and configuration saving
- **Scalability**: Small to large-scale dataset generation

## Recent Achievements

✅ **Non-Topological Machine Support Added**: Full control over transition probabilities
- Custom transition probabilities via YAML configuration
- Random bias generation with configurable strength (0.0=uniform, 1.0=maximum)
- Reproducible bias generation with seeds
- Mixed datasets combining topological and biased machines
- New `--preset biased` for immediate non-uniform experiments

✅ **Successfully Generated**: Small dataset with 2,500 sequences  
- 5 machines (3×2-state + 2×3-state binary)
- Perfect quality score (1.0/1.0)
- Complete multi-format output
- 21.3 second generation time
- All validation checks passed

✅ **Biased Dataset Validation**: Confirmed non-uniform probability generation
- Custom probabilities: 80%/20% bias successfully implemented
- Random bias: 31.8% deviation from uniform achieved
- Statistical analysis automatically detects probability biases
- Framework correctly handles mixed topological/non-topological datasets

## Dependencies and Environment
- **Package Manager**: `uv` (not pip)
- **Python Version**: 3.11+
- **Key Dependencies**: PyTorch, NumPy, PyYAML
- **Installation**: `uv sync`

## Commands to Remember
```bash
# Generate datasets with different scales and probability types
python generate_unified_dataset.py --preset small --output datasets/small      # Uniform
python generate_unified_dataset.py --preset medium --output datasets/medium    # Uniform
python generate_unified_dataset.py --preset large --output datasets/large      # Uniform
python generate_unified_dataset.py --preset biased --output datasets/biased    # Mixed uniform/biased

# Custom experiments with bias control
python generate_unified_dataset.py --config biased_experiment.yaml --output datasets/custom_bias
python generate_unified_dataset.py --config mixed_probabilities.yaml --output datasets/mixed

# Analysis and validation
python analyze_biased_dataset.py datasets/biased_exp    # Analyze symbol frequencies by machine
python test_bias_functionality.py                       # Test bias implementation

# Test classical CSSR (legacy)
uv run python tests/classical_cssr/test_classical_cssr.py

# Install dependencies
uv sync
```

## Known Technical Notes
- PyTorch datasets use `weights_only=False` for compatibility
- All file paths are absolute for cross-platform compatibility
- JSON serialization handles numpy arrays and tuples automatically
- Sequences converted between list and string formats as needed
- Bias functionality validated: 80%/20% custom probabilities work correctly
- Random bias generation tested: up to 31.8% deviation from uniform achieved
- Quality validation thresholds adjusted for non-uniform probability distributions

## Next Steps for Research
1. **Neural Network Training**: Use generated PyTorch datasets for transformer training
2. **Classical Comparison**: Run classical CSSR on raw sequences for baseline comparison
3. **Scaling Studies**: Generate datasets of different scales to study transfer learning
4. **Analysis Framework**: Develop tools for comparing neural vs classical CSSR performance
5. **Evaluation Metrics**: Implement comprehensive evaluation and visualization tools

## Project Progress Updates
- Successfully completed initial dataset generation framework
- Developed robust configuration-driven generation system
- Implemented comprehensive quality validation
- Generated first small-scale experimental dataset
- Prepared infrastructure for scaling experiments

## Framework Status: Production Ready ✅
The unified dataset generation framework is complete and tested. It provides a robust foundation for systematic Neural CSSR research with comprehensive metadata, quality assurance, and reproducibility features.