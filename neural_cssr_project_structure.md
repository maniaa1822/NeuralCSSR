# Neural CSSR Project Structure and Implementation Guide

## Project Directory Structure

```
neural-cssr/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
├── .gitignore
├── config/
│   ├── __init__.py
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── enumeration_config.yaml
├── src/
│   └── neural_cssr/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── epsilon_machine.py        # ε-machine class definition
│       │   ├── causal_states.py          # Causal state operations
│       │   └── information_theory.py     # Entropy, complexity measures
│       ├── classical/
│       │   ├── __init__.py
│       │   ├── cssr.py                   # Classical CSSR implementation
│       │   └── statistical_tests.py      # Sufficiency testing
│       ├── enumeration/
│       │   ├── __init__.py
│       │   ├── enumerate_machines.py     # Generate all ε-machines
│       │   ├── topological_machines.py   # Topological ε-machine utilities
│       │   └── machine_properties.py     # Calculate machine properties
│       ├── neural/
│       │   ├── __init__.py
│       │   ├── transformer.py            # Transformer architecture
│       │   ├── neural_cssr.py           # Neural CSSR algorithm
│       │   ├── causal_probe.py          # Linear probe for state extraction
│       │   └── training.py              # Training loops and objectives
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset_generation.py    # Generate training datasets
│       │   ├── sequence_generation.py   # Generate sequences from machines
│       │   └── data_loaders.py          # PyTorch data loaders
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py               # Evaluation metrics
│       │   ├── validation.py            # Structure recovery validation
│       │   └── visualization.py         # Plotting and analysis tools
│       └── utils/
│           ├── __init__.py
│           ├── io_utils.py              # File I/O utilities
│           ├── math_utils.py            # Mathematical utilities
│           └── logging_utils.py         # Logging configuration
├── scripts/
│   ├── enumerate_machines.py            # Generate machine library
│   ├── create_dataset.py               # Create training dataset
│   ├── train_neural_cssr.py           # Training script
│   ├── evaluate_model.py              # Evaluation script
│   └── run_experiments.py             # Full experimental pipeline
├── notebooks/
│   ├── 01_enumeration_analysis.ipynb   # Analyze enumerated machines
│   ├── 02_dataset_exploration.ipynb    # Explore training data
│   ├── 03_model_training.ipynb         # Interactive training
│   ├── 04_structure_discovery.ipynb    # Test structure discovery
│   └── 05_real_data_experiments.ipynb  # Apply to real datasets
├── tests/
│   ├── __init__.py
│   ├── test_epsilon_machines.py
│   ├── test_classical_cssr.py
│   ├── test_enumeration.py
│   ├── test_neural_cssr.py
│   └── test_evaluation.py
├── data/
│   ├── enumerated_machines/            # Generated machine library
│   ├── datasets/                       # Training/test datasets
│   └── experiments/                    # Experimental results
├── models/
│   ├── checkpoints/                    # Model checkpoints
│   └── trained_models/                 # Final trained models
└── docs/
    ├── theory.md                       # Theoretical background
    ├── api_reference.md                # API documentation
    ├── examples.md                     # Usage examples
    └── paper_summary.md                # Your summary artifact
```

## Implementation Strategy with Claude Code

### Phase 1: Core Infrastructure ✅ **COMPLETED**

**Step 1: Setup Project Structure** ✅ **DONE**
- ✅ Project structure established using `uv` package manager
- ✅ Dependencies configured in `pyproject.toml` (torch, numpy, scipy, etc.)
- ✅ Proper `__init__.py` files in all modules
- ✅ Basic configuration system with YAML files

**Step 2: ε-Machine Core Classes** ✅ **DONE**
- ✅ Implemented `src/neural_cssr/core/epsilon_machine.py`
- ✅ EpsilonMachine class with state transitions and emission probabilities  
- ✅ Sequence generation from machines
- ✅ Causal state computation and machine validation
- ✅ Support for topological machines with uniform distributions

**Step 3: Information Theory Utilities** ✅ **DONE**
- ✅ Statistical complexity calculation in epsilon_machine.py
- ✅ Entropy rate computation  
- ✅ Machine property validation (strongly connected, deterministic)
- ✅ Period calculation and other computational mechanics measures

### Phase 2: Classical CSSR Implementation (Week 2) ⏳ **NOT IMPLEMENTED**

**Step 4: Classical CSSR Algorithm** ❌ **TODO**
```bash
claude-code --file paper_summary.md "Implement classical CSSR algorithm in src/neural_cssr/classical/cssr.py based on the description in the summary. Include: history enumeration, statistical sufficiency testing, state splitting, and convergence detection."
```

**Step 5: Statistical Testing Framework** ❌ **TODO**
```bash
claude-code "Implement statistical tests for distribution comparison in src/neural_cssr/classical/statistical_tests.py. Include chi-square tests, KL-divergence tests, and other methods for determining if two conditional distributions are significantly different."
```

### Phase 3: Enumeration System ✅ **COMPLETED**

**Step 6: Machine Enumeration** ✅ **DONE**
- ✅ Implemented `src/neural_cssr/enumeration/enumerate_machines.py`
- ✅ Generate all accessible DFAs up to specified state constraints
- ✅ Filter for ε-machine properties (strongly connected, deterministic)
- ✅ Enumerated 15 valid machines with up to 3 states and binary alphabet
- ✅ Machine validation and property calculation

**Step 7: Topological ε-Machines** ✅ **DONE**
- ✅ Uniform probability assignment to enumerated machines
- ✅ Canonical form computation and machine validation
- ✅ Statistical complexity and entropy rate calculation
- ✅ Machine library generation with complete metadata

### Phase 4: Neural Components (Week 4-5) ⏳ **NOT IMPLEMENTED**

**Step 8: Transformer Architecture** ❌ **TODO**
```bash
claude-code --file paper_summary.md "Implement the transformer architecture for Neural CSSR in src/neural_cssr/neural/transformer.py. Include: standard autoregressive transformer, probability estimation methods, and hidden state extraction for causal state probing."
```

**Step 9: Neural CSSR Algorithm** ❌ **TODO**
```bash
claude-code --file paper_summary.md "Implement the Neural CSSR algorithm in src/neural_cssr/neural/neural_cssr.py. Replace classical probability estimation with transformer-based estimation while preserving the CSSR clustering logic exactly as described."
```

**Step 10: Causal State Probe** ❌ **TODO**
```bash
claude-code "Implement the linear probe system in src/neural_cssr/neural/causal_probe.py for extracting causal states from transformer hidden states. Include training, inference, and uncertainty estimation for out-of-distribution detection."
```

### Phase 5: Dataset Generation ✅ **COMPLETED**

**Step 11: Dataset Creation Pipeline** ✅ **DONE**
- ✅ Implemented `src/neural_cssr/data/dataset_generation.py`
- ✅ PyTorch-compatible datasets with ground truth causal state labels
- ✅ Generated 1,710 training + 190 validation examples
- ✅ Multi-level structure with sequences from enumerated machines
- ✅ Complete dataset metadata and validation

**Step 12: Data Loaders and Processing** ✅ **PARTIAL**
- ✅ Basic PyTorch dataset implementation (EpsilonMachineDataset)
- ✅ Sequence processing and tensor conversion
- ❌ **TODO**: Advanced data loaders with curriculum learning
- ❌ **TODO**: Multi-task objectives and batch balancing

### Phase 6: Training Infrastructure (Week 7) ⏳ **NOT IMPLEMENTED**

**Step 13: Training Pipeline** ❌ **TODO**
```bash
claude-code --file paper_summary.md "Implement the training pipeline in src/neural_cssr/neural/training.py. Include: multi-task loss functions, curriculum learning, self-supervised refinement, and the two-phase training approach described."
```

**Step 14: Evaluation Framework** ❌ **TODO**
```bash
claude-code "Implement comprehensive evaluation tools in src/neural_cssr/evaluation/: structure recovery metrics, prediction accuracy, sample efficiency comparison, and validation against ground truth from enumerated machines."
```

## Current Implementation Status

✅ **COMPLETED (Phase 1 & Phase 3 & Partial Phase 5)**
- Core ε-machine implementation with all theoretical components
- Complete machine enumeration system (15 machines generated)  
- PyTorch dataset generation (1,900 total examples)
- Working test and inspection scripts
- Proper project structure with `uv` package management

⏳ **IN PROGRESS**
- Advanced data loading features for neural training

❌ **TODO (Phases 2, 4, 6)**
- Classical CSSR algorithm implementation
- Neural network components (transformer, neural CSSR)
- Training pipeline and evaluation framework

## Current Implementation Files

### ✅ Implemented Components
```
src/neural_cssr/
├── core/
│   ├── __init__.py                   ✅ Core module initialization
│   └── epsilon_machine.py            ✅ Complete ε-machine implementation
├── enumeration/
│   ├── __init__.py                   ✅ Enumeration module initialization  
│   └── enumerate_machines.py         ✅ Machine enumeration system
├── data/
│   ├── __init__.py                   ✅ Data module initialization
│   └── dataset_generation.py         ✅ PyTorch dataset generation
└── config/                           ✅ Configuration directory

config/
└── small_test_config.yaml            ✅ Test configuration (3 states, binary alphabet)

# Root level scripts
test_enumeration.py                   ✅ Main enumeration test script
inspect_dataset.py                    ✅ Dataset inspection tool
pyproject.toml                        ✅ Dependencies and project config

# Generated data
data/small_test/
├── train_dataset.pt                  ✅ PyTorch training dataset (246K, 1710 examples)
├── val_dataset.pt                    ✅ PyTorch validation dataset (24K, 190 examples)  
└── metadata.json                     ✅ Dataset metadata
```

### ❌ Missing Components (TODO)
```
src/neural_cssr/
├── classical/                        ❌ Classical CSSR implementation
│   ├── cssr.py                       ❌ Classical CSSR algorithm
│   └── statistical_tests.py          ❌ Statistical testing framework
├── neural/                           ❌ Neural network components
│   ├── transformer.py                ❌ Transformer architecture
│   ├── neural_cssr.py               ❌ Neural CSSR algorithm
│   ├── causal_probe.py              ❌ Linear probe for state extraction
│   └── training.py                  ❌ Training loops and objectives
├── evaluation/                       ❌ Evaluation framework
│   ├── metrics.py                   ❌ Evaluation metrics
│   ├── validation.py                ❌ Structure recovery validation
│   └── visualization.py             ❌ Plotting and analysis tools
└── utils/                           ❌ Utility functions
    ├── io_utils.py                  ❌ File I/O utilities
    ├── math_utils.py                ❌ Mathematical utilities
    └── logging_utils.py             ❌ Logging configuration
```

## Working Commands
- **Test enumeration**: `uv run python test_enumeration.py`
- **Inspect datasets**: `uv run python inspect_dataset.py`
- **Install deps**: `uv sync`

## Key Implementation Files for Claude Code

### Core Configuration File
```yaml
# config/model_config.yaml
model:
  transformer:
    d_model: 512
    num_heads: 8
    num_layers: 6
    dropout: 0.1
    max_seq_length: 1024
  
  neural_cssr:
    max_history_length: 20
    significance_level: 0.05
    min_count: 10
  
  causal_probe:
    hidden_dim: 256
    dropout: 0.2

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 1000
  
  curriculum:
    start_complexity: 1.0
    end_complexity: 3.0
    progression_rate: 0.1
  
  losses:
    autoregressive_weight: 1.0
    causal_state_weight: 0.5
    probability_weight: 0.2

enumeration:
  max_states: 6
  max_alphabet_size: 4
  max_machines_per_family: 1000
```

### Example Claude Code Commands with Context

**Generate Enumeration Script:**
```bash
claude-code --file paper_summary.md --file config/enumeration_config.yaml "Create scripts/enumerate_machines.py that generates the complete library of ε-machines up to specified sizes. Use the enumeration algorithm from the summary and save results in a structured format for dataset creation."
```

**Create Training Script:**
```bash
claude-code --file paper_summary.md --file config/training_config.yaml "Create scripts/train_neural_cssr.py that implements the full training pipeline: load enumerated machines, generate datasets, train transformer with curriculum learning, and validate structure discovery."
```

**Build Evaluation Suite:**
```bash
claude-code --file paper_summary.md "Create scripts/evaluate_model.py that comprehensively tests Neural CSSR: structure recovery accuracy, sample efficiency vs classical CSSR, generalization to unseen machines, and real-world dataset application."
```

## Claude Code Workflow Tips

### 1. Iterative Development
```bash
# Start with core classes
claude-code "Implement basic ε-machine class with state transitions and sequence generation"

# Test and refine
python -m pytest tests/test_enumeration.py

# Extend functionality  
claude-code "Add causal state computation and information-theoretic measures to the ε-machine class"
```

### 2. Use Context Files Effectively
```bash
# Provide theoretical context
claude-code --file docs/paper_summary.md --file docs/theory.md "Implement the classical CSSR algorithm with proper theoretical grounding"

# Provide implementation context
claude-code --file src/neural_cssr/core/epsilon_machine.py "Extend the ε-machine class to support enumeration and topological machines"
```

### 3. Incremental Testing
```bash
# Generate tests alongside implementation
claude-code "Create comprehensive tests for the ε-machine enumeration system, including edge cases and validation against known results"

# Run continuous validation
python -m pytest tests/ -v
```

### 4. Documentation Generation
```bash
# Auto-generate API docs
claude-code "Create comprehensive API documentation for the neural-cssr package with examples and theoretical explanations"
```

## Development Phases Summary

✅ **Phase 1 COMPLETE**: Core infrastructure and ε-machine implementation  
✅ **Phase 3 COMPLETE**: Enumeration system and machine library generation  
✅ **Phase 5 PARTIAL**: Dataset generation (basic implementation done, advanced features TODO)  
❌ **Phase 2 TODO**: Classical CSSR implementation  
❌ **Phase 4 TODO**: Neural components and transformer implementation  
❌ **Phase 6 TODO**: Training pipeline and evaluation framework  

**Current Status**: Successfully implemented the foundational components needed for Neural CSSR. The project can enumerate ε-machines, generate training datasets, and provides a solid base for neural network implementation.

**Next Priority**: Implement neural components (Phase 4) to begin training transformers on the generated datasets.

## Final Integration Commands

**Complete Project Setup:**
```bash
# Initialize full project
claude-code --file paper_summary.md "Set up the complete neural-cssr project structure with all components integrated. Ensure proper imports, configuration loading, and modular design."

# Generate comprehensive README
claude-code --file paper_summary.md "Create a comprehensive README.md explaining the Neural CSSR project, installation, usage examples, and theoretical background."

# Create example notebooks
claude-code --file paper_summary.md "Create Jupyter notebooks demonstrating: machine enumeration, dataset creation, model training, and structure discovery examples."
```

This structure provides a solid foundation for implementing Neural CSSR with Claude Code's assistance, maintaining theoretical rigor while building practical, production-ready software.