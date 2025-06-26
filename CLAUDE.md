# Neural CSSR Project Memory

## Project Overview
Neural CSSR (Causal State Splitting Reconstruction) implementation that enumerates epsilon-machines and generates datasets for training neural networks to learn causal state representations.

## Current Status
✅ **Phase 1 Complete**: Functional epsilon-machine enumeration and PyTorch dataset generation
- Successfully implemented machine enumeration system
- Created PyTorch-compatible datasets with ground truth causal state labels
- Generated 1,710 training + 190 validation examples from enumerated machines
- All core functionality tested and working

✅ **Phase 2 Complete**: Classical CSSR baseline implementation
- Implemented classical CSSR algorithm with multiple statistical tests
- Tested on both small (1,710 examples) and large (28,800 examples) datasets
- Achieved optimal performance with KL divergence (min_count=2, threshold=0.01)
- Large dataset results: 10 discovered states, 65.9% accuracy, 78.3% coverage

## Problem Point 3 and Potential Solutions
- **Issue**: Complexity in state discovery when mixing machines with similar state labels but different behaviors
- **Potential Solution**: Develop more sophisticated state differentiation algorithm
- **Proposed Approach**: Implement multi-dimensional state encoding that captures nuanced behavioral differences
- **Next Step**: Create a more granular state comparison metric beyond current KL divergence method

## Complete Codebase Structure
```
/home/matteo/NeuralCSSR/
├── CLAUDE.md                          # Project memory/instructions
├── claude_test.py                     # Test script
├── inspect_dataset.py                 # Dataset inspection utility
├── neural_cssr_project_structure.md   # Project structure documentation
├── neural_cssr_summary.md            # Project summary documentation
├── pyproject.toml                     # Python project configuration
├── test_enumeration.py                # Main dataset generation script
├── uv.lock                           # UV package manager lock file
├── config/                           # Configuration directory
│   ├── small_test_config.yaml        # Small test (3 states, 1,710 examples)
│   └── large_test_config.yaml        # Large test (4 states, 28,800 examples)
├── data/                             # Generated datasets directory
│   ├── large_test/                   # Large dataset (28,800 examples)
│   │   ├── classical_cssr_results.json # CSSR baseline results
│   │   ├── metadata.json             # Dataset metadata
│   │   ├── train_dataset.pt          # PyTorch training dataset
│   │   └── val_dataset.pt            # PyTorch validation dataset
│   └── small_test/                   # Small dataset (1,710 examples)
│       ├── classical_cssr_results.json # CSSR baseline results
│       ├── metadata.json             # Dataset metadata
│       ├── train_dataset.pt          # PyTorch training dataset
│       └── val_dataset.pt            # PyTorch validation dataset
├── src/neural_cssr/                  # Source code directory
│   ├── __init__.py                   # Package initialization
│   ├── classical/                    # Classical CSSR implementation
│   │   ├── __init__.py
│   │   ├── cssr.py                   # Main classical CSSR algorithm
│   │   └── statistical_tests.py     # Statistical testing framework
│   ├── config/                       # Empty (planned configuration)
│   ├── core/                         # Core epsilon-machine implementation
│   │   ├── __init__.py
│   │   └── epsilon_machine.py        # Core epsilon-machine classes
│   ├── data/                         # Dataset generation
│   │   ├── __init__.py
│   │   └── dataset_generation.py     # PyTorch dataset generation
│   ├── enumeration/                  # Machine enumeration system
│   │   ├── __init__.py
│   │   └── enumerate_machines.py     # Machine enumeration logic
│   └── neural/                       # Empty (planned neural implementation)
└── tests/                            # Test files directory
    └── classical_cssr/               # Classical CSSR tests
        ├── README.md                 # Test documentation
        ├── analyze_cssr.py           # CSSR analysis utilities
        ├── debug_chi_square.py       # Chi-square debugging
        ├── test_classical_cssr.py    # Main CSSR test script
        ├── test_cssr_params.py       # CSSR parameter testing
        └── test_kl_params.py         # KL divergence parameter testing
```

## Key Implementation Files
- **Core**: `src/neural_cssr/core/epsilon_machine.py` - epsilon-machine implementation
- **Enumeration**: `src/neural_cssr/enumeration/enumerate_machines.py` - machine enumeration system
- **Data Generation**: `src/neural_cssr/data/dataset_generation.py` - PyTorch dataset generation
- **Classical CSSR**: `src/neural_cssr/classical/cssr.py` - classical CSSR algorithm
- **Statistical Tests**: `src/neural_cssr/classical/statistical_tests.py` - testing framework
- **Main Scripts**: `test_enumeration.py`, `inspect_dataset.py`, `claude_test.py`

## Ready for Implementation
- **Neural Network**: Empty `src/neural_cssr/neural/` directory ready for transformer implementation
- **Configuration**: Empty `src/neural_cssr/config/` directory ready for neural config classes

## How to Run
- **Generate datasets**: `uv run python test_enumeration.py`
- **Test classical CSSR**: `uv run python tests/classical_cssr/test_classical_cssr.py`
- **Inspect datasets**: `uv run python inspect_dataset.py`
- **Dependencies**: Uses `uv` package manager, dependencies in `pyproject.toml`

## Technical Implementation Details
- **Epsilon-machines**: Topological machines with uniform probability distributions
- **Enumeration**: Generates all valid finite-state machines up to specified constraints
- **Dataset format**: PyTorch tensors with input_ids, attention_mask, target_id, causal_state
- **Ground truth**: Complete causal state labeling for supervised learning
- **Validation**: Machine properties verified (strongly connected, deterministic)

## Configuration
**Small test config** (`config/small_test_config.yaml`):
- Max 3 states, binary alphabet ['0', '1']  
- 5 machines per complexity level
- 10 sequences per machine, length 20
- Max history length 10
- Total: 1,710 training examples

**Large test config** (`config/large_test_config.yaml`):
- Max 4 states, binary alphabet ['0', '1']
- 10 machines per complexity level
- 50 sequences per machine, length 25
- Max history length 12
- Total: 28,800 training examples

## Known Issues Fixed
- PyTorch 2.6 requires `weights_only=False` for custom class loading
- Use `uv run python` instead of plain `python` for proper environment

## Classical CSSR Results Analysis

**Key Finding**: CSSR discovers 10 states from large dataset containing 3 machines (max 4 states each)

**Why More States Than Expected:**
- Dataset mixes sequences from 3 different machines: 2-state, 3-state, and 4-state
- Each machine uses same state labels ("S0", "S1", etc.) but with different behaviors
- CSSR must build unified model capturing all mixed behaviors
- Result: 10 discovered states to distinguish between behaviorally different states across machines

**Performance Metrics:**
- Accuracy: 65.9% (predicting next symbol)
- Coverage: 78.3% (histories assigned to states)
- Statistical test: KL divergence with min_count=2, threshold=0.01

## Next Steps (Not Implemented)
1. Neural CSSR transformer architecture for probability estimation P(next_symbol | history)
2. Integration of transformer probabilities into CSSR algorithm
3. Comparison of neural vs classical CSSR performance
4. Training loop with causal state prediction
5. Evaluation metrics and visualization

## Commands to Remember
- Generate datasets: `uv run python test_enumeration.py`
- Test classical CSSR: `uv run python tests/classical_cssr/test_classical_cssr.py`
- Quick dataset check: `uv run python inspect_dataset.py`
- Install deps: `uv sync`
- Project uses `uv` not `pip`

## Dataset Sample for classical CSSR
uv run python explore_dataset.py
Dataset type: <class 'dict'>
Dataset keys: ['data', 'metadata']

Key: data
  Type: <class 'list'>
  Length: 28800
  First few elements: [{'input_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'target_id': 1, 'machine_id': 10, 'num_states': 3, 'causal_state': 'S0', 'target_prob': 0.5, 'raw_history': ['0'], 'raw_target': '1'}, {'input_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'attention_mask': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'target_id': 0, 'machine_id': 10, 'num_states': 3, 'causal_state': 'S1', 'target_prob': 0.5, 'raw_history': ['0', '1'], 'raw_target': '0'}, {'input_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'attention_mask': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 'target_id': 1, 'machine_id': 10, 'num_states': 3, 'causal_state': 'S0', 'target_prob': 0.5, 'raw_history': ['0', '1', '0'], 'raw_target': '1'}, {'input_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 'attention_mask': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 'target_id': 1, 'machine_id': 10, 'num_states': 3, 'causal_state': 'S1', 'target_prob': 0.5, 'raw_history': ['0', '1', '0', '1'], 'raw_target': '1'}, {'input_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], 'attention_mask': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'target_id': 0, 'machine_id': 10, 'num_states': 3, 'causal_state': 'S2', 'target_prob': 0.5, 'raw_history': ['0', '1', '0', '1', '1'], 'raw_target': '0'}]

Key: metadata
  Type: <class 'dict'>

Data samples analysis:
  Total samples: 28800

First 3 samples:
  Sample 0:
    input_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    attention_mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    target_id: 1
    causal_state: S0
    raw_history: ['0']
    raw_target: 1
  Sample 1:
    input_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    attention_mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    target_id: 0
    causal_state: S1
    raw_history: ['0', '1']
    raw_target: 0
  Sample 2:
    input_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    attention_mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    target_id: 1
    causal_state: S0
    raw_history: ['0', '1', '0']
    raw_target: 1

Token vocabulary: [0, 1]
Target vocabulary: [0, 1]

Sequence lengths (first 100): min=12, max=12

Metadata:
  vocab_size: 2
  alphabet: ['0', '1']
  token_to_id: {'0': 0, '1': 1}
  id_to_token: {0: '0', 1: '1'}
  max_history_length: 12

  ##Core Issue

  Transformers consistently predict ~50% probability for both
  tokens (0 and 1), failing to learn conditional patterns that
  clearly exist in the data.

  What We Know Works

  - Classical CSSR succeeds: Discovers 10 causal states,
  achieves 65.9% accuracy on the same data
  - Empirical patterns exist: Histories like '100' → 45.8% vs
  54.2%, '1101' → 56.8% vs 43.2%
  - Data contains structure: 28,800 samples with measurable
  conditional dependencies

  What Fails

  - Direct autoregressive training: Model learns global marginal
   (~49.4%/50.6%) instead of conditionals
  - Simple RNN: Also gets stuck at uniform despite much simpler
  architecture
  - Sequence-to-symbol: Even with proper sequence context, still
   predicts uniform

  Root Cause Analysis

  1. Data Generation Issue

  - Epsilon-machines use make_topological() → uniform transition
   probabilities (0.5/0.5)
  - Empirical deviations are sampling noise, not real patterns
  - Neural models correctly learn there are no true conditional
  biases

  2. Learning Dynamics Problem

  - Optimization landscape: Global marginal is an easy local
  minimum
  - Signal-to-noise ratio: Real patterns (if any) are too weak
  vs noise
  - Model capacity: Even simple patterns require more
  sophisticated modeling

  3. Task Formulation Issue

  - Autoregressive expansion: Destroys sequential context by
  splitting sequences
  - Context window: Limited history may miss longer-range
  dependencies
  - Target representation: Binary classification may be too
  crude

  Why Classical CSSR Works But Neural Doesn't

  Classical CSSR:
  - Groups histories by statistical similarity (not prediction
  accuracy)
  - Discovers structural patterns (state transitions) not
  probabilistic biases
  - Uses hypothesis testing to find meaningful state
  distinctions

  Neural approaches:
  - Optimize prediction accuracy directly
  - Get stuck in marginal distribution local minimum
  - Need stronger supervision or architectural inductive biases

  Our Neural CSSR Solution

  Stage 1: Learn representations (not direct prediction)Stage 2:
   Cluster to discover causal states (like classical CSSR)Stage 
  3: Predict from discovered states (structured prediction)

  This mimics classical CSSR's approach instead of direct
  end-to-end learning, potentially avoiding the uniform
  distribution trap.