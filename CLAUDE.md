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

## Key Files and Structure
```
src/neural_cssr/
├── core/epsilon_machine.py          # Core epsilon-machine implementation
├── enumeration/enumerate_machines.py # Machine enumeration system
├── data/dataset_generation.py       # PyTorch dataset generation
├── classical/cssr.py                # Classical CSSR implementation
└── classical/statistical_tests.py   # Statistical testing framework

config/
├── small_test_config.yaml           # Small test (3 states, 1,710 examples)
└── large_test_config.yaml           # Large test (4 states, 28,800 examples)

test_enumeration.py                   # Main dataset generation script
tests/classical_cssr/                # Classical CSSR tests
└── test_classical_cssr.py           # CSSR testing and evaluation

data/small_test/                      # Small dataset
├── train_dataset.pt                 # PyTorch training dataset (246K)
├── val_dataset.pt                   # PyTorch validation dataset (24K)
└── metadata.json                    # Dataset metadata

data/large_test/                      # Large dataset
├── train_dataset.pt                 # PyTorch training dataset (28,800 examples)
├── val_dataset.pt                   # PyTorch validation dataset (3,600 examples)
├── metadata.json                    # Dataset metadata
└── classical_cssr_results.json      # CSSR baseline results
```

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