# Neural CSSR Codebase Reorganization Plan

## Current Issues
- Too many test scripts: `test_enumeration.py`, `claude_test.py`, `simple_train.py`, `train_neural_cssr.py`, etc.
- Mixed purposes: dataset generation, model training, analysis scattered across files
- Hard to reproduce experiments
- No clear experimental pipeline

## Proposed Structure

```
/home/matteo/NeuralCSSR/
├── src/neural_cssr/           # Core library (keep as-is)
│   ├── core/                  # Epsilon-machines  
│   ├── enumeration/           # Machine enumeration
│   ├── data/                  # Dataset generation & conversion
│   ├── classical/             # Classical CSSR
│   └── neural/                # Neural models
│
├── experiments/               # NEW: Organized experiments
│   ├── 01_baseline/           # Baseline: classical CSSR
│   ├── 02_autoregressive/     # Direct autoregressive training
│   ├── 03_neural_cssr/        # Clustering-based Neural CSSR
│   ├── 04_advanced/           # Future: memory networks, etc.
│   └── utils/                 # Shared experiment utilities
│
├── datasets/                  # NEW: Centralized datasets
│   ├── small_test/            # Quick testing (existing)
│   ├── large_test/            # Main experiments (existing)  
│   ├── massive_test/          # Pattern validation (new)
│   └── configs/               # Dataset generation configs
│
├── scripts/                   # NEW: Main entry points
│   ├── generate_dataset.py    # Single dataset generation script
│   ├── run_experiment.py      # Single experiment runner
│   └── analyze_results.py     # Results analysis
│
└── results/                   # NEW: Experimental results
    ├── baseline/              # Classical CSSR results
    ├── neural_experiments/    # Neural model results
    └── analysis/              # Comparative analysis
```

## Key Scripts to Create

### 1. `scripts/generate_dataset.py`
- Unified dataset generation
- Replace: `test_enumeration.py`, `generate_massive_dataset.py`
- Usage: `python scripts/generate_dataset.py --config small_test`

### 2. `scripts/run_experiment.py`  
- Unified experiment runner
- Replace: `simple_train.py`, `train_neural_cssr.py`, etc.
- Usage: `python scripts/run_experiment.py --experiment neural_cssr --dataset large_test`

### 3. `experiments/*/train.py`
- Clean, focused training scripts for each approach
- Standardized interface and output format

## Files to Keep, Move, or Remove

### KEEP (core library):
- `src/neural_cssr/` - All core code

### MOVE to `experiments/`:
- `train_neural_cssr.py` → `experiments/03_neural_cssr/train.py`
- `simple_train.py` → `experiments/02_autoregressive/train.py` 
- `sequence_dataset.py` → `experiments/utils/datasets.py`

### MOVE to `datasets/`:
- `data/` → `datasets/`
- `config/` → `datasets/configs/`

### MOVE to `scripts/`:
- `test_enumeration.py` → `scripts/generate_dataset.py` (refactored)
- `generate_massive_dataset.py` → merge into `scripts/generate_dataset.py`

### REMOVE (redundant/outdated):
- `claude_test.py` 
- `explore_dataset.py`
- `inspect_dataset.py`
- `check_empirical_probs.py`
- `analyze_probabilities.py`
- `debug_simple_model.py`
- `test_simple_pattern.py`

### CREATE NEW:
- `experiments/01_baseline/run_classical_cssr.py`
- `experiments/04_advanced/` (for future memory networks, etc.)
- `scripts/analyze_results.py`

## Benefits

1. **Clear experimental pipeline**: Generate → Train → Analyze
2. **Reproducible experiments**: Each approach in its own directory
3. **Easy comparison**: Standardized interfaces and outputs
4. **Future-ready**: Room for new approaches in `experiments/04_advanced/`
5. **Clean separation**: Core library vs experiments vs datasets

## Next Steps

1. Create new directory structure
2. Refactor key scripts with unified interfaces
3. Move existing files to appropriate locations
4. Update documentation
5. Test the new pipeline

Would you like me to implement this reorganization plan?