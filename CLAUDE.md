# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural CSSR discovers epsilon-machines (minimal causal state representations) from sequences using neural networks as probability estimators. The project has two main workflows:

1. **Current Working Pipeline**: Generate data with `pysm_generator.py` → Train nanoGPT model → Discover states with `nanoGPT/js_analysis/unsupervised_fast_original.py`
2. **Legacy/Experimental**: EBM/AR model training in `experiments/ebm/` (see section below)

## Development Environment

### Package Manager
This project uses `uv` (not pip):
```bash
uv sync                    # Install dependencies
uv run python script.py    # Run scripts (optional, python works directly)
```

### Key Dependencies
- **Core**: torch, numpy, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Graph/State Analysis**: networkx, python-igraph>=0.11.8, python-statemachine>=2.5.0

## Current Workflow: nanoGPT + Unsupervised CSSR

This is the **primary working pipeline** for epsilon-machine discovery.

### Step 1: Generate Dataset

Use `pysm_generator.py` to create binary sequences from finite state machines:

```bash
# Seven-state human machine (7 causal states)
uv run python pysm_generator.py --machine seven_state_human --length 100000 \
  --output experiments/datasets --seed 42

# Even process (2 states, infinite memory)
uv run python pysm_generator.py --machine even_process --length 50000 \
  --output experiments/datasets --seed 42

# Golden mean (2 states, finite memory)
uv run python pysm_generator.py --machine golden_mean --length 50000 \
  --output experiments/datasets --seed 42
```

**Available Machines**: `golden_mean`, `even_process`, `seven_state_human`, `sevestateold`, `biased_coin`, `alternating`, `unifilar_3_state`, `distinct_3_state`, `distinct_4_state`, `distinct_6_state`, `anti_compression`, `hierarchical_4_state`

**Output Files** (in `experiments/datasets/<machine>/`):
- `<machine>.dat`: Binary sequence (0s and 1s)
- `<machine>.states`: Ground truth state indices
- `<machine>.states.dat`: State sequence as characters (A, B, C, ...)
- `<machine>.machine.json`: Transition structure
- `<machine>.meta.json`: Metadata and mappings

### Step 2: Prepare nanoGPT Data

Convert `.dat` file to nanoGPT's binary format:

```bash
# Create data directory if needed
mkdir -p nanoGPT/data/seven_state_human

# Copy prepare.py template (or create one)
# The prepare.py script reads the .dat file and creates train.bin, val.bin, meta.pkl

uv run --with numpy python nanoGPT/data/seven_state_human/prepare.py
```

**Key Points**:
- `prepare.py` scripts are machine-specific (see `nanoGPT/data/*/prepare.py` for examples)
- Creates `train.bin`, `val.bin` (torch tensors), and `meta.pkl` (vocabulary)
- Binary sequences use vocab: `{0, 1, 2}` where 2 is typically padding/special token

### Step 3: Train nanoGPT Model

Train a character-level transformer using nanoGPT's config system:

```bash
cd nanoGPT

# Seven-state human (recommended settings)
uv run --with torch --with numpy python train.py \
  config/train_seven_state_human_char.py \
  --device=cuda --dropout=0.0 --max_iters=10000 --lr_decay_iters=10000 \
  --out_dir=out-seven-state-human --always_save_checkpoint=True

# Even process (requires longer context)
uv run --with torch --with numpy python train.py \
  config/train_even_process_char.py \
  --device=cuda --block_size=128 --dropout=0.0 --max_iters=10000 \
  --out_dir=out-even-process --always_save_checkpoint=True
```

**Training Config Files**: `nanoGPT/config/train_<machine>_char.py`
**Checkpoint Output**: `nanoGPT/out-<machine>/ckpt.pt`

**Key Hyperparameters**:
- `block_size`: Context window (64-128 for most machines, ≥128 for even_process)
- `n_layer`, `n_head`, `n_embd`: Model size (default: 4 layers, 4 heads, 128 dim)
- `dropout`: Set to 0.0 for deterministic binary processes
- `max_iters`: Training steps (5000-10000 usually sufficient)

### Step 4: Discover Epsilon Machine

Run unsupervised state discovery using JS divergence clustering:

```bash
# Seven-state human with backward stability (finds minimal suffixes)
uv run --with torch python nanoGPT/js_analysis/unsupervised_fast_original.py \
  --preset seven_state_human_char_large \
  --backward_stability --tolerance_bits 1e-3 --min_suffix_len 2 \
  --stage_a_threshold 0.001 --n_samples 100 --L 5 \
  --output_json results/seven_state_results.json

# Even process (simpler, no backward stability needed)
uv run --with torch python nanoGPT/js_analysis/unsupervised_fast_original.py \
  --preset even_process \
  --stage_a_threshold 0.001 --stage_b_threshold 0.001 \
  --n_samples 100 --L 5 --k_refine 4 \
  --output_json results/even_process_results.json
```

**Key Parameters**:
- `--preset`: Chooses model checkpoint and data paths automatically
- `--L`: History length for sampling (5-6 typical)
- `--k_refine`: Horizon for k-step rollout refinement (3-4)
- `--n_samples`: Number of histories to sample (50-100)
- `--stage_a_threshold`: Emission clustering threshold (0.001-0.025)
- `--stage_b_threshold`: Conditional JS refinement threshold (0.001-0.02)
- `--backward_stability`: Enable minimal suffix detection (useful for high-memory processes)
- `--tolerance_bits`: Tolerance for suffix pruning in bits (1e-3 typical)

**Algorithm Overview**:
1. **Stage A**: Cluster histories by emission distribution (next-token probabilities)
2. **Stage B**: Refine clusters using multi-step (k-step) predictive distributions
3. **Stage C**: Remerge functionally identical states (optional, via `--enable_remerging`)
4. **Backward Stability** (optional): Find minimal suffix that preserves emission probabilities

**Output**: JSON file with discovered clusters, ground truth evaluation, and epsilon machine loss

### Available Presets

The `unsupervised_fast_original.py` script has built-in presets that automatically set model and data paths:
- `seven_state_human_char_large`
- `seven_state_human_large`
- `seven_state_human_100k`
- `sevestateold`
- `even_process`

To use custom paths, override with `--model_ckpt` and `--data` flags.

## Legacy/Experimental: EBM and AR Models

Alternative training pipeline using custom binary language models (not actively used):

### Model Architecture

Two model classes in `experiments/ebm/models.py`:

1. **AutoRegressiveBinaryLM**: Standard transformer-based autoregressive model
2. **EnergyBasedBinaryLM**: Energy-based formulation with energy head

### Training Commands

**Golden Mean:**
```bash
uv run --with torch python experiments/ebm/train_golden_mean.py \
  --preset golden_mean --device auto --context_window 64 --epochs 3 \
  --batch_size 64 --d_model 128 --layers 4 --dropout 0.1
```

**Even Process:**
```bash
uv run --with torch python experiments/ebm/train_golden_mean.py \
  --preset even_process --device auto --context_window 128 --epochs 3 \
  --batch_size 64 --d_model 128 --layers 4 --dropout 0.0
```

## Package Structure

```
pysm_generator.py              # Step 1: Generate datasets
nanoGPT/
├── data/<machine>/prepare.py  # Step 2: Prepare training data
├── train.py                   # Step 3: Train transformer model
├── config/train_*_char.py     # Training configurations
├── model.py                   # nanoGPT transformer architecture
└── js_analysis/
    ├── unsupervised_fast_original.py  # Step 4: MAIN CSSR discovery script
    ├── js_metrics.py          # JS divergence computation
    ├── state_mapping.py       # Ground truth state mappings
    ├── calibration.py         # Platt calibration for probabilities
    └── *.py                   # Visualization and diagnostic tools

experiments/
├── datasets/<machine>/        # Generated datasets
├── ebm/                       # Legacy: EBM/AR model training
└── results/                   # Analysis outputs

transcssr_neural_runner.py     # Alternative: Inject neural probs into transCSSR
```

## Machine Catalog

| Machine | States | Memory | Key Characteristics |
|---------|--------|--------|---------------------|
| `golden_mean` | 2 | Finite (L=1) | No consecutive 0s; recovers easily |
| `even_process` | 2 | Infinite | Runs of 1s have even length; requires long context |
| `seven_state_human` | 7 | Finite (L≤4) | Complex suffix structure; use backward stability |
| `sevestateold` | 7 | Finite (L≤4) | Legacy emission probabilities |
| `distinct_3_state` | 3 | Finite | Well-separated emissions |
| `distinct_4_state` | 4 | Finite | Moderate complexity |
| `hierarchical_4_state` | 4 | Finite | Two-level clustering structure |

## Quick Reference: Complete Pipeline

**Seven-State Human (Full Example)**

```bash
# Step 1: Generate dataset
uv run python pysm_generator.py --machine seven_state_human --length 100000 \
  --output experiments/datasets --seed 42

# Step 2: Prepare nanoGPT data (ensure prepare.py exists)
uv run --with numpy python nanoGPT/data/seven_state_human/prepare.py

# Step 3: Train nanoGPT
cd nanoGPT
uv run --with torch --with numpy python train.py \
  config/train_seven_state_human_char.py \
  --device=cuda --dropout=0.0 --max_iters=10000 \
  --out_dir=out-seven-state-human --always_save_checkpoint=True
cd ..

# Step 4: Discover epsilon machine
uv run --with torch python nanoGPT/js_analysis/unsupervised_fast_original.py \
  --preset seven_state_human_char_large \
  --backward_stability --tolerance_bits 1e-3 --min_suffix_len 2 \
  --stage_a_threshold 0.001 --n_samples 100 --L 5 \
  --output_json results/seven_state_results.json
```

## Key Modules

**Core Pipeline**:
- `pysm_generator.py`: Dataset generation from finite state machines
- `nanoGPT/train.py`: Transformer training (character-level)
- `nanoGPT/js_analysis/unsupervised_fast_original.py`: **Main CSSR discovery algorithm**

**JS Analysis Package** (`nanoGPT/js_analysis/`):
- `js_metrics.py`: JS divergence computation, k-step distributions
- `state_mapping.py`: Ground truth state mappings for evaluation
- `calibration.py`: Platt calibration for neural probabilities
- `plotting.py`, `js_diagnostics.py`: Visualization tools

**Model Implementations**:
- `nanoGPT/model.py`: nanoGPT transformer (used in main pipeline)
- `experiments/ebm/models.py`: EBM/AR models (experimental)

## Algorithm Details

### JS Divergence Clustering

The `unsupervised_fast_original.py` script implements a three-stage algorithm:

**Stage A - Emission Clustering**: Agglomerative clustering based on next-token probability distributions. Merges histories until JS divergence exceeds `stage_a_threshold`.

**Stage B - Rollout Refinement**: Within each emission cluster, refines using k-step conditional JS divergence. Uses representatives to avoid O(n²) comparisons. Controlled by `stage_b_threshold` and `k_refine`.

**Stage C - State Remerging** (optional): Detects functionally identical states by comparing both emissions and multi-step rollouts. Critical for infinite-memory processes. Enable with `--enable_remerging`.

**Backward Stability** (optional): For each history, finds the shortest suffix that preserves emission distribution within `tolerance_bits`. Useful for identifying minimal causal states in high-memory processes.

### Key Insights

- **Emission signatures** group histories with similar immediate predictions
- **Multi-step rollouts** distinguish states with different long-term behavior
- **Backward stability** finds minimal sufficient statistics (e.g., "BA" vs "001001BA")
- **Platt calibration** improves probability estimates from neural logits

## Common Issues and Solutions

**Issue**: nanoGPT training loss plateaus above theoretical entropy
- **Solution**: Increase `block_size`, reduce `dropout`, train longer

**Issue**: CSSR discovers too many states (over-splitting)
- **Solution**: Increase `stage_a_threshold` and `stage_b_threshold`, or enable remerging

**Issue**: CSSR discovers too few states (under-splitting)
- **Solution**: Decrease thresholds, increase `k_refine`, ensure model is well-trained

**Issue**: Even process fails to recover 2 states
- **Solution**: Use `block_size>=128`, enable `--enable_remerging` with low thresholds

**Issue**: Seven-state machine discovers >7 states
- **Solution**: Use `--backward_stability` to find minimal suffixes and deduplicate

## Tips for New Machines

1. **Start simple**: Test with `golden_mean` (2 states, easy to recover)
2. **Check training**: Verify nanoGPT achieves near-theoretical entropy
3. **Calibrate**: Platt calibration is automatic but check diagnostics
4. **Tune thresholds**: Start with `stage_a_threshold=0.001`, adjust based on results
5. **Use ground truth**: Compare discovered states with `state_mapping.py` mappings