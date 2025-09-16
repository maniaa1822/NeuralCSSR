# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural CSSR is a research platform for using neural networks as probability providers for Causal State Splitting Reconstruction (CSSR). The project integrates both nanoGPT transformers and Energy-Based Models (EBM) with classical CSSR to recover epsilon-machines from binary sequences.

## Development Environment

### Package Manager
This project uses `uv` (not pip) for dependency management:
```bash
# Install dependencies
uv sync

# Run scripts
uv run python script.py  # Optional, scripts work directly with python
```

### Dependencies
Key dependencies from pyproject.toml:
- **Core**: torch, numpy, scipy, scikit-learn
- **Data/Config**: pyyaml, tqdm  
- **Visualization**: matplotlib, seaborn
- **Graph Analysis**: networkx, python-igraph>=0.11.8

## Current Focus: Binary Language Models

The project focuses on binary sequence modeling using transformer architectures in `experiments/ebm/models.py`. The main training pipeline uses the **EnergyBasedBinaryLM** architecture.

### Model Architecture

Two model classes in `experiments/ebm/models.py`:

1. **AutoRegressiveBinaryLM**: Standard transformer-based autoregressive model
   - Uses standard `lm_head` for next-token prediction
   - Applies softmax to logits for probability distribution
   - Standard causal transformer with embedding and positional encoding
   
2. **EnergyBasedBinaryLM**: Current main focus - energy-based formulation
   - Uses an `energy_head` instead of standard language model head
   - Computes scores as `-energies` then applies softmax
   - Maintains same transformer encoder architecture as AR model
   - Used by default in `experiments/ebm/train_golden_mean.py`

Key architectural features:
- Both models support causal masking and mixed precision training
- Identical embedding and positional encoding schemes
- Focus on binary sequence modeling (vocab_size=3, output_vocab_size=2)

### Model Training Commands

The training script `experiments/ebm/train_golden_mean.py` supports both Golden Mean and Even Process presets through the `--preset` flag. It trains the **EnergyBasedBinaryLM** model by default.

**Golden Mean Training:**
```bash
uv run --with torch python experiments/ebm/train_golden_mean.py \
  --preset golden_mean --device auto --context_window 64 --epochs 3 --steps_per_epoch 500 \
  --batch_size 64 --d_model 128 --layers 4 --heads 8 --dropout 0.1 --lr 3e-4
```

**Even Process Training:**
```bash
uv run --with torch python experiments/ebm/train_golden_mean.py \
  --preset even_process --device auto --context_window 128 --epochs 3 --steps_per_epoch 500 \
  --batch_size 64 --d_model 128 --layers 4 --heads 8 --dropout 0.0 --lr 3e-4
```

**Key Features of the Training Script:**
- Automatic preset configuration for Golden Mean and Even Process
- Built-in parity-aware metrics for Even Process validation
- Support for mixed precision training (`--amp`) and TF32 (`--tf32`)
- Automatic checkpoint and CSV logging
- Configurable dataset subsets via `--max_tokens`

**Default Dataset Paths (presets):**
- Golden Mean: `/home/matteo/NeuralCSSR/experiments/datasets/golden_mean/golden_mean.dat`
- Even Process: `/home/matteo/NeuralCSSR/experiments/datasets/even_process/even_process.dat`

**Default Output Paths:**
- Checkpoints: `experiments/ebm/checkpoints/{preset}_ebm.pt` 
- Metrics: `experiments/ebm/metrics/{preset}_metrics.csv`

### Model Evaluation and Analysis

**Dataset Validation (Even Process):**
```bash
uv run python experiments/ebm/check_even_dataset.py \
  --data notebook_experiments/even_process/even_process.dat
```

**Cross-Model Evaluation:**
```bash
uv run --with torch python experiments/ebm/eval_ebm.py \
  --ckpt notebook_experiments/even_process/ebm_ckpt.pt \
  --data notebook_experiments/golden_mean/data/golden_mean/golden_mean.dat
```

### Inductive Bias Probing (IBP)

The project includes sophisticated inductive bias analysis for autoregressive models:

**Even Process IBP (with probe head):**
```bash
uv run python notebook_experiments/ibp/ebm_ibp.py \
  --ckpt notebook_experiments/even_process/ebm_ckpt.pt \
  --preset even_process --train_examples 100 --val_examples 2000 \
  --num_probe_datasets 20 --use_probe_head --ft_steps 50 --ft_lr 1e-3 \
  --eval_batch_size 512 --log_next_token_metrics \
  --output notebook_experiments/ibp/ebm_ibp_even_probe_nt.json
```

**Golden Mean IBP (with probe head):**
```bash
uv run python notebook_experiments/ibp/ebm_ibp.py \
  --ckpt notebook_experiments/golden_mean/ebm_ckpt.pt \
  --preset golden_mean --train_examples 100 --val_examples 2000 \
  --num_probe_datasets 20 --use_probe_head --ft_steps 50 --ft_lr 1e-3 \
  --eval_batch_size 512 --log_next_token_metrics \
  --output notebook_experiments/ibp/ebm_ibp_gm_probe_nt.json
```

**Important IBP Guidelines:**
- Always use `--use_probe_head` to preserve next-token behavior
- Avoid `--tune_scope head/full` for IBP as it can degrade LM performance  
- Monitor next-token retention with `--log_next_token_metrics`

**IBP Recommended Parameters:**
- `--train_examples 100`: Small balanced set per task
- `--val_examples 2000`: Fixed validation subset
- `--num_probe_datasets 20..50`: Number of random tasks to probe
- `--ft_steps 50 --ft_lr 1e-3`: Light training for the probe head
- `--eval_batch_size 512`: Efficient batch processing

**IBP Metrics (R-IB/D-IB):**
- R-IB: Fraction of same-state pairs with identical predictions (higher = better)
- D-IB: 1 - fraction of different-state pairs with identical predictions (higher = better)
- Recent results: Even Process R-IB≈0.98, D-IB≈0.96 with probe head retention

## Legacy: nanoGPT-CSSR Pipeline

The project follows the **Neural CSSR research pipeline** documented in `docs/neural_cssr_pipeline.md` - using nanoGPT models as neural probability providers for CSSR analysis.

### Primary Workflow: 8-Step nanoGPT-CSSR Pipeline

Based on `docs/neural_cssr_pipeline.md`:

1. **Generate Dataset**: `pysm_generator.py --machine <machine> --length 50000`
2. **Prepare nanoGPT Data**: `nanoGPT/data/<machine>/prepare.py` creates train.bin, val.bin, meta.pkl
3. **Train nanoGPT**: Use configs in `nanoGPT/config/train_<machine>_char.py`
4. **Analyze Logits**: `nanoGPT/analyze_logits.py` for loss vs context length analysis
5. **Run Neural CSSR**: `run_neural_cssr.py` with `--nanogpt_out_dir` option
6. **Generate DOT**: Automatic FSM visualization via neural_js backend
7. **Render PNG**: Convert DOT to PNG using graphviz
8. **Validate Results**: Compare recovered epsilon-machine with ground truth

### Machine Catalog for nanoGPT Pipeline

| Machine | States | nanoGPT Performance | CSSR Recovery |
|---------|--------|-------------------|---------------|
| `golden_mean` | 2 | Loss ≈ 0.666 bits/char | 2 states recovered |
| `even_process` | 2 | Loss ≈ 0.671 bits/char | 2 states with proper tuning |
| `complex_csm` | Variable | Depends on complexity | Requires JS threshold tuning |

Key parameters for successful recovery:
- `block_size` ≥ 64 for processes requiring long memory
- `js_threshold` 0.02-0.05 (relax if over-splitting occurs)
- `min_count` ≥ 50 for robust statistics

## Package Structure

```
experiments/ebm/    # MAIN FOCUS: Energy-Based Models
├── models.py       # EnergyBasedBinaryLM & AutoRegressiveBinaryLM
├── train_golden_mean.py # EBM training script
├── eval_ebm.py     # Model evaluation and cross-validation
├── check_even_dataset.py # Dataset validation utilities
└── *.yaml         # Configuration files

nanoGPT/           # Legacy nanoGPT integration
├── model.py       # Core nanoGPT transformer
├── train.py       # Training script
└── config/        # Training configurations

notebook_experiments/ # Data and checkpoints
├── golden_mean/   # Golden mean process data
├── even_process/  # Even process data
└── ibp/          # Inductive bias probing results
```

## Quick Start: Complete nanoGPT-CSSR Experiment

**Example: Even Process Recovery**

```bash
# Step 1: Generate dataset (50k symbols)
uv run python pysm_generator.py --machine even_process --length 50000 --output notebook_experiments --seed 42

# Step 2: Prepare nanoGPT data
uv run --with numpy python nanoGPT/data/even_process/prepare.py

# Step 3: Train nanoGPT (GPU recommended)
cd nanoGPT
uv run --with torch --with numpy python train.py config/train_even_process_char.py \
  --device=cuda --dropout=0.0 --max_iters=10000 --lr_decay_iters=10000 \
  --out_dir=out-even-process-cuda-10k --always_save_checkpoint=True

# Step 4: Analyze logits
uv run --with torch --with numpy python analyze_logits.py \
  --out_dir out-even-process-cuda-10k --split val \
  --max_k 128 --num_positions 5000 --stride 5 --device cpu

# Step 5: Run Neural CSSR
uv run --with torch python -u ../run_neural_cssr.py \
  --data ../notebook_experiments/even_process/even_process.dat \
  --L_max 6 --context_window 64 \
  --backend neural_js --state_metric js --js_threshold 0.03 --min_count 50 \
  --nanogpt_out_dir out-even-process-cuda-10k \
  --dot_out out-even-process-cuda-10k/machine.dot --json_only

# Step 6: Render visualization
uv run --with graphviz python - <<'PY'
import subprocess; p='out-even-process-cuda-10k/machine.dot'
subprocess.run(['dot','-Tpng',p,'-o',p.replace('.dot','.png')], check=True)
PY
```

## Key Files

**Core Model Files:**
- **Models**: `experiments/ebm/models.py` - AutoRegressiveBinaryLM (main focus) & EnergyBasedBinaryLM architectures
- **Training**: `experiments/ebm/train_golden_mean.py` - Main AR model training script
- **Evaluation**: `experiments/ebm/eval_ebm.py` - Model evaluation and cross-validation
- **Dataset Validation**: `experiments/ebm/check_even_dataset.py`

**Legacy nanoGPT-CSSR Pipeline:**
- **Main Script**: `run_neural_cssr.py` - Neural CSSR execution with nanoGPT integration  
- **Dataset Generator**: `pysm_generator.py` - Generate binary sequences from finite state machines

## Model Training Guidelines

**Key Parameters for Binary Processes:**
- **Golden Mean**: `context_window=64`, `dropout=0.1`, optimal for 2-state recovery
- **Even Process**: `context_window=128`, `dropout=0.0`, requires longer memory for parity detection
- **Universal**: Use `--device auto` for automatic GPU/CPU selection
- **Performance**: Enable AMP/TF32 for GPU speedups on larger contexts

**Expected Validation Metrics:**
- **Golden Mean**: H≈2/3 bits; p(0|0)≈0, p(0|1)≈0.5  
- **Even Process**: p0|Even≈0.5, p0|Odd≈0.0; fraction_Odd≈1/3

**Performance Tuning Tips:**
- Increase context window (128–256) for Even Process parity detection
- Use `dropout=0.0` and `weight_decay=0.0` for small binary tasks
- Enable AMP/TF32 for GPU speedups: `--amp --tf32`
- For Even Process: requires longer memory, use `context_window=128` minimum

## Command Reference

All commands use `uv` package manager. For training commands, copy-paste directly into terminal.

**Important**: Commands are designed to be copy-pasteable for immediate execution.

## Experimental Results

Based on actual training runs and evaluations in the codebase:

### Model Performance Summary

**Golden Mean Results:**
- **EBM Model**: Converges to ~0.455 nats (0.656 bits), achieving target metrics p(0|0)≈0.004, p(0|1)≈0.553
- **AR Model**: Similar performance ~0.461 nats (0.667 bits), p(0|0)≈0.0001, p(0|1)≈0.491
- **Both models**: Successfully learn Golden Mean structure with zero violations

**Even Process Results:**
- **EBM Model**: Achieves ~0.499 nats (0.720 bits) with proper parity metrics:
  - p0|Even≈0.399, p0|Odd≈0.083, frac_Odd≈0.344 (close to theoretical 1/3)
- **AR Model**: Similar performance ~0.513 nats (0.740 bits) with:
  - p0|Even≈0.435, p0|Odd≈0.151, frac_Odd≈0.333 (exact theoretical)
- **Context dependency**: Even Process requires `context_window≥128` for proper parity learning

### Inductive Bias Probing (IBP) Results

**Golden Mean AR Model (20 probe tasks):**
- R-IB: 0.998 ± 0.001 (excellent state discrimination)
- D-IB: 0.848 ± 0.356 (high variance across tasks)
- Next-token retention: 65.4% accuracy preserved

**Even Process EBM Model (1 probe task):**
- R-IB: 0.981 (strong state consistency)  
- D-IB: 0.963 (excellent differentiation)
- Next-token retention: 64.9% accuracy preserved
- Proper parity detection: argmax0|Odd≈0.005 (near-zero as expected)

### Training Characteristics

**Convergence**: Both models typically converge within 3 epochs (1500 steps total)
**Stability**: EBM models show slightly more stable final metrics
**Memory Requirements**: Even Process needs 2-4x context window vs Golden Mean
**Performance**: Both architectures achieve similar final loss values on both processes