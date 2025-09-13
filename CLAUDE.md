# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural CSSR is a research platform for using neural networks as probability providers for Causal State Splitting Reconstruction (CSSR). The project integrates nanoGPT transformers with classical CSSR to recover epsilon-machines from binary sequences.

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

## Current Best Practice: nanoGPT-CSSR Pipeline

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
src/neural_cssr/
├── core/           # Epsilon machine fundamentals (epsilon_machine.py)
├── data/           # Dataset generation framework
├── classical/      # Classical CSSR implementation (cssr.py, transcssr_wrapper.py)
├── neural/         # Neural probability providers
└── machines/       # Domain-specific machine implementations

nanoGPT/
├── model.py        # Core nanoGPT transformer
├── train.py        # Training script
├── analyze_logits.py # Logit analysis for CSSR
├── data/           # Dataset preparation scripts
└── config/         # Training configurations
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

## Key Files for nanoGPT-CSSR Pipeline

- **Main Script**: `run_neural_cssr.py` - Neural CSSR execution with nanoGPT integration
- **Dataset Generator**: `pysm_generator.py` - Generate binary sequences from finite state machines
- **Pipeline Documentation**: `docs/neural_cssr_pipeline.md` - Complete workflow guide
- **nanoGPT Integration**: Full nanoGPT directory with model, training, and analysis tools
- **Classical CSSR**: `transCSSR/` for comparison and validation

## Current Research Focus: Neural Probability Providers

The branch focuses on using trained neural networks (specifically nanoGPT) as probability providers for classical CSSR algorithms. This approach:

1. **Trains nanoGPT** on binary sequences to learn conditional probabilities
2. **Uses neural probabilities** instead of empirical frequencies in CSSR sufficiency tests
3. **Recovers epsilon-machines** that capture the learned computational structure
4. **Validates results** against ground truth finite state machines

## Expected Results

- **Golden Mean**: 2-state recovery with clean probability boundaries
- **Even Process**: 2-state recovery with proper parity detection (requires block_size ≥ 64)
- **Complex Processes**: Variable recovery quality depending on neural learning and CSSR parameter tuning

## Branch-Specific Notes

This `nanogpt-cssr-pipeline` branch is focused exclusively on the nanoGPT-based neural CSSR approach. It does not include:
- Sliding window transformers
- CSSR-enhanced hybrid extractors  
- Domain-specific machine analysis
- Interactive notebook workflows

For those approaches, see the main branch or other research branches.
- for training runs give the commands so i can paste them in the terminal and run myself
- remember this when givin training commands
- remember this command
- remember the command for the evenprocess