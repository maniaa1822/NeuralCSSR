# CURSOR.md

This file provides guidance for using Cursor when working with the NeuralCSSR repository. It covers project overview, development environment setup, key workflows, and Cursor-specific tips for maximum productivity.

## Project Overview

Neural CSSR discovers epsilon-machines (minimal causal state representations) from sequences using neural networks as probability estimators. The project has a main workflow:

1. **Current Working Pipeline**: Generate data with `pysm_generator_v2.py` → Train nanoGPT model → Discover states with `cssr_discovery/unsupervised_fast_v2.py`

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

## Machine Specifications (NEW!)

All machines are now unified under the `machines/` package, providing a single source of truth for machine properties across the entire pipeline.

### Using Machines

```python
from machines import get_machine, list_machines

# List all available machines
print(list_machines())  # ['even_process', 'golden_mean', 'seven_state_human']

# Get a specific machine
machine = get_machine("seven_state_human")

# Access properties
print(machine.states)           # ['bb', 'aaa', 'aaab', ...]
print(machine.num_states)       # 7
print(machine.memory_length)    # 4
print(machine.emissions)        # {'bb': {'0': 0.9375, '1': 0.0625}, ...}

# Ground truth evaluation
import numpy as np
history = np.array([0, 0, 0, 1])
state = machine.get_gt_state(history)  # Returns 'aaab'

# Theoretical properties
entropy = machine.compute_theoretical_entropy()  # 0.8218 bits
suffixes = machine.get_state_suffixes()  # {'bb': ['11'], 'aaa': ['000'], ...}

# Pipeline integration
dataset_path = machine.get_dataset_path(Path("experiments/datasets"))
model_path = machine.get_model_path(Path("nanoGPT"), variant="char_large")
config = machine.get_training_config(variant="char_large")
```

### Available Machines

- **seven_state_human**: 7 states, finite memory (L≤4), complex suffix structure
- **golden_mean**: 2 states, finite memory (L=1), no consecutive 0s
- **even_process**: 2 states, infinite memory, runs of 1s have even length

Each machine encapsulates:
- Structure (states, transitions, emissions)
- Ground truth state mapping
- Theoretical properties (entropy, memory length)
- Training recommendations (context window, hyperparameters)

## Current Workflow: nanoGPT + Unsupervised CSSR

This is the **primary working pipeline** for epsilon-machine discovery.

### Step 1: Generate Dataset

Use `pysm_generator_v2.py` to create binary sequences from finite state machines:

```bash
# Seven-state human machine (7 causal states)
uv run python pysm_generator_v2.py --machine seven_state_human --length 100000 --seed 42

# Even process (2 states, infinite memory)
uv run python pysm_generator_v2.py --machine even_process --length 50000 --seed 42

# Golden mean (2 states, finite memory)
uv run python pysm_generator_v2.py --machine golden_mean --length 50000 --seed 42
```

**Output Files** (in `experiments/datasets/<machine>/`):
- `<machine>.dat`: Binary sequence (0s and 1s)
- `<machine>.states`: Ground truth state indices
- `<machine>.states.dat`: State sequence as characters (A, B, C, ...)
- `<machine>.machine.json`: Transition structure
- `<machine>.meta.json`: Metadata and mappings

### Step 2: Train nanoGPT Model

Train a character-level transformer using nanoGPT's config system:

```bash
cd nanoGPT

# Seven-state human (recommended settings)
uv run --with torch --with numpy python train.py \
  config/train_seven_state_human_char_large.py \
  --device=cuda --out_dir=out-seven-state-human-char-large \
  --always_save_checkpoint=True

# Even process (requires longer context)
uv run --with torch --with numpy python train.py \
  config/train_even_process_char_large.py \
  --device=cuda --out_dir=out-even-process-char-large \
  --always_save_checkpoint=True

## New Workflow: Per-Layer State Probing

Use the state probing utilities to measure linear vs non-linear probe
performance when predicting machine states from GPT activations:

```bash
# 1) Ensure dependencies are available
uv sync

# 2) Run probes on a baseline checkpoint under IID data
uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_state.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --dataset seven_state_human_mtl100k \
  --machine seven_state_human \
  --tokens 65536 \
  --block-size 64 \
  --batch-size 64 \
  --epochs 5 \
  --mlp-hidden 256 \
  --output runs/probes/state_probe_baseline.csv

# 3) Inspect results
column -t -s, runs/probes/state_probe_baseline.csv | less -S

# 4) Run distance probes (baseline)
uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_distance.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --dataset seven_state_human_mtl100k \
  --machine seven_state_human \
  --tokens 65536 \
  --block-size 64 \
  --batch-size 64 \
  --epochs 5 \
  --mlp-hidden 256 \
  --output runs/probes/state_probe_distance_baseline.csv

# 5) Run probes on multitask eps+dist checkpoint
uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_state.py \
  --model mtl=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --dataset seven_state_human_mtl100k \
  --machine seven_state_human \
  --tokens 65536 \
  --block-size 64 \
  --batch-size 64 \
  --epochs 5 \
  --mlp-hidden 256 \
  --output runs/probes/state_probe_epsdist.csv

uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_distance.py \
  --model mtl=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --dataset seven_state_human_mtl100k \
  --machine seven_state_human \
  --tokens 65536 \
  --block-size 64 \
  --batch-size 64 \
  --epochs 5 \
  --mlp-hidden 256 \
  --output runs/probes/state_probe_distance_epsdist.csv
```

Outputs a CSV with per-layer metrics (loss, accuracy) for both linear and
MLP probes.
```

**Training Config Files**: `nanoGPT/config/train_<machine>_char_large.py`
**Checkpoint Output**: `nanoGPT/out-<machine>-char-large/ckpt.pt`

### Step 3: Discover Epsilon Machine

Run unsupervised state discovery using JS divergence clustering:

```bash
# Seven-state human with backward stability (finds minimal suffixes)
uv run python cssr_discovery/unsupervised_fast_v2.py \
  --machine seven_state_human \
  --variant char_large \
  --history-length 5 \
  --n-samples 100 \
  --backward-stability \
  --enable-remerging \
  --output-json results/seven_state_human_v2.json

# Even process (simpler, no backward stability needed)
uv run python cssr_discovery/unsupervised_fast_v2.py \
  --machine even_process \
  --variant char_large \
  --history-length 6 \
  --n-samples 150 \
  --backward-stability \
  --enable-remerging \
  --output-json results/even_process_v2.json
```

## Cursor-Specific Tips

### 1. Using the File Explorer
- Use the file explorer to navigate the structured directories:
  - `machines/`: Machine definitions
  - `nanoGPT/`: Neural training code
  - `cssr_discovery/`: State discovery algorithms
  - `experiments/`: Generated datasets
  - `results/`: Output from discovery runs

### 2. Leveraging the Chat Panel
- Ask specific questions about any part of the codebase
- Use "@file" to reference specific files in your questions
- Use "@symbol" to ask about specific functions or classes

### 3. Using the Terminal
- Run all commands directly in Cursor's integrated terminal
- Use `uv run` prefix for all Python scripts to ensure correct environment
- Check outputs in the `results/` directory after running discovery

### 4. Code Navigation
- Use "Go to Symbol" (Ctrl/Cmd + Shift + O) to quickly find functions and classes
- Use "Find All References" to understand how components are used across the codebase
- Use "Peek Definition" to view code without leaving your current file

### 5. Working with Notebooks
- The `docs/` directory contains LaTeX notebooks with theoretical background
- Use Cursor's notebook support to analyze results and create visualizations

## Quick Reference: Complete Pipeline

**Seven-State Human (Full Example)**

```bash
# Step 1: Generate dataset
uv run python pysm_generator_v2.py --machine seven_state_human --length 100000 --seed 42

# Step 2: Train nanoGPT (from nanoGPT directory)
cd nanoGPT
uv run --with torch --with numpy python train.py \
  config/train_seven_state_human_char_large.py \
  --device=cuda \
  --out_dir=out-seven-state-human-char-large \
  --always_save_checkpoint=True
cd ..

# Step 3: Discover epsilon machine
uv run python cssr_discovery/unsupervised_fast_v2.py \
  --machine seven_state_human \
  --variant char_large \
  --history-length 5 \
  --n-samples 100 \
  --backward-stability \
  --enable-remerging \
  --output-json results/seven_state_human_v2.json
```

## Key Modules

**Machine Specifications** (`machines/`):
- `base.py`: Abstract Machine class with unified interface
- `seven_state_human.py`, `golden_mean.py`, `even_process.py`: Machine definitions
- **Usage**: `from machines import get_machine; machine = get_machine("seven_state_human")`

**Core Pipeline**:
- `pysm_generator_v2.py`: Dataset generation from finite state machines
- `nanoGPT/train.py`: Transformer training (character-level)
- `cssr_discovery/unsupervised_fast_v2.py`: Main CSSR discovery algorithm

## Common Issues and Solutions

**Issue**: nanoGPT training loss plateaus above theoretical entropy
- **Solution**: Ensure model is well-trained, check that loss is near theoretical entropy

**Issue**: CSSR discovers too many states (over-splitting)
- **Solution**: Adjust thresholds in `unsupervised_fast_v2.py`, or enable remerging

**Issue**: CSSR discovers too few states (under-splitting)
- **Solution**: Ensure model is well-trained, adjust thresholds in discovery script

**Issue**: Even process fails to recover 2 states
- **Solution**: Use sufficient history length and samples, enable remerging

**Issue**: Seven-state machine discovers >7 states
- **Solution**: Use `--backward-stability` to find minimal suffixes and deduplicate
