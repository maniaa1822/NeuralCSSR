# Neural CSSR
## Overview
Neural CSSR reconstructs **epsilon machines**—minimal causal-state models of stochastic processes—by combining neural next-token predictors with information-theoretic clustering. The repository contains:

- Unified machine specifications (`machines/`) for canonical processes such as *Seven-State Human*, *Even Process*, and *Golden Mean*.
- Data generation utilities (`pysm_generator_v2.py`) that emit aligned symbol/state sequences from any registered machine.
- A fork of nanoGPT (`nanoGPT/`) used to train transformer language models on generated datasets.
- Fast, JS-divergence-based epsilon machine discovery (`cssr_discovery/unsupervised_fast_v2.py`).
- Legacy baselines including classical transCSSR and earlier neural adapters.
- Research assets (notebooks, LaTeX presentation) documenting the motivation, analysis, and experimental findings.

## Getting Started

### Prerequisites

- Python ≥ 3.8
- [`uv`](https://github.com/astral-sh/uv) for dependency management (preferred over pip)
- CUDA-capable GPU strongly recommended for transformer training (set `dtype='float16'` for RTX 20-series)

### Environment Setup

```bash
uv sync              # install dependencies into .venv
```

## Repository Structure

| Path | Description |
|------|-------------|
| `machines/` | Machine registry, abstract base class, and concrete epsilon-machine specifications |
| `pysm_generator_v2.py` | Streamlined CLI for dataset generation using unified machines |
| `nanoGPT/` | Transformer training code (submodule-style fork) |
| `cssr_discovery/` | JS-divergence clustering, calibration, and ground-truth mapping tools |
| `transcssr_baseline/`, `transCSSR/` | Classical CSSR baselines and neural adapters |
| `docs/` | Presentations, LaTeX notebook (`neuralcssr_notebook.tex`), and supporting figures |
| `experiments/` | Generated datasets (symbol traces, state traces, metadata) |

## Workflow

1. **Generate data** from a named machine
2. **Train** a nanoGPT model on the generated binary sequence
3. **Discover** epsilon-machine states using unsupervised JS clustering
4. **Evaluate** purity, coverage, and epsilon machine loss; optionally compare against ground truth

### 1. Dataset Generation

```bash
# Seven-State Human (100k symbols, reproducible seed)
uv run python pysm_generator_v2.py \
  --machine seven_state_human \
  --length 100000 \
  --seed 42

# Even Process with default length (100000)
uv run python pysm_generator_v2.py --machine even_process
```

Outputs (under `experiments/datasets/<machine>/`):

- `<machine>.dat` – binary sequence
- `<machine>.states` – integer state indices
- `<machine>.states.dat` – character-coded states (A, B, …)
- `<machine>.machine.json` – transition graph
- `<machine>.meta.json` – metadata including state mappings and generation config

### 2. nanoGPT Training

Train from the repository root or from inside `nanoGPT/`:

```bash
# run within nanoGPT directory

cd nanoGPT
uv run --with torch --with numpy python train.py \
  config/train_seven_state_human_char_large.py \
  --device=cuda \
  --out_dir=out-seven-state-human-char-large \
  --always_save_checkpoint=True
```

Key config highlights (`config/train_seven_state_human_char_large.py`):

- Larger architecture (`n_layer=8`, `n_head=8`, `n_embd=64`)
- Mixed precision via `dtype='float16'` for RTX 20-series GPUs
- Example command stored inline for reference

`machines/<process>.py` exposes `get_training_config(variant)` and `get_model_path(...)` helpers to keep checkpoint naming consistent.

### 3. Fast JS-Divergence Discovery

`cssr_discovery/unsupervised_fast_v2.py` runs the three-stage clustering algorithm:

```bash
uv run python cssr_discovery/unsupervised_fast_v2.py \
  --machine seven_state_human \
  --variant char_large \
  --history-length 5 \
  --n-samples 100 \
  --backward-stability \
  --enable-remerging \
  --output-json results/seven_state_human_v2.json

uv run python cssr_discovery/unsupervised_fast_v2.py \
  --machine even_process \
  --variant char_large \
  --history-length 6 \
  --n-samples 150 \
  --backward-stability \
  --enable-remerging \
  --output-json results/even_process_v2.json
```

Stages:

1. **Emission clustering** (Stage A) by JS divergence on next-token probabilities
2. **Backward stability** (Stage A+) to shrink contexts to minimal predictive suffixes
3. **Representative refinement** (Stage B) using conditional JS over k-step rollouts with caching
4. **State remerging** (Stage C) to merge functionally equivalent states—critical for infinite-memory processes

Outputs include:

- Cluster purity vs. ground truth
- Discovered state count vs. expected
- Weighted purity and coverage metrics
- Jensen-Shannon diagnostics and cache performance
- Epsilon machine loss compared to original neural model

### 4. Evaluation & Diagnostics

- `cssr_discovery/state_mapping.py` bridges legacy presets to unified machines for consistent ground-truth evaluation.
- `docs/presentations/neuralcssr-notebook/neuralcssr_notebook.tex` provides a comprehensive research narrative covering classical CSSR brittleness, neural probability estimation, JS-based clustering, backward stability, and experimental diagnostics (JS matrices, loss comparisons, etc.).

## Legacy Components

- `pysm_generator.py` retains extensive state machine implementations built on `statemachine` for older experiments. Prefer `pysm_generator_v2.py` for unified workflow.
- `transcssr_baseline/` contains neural adapters that plug transformer probabilities into classical transCSSR.
- `transCSSR/` is the original Causal State Splitting Reconstruction library for reference and comparison.

## Tips & Best Practices

- **GPU precision**: For GPUs lacking native bfloat16 (e.g., RTX 2060), set `dtype='float16'` to avoid warnings and still gain mixed-precision speedups.
- **Backward stability**: Always enable `--backward-stability` and `--enable-remerging` when running `unsupervised_fast_v2.py`; they dramatically reduce context length and solve infinite-memory over-segmentation.
- **Sampling**: `--sampling-strategy emission_stratified` (default) ensures representative coverage of emission probabilities; adjust `--n-samples` (50–200) based on process complexity.
- **Thresholds**: Start with `stage_a_threshold=stage_b_threshold=0.001`; the JS-based heuristics are far less brittle than classical CSSR significance levels.

## Further Reading

The LaTeX notebook under `docs/presentations/neuralcssr-notebook/` captures:

- Motivation and limitations of classical CSSR (parameter brittleness, infinite memory issues)
- Neural CSSR’s probability-oracle perspective
- JS divergence analysis, histograms, and figures quantifying state separability
- Backward stability token-dropping statistics and stage-wise clustering diagnostics
- Experimental case studies: Seven-State Human, Even Process, Golden Mean
- Future directions (multi-process transfer, in-context learning, representation shaping)

## Contributing

- Run `python3 -m compileall ...` or relevant lint/tests before submitting changes.
- Follow the unified machine framework (`Machine` subclasses + `@register_machine` decorator) when adding new processes.
- Keep checkpoint and dataset paths consistent by reusing `Machine.get_model_path()` and `Machine.get_dataset_path()`.

## License

Project metadata and licensing terms are maintained via `pyproject.toml`. Refer to upstream component licenses (e.g., nanoGPT, transCSSR) for their specific terms.