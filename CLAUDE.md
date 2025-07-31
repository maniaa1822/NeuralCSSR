# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural CSSR is a comprehensive research platform for studying both classical and neural approaches to Causal State Splitting Reconstruction (CSSR) with epsilon-machines. The project provides end-to-end pipelines for generating synthetic datasets, running classical CSSR analysis, and evaluating machine reconstruction quality through quantitative distance metrics.

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

## Core Architecture

The project consists of four main analysis pipelines:

### 1. Dataset Generation (`generate_unified_dataset.py`)
Unified framework for creating synthetic FSM datasets with multiple output formats.

### 2. Domain-Specific Dataset Generation (`generate_domain_dataset.py`)
Streamlined generator for single-machine datasets with aligned state trajectories for neural training and linear probe experiments.

### 3. Dataset Format Conversion (`convert_to_transcssr.py`)
Converts NeuralCSSR datasets to transCSSR-compatible .dat format with optional burn-in trimming.

### 4. Classical CSSR Analysis (`analyze_classical_cssr.py`) 
Comprehensive classical CSSR analysis with parameter sweep optimization and ground truth evaluation.

### 5. Machine Distance Analysis (`analyze_machine_distances.py`)
Quantitative comparison framework using 6 distance metrics between reconstructed and ground truth machines.

## Package Structure

```
src/neural_cssr/
├── core/           # Epsilon machine fundamentals (epsilon_machine.py)
├── data/           # Dataset generation framework (dataset_generator.py, sequence_processor.py)
├── enumeration/    # Machine enumeration algorithms (enumerate_machines.py)
├── classical/      # Classical CSSR implementation (cssr.py, transcssr_wrapper.py)
├── neural/         # Neural CSSR components (transformer.py)
├── analysis/       # Classical CSSR analysis pipeline (classical_analyzer.py)
├── evaluation/     # Machine distance analysis system (machine_distance.py)
├── config/         # Configuration schemas and presets (generation_schemas.py)
└── machines/       # Domain-specific machine implementations
```

## Project Memories

- Remember we are working with domain specific machines for now
- We will refactor the transformer file remembering all the related things we need to account for
- Remember the iter to create the dataset, convert it, and perform cssr on
- We should train with chunk size of approx 20 L
- Remember the parameters scope and importance

## Training Runs

- Use this command for training runs to paste in the terminal:
```bash
python time_delay_transformer.py --train data/golden_mean/golden_mean.dat --mode ar --epochs 20 --batch 128 --d_model 64 --layers 2 --heads 4 --lr 1e-3
```

## Memories

- Used commands to create a new file for storing commands and steps for the project
- Created a memory file with the commands used for training and project setup