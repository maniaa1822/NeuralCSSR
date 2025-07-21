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

The project consists of three main analysis pipelines:

### 1. Dataset Generation (`generate_unified_dataset.py`)
Unified framework for creating synthetic FSM datasets with multiple output formats.

### 2. Classical CSSR Analysis (`analyze_classical_cssr.py`) 
Comprehensive classical CSSR analysis with parameter sweep optimization and ground truth evaluation.

### 3. Machine Distance Analysis (`analyze_machine_distances.py`)
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

## Common Development Commands

### Setup
```bash
uv sync                                    # Install dependencies
```

### Dataset Generation
```bash
# Built-in presets
python generate_unified_dataset.py --preset small --output datasets/small_exp      # 2.5K sequences, 5 machines
python generate_unified_dataset.py --preset medium --output datasets/medium_exp    # 25K sequences, 16 machines
python generate_unified_dataset.py --preset large --output datasets/large_exp      # 122K sequences, 33 machines
python generate_unified_dataset.py --preset biased --output datasets/biased_exp    # 6.5K sequences, mixed probability types

# Custom configuration
python generate_unified_dataset.py --config my_config.yaml --output datasets/custom_exp
```

### Classical CSSR Analysis
```bash
# Single dataset with parameter sweep
python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/cssr_analysis --parameter-sweep

# Batch analysis of multiple datasets
python analyze_classical_cssr.py --batch --datasets-dir datasets --output results/batch_analysis
```

### Machine Distance Analysis
```bash
# Comprehensive distance analysis
python analyze_machine_distances.py biased_exp --comprehensive --output-dir results/distance_analysis

# Quick analysis for scripting
python analyze_machine_distances.py biased_exp --quiet
```

### Testing
```bash
# Classical CSSR tests (no formal test framework yet)
uv run python tests/classical_cssr/test_classical_cssr.py
uv run python tests/classical_cssr/analyze_cssr.py
```

### Complete Research Workflow
```bash
# 1. Generate dataset
python generate_unified_dataset.py --preset biased --output datasets/experiment

# 2. Run classical CSSR analysis
python analyze_classical_cssr.py --dataset datasets/experiment --output results/cssr --parameter-sweep

# 3. Evaluate reconstruction quality
python analyze_machine_distances.py experiment --comprehensive --output-dir results/distances
```

## Configuration System

The project uses YAML-based configuration for experiments:

### Built-in Presets
- **small**: 5 machines, uniform probabilities, fast testing
- **medium**: 16 machines, uniform probabilities, standard research
- **large**: 33 machines, uniform probabilities, comprehensive studies  
- **biased**: 6 machines, mixed uniform/biased probabilities, bias studies

### Custom Configuration
Create YAML files following schema in `src/neural_cssr/config/dataset_configs/`:

```yaml
experiment_name: "my_experiment"
machine_specs:
  - complexity_class: "2-state-binary"
    machine_count: 3
    samples_per_machine: 1000
    topological: true                    # Uniform probabilities
  
  - complexity_class: "3-state-binary" 
    machine_count: 2
    samples_per_machine: 1500
    topological: false                   # Custom probabilities
    bias_strength: 0.7                   # Random bias strength
    probability_seed: 123                # Reproducible bias
```

## Output Structure

### Generated Datasets (`datasets/[name]/`)
- `raw_sequences/` - Text sequences for classical CSSR
- `neural_format/` - PyTorch datasets for neural training
- `ground_truth/` - Machine definitions and state trajectories
- `statistical_analysis/` - Information-theoretic measures
- `quality_reports/` - Coverage validation and quality scores
- `experiment_config.yaml` - Configuration used
- `generation_info.yaml` - Generation metadata

### Analysis Results (`results/[analysis]/`)
- `classical_cssr_analysis_report.html` - Comprehensive HTML report
- `classical_cssr_results.json` - Detailed JSON results
- `machine_distance_report.md` - Distance analysis report
- `machine_distance_report.json` - Distance metrics in JSON
- `*.png` - Professional visualizations

## Neural CSSR Implementations

The repository includes several neural approaches (experimental stage):

### Available Implementations
- `time_delay_transformer.py` - Transformer trained on FSM sequences
- `extract_epsilon_machine_from_transformer.py` - Extract ε-machines from transformers  
- `analyze_internal_states.py` - Analyze transformer hidden states for causal structure
- `investigate_fsm_learning.py` - Study FSM learning in transformers

### Usage
```bash
python time_delay_transformer.py
python investigate_fsm_learning.py
python analyze_internal_states.py
```

## Key Technical Notes

### File Handling
- All file paths are absolute for cross-platform compatibility
- PyTorch datasets use `weights_only=False` for compatibility
- JSON serialization handles numpy arrays automatically

### Machine Types
- **Topological machines**: Uniform transition probabilities
- **Biased machines**: Custom or randomly biased probabilities
- Both types supported throughout the framework

### Quality Validation
- Automated coverage analysis (states, transitions)
- Quality scores adapted for non-uniform probability distributions
- Comprehensive validation reports with recommendations

## Project Status

### Production Ready
- **Dataset Generation**: Complete framework with quality validation
- **Classical CSSR**: Parameter sweep optimization with ground truth evaluation
- **Machine Distance Analysis**: 6-metric comprehensive evaluation framework
- **Professional Reporting**: HTML reports and publication-quality visualizations

### Experimental
- **Neural CSSR**: Multiple implementations available but not integrated into main analysis pipeline
- **FSM Extraction**: Various approaches to extract FSMs from neural networks
- **Transfer Learning**: Preliminary studies of scaling laws between machine types

### Missing Infrastructure
- No formal test framework (pytest)
- No CI/CD pipeline
- No pre-commit hooks or automated code quality tools
- Neural results not integrated with distance analysis framework

## Documentation

Comprehensive documentation available in `plans_and_guides/`:
- `README.md` - Documentation index
- `MACHINE_DISTANCE_README.md` - Distance analysis usage
- `MACHINE_DISTANCE_INTERPRETATION_GUIDE.md` - Results interpretation
- Various research plans and implementation guides

## Research Workflow

This project provides a complete research ecosystem for epsilon-machine analysis:

1. **Generate synthetic datasets** with controlled properties
2. **Run classical CSSR baselines** with parameter optimization
3. **Evaluate reconstruction quality** using quantitative distance metrics
4. **Generate professional reports** suitable for publications
5. **Experiment with neural approaches** for comparison studies

The framework is designed for systematic research in computational mechanics and transfer learning scaling laws between synthetic finite state machine datasets.