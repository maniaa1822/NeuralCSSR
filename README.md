# Neural CSSR: Bridging Classical Computational Mechanics with Modern Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Neural CSSR is a research project that enhances classical Causal-State Splitting Reconstruction (CSSR) by replacing count-based probability estimation with neural networks (transformers). This approach achieves exponentially better sample efficiency while preserving theoretical guarantees for discovering minimal sufficient statistics in dynamical systems.

The central question addressed: **"What is the minimal information we need to predict a system's future behavior?"**

## Key Innovations

- **Enhanced Sample Efficiency**: Uses transformer generalization to achieve O(N) sample complexity instead of classical CSSR's O(k^L)
- **Preserved Theoretical Guarantees**: Maintains optimality properties of classical CSSR while leveraging neural network capabilities
- **Comprehensive Dataset Framework**: Systematic generation of ground-truth datasets from enumerated ε-machines
- **Multi-Format Output**: Supports both classical CSSR and modern neural network training formats

## Quick Start

### Installation

Using `uv` (recommended):
```bash
git clone https://github.com/your-username/NeuralCSSR.git
cd NeuralCSSR
uv sync
```

Using pip:
```bash
git clone https://github.com/your-username/NeuralCSSR.git
cd NeuralCSSR
pip install -e .
```

### Generate Your First Dataset

```bash
# Small experiment (2,500 sequences, 5 machines)
python generate_unified_dataset.py --preset small --output datasets/small_exp

# Medium experiment (25,000 sequences, 16 machines)
python generate_unified_dataset.py --preset medium --output datasets/medium_exp

# Biased experiment (non-uniform probabilities)
python generate_unified_dataset.py --preset biased --output datasets/biased_exp
```

### Analyze Classical CSSR Performance

```bash
# Run classical CSSR analysis on generated dataset
python analyze_classical_cssr.py datasets/small_exp
```

## Architecture

### Project Structure

```
NeuralCSSR/
├── src/neural_cssr/          # Core package
│   ├── core/                 # Fundamental classes (ε-machines, etc.)
│   ├── data/                 # Dataset generation framework
│   ├── enumeration/          # Machine enumeration algorithms
│   ├── classical/            # Classical CSSR implementation
│   ├── neural/               # Neural CSSR components
│   ├── analysis/             # Analysis and visualization tools
│   └── config/               # Configuration schemas and presets
├── datasets/                 # Generated datasets
├── results/                  # Experimental results
├── tests/                    # Test suite
├── fsm_transformer/          # Transformer implementation
└── docs/                     # Documentation
```

### Core Components

#### 1. ε-Machine Framework (`src/neural_cssr/core/`)
- **EpsilonMachine**: Core ε-machine implementation with topological and biased variants
- **CausalState**: Representation of causal states and their properties

#### 2. Dataset Generation (`src/neural_cssr/data/`)
- **UnifiedDatasetGenerator**: Main orchestrator for dataset creation
- **SequenceProcessor**: Raw sequence generation and processing
- **NeuralFormatter**: PyTorch dataset creation for transformer training
- **MetadataComputer**: Information-theoretic and complexity measures
- **QualityValidator**: Automated quality assurance and validation

#### 3. Classical CSSR (`src/neural_cssr/classical/`)
- **ClassicalCSSR**: Full implementation of classical CSSR algorithm
- **StatisticalTests**: Chi-square, KL-divergence, and permutation tests
- Compatible with both empirical and neural probability estimation

#### 4. Neural Components (`src/neural_cssr/neural/`)
- **NeuralCSSR**: Neural-enhanced CSSR implementation
- **TransformerProbe**: Linear probes for causal state extraction
- **ProbabilityEstimator**: Neural probability estimation modules

## Dataset Generation Framework

### Built-in Presets

| Preset | Machines | Sequences | Probabilities | Use Case |
|--------|----------|-----------|---------------|----------|
| `small` | 5 | 2,500 | Uniform | Quick testing |
| `medium` | 16 | 25,000 | Uniform | Standard experiments |
| `large` | 33 | 122,000 | Uniform | Large-scale studies |
| `biased` | 6 | 6,500 | Mixed uniform/biased | Non-topological studies |

### Custom Configuration

Create YAML configurations for custom experiments:

```yaml
# custom_experiment.yaml
experiment_name: "my_custom_experiment"
random_seed: 42

machine_specs:
  # Uniform (topological) machines
  - complexity_class: "2-state-binary"
    machine_count: 3
    samples_per_machine: 1000
    topological: true
    
  # Biased (non-topological) machines  
  - complexity_class: "3-state-binary"
    machine_count: 2
    samples_per_machine: 800
    probability_seed: 123  # For reproducible random bias
    
sequence_spec:
  length_distribution: [50, 500]
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

quality_spec:
  min_state_coverage: 100
  length_diversity_threshold: 0.4
```

Run with:
```bash
python generate_unified_dataset.py --config custom_experiment.yaml --output datasets/custom
```

### Output Structure

Each generated dataset includes:

```
datasets/experiment_name/
├── raw_sequences/           # Plain text for classical CSSR
│   ├── train_sequences.txt
│   ├── val_sequences.txt
│   ├── test_sequences.txt
│   └── sequence_metadata.json
├── neural_format/           # PyTorch datasets
│   ├── train_dataset.pt
│   ├── val_dataset.pt
│   ├── test_dataset.pt
│   └── vocab_metadata.json
├── ground_truth/            # Complete causal state information
│   ├── machine_definitions.json
│   ├── causal_state_labels.json
│   └── transition_matrices.json
├── statistical_analysis/    # Information-theoretic measures
│   ├── information_measures.json
│   ├── complexity_metrics.json
│   └── sequence_statistics.json
├── quality_reports/         # Validation and quality metrics
│   ├── coverage_analysis.json
│   ├── distribution_validation.json
│   └── generation_diagnostics.json
└── experiment_config.yaml   # Complete configuration used
```

## Usage Examples

### 1. Basic Dataset Generation

```python
from neural_cssr.data.dataset_generator import UnifiedDatasetGenerator

# Generate dataset from configuration
generator = UnifiedDatasetGenerator(
    config_path="config/small_experiment.yaml",
    output_dir="datasets/my_experiment",
    seed=42
)

report = generator.generate_dataset()
print(f"Generated {report['total_sequences']} sequences")
print(f"Quality score: {report['quality_score']:.3f}")
```

### 2. Classical CSSR Analysis

```python
from neural_cssr.classical.cssr import ClassicalCSSR
from neural_cssr.data.dataset_generation import load_dataset

# Load generated dataset
dataset = load_dataset("datasets/small_exp")

# Run classical CSSR
cssr = ClassicalCSSR(significance_level=0.05)
cssr.load_from_dataset(dataset)
success = cssr.run_cssr(max_iterations=20)

# Evaluate against ground truth
results = cssr.evaluate_against_ground_truth(dataset)
print(f"State recovery accuracy: {results['state_accuracy']:.3f}")
```

### 3. Neural CSSR Training

```python
from neural_cssr.neural.neural_cssr import NeuralCSSR
from neural_cssr.neural.transformer_probe import TransformerProbe

# Load neural format dataset
train_dataset = torch.load("datasets/small_exp/neural_format/train_dataset.pt")

# Train transformer for probability estimation
neural_cssr = NeuralCSSR(
    vocab_size=2,  # Binary alphabet
    hidden_size=512,
    num_layers=6
)

# Train on autoregressive task
neural_cssr.train(train_dataset, epochs=50)

# Extract causal states using linear probe
probe = TransformerProbe(neural_cssr.transformer)
causal_states = probe.extract_causal_states(train_dataset)
```

## Theoretical Foundation

### Causal States and ε-Machines

**Causal State Definition**: Two histories belong to the same causal state if and only if:
```
P(Future | History₁) = P(Future | History₂)
```

**ε-Machine Properties**:
- Provably minimal sufficient statistics for prediction
- Markovian dynamics (future depends only on current causal state)
- Optimal information compression for prediction

### Key Advantages of Neural CSSR

1. **Sample Efficiency**: O(N) vs O(k^L) scaling through neural generalization
2. **Pattern Recognition**: Handles similar predictive patterns across different machines
3. **Theoretical Guarantees**: Preserves optimality properties of classical CSSR
4. **Scalability**: Works with limited data where classical methods fail

## Advanced Features

### Non-Topological Machines

Support for ε-machines with custom transition probabilities:

```yaml
machine_specs:
  - complexity_class: "2-state-binary"
    custom_probabilities:
      "S0": {"0": 0.8, "1": 0.2}  # 80%/20% bias
      "S1": {"0": 0.3, "1": 0.7}  # 30%/70% bias
```

### Quality Assurance

Automated validation includes:
- **Coverage Analysis**: State and transition coverage verification
- **Distribution Validation**: Statistical consistency checks
- **Length Diversity**: Sequence length distribution analysis
- **Information Content**: Entropy and complexity validation

### Baseline Computation

Automatic computation of baseline metrics:
- Random prediction baselines
- N-gram model performance
- Theoretical optimal bounds
- Classical CSSR oracle performance

## Contributing

### Development Setup

```bash
git clone https://github.com/your-username/NeuralCSSR.git
cd NeuralCSSR
uv sync --dev
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_classical_cssr.py
pytest tests/test_dataset_generation.py
pytest tests/test_neural_components.py
```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

## Documentation

- [**Neural CSSR Summary**](neural_cssr_summary.md): Comprehensive project overview
- [**Dataset Framework**](neural_cssr_dataset_framework.md): Technical implementation details
- [**Classical Analysis Framework**](classical_cssr_analysis_framework.md): Classical CSSR integration
- [**Dataset Metadata Guide**](DATASET_METADATA_GUIDE.md): Metadata specification

## Research Applications

This framework enables research in:

- **Transfer Learning**: Scaling laws between synthetic FSM datasets
- **Scientific Discovery**: Automated discovery of minimal governing equations
- **Information Theory**: Empirical validation of theoretical predictions
- **Computational Mechanics**: Modern approaches to classical problems

## Citation

If you use this software in your research, please cite:

```bibtex
@software{neural_cssr,
  title={Neural CSSR: Bridging Classical Computational Mechanics with Modern Machine Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/NeuralCSSR}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Santa Fe Institute for foundational work on ε-machine enumeration
- Classical CSSR algorithm by Shalizi and Klinkner
- Transformer architecture by Vaswani et al.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/NeuralCSSR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/NeuralCSSR/discussions)
- **Email**: your.email@domain.com

---

**Neural CSSR** - Discovering minimal causal representations through the synthesis of classical computational mechanics and modern machine learning.
