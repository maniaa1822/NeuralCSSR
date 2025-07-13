# Neural CSSR Project Structure

This document describes the organized structure of the Neural CSSR repository after cleanup.

## Core Experiment Files

### 📖 Documentation
- `EXPERIMENT_GUIDE.md` - Complete step-by-step experiment reproduction guide
- `PROJECT_STRUCTURE.md` - This file describing repository organization
- `CLAUDE.md` - Project memory and development history
- `pyproject.toml` - Python project configuration

### 🧪 Main Experiment Scripts
- `generate_unified_dataset.py` - Generate synthetic FSM datasets
- `analyze_classical_cssr.py` - Classical CSSR analysis with parameter sweep
- `analyze_machine_distances.py` - Machine distance analysis framework
- `time_delay_transformer.py` - Transformer training for sequence prediction
- `investigate_fsm_learning.py` - Validate FSM learning vs pattern memorization
- `analyze_internal_states.py` - Extract and analyze transformer internal representations
- `extract_epsilon_machine_from_transformer.py` - Convert neural states to epsilon machines
- `extract_fsm_from_transformer.py` - FSM extraction quality analysis
- `simple_fsm_comparison.py` - Compare neural vs classical CSSR approaches

## Directory Structure

```
NeuralCSSR/
├── src/neural_cssr/                   # Core framework source code
│   ├── classical/                     # Classical CSSR implementation
│   ├── config/                        # Configuration system
│   ├── core/                          # Epsilon-machine core classes
│   ├── data/                          # Dataset generation framework
│   ├── enumeration/                   # Machine enumeration system
│   ├── evaluation/                    # Machine distance analysis
│   ├── analysis/                      # Classical CSSR analysis tools
│   └── neural/                        # Neural network implementations
├── datasets/                          # Generated datasets
│   └── biased_exp/                    # Main experiment dataset
│       ├── raw_sequences/             # Text sequences for classical CSSR
│       ├── neural_format/             # PyTorch datasets
│       ├── ground_truth/              # True FSM definitions
│       ├── statistical_analysis/     # Information theory metrics
│       └── quality_reports/           # Dataset validation
├── results/                           # Core experiment results
│   ├── fixed_theoretical_viz/         # Classical CSSR analysis results
│   ├── internal_analysis/             # Transformer internal state analysis
│   ├── epsilon_machine_extraction/    # Extracted epsilon machines
│   ├── neural_vs_classical_fsm_comparison/ # Method comparison
│   ├── fsm_learning_investigation.txt # FSM learning validation
│   └── fsm_learning_investigation.json
├── plans_and_guides/                  # Project documentation
│   ├── README.md                      # Documentation index
│   ├── MACHINE_DISTANCE_INTERPRETATION_GUIDE.md # Distance analysis guide
│   ├── complete_time_delay_neural_cssr_spec.md # Time-delay research spec
│   └── completed/                     # Completed implementation plans
├── tests/                             # Test files
├── old_results/                       # Previous experiment results (archived)
└── old_experiments/                   # Previous experimental scripts (archived)
```

## Experiment Workflow

The main experiment follows this sequence:

1. **Dataset Generation**
   ```bash
   python generate_unified_dataset.py --preset biased --output datasets/biased_exp
   ```

2. **Classical CSSR Analysis**
   ```bash
   python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/classical_analysis --parameter-sweep
   python analyze_machine_distances.py biased_exp --output-dir results/classical_distance_analysis
   ```

3. **Neural Training & Analysis**
   ```bash
   python time_delay_transformer.py --train datasets/biased_exp/neural_format/train_dataset.pt --dev datasets/biased_exp/neural_format/val_dataset.pt --mode ar --epochs 3 --batch 32 --d_model 32 --layers 3 --heads 8 --lr 1e-3
   ```

4. **FSM Learning Investigation**
   ```bash
   python investigate_fsm_learning.py --model checkpoints/best_model.pt --dataset datasets/biased_exp --d_model 32 --layers 3 --heads 8
   ```

5. **Internal State Analysis**
   ```bash
   python analyze_internal_states.py --model checkpoints/best_model.pt --dataset datasets/biased_exp --d_model 32 --layers 3 --heads 8
   ```

6. **Epsilon Machine Extraction**
   ```bash
   python extract_epsilon_machine_from_transformer.py --model checkpoints/best_model.pt --dataset datasets/biased_exp --d_model 32 --layers 3 --heads 8 --layer 2
   ```

7. **Comparison Analysis**
   ```bash
   python simple_fsm_comparison.py
   ```

## Key Results

### Classical CSSR
- **States discovered**: 5 (exact ground truth match)
- **Quality score**: 0.789 (good recovery)
- **Method**: Statistical hypothesis testing

### Neural CSSR (Transformer)
- **Training accuracy**: 99.33% (vs 53.96% marginal baseline)
- **Internal states**: 8 per layer (60% enhancement)
- **Epsilon machine**: Valid FSM structure with symbol emission probabilities
- **Method**: Representation learning + clustering

### Key Findings
- ✅ **Convergent validation**: Both methods independently discover FSM structure
- ✅ **Enhanced representations**: Neural approach learns richer state space
- ✅ **Genuine learning**: Evidence against pattern memorization
- ✅ **Implementation insights**: Shows how neural networks simulate FSMs

## Archive Folders

### `old_results/`
Contains results from previous experimental iterations, including:
- Various classical CSSR analysis attempts
- Neural oracle experiments
- Mixed experimental approaches
- Preliminary distance analyses

### `old_experiments/`
Contains scripts from previous experimental approaches, including:
- Debug and diagnostic scripts
- Alternative config files
- Experimental neural oracle implementations
- Preliminary analysis tools

These archives preserve the development history while keeping the main repository focused on the core validated experiment.

## File Naming Conventions

- `analyze_*.py` - Analysis and evaluation scripts
- `extract_*.py` - Data/structure extraction tools
- `generate_*.py` - Data generation utilities
- `investigate_*.py` - Hypothesis testing and validation tools
- `*_comparison.py` - Comparative analysis tools

## Dependencies

All dependencies are managed through `uv` and specified in `pyproject.toml`. Install with:
```bash
uv sync
```

Key dependencies:
- PyTorch (neural networks)
- NumPy, SciPy (numerical computing)
- Matplotlib, Seaborn (visualization)
- scikit-learn (clustering and analysis)
- NetworkX (graph analysis)
- PyYAML (configuration)

## Getting Started

1. Follow the complete experiment guide in `EXPERIMENT_GUIDE.md`
2. Check `plans_and_guides/README.md` for additional documentation
3. Review `CLAUDE.md` for project development history and context

For questions or issues, refer to the troubleshooting section in `EXPERIMENT_GUIDE.md`.