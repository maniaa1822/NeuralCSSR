# Entanglement Analysis Suite

Quantifies entanglement and nonlinearity in nanoGPT's latent space by analyzing how well simple linear readouts can recover ε-states and their primitive factors.

## Goal

Check how well simple linear readouts can recover ε-state and its primitive factors—and how much "extra nonlinearity" is needed when linear fails.

## Installation

This package is designed to work within the NeuralCSSR project:

```bash
cd /path/to/NeuralCSSR
uv sync
```

## Quick Start

```python
import numpy as np
from machines import get_machine, generate_sequence_with_states
from entanglement_analysis.analysis_pipeline import EntanglementAnalysisPipeline

# Load machine and model
machine = get_machine("seven_state_human")
pipeline = EntanglementAnalysisPipeline(
    machine=machine,
    model_path="nanoGPT/out-seven-state-human-multiseed-icl/ckpt.pt"
)

# Generate data (convert generated strings to numpy arrays)
sequences = []
for seed in range(50):
    seq_str, _ = generate_sequence_with_states(machine, length=64, seed=seed)
    sequences.append(np.fromiter(seq_str, dtype=int))

# Run analysis
results = pipeline.run_full_analysis(sequences, n_samples=5000, n_formulas=200)

# View FER scores
print("FER Scores:", results['fer_scores'])
```

## Command Line Usage

```bash
# Run full analysis
uv run python entanglement_analysis/run_analysis.py \\
    --model_path nanoGPT/out-seven-state-human-multiseed-icl/ckpt.pt \\
    --machine seven_state_human \\
    --output_dir ./entanglement_results \\
    --n_samples 5000 \\
    --n_formulas 200
```

## Components

### 1. Data Extraction (`data/`)
- **DataExtractor**: Extracts histories, ε-states, and primitive factors
- **PrimitiveFactors**: Computes 8 primitive factors:
  - Entropy tertile
  - Emission-bias bucket
  - Community ID
  - Parity mod-2
  - Mod-3
  - Stationary tertile
  - Out-degree class
  - Self-loop flag

### 2. Feature Extraction (`features/`)
- **FeatureExtractor**: Hooks into nanoGPT to extract activations from all transformer blocks (residual outputs) plus the final layer norm feeding the LM head
- Additional hook points (pre/post attention or MLP) can be enabled by extending `FeatureExtractor` if needed

### 3. Probing Suite (`probes/`)
- **LinearProbeSuite**: Multiclass logistic regression probes with L2 regularization
- **MLPProbeSuite**: Small 1-hidden-layer MLPs (4-16 hidden units)
- **CompositionEvaluator**: Zero-shot and few-shot logical composition tests
- **ProbeEvaluator**: Nonlinearity index and comparative metrics

### 4. Entanglement Metrics (`metrics/`)
- **EntanglementMetrics**: Core entanglement measures
  - ε-state linearity
  - Primitive synergy
  - Fragmentation analysis
  - k-step reachability extrapolation
- **FragmentationAnalyzer**: Cluster-based fragmentation analysis
- **TrajectoryAnalyzer**: Successor alignment and curvature
- **FERCalculator**: Composite FER score aggregation

### 5. Analysis Pipeline (`analysis_pipeline.py`)
- **EntanglementAnalysisPipeline**: Orchestrates the complete analysis
- Handles data preprocessing, feature extraction, probing, and metric computation

## Output Metrics

### FER Score
Composite metric quantifying representation quality:
```
FER = w₁(1 - LinAcc_prims) + w₂ NLI_avg + w₃ (Frag - 1) + w₄ Synergy + w₅ Curvature
```
Lower is better (less fractured/entangled).

### Key Metrics
- **ε-state linearity**: Linear separability of causal states
- **Nonlinearity Index (NLI)**: `Acc_MLP - Acc_Linear` (higher = more entanglement)
- **Composition gap**: Zero-shot vs few-shot logical composition
- **Fragmentation**: How many clusters each ε-state occupies
- **Successor alignment**: Geometric encoding of transitions
- **k-extrapolation**: Reachability generalization

## Reporting

Use the Jupyter notebook for comprehensive visualization:

```bash
cd entanglement_analysis/notebooks
jupyter notebook entanglement_report.ipynb
```

The notebook generates:
- FER scores by layer
- Linear vs MLP accuracy heatmaps
- Few-shot learning curves
- Composition performance
- Fragmentation analysis
- Trajectory geometry plots
- Reachability extrapolation curves

## Acceptance Thresholds

Based on the specification:
- ε-state linear probe ≥ 90% → "good base"
- Primitive linear probes ≥ 80% on ≥4/7 primitives
- Average NLI ≤ 5% for primitives
- Composition zero-shot ≥ 65%, few-shot ≥ 80%
- Fragmentation median ≤ 1.6 clusters/state
- Successor alignment cosine ≥ 0.35

## Interpretation

### FER Score Ranges
- < 0.2: Excellent (well-factorized)
- 0.2-0.4: Good (mostly disentangled)
- 0.4-0.6: Moderate (some entanglement)
- 0.6-0.8: Poor (significant entanglement)
- > 0.8: Very Poor (highly entangled)

### What This Tells You
1. **Layer-wise maps**: Where information becomes linearly accessible
2. **Nonlinearity requirements**: How much extra nonlinearity is needed
3. **Geometric vs tabular**: Whether model encodes graph geometry vs label lookups
4. **Primitive guidance**: How fragmented each ε-state is for MTL targeting

## File Structure

```
entanglement_analysis/
├── __init__.py
├── README.md
├── run_analysis.py              # Command-line interface
├── analysis_pipeline.py         # Main pipeline orchestrator
├── data/                        # Data extraction and primitives
├── features/                    # Feature extraction from nanoGPT
├── probes/                      # Linear/MLP probes and composition
├── metrics/                     # Entanglement metrics and FER score
├── notebooks/                   # Reporting and visualization
└── utils/                       # Utility functions
```

## Dependencies

- torch
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- umap-learn (optional, for 2D projections)
- pandas

All dependencies should be available through the main project's `uv` environment.
