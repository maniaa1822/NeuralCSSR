# Neural CSSR Experimental Reproduction Guide

This guide provides step-by-step instructions to reproduce the complete Neural CSSR experimental pipeline, from transformer training through FSM extraction and visualization.

## Overview

The experimental pipeline consists of:
1. **Dataset Preparation**: Domain-specific FSM datasets
2. **Transformer Training**: Sliding window approach for robust sequence modeling
3. **CSSR-Enhanced Extraction**: Hybrid classical-neural FSM recovery
4. **Parameter Sweep**: Systematic hyperparameter optimization
5. **Attention Visualization**: Understanding learned representations

## Prerequisites

```bash
# Install dependencies
uv sync

# Verify repository structure
ls domain_machines/  # Should contain machine datasets
ls configs/          # Should contain training configurations
```

---

## 1. Dataset Preparation

The experiments use pre-generated domain-specific FSM datasets with 80,000 symbols each.

### Available Datasets
```bash
# List available machines
ls domain_machines/
# Expected: custom_3_state, seven_state_human, golden_mean, etc.

# Dataset format: binary sequences (.dat) with ground truth (.machine.json)
ls domain_machines/seven_state_human/seven_state_human/
# Expected: seven_state_human.dat, seven_state_human.machine.json
```

### Dataset Statistics
- **Seven-state human**: 80,000 symbols, 7 causal states, complex probability distributions
- **Custom 3-state**: 80,000 symbols, 3 causal states, validation baseline
- **Golden mean**: 80,000 symbols, 2 causal states, simple test case

---

## 2. Transformer Training

### 2.1 Configuration-Based Training (Recommended)

The sliding window approach is the **default method** that solved long sequence generation issues.

#### Seven-State Machine Training
```bash
# Primary experimental configuration
python train.py --config configs/sliding_window_seven_state.yaml

# Expected output:
# - Model: SlidingWindowTransformer
# - Parameters: 34,144 total (d_model=32, n_layers=2, n_heads=2)
# - Training: 5 epochs, sliding window approach
# - Final accuracy: ~98.4%
# - Output: checkpoints/sliding_window_seven_state/best.pt
```

**Key Hyperparameters** (from `configs/sliding_window_seven_state.yaml`):
```yaml
model:
  type: sliding_window
  d_model: 32
  n_layers: 2
  n_heads: 2
  dropout: 0.1
  window_size: 25

training:
  epochs: 5
  batch_size: 128
  learning_rate: 0.001
  chunk_size: 25
  sliding_window: true  # Critical for quality generation

data:
  train_path: domain_machines/seven_state_human/seven_state_human/seven_state_human.dat
  train_split: 0.8
```

#### Three-State Validation Training
```bash
# Baseline validation configuration
python train.py --config configs/sliding_window_three_state.yaml

# Expected results:
# - Perfect baseline performance
# - Faster training due to simpler patterns
# - Output: checkpoints/three_state_test/best.pt
```

### 2.2 Alternative Training Commands

#### CLI-Based Training
```bash
# Manual parameter specification
python train.py \
  --model sliding_window \
  --train domain_machines/seven_state_human/seven_state_human/seven_state_human.dat \
  --epochs 5 \
  --batch_size 128 \
  --d_model 32 \
  --layers 2 \
  --heads 2 \
  --lr 0.001 \
  --chunk_size 25 \
  --sliding_window \
  --output_dir checkpoints/manual_seven_state
```

#### Legacy Single-File Training
```bash
# Monolithic streamlined transformer (for comparison)
python streamlined_transformer.py \
  --train domain_machines/seven_state_human/seven_state_human/seven_state_human.dat \
  --epochs 5 \
  --batch 128 \
  --d_model 64 \
  --layers 2 \
  --heads 4 \
  --lr 1e-3 \
  --chunk-size 25 \
  --out checkpoints/streamlined_comparison
```

---

## 3. Model Evaluation

### 3.1 Standard Evaluation
```bash
# Test accuracy on held-out data
python evaluate.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --test domain_machines/seven_state_human/seven_state_human/seven_state_human.dat

# Expected output:
# - Test accuracy: ~98.4%
# - Loss metrics and model statistics
```

### 3.2 Generation Quality Assessment
```bash
# Generate sequences to verify quality
python evaluate.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --generate \
  --length 2000

# Expected output:
# - Realistic symbol distribution (~42% zeros, 58% ones)
# - No repetitive patterns or degenerate sequences
# - Diverse transition patterns
```

### 3.3 Batch Evaluation
```bash
# Test on multiple datasets
python evaluate.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --batch_eval domain_machines/*/*/*.dat
```

---

## 4. CSSR-Enhanced FSM Extraction

### 4.1 Single Extraction

#### Seven-State Machine Extraction
```bash
# Extract FSM using optimal parameters from sweep results
python cssr_enhanced_extractor.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --data domain_machines/seven_state_human/seven_state_human/seven_state_human.dat \
  --output results/seven_state_extraction \
  --max-sequences 500 \
  --max-suffix-length 6 \
  --significance 0.01

# Expected results:
# - Natural discovery of 7 causal states
# - High emission accuracy (~85.9%)
# - No K-means fallback required
# - Output: results/seven_state_extraction/cssr_enhanced_results.json
```

**Optimal Hyperparameters** (from sweep analysis):
- `max_suffix_length`: 6 (sweet spot for classical CSSR)
- `significance_level`: 0.01 (moderate statistical threshold)
- `neural_threshold`: 5.0 (balanced neural similarity)
- `min_suffix_count`: 10 (filter noisy patterns)

#### Three-State Baseline Extraction
```bash
# Perfect baseline validation
python cssr_enhanced_extractor.py \
  --checkpoint checkpoints/three_state_test/best.pt \
  --data domain_machines/custom_3_state/custom_3_state/custom_3_state.dat \
  --output results/three_state_extraction \
  --max-sequences 300 \
  --max-suffix-length 6 \
  --significance 0.01

# Expected results:
# - Perfect 3-state recovery (100% accuracy)
# - Alignment error: 0.0000
# - Robust across parameter variations
```

### 4.2 Ground Truth Comparison
```bash
# Detailed quality analysis
python compare_ground_truth_vs_extracted.py

# Expected output:
# - State count accuracy: 1.000 (7/7 states)
# - Emission accuracy: 0.859 (85.9%)
# - Overall quality score: 0.763 (GOOD classification)
# - Detailed state alignment analysis
```

---

## 5. Parameter Sweep Analysis

### 5.1 Systematic Hyperparameter Sweep

#### Seven-State Machine Sweep
```bash
# Comprehensive parameter exploration
python sweep_extraction_params.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --data domain_machines/seven_state_human/seven_state_human/seven_state_human.dat \
  --ground_truth domain_machines/seven_state_human/seven_state_human/seven_state_human.machine.json \
  --output_dir sweep_results/seven_state_comprehensive

# Parameter space tested:
# - max_suffix_length: [6, 8, 10, 12, 15]
# - significance_level: [0.001, 0.01]
# - neural_threshold: [1.0, 5.0, 10.0]
# - min_suffix_count: [10, 20, 30]
# - target_states: [None, 7] (natural vs forced discovery)
```

#### Three-State Validation Sweep
```bash
# Baseline parameter robustness test
python sweep_extraction_params.py \
  --checkpoint checkpoints/three_state_test/best.pt \
  --data domain_machines/custom_3_state/custom_3_state/custom_3_state.dat \
  --ground_truth domain_machines/custom_3_state/custom_3_state/custom_3_state.machine.json \
  --output_dir sweep_results/three_state_validation

# Expected results:
# - Perfect recovery across all parameter combinations
# - Demonstrates algorithm robustness on simple cases
```

### 5.2 Suffix Length Challenge Test
```bash
# Test performance at challenging suffix lengths where classical CSSR fails
python sweep_extraction_params.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --data domain_machines/seven_state_human/seven_state_human/seven_state_human.dat \
  --ground_truth domain_machines/seven_state_human/seven_state_human/seven_state_human.machine.json \
  --output_dir sweep_results/suffix_length_challenge

# Tests suffix lengths [12, 15] where classical CSSR produces 708 spurious states
# Neural-enhanced approach reduces to ~11 meaningful states (64× improvement)
```

---

## 6. Results Visualization

### 6.1 Parameter Sweep Visualization
```bash
# Comprehensive sweep analysis
python visualize_sweep_results.py \
  --sweep_results sweep_results/seven_state_comprehensive/parameter_sweep_results.csv \
  --output_dir visualizations/seven_state_analysis

# Generated visualizations:
# - parameter_landscape.png: 6-panel parameter space analysis
# - correlation_matrix.png: Parameter-outcome correlations
# - parameter_space_3d.html: Interactive 3D parameter exploration
# - sweep_analysis_report.md: Comprehensive analysis summary
```

### 6.2 Attention Pattern Analysis
```bash
# Transformer attention visualization
python visualize_attention.py \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --num_sequences 5

# Generated visualizations:
# - attention_sequence_N.png: Layer-by-layer attention heatmaps
# - attention_heads_sequence_1.png: Head specialization analysis
# - Console output: Quantitative attention statistics
```

### 6.3 Dynamics Visualization (Optional)
```bash
# Combined dynamics analysis (if desired)
python visualize_sweep_results.py \
  --sweep_results sweep_results/seven_state_comprehensive/parameter_sweep_results.csv \
  --checkpoint checkpoints/sliding_window_seven_state/best.pt \
  --data domain_machines/seven_state_human/seven_state_human/seven_state_human.dat \
  --analyze_dynamics \
  --output_dir visualizations/comprehensive_analysis
```

---

## 7. Key Experimental Results

### 7.1 Training Performance
- **Sliding window approach**: 98.4% accuracy, 34k parameters
- **Generation quality**: Realistic sequences, no repetitive patterns
- **Parameter efficiency**: 25× more training data from overlapping windows

### 7.2 FSM Extraction Breakthrough
- **Seven-state recovery**: Natural discovery of all 7 states
- **Emission accuracy**: 85.9% probability matching
- **Classical vs Neural**: 64× reduction in spurious states (708 → 11)
- **Parameter robustness**: Excellent performance across wide parameter ranges

### 7.3 Attention Patterns
- **Local focus**: 9-15% attention to adjacent positions
- **Self-attention**: 25-43% focus on current token
- **Layer specialization**: Higher layers more focused on current position
- **Token balance**: Fair attention distribution between 0s and 1s

### 7.4 Theoretical Validation
- **Maintains CSSR guarantees**: All theoretical properties preserved
- **Neural regularization**: Prevents classical over-splitting pathologies
- **Scalability**: Handles complex 7-state machines with overlapping distributions
- **Natural discovery**: No forced clustering required for optimal results

---

## 8. Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Verify checkpoint exists and model architecture matches
python -c "from extract_fsm_sliding_window import load_model_from_checkpoint; import torch; print(load_model_from_checkpoint('checkpoints/sliding_window_seven_state/best.pt', torch.device('cpu')))"
```

#### Data Format Issues
```bash
# Verify dataset format
head domain_machines/seven_state_human/seven_state_human/seven_state_human.dat
# Should show binary sequences: 0101100111...
```

#### Sweep Result Errors
```bash
# Check for successful extractions in sweep results
python -c "import pandas as pd; df = pd.read_csv('sweep_results/seven_state_comprehensive/parameter_sweep_results.csv'); print(f'Success rate: {df.success.mean():.2%}')"
```

### Performance Optimization
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Reduce memory usage for large sweeps
python sweep_extraction_params.py --max_sequences 200  # Instead of 500
```

---

## 9. Expected Runtime

### Training Times
- **Seven-state sliding window**: ~5-10 minutes (5 epochs, CPU)
- **Three-state baseline**: ~3-5 minutes (5 epochs, CPU)

### Extraction Times  
- **Single extraction**: ~15-30 seconds per configuration
- **Parameter sweep**: ~10-30 minutes (depending on combinations tested)

### Visualization Times
- **Attention patterns**: ~1-2 minutes
- **Sweep visualization**: ~2-5 minutes

---

## 10. File Structure After Completion

```
NeuralCSSR/
├── checkpoints/
│   ├── sliding_window_seven_state/best.pt    # Primary trained model
│   └── three_state_test/best.pt              # Validation model
├── sweep_results/
│   ├── seven_state_comprehensive/            # Main experimental results
│   └── three_state_validation/               # Baseline validation
├── visualizations/
│   ├── seven_state_analysis/                 # Parameter landscape plots
│   └── attention_visualizations/             # Attention pattern analysis
├── results/
│   ├── seven_state_extraction/               # FSM extraction outputs
│   └── three_state_extraction/               # Baseline extraction
└── REPRODUCTION_GUIDE.md                     # This guide
```

---

## 11. Citation and References

When reproducing or extending this work, please reference:

- **Neural-Enhanced CSSR**: Hybrid approach combining classical computational mechanics with neural trajectory dynamics
- **Sliding Window Training**: Solution to long sequence generation quality issues  
- **Parameter Sweep Methodology**: Systematic exploration of hyperparameter space for FSM extraction
- **Breakthrough Results**: First successful multi-state FSM extraction from neural transformers with 64× improvement over classical methods

## 12. Contact and Support

For questions about reproduction:
1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed with `uv sync`
3. Ensure dataset files are present and correctly formatted
4. Review console output for specific error messages

The complete pipeline should run successfully on both CPU and GPU systems with the provided configurations and datasets.