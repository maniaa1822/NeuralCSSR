# Two-Stage Oracle CSSR

Oracle-driven two-stage CSSR implementation for discovering epsilon-machines from sequences using neural network probability estimates.

## Overview

This implementation uses a two-stage approach to discover causal state representations:

**Stage 1: Oracle Clustering**
- Clusters fixed-length histories using oracle rollouts at increasing prediction horizons
- Compares k-step predictive distributions (JS divergence) to group histories with equivalent futures
- Progressively refines the partition by increasing k until convergence

**Stage 2: Minimal Suffix Discovery**
- Discovers minimal synchronizing suffixes for each state
- These suffixes serve as state labels and enable state identification from arbitrary contexts
- Ensures all states correspond to full-length (L_max) histories

## Key Features

- **No short pre-sync contexts**: All states are defined by histories of length L_max
- **Oracle-driven**: Uses neural network k-step predictions instead of empirical counts
- **Platt calibration**: Optional probability calibration for improved accuracy
- **Epsilon machine loss**: Computes negative log-likelihood under discovered machine

## Dependencies

### Required Python Modules

All dependencies are in the repository:
- `cssr_discovery/calibration.py` - Platt calibration
- `cssr_discovery/js_metrics.py` - JS divergence and distribution utilities
- `transcssr_baseline/transcssr_neural_runner.py` - Model loading utilities
- `nanoGPT/model.py` - Transformer model definitions

### Required Python Packages

Install via `uv sync`:
- `torch` - Neural network framework
- `numpy` - Numerical computing
- `scipy` - Scientific computing (not directly used but may be needed by imports)
- `scikit-learn` - For Platt calibration (Gaussian mixture)

### Data Requirements

**Model checkpoint**: Trained nanoGPT model (default: `nanoGPT/out-seven-state-human-char-large/ckpt.pt`)

**Dataset**: Binary sequence file (default: `experiments/datasets/seven_state_human/seven_state_human.dat`)

## Usage

### Basic Usage

```bash
# Use default preset (seven_state_human_char_large)
uv run python cssr_discovery/two_stage_oracle_cssr.py

# Specify custom model and data
uv run python cssr_discovery/two_stage_oracle_cssr.py \
  --model_ckpt path/to/model.pt \
  --data path/to/data.dat \
  --L_max 5 \
  --tolerance_bits 1e-3
```

### Available Presets

- `seven_state_human_char_large` (default) - Seven-state human machine, large model
- `seven_state_human_100k` - Seven-state human, 100k training steps
- `sevestateold` - Legacy seven-state variant
- `even_process` - Even process (infinite memory)

### Command-Line Arguments

**Data & Model**:
- `--preset` - Preset configuration (default: `seven_state_human_char_large`)
- `--model_ckpt` - Path to nanoGPT checkpoint (overrides preset)
- `--data` - Path to binary .dat file (overrides preset)

**CSSR Parameters**:
- `--L_max` - History length for defining states (default: 5)
- `--metrics_k` - Sequence of k horizons for refining partition (default: `1 2 3`)
- `--fixed_k` - Use single k horizon (overrides --metrics_k)
- `--tolerance_bits` - JS divergence tolerance in bits (default: 1e-3)

**Calibration**:
- `--disable_platt` - Disable Platt calibration (enabled by default)

**Evaluation**:
- `--compute_loss` - Compute epsilon-machine NLL on dataset
- `--output_json` - Path to save JSON summary

### Example Commands

```bash
# Discover states with L=5, tolerance 0.001 bits
uv run python cssr_discovery/two_stage_oracle_cssr.py \
  --L_max 5 \
  --tolerance_bits 1e-3 \
  --metrics_k 1 2 3 4 5

# Use single prediction horizon
uv run python cssr_discovery/two_stage_oracle_cssr.py \
  --fixed_k 3 \
  --L_max 6

# Compute loss and save results
uv run python cssr_discovery/two_stage_oracle_cssr.py \
  --L_max 5 \
  --compute_loss \
  --output_json results/two_stage_discovery.json

# Disable calibration
uv run python cssr_discovery/two_stage_oracle_cssr.py \
  --disable_platt
```

## Output

### Console Output

```
Collected 32768 length-5 histories.
Unique histories: 32

=== Stage 1: oracle clustering ===
States discovered: 7
  State 0: 4 histories (example suffix 00000)
  State 1: 5 histories (example suffix 11100)
  ...

=== Stage 2: synchronizing suffixes ===
  State 0: minimal suffixes ['2:00']
  State 1: minimal suffixes ['3:111']
  ...
```

### JSON Output

When using `--output_json`, the script saves:

```json
{
  "preset": "seven_state_human_char_large",
  "L_max": 5,
  "metrics_k": [1, 2, 3],
  "tolerance_bits": 0.001,
  "num_states": 7,
  "state_sizes": [4, 5, 3, 6, 4, 5, 5],
  "minimal_suffixes": [
    [{"length": 2, "suffix": "00"}],
    [{"length": 3, "suffix": "111"}],
    ...
  ],
  "histories_considered": 32,
  "total_histories": 32768,
  "loss": {
    "total_loss": 12345.67,
    "avg_loss_per_symbol": 0.456,
    "avg_loss_per_symbol_bits": 0.658,
    "num_predictions": 99000,
    "num_states": 7
  }
}
```

## Algorithm Details

### Stage 1: Clustering

1. Extract all length-L_max histories from data
2. For k=1: cluster histories by JS divergence of 1-step predictions
3. For k=2,3,...: refine clusters using k-step predictions
4. Stop when partition doesn't change

### Stage 2: Suffix Discovery

1. For each state, collect all suffixes of member histories
2. Identify suffixes that uniquely identify each state
3. Remove suffixes that are subsumed by shorter unique suffixes
4. Return minimal synchronizing suffix set per state

### Loss Computation

If `--compute_loss` is enabled:
1. Compute emission distribution for each state (from representative)
2. Map each context in dataset to state via minimal suffixes
3. Compute negative log-likelihood: -Σ log P(symbol|state)
4. Report average loss in nats and bits

## Troubleshooting

**"Checkpoint not found"**
- Check that model checkpoint exists at specified path
- Verify preset name matches available configurations
- Use `--model_ckpt` to override preset path

**"No binary tokens in data file"**
- Ensure data file contains only '0' and '1' characters
- Check file encoding (should be plain text)

**"Contexts will be truncated"**
- Warning when L_max > model's block_size
- Model will only see rightmost block_size tokens
- Consider reducing L_max or using larger model

**CUDA out of memory**
- Reduce L_max to decrease number of histories
- Use CPU: the script auto-detects CUDA availability

## Performance Tips

1. **Start small**: Begin with L_max=3-4 to verify behavior
2. **Use fixed_k**: Single horizon is faster than multiple
3. **Increase tolerance**: Larger tolerance_bits finds fewer states
4. **Calibration**: Platt calibration improves discovery accuracy

## Relationship to Classical CSSR

This implementation differs from classical CSSR:
- Uses neural probabilities instead of empirical counts
- Employs k-step predictions for equivalence testing
- All states have full-length (L_max) histories
- No transient states during discovery

## Citation

If using this code, please cite the Neural CSSR paper:
[Citation info to be added]

## See Also

- `unsupervised_fast_original.py` - Alternative CSSR implementation
- `cssr_neural_js.py` - JS-based neural CSSR variant
- `../transcssr_baseline/` - Classical CSSR with neural probabilities
