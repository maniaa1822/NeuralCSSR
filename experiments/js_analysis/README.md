# JS Divergence Analysis Results

This directory contains results from the JS divergence analysis script (`nanoGPT/plot_js_vs_L.py`) which analyzes Jensen-Shannon divergence distributions vs history length for neural CSSR models.

## Directory Structure

```
experiments/js_analysis/
├── golden_mean/               # Golden Mean process results
├── seven_state_human/         # Seven-state human machine results
├── even_process/              # Even process results
├── custom_models/             # Custom model results
├── comparative_studies/       # Cross-preset comparisons
├── raw_vs_calibrated/         # Raw vs Platt-calibrated comparisons
└── README.md                  # This file
```

### Analysis Type Subdirectories

Each preset directory contains:

- **random_pairs/**: Random history pair JS divergence analysis
  - Files: `{preset}_L{max}_raw_js_histograms.png`, `{preset}_L{max}_raw_js_thresholds_vs_L.png`
  - Analyzes JS divergence between randomly sampled history pairs

- **instate_analysis/**: Within-state JS divergence analysis
  - Files: `{preset}_L{max}_instate_raw_js_histograms.png`, `{preset}_L{max}_instate_raw_js_thresholds_vs_L.png`
  - Analyzes JS divergence between histories within the same ground truth state

- **cross_state_analysis/**: Between-state JS divergence analysis
  - Files: `{preset}_cross_raw_L{L}_cross_state_mean.png`, `{preset}_cross_raw_L{L}_cross_state_hist.png`
  - Analyzes JS divergence between histories from different ground truth states

- **k_step_analysis/**: k-step rollout JS divergence analysis
  - Files: `{preset}_L{max}_k{k}_raw_js_histograms.png`, `{preset}_L{max}_k{k}_raw_js_thresholds_vs_L.png`
  - Analyzes JS divergence using k-step model rollouts instead of single-step predictions

- **optimal_L_analysis/**: Optimal history length analysis
  - Files: `{preset}_L{max}_optimal_L_raw_optimal_L_analysis.png`
  - Determines optimal history length L where JS thresholds stabilize

## Generated File Types

### PNG Files
- **js_histograms.png**: Histograms of JS divergence values with GMM fitting
- **js_thresholds_vs_L.png**: JS threshold values vs history length L
- **instate_js_histograms.png**: Within-state JS divergence histograms
- **instate_js_thresholds_vs_L.png**: Within-state threshold vs L
- **instate_histories_vs_L.png**: Number of available histories per state vs L
- **L{L}_cross_state_mean.png**: Cross-state mean JS divergence heatmaps
- **L{L}_cross_state_hist.png**: Cross-state JS divergence histograms
- **optimal_L_analysis.png**: Threshold and separation stability for optimal L detection

## Usage Examples

### Basic Random Pairs Analysis
```bash
cd nanoGPT
python plot_js_vs_L.py --preset golden_mean --analysis_type random --L_max 6
```

### Instate Analysis for Seven-State Machine
```bash
cd nanoGPT
python plot_js_vs_L.py --preset seven_state_human --analysis_type instate --L_max 8
```

### Complete Analysis (All Types)
```bash
cd nanoGPT
python plot_js_vs_L.py --preset even_process --analysis_type all --L_max 6 --k 3
```

### Custom Model Analysis
```bash
cd nanoGPT
python plot_js_vs_L.py \
  --model_ckpt /path/to/custom_model.pt \
  --data /path/to/custom_data.dat \
  --analysis_type both --L_max 5
```

## Analysis Types

1. **random**: Random history pairs (works with any preset)
2. **instate**: Within same ground truth state (requires preset with GT state mapping)
3. **cross**: Between different ground truth states (requires preset with GT state mapping)
4. **both**: Both random and instate analysis
5. **all**: Random, instate, and cross-state analysis

## Preset Requirements

- **golden_mean, even_process, seven_state_human**: Support all analysis types
- **Custom presets**: Only support random analysis unless GT state mapping is implemented

## Key Parameters

- `--L_max`: Maximum history length to analyze (default: 4)
- `--n_samples`: Number of sample pairs for analysis (default: 4000)
- `--k`: k-step rollout length (default: 0, disabled)
- `--platt_min_count`: Minimum count for Platt calibration (default: 5)

## Output Organization

Results are automatically saved to subdirectories based on:
- **Preset type**: Determines main directory
- **Analysis type**: Determines subdirectory
- **Calibration**: Raw vs calibrated results use different prefixes
- **Parameters**: L_max, k values included in filenames

## File Naming Convention

```
{preset}_L{L_max}_{calibration}_{analysis_type}_{plot_type}.png

Examples:
- golden_mean_L4_raw_js_histograms.png
- seven_state_human_L6_instate_calibrated_js_thresholds_vs_L.png
- even_process_cross_raw_L4_cross_state_mean.png
```

## Integration with Neural CSSR Pipeline

This analysis helps determine optimal parameters for the Neural CSSR pipeline:
- **Optimal L**: History length where JS thresholds stabilize
- **JS Thresholds**: For state splitting decisions in CSSR
- **State Quality**: Instate analysis validates learned representations
- **Cross-State Discrimination**: Confirms state separability