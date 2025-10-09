# Unified Evaluation Framework - Implementation Summary

## What Was Built

A new unified evaluation framework in `evaluation/` that consolidates three previously separate codebases:
1. MTL/OOD evaluation (`nanoGPT/mtl/eval_ood.py`)
2. State probing (`nanoGPT/probes/state_probing/probe_state.py`)
3. Entanglement analysis (`entanglement_analysis/`)

## Directory Structure

```
evaluation/
├── __init__.py
├── README.md                       # Comprehensive user guide
├── IMPLEMENTATION_SUMMARY.md       # This file
├── core/
│   ├── __init__.py
│   ├── models.py                   # Unified model loading (baseline + MTL)
│   ├── data.py                     # IID + OOD data generation (OODGenerator)
│   ├── activations.py              # Layer activation extraction
│   └── metrics.py                  # Shared metric computations
├── probes/
│   ├── __init__.py
│   ├── linear.py                   # Linear probe training
│   ├── mlp.py                      # MLP probe training
│   └── evaluator.py                # Probe comparison (NLI)
├── ood/
│   ├── __init__.py
│   └── evaluation.py               # OOD evaluation logic
├── analysis/
│   └── __init__.py                 # Placeholder for future entanglement metrics
├── runners/
│   └── __init__.py                 # Runner modules
└── scripts/
    ├── run_probes.py               # CLI: probe experiments
    ├── run_ood.py                  # CLI: OOD experiments
    ├── run_all_probes.sh           # Batch: all models, probe eval
    └── run_all_ood.sh              # Batch: all models, all OOD regimes
```

## Key Design Decisions

### 1. No Modifications to Existing Code
- Created fresh `evaluation/` directory
- Original codebases remain untouched
- Can validate new implementation against old outputs

### 2. Extracted Best Implementations
- **Model loading**: From `nanoGPT/mtl/eval_ood.py` (lines 240-339)
- **OOD generation**: From `nanoGPT/mtl/eval_ood.py` (lines 81-204)
- **Activation extraction**: From `nanoGPT/probes/state_probing/probe_state.py` (lines 221-269)
- **Probe training**: From `entanglement_analysis/probes/`

### 3. Unified Interfaces
- All scripts use `machines/` API for ground truth
- Consistent CLI argument naming across scripts
- Single `--model` flag (repeatable) for multi-model comparison
- CSV output format for easy downstream analysis

### 4. Support for Both Model Types
- Baseline GPT: LM metrics only
- MultitaskGPT: LM + epsilon + distance metrics
- Automatic detection of model type from checkpoint
- Auto-extraction of GPT core from MultitaskGPT wrapper

## What Each Component Does

### Core Infrastructure (`evaluation/core/`)

**models.py**:
- `ModelHandle`: Container for loaded model + metadata
- `load_model()`: Loads baseline or MTL checkpoint, auto-detects type
- `compute_state_distances()`: Computes state distance matrix for geometry head
- `parse_model_spec()`: Parses "name=path" model specifications

**data.py**:
- `OODGenerator`: Generates sequences under various OOD regimes
- `generate_iid_data()`: IID sequences matching training distribution
- `generate_ood_data()`: OOD sequences under specified regime
- `build_eval_tensors()`: Converts generated data to torch tensors

**activations.py**:
- `collect_layer_activations()`: Extracts activations from all transformer layers
- Works with both baseline and MTL models (auto-detects GPT core)
- Returns OrderedDict: {layer_name: features [N, D]}

**metrics.py**:
- `compute_lm_metrics()`: Language modeling loss, bits/token, perplexity
- `compute_classification_metrics()`: Classification loss, accuracy
- `compute_distance_metrics()`: Distance prediction metrics

### Probing Infrastructure (`evaluation/probes/`)

**linear.py**:
- `LinearProbeSuite`: Trains linear probes for multiple layers/labels
- `train_linear_probe()`: Train single linear probe with train/val split
- Returns `ProbeMetrics`: loss, accuracy, samples

**mlp.py**:
- `MLPProbeSuite`: Trains MLP probes (1 hidden layer)
- `train_mlp_probe()`: Train single MLP probe
- Same interface as linear probes

**evaluator.py**:
- `ProbeEvaluator.compute_nonlinearity_index()`: NLI = (mlp_acc - linear_acc) / (1 - linear_acc)
- `ProbeEvaluator.summarize_probe_performance()`: Per-layer summaries

### OOD Infrastructure (`evaluation/ood/`)

**evaluation.py**:
- `evaluate_model_ood()`: Evaluates model on OOD data
- Handles both baseline and MTL models
- Returns `OODMetrics`: LM loss, epsilon accuracy, distance accuracy

### CLI Scripts (`evaluation/scripts/`)

**run_probes.py**:
- Trains linear + MLP probes on epsilon states
- Outputs CSV with per-layer results and NLI
- Works with both baseline and MTL models

**run_ood.py**:
- Evaluates models on OOD regimes
- Supports multiple models in single run (for comparison)
- Outputs CSV with all metrics

**run_all_probes.sh**:
- Batch script to evaluate all 3 MTL models
- Creates `results/probes/` directory

**run_all_ood.sh**:
- Batch script to evaluate all 3 MTL models on multiple OOD regimes
- Tests: emission_mix, transition_noise, start_state_uniform, alphabet_swap, emission_bias
- Creates `results/ood/` directory

## Target Models

The framework is designed to evaluate these 3 models (trained via `scripts/train_mtl.sh`):

1. **Baseline**: `nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt`
   - LM-only training (10k iters)
   - No epsilon or distance heads

2. **ε-only**: `out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt`
   - Multitask finetuning with epsilon-state head
   - Taps from layers: block_1, lm_head_input

3. **ε+distance**: `out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt`
   - Multitask finetuning with epsilon + distance heads
   - Taps from layers: block_1, lm_head_input

## Usage Examples

### Single Model Probe Evaluation
```bash
uv run python evaluation/scripts/run_probes.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --tokens 65536 \
  --output results/probes/baseline.csv
```

### Multi-Model OOD Evaluation
```bash
uv run python evaluation/scripts/run_ood.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --regime emission_mix --severity 0.4 --tokens 65536 \
  --output results/ood/emission_mix_0p4.csv
```

### Batch Evaluation
```bash
# Probes on all 3 models
./evaluation/scripts/run_all_probes.sh

# OOD on all 3 models, multiple regimes
./evaluation/scripts/run_all_ood.sh
```

## Output Formats

### Probe Results CSV
Columns: `layer_idx`, `layer_name`, `probe_type`, `hidden_dim`, `loss`, `accuracy`, `num_tokens`, `nli`

Example:
```
0,embedding,linear,,1.184360,0.486115,6554,
0,embedding,mlp,256,1.105158,0.490082,6554,0.007603
1,layer_0,linear,,1.000674,0.555844,6554,
1,layer_0,mlp,256,0.646233,0.700336,6554,0.325042
```

### OOD Results CSV
Columns: `model`, `kind`, `regime`, `severity`, `token_count`, `lm_loss`, `bits_per_token`, `epsilon_loss`, `epsilon_accuracy`, `distance_loss`, `distance_accuracy`

Example:
```
baseline,baseline,emission_mix,0.400000,65536,0.950123,1.369876,,,
eps,mtl,emission_mix,0.400000,65536,0.948234,1.367153,0.234567,0.923456,
epsdist,mtl,emission_mix,0.400000,65536,0.952345,1.373214,0.212345,0.934567,0.123456,0.876543
```

## OOD Regimes Supported

1. **emission_mix** (severity ∈ [0,1]): Mix emissions toward uniform
2. **start_state_uniform**: Random initial state
3. **transition_noise** (severity = probability): Random state transitions
4. **alphabet_swap**: Swap 0/1 emissions
5. **emission_bias** (severity ∈ [0,1]): Push emissions toward extremes
6. **state_dependent_swap** (severity = fraction): Swap emissions for subset of states
7. **temporal_swap** (severity = probability): Per-timestep emission swap
8. **mixed_regime**: Apply multiple regimes (requires --mixed-regimes and --mixed-severities)

## Testing Status

- [x] Core infrastructure implemented
- [x] Probe scripts implemented
- [x] OOD scripts implemented
- [x] Batch scripts created
- [x] Documentation complete
- [ ] **TODO: Run scripts on actual models to validate outputs**
- [ ] **TODO: Compare outputs with old implementations**

## Next Steps (For Validation)

1. **Test probe script**:
   ```bash
   uv run python evaluation/scripts/run_probes.py \
     --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
     --tokens 1000 \
     --output results/probes/test_baseline.csv
   ```

2. **Test OOD script**:
   ```bash
   uv run python evaluation/scripts/run_ood.py \
     --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
     --regime emission_mix --severity 0.4 --tokens 1000 \
     --output results/ood/test_emission_mix.csv
   ```

3. **Compare with old implementation**:
   - Run old `nanoGPT/probes/state_probing/probe_state.py` on same model
   - Compare probe accuracies (should match within numerical precision)
   - Run old `nanoGPT/mtl/eval_ood.py` on same models/regimes
   - Compare LM loss, epsilon accuracy (should match)

4. **Run full batch evaluation**:
   ```bash
   ./evaluation/scripts/run_all_probes.sh
   ./evaluation/scripts/run_all_ood.sh
   ```

## Future Extensions

1. **Unified runner**: Combine probes + OOD + entanglement in single script
2. **Entanglement metrics**: Integrate `entanglement_analysis/` metrics (fragmentation, NLI, FER)
3. **Compositionality**: Add logical composition tests from `entanglement_analysis/probes/composition.py`
4. **Plotting**: Auto-generate comparison plots (probe accuracy curves, OOD robustness)
5. **Report generation**: LaTeX/Markdown reports from CSV results

## Key Advantages

1. **Single source of truth**: No more duplicated model loading, activation extraction, data generation
2. **Consistent API**: Same argument naming across all scripts
3. **Easy comparison**: Multi-model evaluation in single command
4. **Extensible**: Easy to add new OOD regimes, probe types, metrics
5. **Documented**: Comprehensive README and inline documentation
6. **Tested design**: Extracted from proven implementations
7. **No disruption**: Original codebases unchanged, can validate outputs

## Code Reuse Statistics

- **core/models.py**: ~200 lines (from eval_ood.py)
- **core/data.py**: ~250 lines (from eval_ood.py)
- **core/activations.py**: ~80 lines (from probe_state.py)
- **core/metrics.py**: ~80 lines (new consolidation)
- **probes/linear.py**: ~150 lines (from entanglement_analysis)
- **probes/mlp.py**: ~140 lines (from entanglement_analysis)
- **probes/evaluator.py**: ~80 lines (from entanglement_analysis)
- **ood/evaluation.py**: ~150 lines (from eval_ood.py)
- **scripts/run_probes.py**: ~180 lines (replaces probe_state.py)
- **scripts/run_ood.py**: ~250 lines (replaces eval_ood.py)

**Total**: ~1560 lines of clean, unified, well-documented code

**Replaces**: ~2000+ lines spread across 3 separate codebases with overlapping functionality
