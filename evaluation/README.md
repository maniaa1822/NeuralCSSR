# Unified Evaluation Framework

Consolidated evaluation infrastructure for MTL/OOD/Probe analysis.

## Overview

This framework unifies three previously separate codebases:
1. **MTL/OOD evaluation** (`nanoGPT/mtl/eval_ood.py`)
2. **State probing** (`nanoGPT/probes/state_probing/probe_state.py`)
3. **Entanglement analysis** (`entanglement_analysis/`)

## Target Models

These scripts are designed to evaluate models trained via `scripts/train_mtl.sh`:

- **Baseline**: `nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt`
- **ε-only**: `out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt`
- **ε+distance**: `out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt`

## Quick Start

### 1. Probe Evaluation

Train linear and MLP probes on layer activations to predict epsilon states:

```bash
# Baseline model
uv run python evaluation/scripts/run_probes.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --tokens 65536 \
  --output results/probes/baseline_probes.csv

# ε-only model
uv run python evaluation/scripts/run_probes.py \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --tokens 65536 \
  --output results/probes/eps_probes.csv

# ε+distance model
uv run python evaluation/scripts/run_probes.py \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --tokens 65536 \
  --output results/probes/epsdist_probes.csv
```

**Output**: CSV with columns `layer_idx`, `layer_name`, `probe_type`, `hidden_dim`, `loss`, `accuracy`, `num_tokens`, `nli`

### 2. OOD Evaluation

Evaluate models on out-of-distribution regimes:

```bash
# Emission mix (severity 0.4)
uv run python evaluation/scripts/run_ood.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --regime emission_mix --severity 0.4 --tokens 65536 \
  --output results/ood/emission_mix_0p4.csv

# Alphabet swap
uv run python evaluation/scripts/run_ood.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --regime alphabet_swap --severity 1.0 --tokens 65536 \
  --output results/ood/alphabet_swap.csv

# Transition noise (severity 0.1)
uv run python evaluation/scripts/run_ood.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --regime transition_noise --severity 0.1 --tokens 65536 \
  --output results/ood/transition_noise_0p1.csv
```

**Output**: CSV with columns `model`, `kind`, `regime`, `severity`, `token_count`, `lm_loss`, `bits_per_token`, `epsilon_loss`, `epsilon_accuracy`, `distance_loss`, `distance_accuracy`

### 3. Batch Evaluation

Run multiple OOD regimes at once:

```bash
#!/bin/bash
# evaluation/scripts/run_all_ood.sh

MODELS="--model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
        --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
        --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt"

# Emission mix
for sev in 0.2 0.4 0.6; do
  uv run python evaluation/scripts/run_ood.py $MODELS \
    --regime emission_mix --severity $sev --tokens 65536 \
    --output results/ood/emission_mix_${sev}.csv
done

# Transition noise
for sev in 0.05 0.1 0.15; do
  uv run python evaluation/scripts/run_ood.py $MODELS \
    --regime transition_noise --severity $sev --tokens 65536 \
    --output results/ood/transition_noise_${sev}.csv
done

# Start state uniform
uv run python evaluation/scripts/run_ood.py $MODELS \
  --regime start_state_uniform --severity 0.0 --tokens 65536 \
  --output results/ood/start_state_uniform.csv

# Alphabet swap
uv run python evaluation/scripts/run_ood.py $MODELS \
  --regime alphabet_swap --severity 1.0 --tokens 65536 \
  --output results/ood/alphabet_swap.csv
```

## Package Structure

```
evaluation/
├── core/                   # Shared infrastructure
│   ├── models.py          # Model loading (baseline + MTL)
│   ├── data.py            # IID + OOD data generation
│   ├── activations.py     # Layer activation extraction
│   └── metrics.py         # Shared metrics
├── probes/                # Probing infrastructure
│   ├── linear.py          # Linear probes
│   ├── mlp.py             # MLP probes
│   └── evaluator.py       # Probe comparison (NLI, etc.)
├── ood/                   # OOD evaluation
│   └── evaluation.py      # OOD evaluation logic
├── analysis/              # Placeholder for entanglement metrics
├── runners/               # Runner modules (for imports)
└── scripts/               # CLI scripts
    ├── run_probes.py      # Probe experiments
    └── run_ood.py         # OOD experiments
```

## CLI Options

### run_probes.py

```
--model MODEL           Model spec: name=path/to/ckpt.pt (required)
--dataset DATASET       Dataset name for vocab (default: seven_state_human_mtl100k)
--machine MACHINE       Machine name (default: seven_state_human)
--tokens TOKENS         Number of tokens to sample (default: 65536)
--block-size SIZE       Sequence length per window (default: 64)
--batch-size SIZE       Batch size (default: 64)
--epochs EPOCHS         Training epochs per probe (default: 5)
--learning-rate LR      Learning rate (default: 1e-3)
--mlp-hidden DIM        MLP hidden dimension (default: 256)
--val-fraction FRAC     Validation fraction (default: 0.1)
--device DEVICE         Device: cpu or cuda (default: auto)
--seed SEED             Random seed (default: 42)
--output PATH           Optional CSV output path
```

### run_ood.py

```
--model MODEL                       Model spec (repeatable): name=path/to/ckpt.pt
--regime REGIME                     OOD regime (required, see below)
--severity SEVERITY                 Regime severity parameter (default: 0.0)
--tokens TOKENS                     Number of evaluation tokens (default: 65536)
--block-size SIZE                   Sequence length per window (default: 64)
--batch-size SIZE                   Batch size (default: 64)
--dataset DATASET                   Dataset name for vocab (default: seven_state_human_mtl100k)
--machine MACHINE                   Machine name (default: seven_state_human)
--seed SEED                         Random seed (default: 42)
--device DEVICE                     Device: cpu or cuda (default: auto)
--output PATH                       Optional CSV output path
--mixed-regimes REGIMES             Comma-separated regime names for mixed_regime
--mixed-severities SEVERITIES       Comma-separated severities for mixed_regime
```

### OOD Regimes

- `emission_mix`: Mix emissions toward uniform (severity ∈ [0,1])
- `start_state_uniform`: Random initial state
- `transition_noise`: Random state transitions (severity = probability)
- `alphabet_swap`: Swap 0/1 emissions
- `emission_bias`: Push emissions toward extremes (0.9 or 0.1)
- `state_dependent_swap`: Swap emissions for fraction of states
- `temporal_swap`: Per-timestep emission swap (severity = probability)
- `mixed_regime`: Apply multiple regimes (requires --mixed-regimes and --mixed-severities)

## Key Features

1. **Unified model loading**: Automatically handles baseline GPT and MultitaskGPT checkpoints
2. **Consistent data generation**: Shared OODGenerator for IID and OOD sequences
3. **Layer activation extraction**: Works with both baseline and MTL models (auto-detects GPT core)
4. **Composable metrics**: LM loss, epsilon accuracy, distance accuracy, probe accuracy, NLI
5. **CSV output**: Easy downstream analysis and plotting

## Comparison with Old Code

| Feature | Old Code | New Unified Framework |
|---------|----------|----------------------|
| Model loading | Duplicated in 3 places | `evaluation.core.models` |
| Activation extraction | 2 implementations | `evaluation.core.activations` |
| OOD generation | Only in eval_ood.py | `evaluation.core.data` |
| Probes | Separate scripts | `evaluation.probes` |
| CLI consistency | Different APIs | Unified argument naming |
| Multi-model comparison | Manual | Single `--model` flag (repeatable) |

## Future Extensions

- **Unified runner**: Combine probes + OOD + entanglement in single script
- **Entanglement metrics**: Integrate fragmentation, NLI, FER scores
- **Compositionality**: Add logical composition tests
- **Plotting**: Auto-generate comparison plots
- **Report generation**: LaTeX/Markdown reports from CSV results

## Notes

- All scripts use `machines/` API for ground truth labels
- MTL models automatically extract epsilon/distance heads if available
- Baseline models only report LM metrics (no epsilon/distance)
- NLI (Nonlinearity Index) = (mlp_acc - linear_acc) / (1 - linear_acc)
