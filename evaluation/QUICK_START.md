# Quick Start Guide

## TL;DR

```bash
# Probe evaluation (all 3 models)
./evaluation/scripts/run_all_probes.sh

# OOD evaluation (all 3 models, all regimes)
./evaluation/scripts/run_all_ood.sh

# Results:
#   results/probes/*.csv
#   results/ood/*.csv
```

## Single Model Examples

### Probes
```bash
uv run python evaluation/scripts/run_probes.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --output results/probes/baseline.csv
```

### OOD (single model)
```bash
uv run python evaluation/scripts/run_ood.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --regime emission_mix --severity 0.4 \
  --output results/ood/emission_mix_baseline.csv
```

### OOD (compare all 3 models)
```bash
uv run python evaluation/scripts/run_ood.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --regime alphabet_swap --severity 1.0 \
  --output results/ood/alphabet_swap_all.csv
```

## The 3 Models

1. **baseline**: LM-only (no MTL heads)
   - Path: `nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt`
   - Metrics: LM loss, bits/token

2. **eps**: ε-state head only
   - Path: `out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt`
   - Metrics: LM loss, epsilon accuracy

3. **epsdist**: ε-state + distance heads
   - Path: `out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt`
   - Metrics: LM loss, epsilon accuracy, distance accuracy

## OOD Regimes

| Regime | Description | Severity |
|--------|-------------|----------|
| `emission_mix` | Mix toward uniform | 0.0-1.0 |
| `alphabet_swap` | Swap 0/1 emissions | 1.0 |
| `transition_noise` | Random state jumps | 0.0-1.0 (probability) |
| `start_state_uniform` | Random initial state | 0.0 |
| `emission_bias` | Push toward extremes | 0.0-1.0 |

## Common Options

```bash
--tokens 65536        # Number of evaluation tokens
--block-size 64       # Sequence length
--batch-size 64       # Batch size
--seed 42             # Random seed
--device cuda         # Device (auto-detect by default)
```

## Output Format

### Probe CSV
```
layer_idx,layer_name,probe_type,hidden_dim,loss,accuracy,num_tokens,nli
0,embedding,linear,,1.18,0.49,6554,
0,embedding,mlp,256,1.11,0.49,6554,0.01
```

### OOD CSV
```
model,kind,regime,severity,token_count,lm_loss,bits_per_token,epsilon_accuracy,distance_accuracy
baseline,baseline,emission_mix,0.4,65536,0.95,1.37,,
eps,mtl,emission_mix,0.4,65536,0.95,1.37,0.92,
epsdist,mtl,emission_mix,0.4,65536,0.95,1.37,0.93,0.88
```

## See Also

- [README.md](README.md) - Full documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
