#!/bin/bash
# Batch OOD evaluation for all 3 MTL models

set -e

cd "$(dirname "$0")/../.."

MODELS="--model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
        --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
        --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt"

mkdir -p results/ood

echo "Running OOD evaluations on 3 models..."
echo "Results will be saved to results/ood/"
echo ""

# Emission mix
echo "=== Emission Mix ==="
for sev in 0.2 0.4 0.6; do
  echo "  Severity: $sev"
  uv run python evaluation/scripts/run_ood.py $MODELS \
    --regime emission_mix --severity $sev --tokens 65536 \
    --output results/ood/emission_mix_${sev}.csv
done

# Transition noise
echo ""
echo "=== Transition Noise ==="
for sev in 0.05 0.1 0.15; do
  echo "  Severity: $sev"
  uv run python evaluation/scripts/run_ood.py $MODELS \
    --regime transition_noise --severity $sev --tokens 65536 \
    --output results/ood/transition_noise_${sev}.csv
done

# Start state uniform
echo ""
echo "=== Start State Uniform ==="
uv run python evaluation/scripts/run_ood.py $MODELS \
  --regime start_state_uniform --severity 0.0 --tokens 65536 \
  --output results/ood/start_state_uniform.csv

# Alphabet swap
echo ""
echo "=== Alphabet Swap ==="
uv run python evaluation/scripts/run_ood.py $MODELS \
  --regime alphabet_swap --severity 1.0 --tokens 65536 \
  --output results/ood/alphabet_swap.csv

# Emission bias
echo ""
echo "=== Emission Bias ==="
for sev in 0.3 0.5 0.7; do
  echo "  Severity: $sev"
  uv run python evaluation/scripts/run_ood.py $MODELS \
    --regime emission_bias --severity $sev --tokens 65536 \
    --output results/ood/emission_bias_${sev}.csv
done

echo ""
echo "All OOD evaluations complete!"
echo "Results saved to results/ood/"
