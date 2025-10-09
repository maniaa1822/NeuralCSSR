#!/bin/bash
# Batch probe evaluation for all 3 MTL models

set -e

cd "$(dirname "$0")/../.."

mkdir -p results/probes

echo "Running probe evaluations on 3 models..."
echo "Results will be saved to results/probes/"
echo ""

# Baseline model
echo "=== Baseline Model ==="
uv run python evaluation/scripts/run_probes.py \
  --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
  --tokens 65536 \
  --output results/probes/baseline_probes.csv

# ε-only model
echo ""
echo "=== ε-only Model ==="
uv run python evaluation/scripts/run_probes.py \
  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \
  --tokens 65536 \
  --output results/probes/eps_probes.csv

# ε+distance model
echo ""
echo "=== ε+distance Model ==="
uv run python evaluation/scripts/run_probes.py \
  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
  --tokens 65536 \
  --output results/probes/epsdist_probes.csv

echo ""
echo "All probe evaluations complete!"
echo "Results saved to results/probes/"
