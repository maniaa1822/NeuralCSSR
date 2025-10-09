#!/bin/bash
# Test script for new OOD regimes

set -e

cd /home/matteo/NeuralCSSR
export PYTHONPATH=.

MODELS="--model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt --model eps=nanoGPT/out-seven-state-human-mtl100k-mtl/ckpt.pt"

echo "Running emission_bias tests..."
uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime emission_bias --severity 0.3 --tokens 65536 \
  --output results/emission_bias_0p3.csv

uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime emission_bias --severity 0.5 --tokens 65536 \
  --output results/emission_bias_0p5.csv

uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime emission_bias --severity 0.7 --tokens 65536 \
  --output results/emission_bias_0p7.csv

echo "Running multistep_prediction tests..."
uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime multistep_prediction --k-step 2 --tokens 65536 \
  --output results/multistep_k2.csv

uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime multistep_prediction --k-step 3 --tokens 65536 \
  --output results/multistep_k3.csv

uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime multistep_prediction --k-step 4 --tokens 65536 \
  --output results/multistep_k4.csv

echo "Running mixed_regime tests..."
uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime mixed_regime \
  --mixed-regimes "emission_mix,transition_noise" \
  --mixed-severities "0.3,0.05" \
  --tokens 65536 \
  --output results/mixed_emission0p3_transition0p05.csv

uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime mixed_regime \
  --mixed-regimes "emission_bias,transition_noise" \
  --mixed-severities "0.5,0.1" \
  --tokens 65536 \
  --output results/mixed_bias0p5_transition0p1.csv

uv run --with torch --with numpy python nanoGPT/mtl/eval_ood.py $MODELS \
  --regime mixed_regime \
  --mixed-regimes "start_state_uniform,emission_mix" \
  --mixed-severities "0.0,0.4" \
  --tokens 65536 \
  --output results/mixed_start_uniform_emission0p4.csv

echo "All tests completed successfully!"
