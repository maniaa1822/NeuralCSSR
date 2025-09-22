---
alwaysApply: True
---

# NanoGPT Pipeline (Default Neural Provider)

This document captures the standard pipeline to prepare data, configure NanoGPT, and train checkpoints for use as the neural probability provider in Neural CSSR.

## 1) Data Preparation

Seven-state human dataset:

```bash
# Ensure dataset exists as a single line of 0/1 characters
ls experiments/datasets/seven_state_human/seven_state_human.dat

# Prepare nanoGPT binary files
uv run --with torch python nanoGPT/data/seven_state_human/prepare.py
# Produces nanoGPT/data/seven_state_human/{train.bin,val.bin}
```

Golden mean dataset (example):

```bash
uv run --with torch python nanoGPT/data/golden_mean/prepare.py
```

## 2) Training Configuration

Use the provided config files:
- nanoGPT/config/train_seven_state_char.py
- nanoGPT/config/train_golden_mean_char.py

Adjust key options inside the config if needed:
- dataset (name)
- out_dir (output directory)
- block_size (context window)
- batch_size, max_iters

## 3) Train NanoGPT

Seven-state human:

```bash
uv run --with torch python nanoGPT/train.py nanoGPT/config/train_seven_state_char.py
```

Golden mean:

```bash
uv run --with torch python nanoGPT/train.py nanoGPT/config/train_golden_mean_char.py
```

Notes:
- Avoid unrecognized CLI overrides; edit the config file instead if necessary.
- The resulting checkpoint is out-<dataset>-char/ckpt.pt.

## 4) Provider Evaluation (No CSSR)

```bash
uv run --with torch python evaluate_neural_cssr_model.py \
  --model_ckpt nanoGPT/out-seven-state-char/ckpt.pt \
  --data experiments/datasets/seven_state_human/seven_state_human.dat \
  --output_dir experiments/results/provider_eval_nanogpt_seven \
  --context_window 32 --L_max 6 --diff_min_count 5 --platt_fit --gt_machine seven_state_human
```

Outputs:
- neural_vs_empirical_probs_nomix.csv including p_gt from the exact seven-state machine
- provider_evaluation_report.json with WMSE metrics and alpha suggestion

## 5) Running transCSSR with NanoGPT

```bash
uv run --with torch python transcssr_neural_runner.py \
  --data experiments/datasets/seven_state_human/seven_state_human.dat \
  --model_ckpt nanoGPT/out-seven-state-char/ckpt.pt \
  --backend neural --L_max 12 --alpha 0.01 \
  --context_window 32 --mix_empirical 0.0 \
  --pseudo_count_scale 90 --gt_machine seven_state_human \
  --platt_fit --test_method G \
  --output_json experiments/results/provider_eval_nanogpt_seven/cssr_neural_pure.json \
  --dump_prob_diff experiments/results/provider_eval_nanogpt_seven/neural_vs_empirical_probs_nomix.csv
```

## 6) Recommended Defaults (Neural-only)
- alpha: 0.01 (tune 0.005–0.02)
- L_max: 10–12
- pseudo_count_scale: 90 (cap on)
- mix_empirical: 0.0
- test_method: G
- platt_fit: on
- context_window: 32

## 7) L-Robustness Check (Optional)
Run CSSR across L ∈ {6,8,10,12} and compare state counts and WMSE-to-GT for stability.
