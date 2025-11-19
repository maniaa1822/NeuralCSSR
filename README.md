# Neural CSSR (minimal modal branch)

This branch is trimmed to only what’s needed to:
- generate a binary sequence from a registered machine,
- train a small nanoGPT model on it, and
- run the two-stage oracle CSSR (`cssr_discovery/two_stage_oracle_cssr.py`) to recover causal states.

Everything else (legacy baselines, checkpoints, bulky datasets) has been dropped for a slim Modal deploy.

## Contents
- `generate_dataset.py`, `machines/`: data generation from unified machine specs (e.g., `seven_state_human`).
- `nanoGPT/`: minimal training code + one reference config (`config/train_seven_state_human_char_100k.py`) and the prep script (`data/seven_state_human/prepare.py`).
- `cssr_discovery/`: two-stage oracle CSSR (code + short docs).
- `docs/`: small design/plan notes (kept as text only).
- Root deps: `pyproject.toml`, `uv.lock`, `neural_cssr_utils.py`.

## Quickstart
Prereqs: Python ≥3.8 and [`uv`](https://github.com/astral-sh/uv). Use a GPU for training.

```bash
uv sync
```

### 1) Generate data
```bash
uv run python generate_dataset.py --machine seven_state_human --length 100000 --seed 42
```
Outputs under `experiments/datasets/seven_state_human/` (sequence + metadata).

### 2) Prep nanoGPT bins
```bash
cd nanoGPT
uv run python data/seven_state_human/prepare.py
```

### 3) Train nanoGPT
```bash
uv run --with torch --with numpy python train.py \
  config/train_seven_state_human_char_100k.py \
  --device=cuda \
  --out_dir=out-seven-state-human-char-100k
```

### 4) Two-stage oracle CSSR
From repo root:
```bash
uv run python cssr_discovery/two_stage_oracle_cssr.py \
  --preset seven_state_human_char_large \
  --model_ckpt nanoGPT/out-seven-state-human-char-100k/ckpt.pt \
  --data experiments/datasets/seven_state_human/seven_state_human.dat \
  --L_max 5 \
  --metrics_k 1 2 3 \
  --tolerance_bits 1e-3 \
  --compute_loss \
  --output_json results/two_stage.json
```
Adjust paths/preset as needed.

## Notes
- `nanoGPT/` is vendored (no submodule) and stripped of checkpoints/bins.
- Only the two-stage CSSR code path is kept; classical/unsupervised variants and baselines are removed.
- If you see `results/` or other artifacts in your IDE, ensure they are not carried into the Modal copy.
