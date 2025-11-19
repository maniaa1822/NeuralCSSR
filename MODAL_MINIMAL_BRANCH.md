# Modal minimal branch plan (safe, non-destructive)

Goal: prepare a slim branch that runs the two-stage oracle CSSR (`cssr_discovery/two_stage_oracle_cssr.py`) plus the minimal data-gen + nanoGPT training workflow, without touching the current dirty tree. Commands below copy out a filtered snapshot; they do **not** delete anything here.

Note: `nanoGPT/` is now vendored into this tree (no submodule) with only code/config and prepare scripts; all checkpoints/bins are stripped.

## Keep (core code needed to run two_stage_oracle_cssr.py)
- Root: `pyproject.toml`, `uv.lock`, `README.md`, `generate_dataset.py`, `neural_cssr_utils.py`
- CSSR discovery: `cssr_discovery/two_stage_oracle_cssr.py`, `cssr_discovery/js_metrics.py`, `cssr_discovery/calibration.py`, `cssr_discovery/__init__.py`, `cssr_discovery/2-stage.md`
- Baseline runner stub: `transcssr_baseline/__init__.py`, `transcssr_baseline/transcssr_neural_runner.py`
- Machines: entire `machines/`
- nanoGPT: configs/code only (no heavy data/checkpoints)
  - Keep: `nanoGPT/config/train_seven_state_human_char_100k.py` (guide/config), `nanoGPT/model.py`, `nanoGPT/train.py`, `nanoGPT/sample.py`, `nanoGPT/configurator.py`, `nanoGPT/data/seven_state_human/prepare.py`
  - Drop: any other configs, `.bin`, `.pkl`, `.dat`, `out*`, and `mtl/`
- Docs (optional, small): `docs/pipelines/01-unsupervised-cssr.md`, `cssr_discovery/README_v2.md`

## Drop/exclude (for the slim copy)
- Generated artifacts: `experiments/`, `nanoGPT/out*`, root `out-*`, `runs/`, `results/` (if you want ultra-slim), `complete_prediction_dataset/`
- Large/legacy: `archived/`, `archived_models/`, `analysis/`, `blog/`, `notebooks/`, `old_rules/`, `ood_supervised/`, `data/` dumps, `transCSSR/` (unless classical baseline is required)
- Heavy nanoGPT data: `nanoGPT/archived_runs`, `nanoGPT/data/*/*.bin`, `nanoGPT/data/*/meta.pkl`, `nanoGPT/data/*/*.dat`
- Misc: `__pycache__/`, `*.pyc`, `.venv/` (if present)

## Safe copy recipe (whitelist-first) — from repo root
```bash
mkdir -p ../neuralcssr_modal_min
# 1) Whitelist the essentials
rsync -av \
  --include 'pyproject.toml' \
  --include 'uv.lock' \
  --include 'README.md' \
  --include 'generate_dataset.py' \
  --include 'neural_cssr_utils.py' \
  --include 'cssr_discovery/' \
  --include 'cssr_discovery/***' \
  --include 'transcssr_baseline/' \
  --include 'transcssr_baseline/***' \
  --include 'machines/' \
  --include 'machines/***' \
  --include 'nanoGPT/' \
  --include 'nanoGPT/config/***' \
  --include 'nanoGPT/*.py' \
  --include 'nanoGPT/ckpt_utils.py' \
  --include 'nanoGPT/sample.py' \
  --include 'nanoGPT/trainer.py' \
  --include 'nanoGPT/data/**/prepare.py' \
  --include 'nanoGPT/data/seven_state_human/***' \
  --include 'nanoGPT/data/shakespeare_char/***' \
  --include 'docs/pipelines/01-unsupervised-cssr.md' \
  --include 'cssr_discovery/README_v2.md' \
  --exclude '*' \
  . ../neuralcssr_modal_min

# 2) Defensive excludes to keep the copy lean (in case extra paths slipped through)
rsync -av \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'experiments' \
  --exclude 'data' \
  --exclude 'archived' \
  --exclude 'archived_models' \
  --exclude 'analysis' \
  --exclude 'blog' \
  --exclude 'notebooks' \
  --exclude 'old_rules' \
  --exclude 'ood_supervised' \
  --exclude 'transCSSR' \
  --exclude 'nanoGPT/out*' \
  --exclude 'nanoGPT/archived_runs' \
  --exclude 'nanoGPT/data/*/*.bin' \
  --exclude 'nanoGPT/data/*/meta.pkl' \
  --exclude 'nanoGPT/data/*/*.dat' \
  --exclude 'out-*' \
  --exclude 'runs' \
  --exclude 'results' \
  . ../neuralcssr_modal_min
# (optional) prune other heavy bits after the copy if needed
```

### If you still see heavy folders in the slim copy (expected in the original repo)
Only operate inside the copy (`../neuralcssr_modal_min`), **not** the original tree. From the copy root:
```bash
# double-check size
du -sh .
# remove any lingering training artifacts that slipped through
rm -rf runs results out-* nanoGPT/out* \
       nanoGPT/data/*/*.bin nanoGPT/data/*/meta.pkl nanoGPT/data/*/*.dat
```
If your IDE is showing `runs/` in the original repo, that’s fine—don’t delete it here. Just ensure the copy you push to Modal is clean.

After the copy:
```bash
cd ../neuralcssr_modal_min
git init && git add .
```

## Minimal run workflow on Modal (in the slim copy)
1) Generate data: `uv run python generate_dataset.py --machine seven_state_human --length 100000 --seed 42`
2) Prep bins: `cd nanoGPT && uv run python data/seven_state_human/prepare.py`
3) Train: `uv run --with torch --with numpy python train.py config/train_seven_state_human_char_large.py --device=cuda --out_dir=out-seven-state-human-char-large`
4) Two-stage CSSR:  
   `uv run python cssr_discovery/two_stage_oracle_cssr.py --preset seven_state_human_char_large --model_ckpt nanoGPT/out-seven-state-human-char-large/ckpt.pt --data experiments/datasets/seven_state_human/seven_state_human.dat --L_max 5 --metrics_k 1 2 3 --tolerance_bits 1e-3 --compute_loss --output_json results/two_stage.json`

Adjust `--preset/--data/--model_ckpt` for other machines as needed.
