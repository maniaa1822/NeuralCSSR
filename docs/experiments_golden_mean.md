## Golden Mean Experiments: Data, Training, Probing, and Analysis

### Dataset generation and preparation

1) Generate Golden Mean raw data + states via python-statemachine
```bash
uv run python pysm_generator.py --machine golden_mean --length 80000 \
  --output /home/matteo/NeuralCSSR/notebook_experiments/golden_mean/data --seed 42
```

2) Prepare NanoGPT bins from raw `.dat`
```bash
cd /home/matteo/NeuralCSSR/nanoGPT/data/golden_mean
uv run python prepare.py
```

Sanity of raw data:
- p(0|0)=0.0, p(0|1)≈0.497

### Baseline NanoGPT training (autoregressive only)

Train on corrected bins:
```bash
cd /home/matteo/NeuralCSSR/nanoGPT
uv run python train.py config/train_golden_mean_char.py --device=cuda --compile=True
```

Next-token analysis:
```bash
uv run python analyze_logits.py --out_dir out-golden-mean-char --dataset golden_mean \
  --split val --max_k 16 --num_positions 1000 --stride 1 --device cpu
```
Result (expected Golden Mean):
- NLL≈0.674 bits; p(0|0)=0.000, p(0|1)≈0.498

Prefix-only pre-state linear probe (final layer):
```bash
uv run python probe_linear_state.py --ckpt out-golden-mean-char/ckpt.pt \
  --data_dir data/golden_mean --device cpu --epochs 8 --lr 5e-3 \
  --batch_size 8192 --class_weight --target_state pre --layer final
```
Result:
- Acc≈0.664 (majority baseline) → states not linearly decodable (FER)

### Auxiliary state head training

Config file: `nanoGPT/config/train_golden_mean_state.py` (state_head_classes=2)

Short run (1500 iters, reduced weight):
```bash
uv run python train.py config/train_golden_mean_state.py \
  --device=cuda --compile=True \
  --state_loss_weight=0.1 \
  --out_dir=out-golden-mean-state-aux01 \
  --max_iters=1500 --eval_interval=250
```

Next-token analysis (should remain correct):
```bash
uv run python analyze_logits.py --out_dir out-golden-mean-state-aux01 --dataset golden_mean \
  --split val --max_k 16 --num_positions 1000 --stride 1 --device cpu
```
Observed:
- NLL≈0.675 bits; p(0|0)≈0.001, p(0|1)≈0.50

Prefix-only pre-state probe (final layer):
```bash
uv run python probe_linear_state.py --ckpt out-golden-mean-state-aux01/ckpt.pt \
  --data_dir data/golden_mean --device cpu --epochs 8 --lr 5e-3 \
  --batch_size 8192 --class_weight --target_state pre --layer final
```
Observed:
- Acc≈0.841 (substantial improvement vs baseline 0.664)

### Representation geometry

Per-layer silhouette/inter-intra/PR + PCA (prefix-only features):
```bash
uv run python analyze_rep_geometry.py --ckpt out-golden-mean-char/ckpt.pt \
  --data_dir data/golden_mean \
  --states_dat ../notebook_experiments/golden_mean/data/golden_mean/golden_mean.states.dat \
  --layer all --device cpu --out out-golden-mean-char/rep_geom.json --pca_png out-golden-mean-char/rep_pca.png

uv run python analyze_rep_geometry.py --ckpt out-golden-mean-state-aux01/ckpt.pt \
  --data_dir data/golden_mean \
  --states_dat ../notebook_experiments/golden_mean/data/golden_mean/golden_mean.states.dat \
  --layer all --device cuda --out out-golden-mean-state-aux01/rep_geom.json --pca_png out-golden-mean-state-aux01/rep_pca.png
```

### Key findings

- AR-only model: perfect behavior; pre-state not linearly decodable (≈0.664), geometry consistent with FER.
- With auxiliary state head (0.1): behavior preserved; pre-state linear decodability rises to ≈0.80.

### Notes

- If next-token analysis shows p(0|0)≈0.5 and p(0|1)≈1.0, re-prepare data: `uv run python prepare.py`.
- Use `sanity_logits.py` to double-check mapping and accuracies.



