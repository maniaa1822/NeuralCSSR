## Neural CSSR research pipeline

This guide documents the end-to-end workflow we used to recover epsilon-machines from binary processes using a nanoGPT model as a neural probability provider for CSSR.

### Prerequisites
- Python managed by `uv` (used in examples)
- PyTorch, numpy, tiktoken pulled automatically via `uv run --with ...`

### 1) Datasets
You can use any binary `.dat` sequence, or generate one with the included FSM generator.

- Generate Even Process (50k symbols):
```bash
uv run python pysm_generator.py --machine even_process --length 50000 --output notebook_experiments --seed 42
```
- Existing datasets we used:
  - Golden Mean: `notebook_experiments/golden_mean/data/golden_mean/golden_mean.dat`
  - Complex CSM: `transCSSR/data/complex-csm.dat`

### 2) nanoGPT data preparation (binary char-level)
Each dataset has a prep script that creates `train.bin`, `val.bin`, and `meta.pkl` (vocab={0,1}). The script autodiscovers the input from common locations.

- Golden Mean:
```bash
uv run --with numpy python nanoGPT/data/golden_mean/prepare.py
```
- Complex CSM:
```bash
uv run --with numpy python nanoGPT/data/complex_csm/prepare.py
```
- Even Process:
```bash
uv run --with numpy python nanoGPT/data/even_process/prepare.py
```

### 3) Train nanoGPT
Configs (tiny models):
- `nanoGPT/config/train_golden_mean_char.py`
- `nanoGPT/config/train_complex_csm_char.py`
- `nanoGPT/config/train_even_process_char.py`

GPU (recommended for longer runs):
```bash
cd nanoGPT
uv run --with torch --with numpy python train.py config/train_even_process_char.py \
  --device=cuda --dropout=0.0 --max_iters=10000 --lr_decay_iters=10000 \
  --out_dir=out-even-process-cuda-10k --always_save_checkpoint=True
```

CPU quick tests:
```bash
uv run --with torch --with numpy python train.py config/train_even_process_char.py \
  --device=cpu --compile=False --eval_iters=20 --log_interval=1 \
  --block_size=128 --batch_size=24 --n_layer=4 --n_head=4 --n_embd=128 \
  --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Notes on block_size:
- Golden Mean requires only short memory; small block_size (e.g., 32) is fine.
- Even Process needs correct 1-run parity; use block_size ≥ 64 (we used 64–256).

### 4) Logit/context-length analysis
Analyze loss vs context length and conditional probabilities; optionally dump per-suffix probabilities for CSSR.
```bash
cd nanoGPT
uv run --with torch --with numpy python analyze_logits.py \
  --out_dir out-even-process-cuda-10k --split val \
  --max_k 128 --num_positions 5000 --stride 5 --device cpu \
  --dump_csv --csv_out out-even-process-cuda-10k/suffix_logits.csv
```
Outputs:
- `logit_analysis.json`, `logit_analysis.png`, optional `suffix_logits.csv`.

Expected entropy rates (bits/char):
- Golden Mean: ≈ 2/3
- Even Process (p(1|E)=p): H_rate = H2(p)/(1+p); for p=0.5 → 2/3

### 5) Neural CSSR with nanoGPT probabilities
Use the trained nanoGPT checkpoint directly as the neural provider via `--nanogpt_out_dir`.

Canonical run (JS metric):
```bash
cd nanoGPT
uv run --with torch python -u ../run_neural_cssr.py \
  --data ../notebook_experiments/even_process/even_process.dat \
  --L_max 6 --alpha 0.05 --context_window 64 \
  --backend neural_js --state_metric js --js_threshold 0.03 --min_count 50 \
  --nanogpt_out_dir out-even-process-cuda-10k \
  --dot_out out-even-process-cuda-10k/machine.dot --json_only
```

Tips:
- If CSSR over-splits, relax `--js_threshold` (e.g., 0.02–0.05) and/or increase `--min_count`.
- JS-H (short horizon) can improve robustness:
```bash
uv run --with torch python -u ../run_neural_cssr.py \
  --data ../notebook_experiments/even_process/even_process.dat \
  --L_max 6 --context_window 64 \
  --backend neural_js --state_metric jsh --horizon 2 --js_threshold 0.02 --min_count 50 \
  --nanogpt_out_dir out-even-process-cuda-10k \
  --dot_out out-even-process-cuda-10k/machine.dot --json_only
```

Render DOT:
```bash
uv run --with graphviz python - <<'PY'
import subprocess; p='nanoGPT/out-even-process-cuda-10k/machine.dot'
subprocess.run(['dot','-Tpng',p,'-o',p.replace('.dot','.png')], check=True)
PY
```

### 6) Findings
- Golden Mean:
  - Loss ≈ 0.666 bits/char; p(0|last=0)≈0, p(0|last=1)≈0.5; CSSR recovers 2 states.
- Even Process:
  - With CUDA 10k iters, loss ≈ 0.4655 nats ≈ 0.671 bits/char (near 2/3 target).
  - p(0|last=1) ≈ 0, p(0|last=0) ≈ 0.5. CSSR merges to 2 states with `js_threshold≈0.03, min_count≥50`.
  - Smaller block_size can smear parity and cause over-splitting; use block_size ≥ 64 and set CSSR `context_window` ≤ block_size.

### 7) Directory landmarks
- nanoGPT outputs:
  - `nanoGPT/out-<dataset>-char/` or `nanoGPT/out-even-process-cuda-10k/`: checkpoints, DOT, analysis artifacts
- Datasets:
  - Generated: `notebook_experiments/<machine>/`
  - Prepared binaries: `nanoGPT/data/<dataset>/`

### 8) Repro: Even Process (concise)
```bash
# 1) generate (50k)
uv run python pysm_generator.py --machine even_process --length 50000 --output notebook_experiments --seed 42
# 2) prepare
uv run --with numpy python nanoGPT/data/even_process/prepare.py
# 3) train (GPU, 10k iters)
cd nanoGPT
uv run --with torch --with numpy python train.py config/train_even_process_char.py \
  --device=cuda --dropout=0.0 --max_iters=10000 --lr_decay_iters=10000 \
  --out_dir=out-even-process-cuda-10k --always_save_checkpoint=True
# 4) analyze
uv run --with torch --with numpy python analyze_logits.py --out_dir out-even-process-cuda-10k --split val \
  --max_k 128 --num_positions 5000 --stride 5 --device cpu
# 5) CSSR + DOT
uv run --with torch python -u ../run_neural_cssr.py --data ../notebook_experiments/even_process/even_process.dat \
  --L_max 6 --context_window 64 --backend neural_js --state_metric js --js_threshold 0.03 --min_count 50 \
  --nanogpt_out_dir out-even-process-cuda-10k --dot_out out-even-process-cuda-10k/machine.dot --json_only
```


