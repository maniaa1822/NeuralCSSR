# Factorization Probe – Stage A Experiment Log

This document records the concrete implementation and results for the Stage A factorization probe described in `docs/factorization_probe.md`.

---

## 1. Environment + Agent Setup

- Env states: `E = {L, R}`
- Observations: `O = {obs_L, obs_R}` (deterministic: encodes env state)
- Actions: `A = {left, right}`
- Dynamics:
  - If `L`: `left → L`, `right → R`
  - If `R`: `right → R`, `left → L`
- Agent modes: `M = {GoRight, GoLeft}`
  - `GoRight`: `P(right) = 0.9`, `P(left) = 0.1`
  - `GoLeft`: `P(left) = 0.9`, `P(right) = 0.1`
  - Mode switching: w.p. `p_switch = 0.01` per step, flip mode; otherwise keep.

Joint hidden states: `(L,GoRight), (R,GoRight), (L,GoLeft), (R,GoLeft)` encoded as `{0,1,2,3}`.

Implementation: `machines/factorization_probe.py`.

---

## 2. Dataset + Tokenization

Sequence format (per episode):

- Token vocabulary: `{obs_L, obs_R, act_left, act_right}`.
- At each step `t`, emit two tokens: `obs_t, act_t`.
- Episode length: `T = 100` steps → `200` tokens.

Generated splits (`data/factorization_probe/`):

- Episodes:
  - Train: 20,000
  - Val: 2,000
  - Test: 2,000
- Files per split:
  - `*.jsonl`: one episode per line with fields:
    - `tokens` (length 200)
    - `env_states` (length 100, values in `{L,R}`)
    - `agent_modes` (length 100, values in `{GoRight,GoLeft}`)
    - `joint_states` (length 100, values in `{0,1,2,3}`)
    - `obs_tokens`, `act_tokens`
  - `*_per_episode.txt`: space-delimited token sequences, one episode per line.
  - `*.txt`: flattened token streams (single long sequence) for nanoGPT.
  - `metadata.json`: config + optimal-loss estimates.

Optimal next-token loss (from `metadata.json`):

- Action-level conditional entropy (nats): ≈ 0.37
- Per-token next-token CE (obs + act): ≈ 0.185 nats (≈ 0.267 bits)
- Per-token perplexity: ≈ 1.20

---

## 3. nanoGPT Data Prep + Model

Data prep script: `nanoGPT/data/factorization_probe/prepare.py`

- Reads flattened `{train,val,test}.txt` from `data/factorization_probe`.
- Maps tokens to IDs with fixed vocab:
  - `obs_L → 0`, `obs_R → 1`, `act_left → 2`, `act_right → 3`.
- Writes:
  - `train.bin, val.bin, test.bin` (uint16 token streams).
  - `meta.pkl` with `stoi/itos` and `vocab_size=4`.

Model config: `nanoGPT/config/train_factorization_probe.py`

- Dataset: `factorization_probe`
- Context: `block_size = 64` (later, we probed with chunked sequences of length 64)
- Architecture:
  - `n_layer = 6`
  - `n_head = 3`
  - `n_embd = 24`
  - `dropout = 0.0`
- Optimizer / schedule:
  - `batch_size = 64`, `gradient_accumulation_steps = 1`
  - `learning_rate = 3e-3`
  - `max_iters = 8000`, `lr_decay_iters = max_iters`, `min_lr = 3e-4`
  - `warmup_iters = 200`

Training command (via `uv` from `nanoGPT/`):

```bash
uv run python train.py config/train_factorization_probe.py
```

Final language-model performance (from training logs):

- At `iter 8000`: `val lm loss ≈ 0.21` nats/token.
- This is close to the theoretical optimum ≈ 0.185 nats/token.

---

## 4. Feature Extraction for Probing

Extraction script: `evaluation/factorization_probe/probe_extract.py`

Key behaviors:

- Loads checkpoint: `nanoGPT/out-factorization-probe/ckpt.pt`.
- Loads dataset JSONL splits from `data/factorization_probe`.
- Runs the model over each episode, chunked into blocks of length `block_size` (here 64).
- Registers a hook on `transformer.ln_f` to capture final hidden states.
- Supports two labeling modes:
  - Per-step labels (length 100) – not used for final probes.
  - `--label_per_token`: duplicate per-step labels for both obs and act tokens (length 200), used for linear probes.
- Flattens across episodes and saves `.npz` with:
  - `features`: `(N_tokens, d_model)` — here `(400000, 24)` for 2000 episodes.
  - `joint_labels`: `(N_tokens,)` in `{0,1,2,3}`.
  - `env_labels`: `(N_tokens,)` in `{0,1}` (`L=0`, `R=1`).
  - `mode_labels`: `(N_tokens,)` in `{0,1}` (`GoLeft=0`, `GoRight=1`).
  - `tokens`: `(N_tokens,)` token IDs.

Commands used:

- Train subset (first 2000 train episodes):

```bash
PYTHONPATH=. uv run python evaluation/factorization_probe/probe_extract.py \
  --ckpt nanoGPT/out-factorization-probe/ckpt.pt \
  --data data/factorization_probe \
  --meta nanoGPT/data/factorization_probe/meta.pkl \
  --split train \
  --max_episodes 2000 \
  --output nanoGPT/out-factorization-probe/probe_features_train_2000.npz \
  --device cuda \
  --block_size 64 \
  --label_per_token
```

- Full test split:

```bash
PYTHONPATH=. uv run python evaluation/factorization_probe/probe_extract.py \
  --ckpt nanoGPT/out-factorization-probe/ckpt.pt \
  --data data/factorization_probe \
  --meta nanoGPT/data/factorization_probe/meta.pkl \
  --split test \
  --output nanoGPT/out-factorization-probe/probe_features_test.npz \
  --device cuda \
  --block_size 64 \
  --label_per_token
```

Validation check:

- Confirmed that:
  - `features.shape[0] == tokens.shape[0]`
  - Token IDs match the original `test.txt` stream exactly.

---

## 5. Linear Probe Training and Results

Probe training script: `evaluation/factorization_probe/probe_train.py`

- Loads an NPZ with `features`, `joint/env/mode` labels.
- Splits into train/validation (80/20) with `train_test_split`.
- Trains three separate `LogisticRegression` probes (lbfgs, `max_iter=500` or 1000):
  - `joint` (4 classes),
  - `env` (2 classes),
  - `mode` (2 classes).
- Reports train/val accuracy for each probe.

Example command (subset only):

```bash
PYTHONPATH=. uv run python evaluation/factorization_probe/probe_train.py \
  --npz nanoGPT/out-factorization-probe/probe_features_train_2000.npz \
  --test-size 0.2 \
  --seed 0
```

Basic result on this subset:

- `joint`: train ≈ 0.94, val ≈ 0.94
- `env`: train ≈ 0.97, val ≈ 0.97
- `mode`: train ≈ 0.96, val ≈ 0.96

Probe analysis script: `evaluation/factorization_probe/probe_analysis.py`

- Trains joint/env/mode probes on a train NPZ.
- Evaluates on both train and test NPZs.
- Writes metrics + confusion matrices to JSON.

Command:

```bash
PYTHONPATH=. uv run python evaluation/factorization_probe/probe_analysis.py \
  --train-npz nanoGPT/out-factorization-probe/probe_features_train_2000.npz \
  --test-npz nanoGPT/out-factorization-probe/probe_features_test.npz \
  --max-iter 1000
```

Key metrics (from `nanoGPT/out-factorization-probe/probe_metrics.json`):

- Joint state probe:
  - train accuracy: 0.9370
  - test accuracy: 0.9355
- Env state probe:
  - train accuracy: 0.9750
  - test accuracy: 0.9747
- Agent mode probe:
  - train accuracy: 0.9548
  - test accuracy: 0.9536

Confusion matrices for all three probes are stored in `probe_metrics.json`.

---

## 6. Takeaways (Stage A)

- The last-layer representation of this small GPT trained only on next-token prediction:
  - Makes env state linearly decodable at ≈ 97% accuracy.
  - Makes agent mode linearly decodable at ≈ 95% accuracy.
  - Supports high joint-state decodability at ≈ 94%.
- Train/test gaps are small, suggesting the probes are not simply overfitting idiosyncrasies of the train subset.
- Combined with the near-optimal LM performance (~0.21 vs ~0.185 nats/token), this indicates that the model’s predictive representation encodes both env dynamics and agent dynamics in a way that is largely linearly separable.

These results provide a concrete baseline for Stage A of the factorization probe, and set up the next steps:

- Layer-wise probing (how factorization emerges across depth).
- Geometry of env vs mode subspaces (e.g., directions/angles, CCA).
- Robustness to changes in `p_switch`, episode length, or reward structure.

---

## 7. Geometry of Factorization (first cut)

Script: `evaluation/factorization_probe/probe_geometry.py`

Inputs:

- Train features: `nanoGPT/out-factorization-probe/probe_features_train_2000.npz` (2k train episodes, token-aligned labels).
- Test features: `nanoGPT/out-factorization-probe/probe_features_test.npz` (full test split, token-aligned labels).

What it computes:

- Env vs mode probe direction (binary LR): cosine + angle between weight vectors.
- Principal angles between the span of {env, mode} directions and the joint-probe subspace.
- Factorized joint (env×mode) vs monolithic joint accuracy on test, plus KL between their predicted distributions.

Command:

```bash
PYTHONPATH=. uv run python evaluation/factorization_probe/probe_geometry.py \
  --train-npz nanoGPT/out-factorization-probe/probe_features_train_2000.npz \
  --test-npz nanoGPT/out-factorization-probe/probe_features_test.npz \
  --max-iter 1000
```

Key outputs (see `nanoGPT/out-factorization-probe/probe_geometry.json`):

- Env vs mode direction:
  - cos = 0.1765 → angle ≈ 79.8° (nearly orthogonal).
- Principal angles (span(env, mode) vs joint subspace):
  - ≈ [4.87°, 24.84°] — joint probe lives close to the env/mode plane.
- Joint accuracy:
  - Monolithic joint: 0.9355
  - Factorized joint (env×mode logits): 0.9318
- Avg KL(p_joint || p_fact): ≈ 0.0092 nats

Interpretation:

- Env and mode probe directions are almost orthogonal, suggesting a fairly clean factorization in the last-layer representation.
- A factorized joint predictor (independence assumption) is only ~0.36% absolute accuracy below the monolithic joint probe, and the KL is tiny, reinforcing that the joint information is largely captured by combining env + mode signals.
