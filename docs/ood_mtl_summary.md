# OOD Robustness Summary: Baseline vs Multitask (ε, ε+dist)

## TL;DR

- Multitask fine-tuning (ε and ε+distance) yields more factorized, linearly separable latent spaces and improves auxiliary ε/geometry accuracy under OOD.
- LM (next-token) robustness does not improve under the tested OOD regimes. In severe shifts (alphabet swap), baseline has the best LM; in moderate shifts (emission mix, start-state, transition noise) all models are effectively tied on LM.
- If LM robustness is the goal, adapt the emission mapping (LM head) via small post‑hoc calibration or multi‑environment adapters; the factorized backbone alone is not sufficient.

---

## Models Trained

- Baseline LM-only (10k iters total)
  - 5k pretrain seed → `nanoGPT/out-seven-state-human-mtl100k-baseline-pretrain/ckpt.pt`
  - 10k final → `nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt`
- ε-only finetune (5k iters from 5k seed)
  - `out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt`
- ε + distance finetune (5k iters from 5k seed)
  - `out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt`

All models share the same small architecture: `n_layer=2, n_head=4, n_embd=32`, `block_size=64`, `vocab=2`.

---

## How to Reproduce

Prereqs: use the provided wrappers to avoid env issues.

- Baseline 5k pretrain (separate out_dir):
  - `./scripts/train.sh config/train_seven_state_human_mtl100k_baseline.py \`
  - `  --device=cuda --init_from=scratch \`
  - `  --out_dir=out-seven-state-human-mtl100k-baseline-pretrain \`
  - `  --max_iters=5000 --lr_decay_iters=5000 \`
  - `  --dtype=float32 --compile=False`

- Resume baseline to 10k total:
  - `./scripts/train.sh config/train_seven_state_human_mtl100k_baseline.py \`
  - `  --device=cuda --init_from=resume \`
  - `  --max_iters=10000 --lr_decay_iters=10000 \`
  - `  --dtype=float32 --compile=False`

- ε-only finetune from 5k seed (5k iters):
  - `./scripts/train_mtl.sh nanoGPT/mtl/config/train_multitask_seven_state.py \`
  - `  --device=cuda`

- ε+distance finetune from 5k seed (5k iters):
  - `./scripts/train_mtl.sh nanoGPT/mtl/config/train_multitask_seven_state_mtl100k_eps_dist.py \`
  - `  --device=cuda`

Latent space plots:
- Examples created by: `entanglement_analysis/plots/latent_space.py`
  - Baseline vs ε: `entanglement_plots/latent_space_baseline_vs_eps.png`
  - Baseline vs ε+dist: `entanglement_plots/latent_space_baseline_vs_epsdist.png`

OOD evaluation (zero-shot; no ICL):
- Use: `./scripts/eval_ood.sh` (wraps `nanoGPT/mtl/eval_ood.py`)
- Examples:
  - `./scripts/eval_ood.sh --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \`
  - `  --model eps=out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt \`
  - `  --model epsdist=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \`
  - `  --regime emission_mix --severity 0.4 --tokens 65536 --output results/emission_mix_0p4_new.csv`
- Prefix-only scoring: add `--context-tokens K` to treat the first `K` tokens per window as observed context (ignored for LM/aux metrics). Useful for few-shot OOD tests.
- Context-fit remap: `--fit-remap {global,state}` learns a probability remap from the context tokens (`state` variant uses ground-truth ε-state labels per position).

Post-hoc LM head remap (binary vocab):
- Optional flags to remap softmax probs with a 2×2 matrix before computing LM loss.
  - Swap classes: add `--remap-swap` (useful for `alphabet_swap`).
  - Custom matrix: `--remap-matrix a,b,c,d` applies `p' = p @ [[a,b],[c,d]]` with renormalization.
- Example (alphabet swap):
  - `./scripts/eval_ood.sh --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \`
  - `  --regime alphabet_swap --severity 1.0 --tokens 65536 --remap-swap --output results/alphabet_swap_calibrated.csv`

OOD regimes implemented:
- `emission_mix` (severity α): soften emissions toward uniform
- `start_state_uniform`: random initial causal state
- `transition_noise` (p): random state jumps
- `alphabet_swap`: swap 0/1 emissions
- `state_dependent_swap` (fraction): swap emissions for a subset of states only (breaks global 2×2 remap)
- `temporal_swap` (p): at each timestep, swap emissions with probability p (breaks static remaps; time-varying mapping)

---

## Results (LM-centric)

- Emission mix (α = 0.2/0.4/0.6): baseline, ε, and ε+dist are effectively tied on LM (≈0.88–1.10 bits/token). No LM benefit from MTL.
- Start-state uniform: all tied on LM (≈0.78 bits/token).
- Transition noise (p = 0.05/0.10): all tied on LM (≈0.84–0.89 bits/token).
- Alphabet swap: baseline best LM (≈2.66 bits) < ε (≈2.73) < ε+dist (≈2.82).

Conclusion (LM): For next-token robustness, LM-only baseline is strongest or tied across regimes. Multitask did not reduce LM loss under OOD.

---

## Results (Structure-centric)

- Across regimes, ε+distance has the best ε-state and distance (geometry) accuracy; ε-only is second; baseline has no auxiliary heads.
- Alphabet swap shows the clearest separation: ε+dist maintains much higher ε/geometry accuracy than ε-only, confirming structural robustness.

Conclusion (Structure): Multitask, especially ε+distance, meaningfully improves causal structure readouts under OOD.

---

## Interpretation & Next Steps

Why LM doesn’t improve: the LM head maps hidden states → tokens. When emissions shift (especially permutations), the head is miscalibrated even if the backbone’s state representation is stable.

To get LM robustness from factorized latents:
- Post‑hoc head calibration (tiny 2×2 remap for binary alphabet; frozen backbone).
- Multi‑environment finetuning (sample emission mixes/permutations; shared backbone, small head adapters per environment).
- Test‑time calibration (temperature/BN on logits with unlabeled OOD text).

Reproduce OOD tables:
- See CSVs in `results/` (e.g., `emission_mix_0p4_new.csv`, `alphabet_swap_new.csv`).
