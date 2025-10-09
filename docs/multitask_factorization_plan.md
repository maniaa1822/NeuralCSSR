# Multitask Fine-Tuning Plan for nanoGPT Representations

## Objective

Shape nanoGPT's internal representations so that ε-states and primitive factors become linearly accessible and compositionally structured, using only standard multitask cross-entropy losses (no auxiliary "loss hacks").

## Core Idea

Jointly fine-tune nanoGPT on three families of labels:

1. **Next-token prediction** (existing LM loss).
2. **ε-state classification** at selected transformer layers.
3. **Primitive factor classification** (same layers as ε-state).
4. *(Optional extension)* Logical composition labels derived from primitive factors.

Each task supplies a conventional cross-entropy head attached to the latent representation. All heads share gradients through the underlying transformer so multitask supervision shapes the shared latent space.

## Data Requirements

- **Sequences:** Use existing machine-generated traces (e.g., seven_state_human). Keep original token order.
- **Per-token ε-state labels:** Reuse `DataExtractor` logic to map each timestep to its ground-truth ε-state using suffix matching.
- **Primitive labels:** Compute the eight primitive factors per timestep via `PrimitiveFactors`. For primitives defined per state, broadcast the state’s factor value to all positions aligned with that state.
- **Batch format:** Pack into tensors `(input_ids, targets_next_token, targets_epsilon, targets_primitives)` with masking for positions lacking sufficient history.

## Model Modifications (Non-Invasive)

1. **Separate MTL Wrapper**
   - Add a parallel module (e.g., `nanoGPT/mtl/`) that imports the standard `GPT` without editing core files.
   - Register forward hooks inside the wrapper to capture residual streams from target layers (`block_1`, `block_2`, `lm_head_input`). Hooks propagate gradients back into the base model while keeping vanilla code unchanged.

2. **Shared Projections & Heads**
   - Inside the wrapper, define a small shared projection per tapped layer (`nn.Linear(n_embd, bottleneck_dim)`).
   - Attach task-specific linear classifiers on the shared projection:
     - ε-state logits (`bottleneck_dim → n_states`).
     - Primitive logits (categorical factors share head, numerical factors discretized if needed).
     - *(Optional)* Composition head for logical formulas.

3. **MTL Config & Trainer**
   - Create dedicated config files (e.g., `nanoGPT/mtl/config/train_multitask_seven_state.py`) extending the standard hyperparameters with `multitask_layers`, `bottleneck_dim`, `task_weights`, `include_composition`.
   - Implement a separate trainer (`nanoGPT/mtl/train.py`) that mirrors the vanilla training loop but sums the additional cross-entropy losses before backprop.

## Training Loop Changes

- **Loss Computation**
  - Standard next-token CE using existing LM head.
  - For each tapped layer, compute:
    - ε-state CE (mask positions without valid labels).
    - Primitive factor CE (one loss per factor; skip non-varying factors such as `out_degree_class`).
  - Sum weighted losses: `loss_total = w_lm * L_lm + Σ w_eps * L_eps + Σ w_prim * L_prim [+ w_comp * L_comp]`.

- **Gradients**
  - Backprop through transformer and shared projections. No additional regularizers.
  - Use AdamW with learning rate tuned for fine-tuning (start ~1e-4) and per-task gradient clipping if needed.

- **Batching**
  - Reuse data loader infrastructure; ensure aligned labels per token.
  - Optionally oversample underrepresented states to keep primitive class balance.

## Implementation Steps

1. **Model plumbing (wrapper)**
   - Scaffold `nanoGPT/mtl/` with a wrapper that taps hidden states via hooks and hosts the multitask heads.
   - Ensure core `nanoGPT/model.py` stays untouched.

2. **Data pipeline**
   - Build dataset wrapper producing per-token ε-state/primitive labels.
   - Serialize cached datasets for reproducibility.

3. **Training script**
   - Clone the standard trainer into `nanoGPT/mtl/train.py`, add multitask loss aggregation, per-task logging, and checkpointing.

4. **Evaluation**
   - After fine-tuning, run `entanglement_analysis/run_analysis.py` to recompute FER/NLI/fragmentation.
   - Compare against baseline (next-token only) checkpoints.

5. **Ablations**
   - Tasks: LM only vs LM+ε vs LM+ε+primitives. Optionally include composition head.
   - Layers: test different tap points to identify where supervision is most effective.

## Expected Outcomes

- Earlier layers become more linearly separable for ε-states and primitives.
- Reduced NLI gaps and FER scores relative to baseline.
- Improved zero-shot composition accuracy when composition heads are included.
- Minimal degradation (or improvement) in language modeling perplexity if task weights are balanced.

## Open Questions

- Best strategy for numerical primitives (`out_degree_class`)—binary or regression?
- Data efficiency: how many supervised examples are required before multitask benefits plateau?
- Whether to freeze lower transformer blocks for stability during fine-tuning.

## Baseline Checkpoints

- Next-token-only reference: `nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt` (100K-token dataset).
- Multitask (ε-state only): `nanoGPT/mtl/config/train_multitask_seven_state_mtl100k.py`.
- Multitask (ε-state + distance): `nanoGPT/mtl/config/train_multitask_seven_state_mtl100k_eps_dist.py` (loads baseline checkpoint, enables structural distance head).

## Next Actions

1. Prototype hidden-state return path in `nanoGPT/model.py`.
2. Implement dataset generation with per-token ε-state/primitive labels.
3. Stand up a minimal multitask head module and training loop skeleton.
4. Run a small-scale experiment (short sequences, few epochs) to validate plumbing before scaling.

## Next Session: Extend Tasks

- Add primitive heads (categorical first): `entropy_tertile`, `emission_bias_bucket`, `community_id`, `parity_mod2`. Skip `out_degree_class` (constant here).
- Dataset/cache: extend `nanoGPT/mtl/data.py` to store per-token primitive labels aligned to post-token states. Keep class indices consistent with `PrimitiveFactors.get_primitive_info()`.
- Model: attach per-primitive linear heads at the same taps (start with `lm_head_input`, optionally `block_1`), reusing identity projection (no bottleneck). Expose per-task weights (start 0.05–0.1 each) in config.
- Training: keep ε weight at 0.1, primitives small; ensure LM loss remains near baseline. Optionally warm up primitives after a few hundred steps.
- Logging: report per-primitive losses during training; keep LM and ε traces.
- Validation: use `entanglement_analysis/eval_linear_probes.py` to check ε + each primitive; then full analysis with `--n_samples 5000 --n_formulas 200` and regenerate latent plots.
