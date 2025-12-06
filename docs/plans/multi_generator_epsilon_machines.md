# Plan: Multi-generator epsilon-machine discovery

## 1) Problem framing

- Decide which mixed-process scenario to focus on first:
  - A) Union of sequences: dataset = concatenation of sequences, each drawn from one machine (mixture-of-tasks).
  - B) Single long sequence with switching: one sequence that switches between machines over time (regime switching).
  - C) Other structured settings (e.g., joint/emission factorization).
- Clarify what the “epsilon machine” should represent:
  - Only the marginal bit process (ignore which generator produced each bit).
  - Or a meta-machine whose states encode (generator, local state) and possibly the switching dynamics.

**Belief-space hypothesis:** working assumption is that the transformer (and downstream CSSR) learns a belief-space epsilon machine over the joint process—i.e., discovered states encode the posterior over `(generator, generator_state)`. Keep this front-of-mind when defining evaluation metrics.

**Outcome:** a concrete target scenario (A/B/C) and a precise definition of the ground-truth epsilon machine for evaluation, tied to the belief-state interpretation.

---

## 2) Mixed dataset design

See `mixed_machine_regimes/README.md` for the live checklist and artifacts.

- ✅ Added a mixed dataset generator (union + switching) wired into `generate_dataset.py` with:
  - `combined.dat`, `combined.machine_ids.dat`, `combined.state_ids.dat`, `combined.meta.json`.
  - Metadata now includes per-machine stats, belief-machine summary, and a Markov-L baseline loss.
- ✅ Generated concrete datasets:
  - `two_machine_union`, `two_machine_switch`, `gm_seven_union`, `gm_seven_switch` (100k tokens).
- ✅ Verified metadata/segment alignment and expected loss entries via smoke tests.

**Outcome:** reproducible combined datasets with explicit ground-truth metadata (segments, belief-machine stats, markov baseline, state labels) to drive later stages.

---

## 3) nanoGPT training on mixed data

- Add a prep script in `nanoGPT/data/<mixed_name>/prepare.py` that:
  - Reads `combined.dat`.
  - Builds binary `{0,1}` vocab as usual.
  - Produces `train.bin`, `val.bin`, `meta.pkl`.
- Start with **pure bit-level training**:
  - Ignore machine IDs in the first pass; treat data as one unlabeled process.
  - Reuse a small config based on `train_seven_state_human_char_100k.py`:
    - Adjust dataset name and paths.
    - Set `block_size` ≥ max relevant memory across participating machines.
- Train a small nanoGPT model on the mixed dataset:
  - First target: CPU-friendly run to validate plumbing.
  - Then scale to a GPU run for realistic results once everything is wired up.

**Progress:**  
- ✅ Added `nanoGPT/data/gm_seven_switch/prepare.py` and trained `train_gm_seven_switch_char.py`.  
- ✅ Ran GPU training (2000 steps initially, then extended to 10k+) producing `nanoGPT/out-gm-seven-switch-char/ckpt.pt`.  
- ✅ Measured model loss (~0.52 nats) vs metadata expectation (~0.50 nats).

**Outcome:** We have a trained mixed-process nanoGPT checkpoint with logged loss, ready for CSSR/probe experiments.

---

## 4) CSSR on the mixed process

- Treat the mixed data as one unlabeled process:
  - Run `cssr_discovery/two_stage_oracle_cssr.py` on `combined.dat`.
  - Choose `L_max` large enough to cover the most memory-hungry machine.
  - Use sensible defaults for `metrics_k`, `branch_topk`, and JS thresholds; record them in this plan once fixed.
- Baseline / oracle comparisons:
  - Run the same two-stage CSSR pipeline separately on each constituent machine’s own dataset.
  - Optionally, run CSSR on segments of `combined.dat` corresponding to each machine (using metadata) to confirm that “cropped” mixed data behaves like pure single-machine data.

**Progress:**  
- ✅ Ran two-stage CSSR on `gm_seven_switch` at multiple settings:  
  - `L=6`, `k={1,2,3}`, tolerance `1e-3` and `5e-4`;  
  - `k={1..5}`, tolerance loosening (`1e-2`) to inspect merging behavior;  
  - `L=8` run to reduce loss.  
- ✅ Added metadata ingestion + state-id evaluation inside two-stage CSSR, producing JSON summaries with state alignment and timing.  
- ✅ Added `merge_states.py` to cluster CSSR states by JS divergence post-hoc and recompute loss.  
- ✅ Explored state counts/loss tradeoffs (e.g., `39`→`48` states vs. `11` states after loose merges, and `56` states at `L=8` merging to `11` clusters).  
- ✅ Computed empirical Markov-L baselines for reference in metadata + standalone script.

**Outcome:** CSSR outputs + merged-state summaries for the mixed process, with explicit loss numbers and state-alignment diagnostics.

---

## 5) Evaluation and diagnostics

- Construct ground-truth objects:
  - For Mode B (switching), explicitly define a meta epsilon machine whose states are `(generator, generator_state)` plus transitions encoding switching.
  - For Mode A (union), treat the process as a mixture-of-tasks and clarify what “ground truth” means (e.g., union of per-machine states vs. a compressed representation).
- Use metadata to evaluate discovered states:
  - From `combined.machine_ids.dat` and (if available) `get_gt_state` from `machines`, compute:
    - Mutual information between discovered states and `(machine_id, gt_state)`.
    - Confusion matrices: for each discovered state, empirical distribution over generators and gt states.
  - Visual diagnostics:
    - DOT diagram of the discovered machine.
    - DOT diagram (or analytic description) of the constructed meta-machine.
    - Tables/plots summarizing which generators/states are mixed or separated.
- Characterize failure modes:
  - Over-splitting vs. under-splitting.
  - Sensitivity to `L_max`, metrics_k, thresholds.
  - Dependence on dataset length and mixing policy.

- **Progress & Findings:**  
  - ✅ Derived belief-machine metadata (joint states + expected loss) and recorded it in `combined.meta.json`.  
  - ✅ Added ground-truth state alignment to two-stage CSSR outputs (per-state distributions over `(machine,state)`).  
  - ✅ Identified two failure modes:  
    1. **Over-splitting**: suffix-specific duplicates when tolerance/k is strict.  
    2. **Pre-sync states**: extra states that capture “am I synchronized yet?” contexts post-switch.  
  - ✅ Demonstrated linear probes on the model’s hidden states:  
    - Machine ID ≈ 98% accuracy, generator state ≈ 92%, sync status ≈ 87% (even when probing the mixed model).  
  - ✅ Computed Markov-L baselines showing any length-L predictor’s best achievable loss (`L=8` gives 0.5358 nats), providing a faithful lower bound.
  - ✅ Investigated thresholds/k expansions, showing trade-offs in state count vs loss and evidence that the transformer internalizes belief dynamics more finely than the extracted machine.
  - ✅ Tested cross-domain reuse: running CSSR on pure seven-state / Golden Mean datasets using the mixed-model checkpoint results in many more states (32 or 10) and higher loss, confirming the mixed model’s belief state always keeps generator uncertainty when it hasn’t been trained on single-regime data.
  - ✅ Added JS-based state merging (`merge_states.py`) to post-process CSSR outputs, collapsing suffix duplicates for both mixed-regime and single-machine experiments and recomputing merged-machine loss.
  - ✅ Confirmed that even after merging, mixed-model probes on single-machine data retain higher loss and extra belief states; this lines up with the linear probe results showing a persistent latent “which generator” dimension.

**Outcome:** We can now quantify how close CSSR comes to the neural belief space, understand remediations (JS-based merging, higher `L`), and contextualize losses against Markov baselines and neural probes.

---

## 6) Extensions and follow-ups

- **Conditional / multi-task models:**
  - Extend the dataset and nanoGPT training to include machine-ID conditioning:
    - e.g., prepend a special token per sequence, or add an auxiliary head.
  - Re-run CSSR and compare whether per-generator structures become cleaner.
- **Shared structure & factorization:**
  - Examine whether the mixed-process epsilon machine reuses substructure across generators.
  - Explore factorizations where some states correspond to shared predictive contexts.
- **Regime detection:**
  - Use discovered states and histories as features for inferring the active generator over time.
  - Measure accuracy at detecting regime switches using the metadata.

**Outcome:** a roadmap of research directions beyond the initial mixed-process experiment, grounded in the same code path.
