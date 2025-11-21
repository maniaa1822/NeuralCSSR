# Research Plan: AR Models as Epsilon-Machine Learners

This document sketches a concrete plan for tightening the theory and experiments around the central question:

> Do autoregressive (AR) models internalize epsilon / belief-state machines, and can we reconstruct those machines from their predictions?

It references the existing theory and experiment notes:

- `docs/transformers_and_causal_states.md`
- `docs/amortized_epsilon_machines.md`
- `docs/factorized_world_models.md`
- `docs/experiments_golden_mean.md`

---

## 1. Theory: formalizing the hypothesis

Goal: turn the informal story in `transformers_and_causal_states.md` into a set of clear assumptions and convergence statements.

### 1.1 Formal objects and assumptions

- Processes:
  - Single-process case: stationary, ergodic, finite-state, synchronizable epsilon-machine.
  - Mixture case: finite mixture of such processes (belief-state machine exists and is finite).
  - Controlled case: input–output processes over `(O, A)` with a finite controlled epsilon-machine.
- Oracle transformer:
  - Define an ε-accurate oracle: for histories of length ≤ `L_max`, its k-step predictive distributions differ from the true process by at most ε in JS or total variation.
  - Assume `L_max` is at least the relevant synchronization length for the view we care about.

Deliverable:

- A short theory section that names these assumptions explicitly and motivates them with references to classical CSSR / computational mechanics.

### 1.2 Oracle CSSR as a first-class algorithm

Based on `cssr_discovery/2-stage.md` and `two_stage_oracle_cssr.py`:

- Define “oracle CSSR” as CSSR where probability estimates come from an oracle (transformer or ground-truth generator) instead of counts.
- State a convergence claim:
  - If the oracle is ε-accurate and thresholds scale appropriately with ε and sample size, the 2-stage partition converges (in probability) to:
    - the true causal partition (single-process),
    - the belief-state partition (mixture),
    - the controlled epsilon-machine partition (Env/Agent views).

This does not need full proofs but should:

- Make the link to classical CSSR theorems explicit.
- Argue why restricting to fixed `L_max` and using horizon refinement + minimal synchronizing suffixes preserves the key properties.

### 1.3 Env / Agent views as distinct epsilon-machines

Building on `factorized_world_models.md`:

- Formalize three processes:
  - Joint process over `(O, A)` → joint epsilon-machine.
  - Environment process under interventions: `P(O_next | history, do(A=a))` → controlled epsilon-machine.
  - Agent process: `P(A_next | history)` or `P(A_next | S_env, history)` → policy epsilon-machine.
- Show that the Env and Agent equivalence relations used in the experiments are exactly the causal equivalence relations for these processes.

Deliverable:

- A compact subsection in `transformers_and_causal_states.md` (or a new “theory” section) that formally defines these three machines and connects them to the equivalence relations in `factorized_world_models.md`.

### 1.4 Hidden states vs causal states

Formal conjecture:

- Given a process with a finite epsilon-machine and an ε-accurate AR model:
  - There exists a (measurable) function `g` such that `S_t = g(h_t)` almost surely, where `S_t` is the (true or CSSR) causal/belief state and `h_t` is the model’s hidden state.

Empirical proxy:

- Probes `g_φ(h_t) ≈ S_t^CSSR` approximate `g`.
- We can study:
  - probe accuracy,
  - linear vs non-linear `g`,
  - extra information stored in `h_t` beyond `S_t`.

Deliverable:

- Add a short “Representation conjecture” subsection to `transformers_and_causal_states.md` that states this formally and sets up the probing experiments.

---

## 2. Experiments: single-process and mixtures

Goal: validate the hypothesis in clean settings where the ground-truth epsilon-machine is known.

### 2.1 Single-process benchmarks (obs-only)

Targets (see `experiments_golden_mean.md`):

- Golden Mean, Even Process, Seven-State Human, and a couple of additional finite-state synthetic processes.

For each:

1. Generate long sequences from the known machine.
2. Train a small AR transformer (nanoGPT variant) on the sequence.
3. Run oracle CSSR (`two_stage_oracle_cssr.py`) on the trained model.
4. Evaluate:
   - State count and minimal suffix sets vs ground truth.
   - Predictive loss of the recovered machine vs the true machine.
   - Stability vs `L_max`, k-horizons, and tolerance.
5. Probing:
   - Log hidden states `h_t` and CSSR state IDs `S_t^CSSR`.
   - Train probes `g_φ(h_t)` (linear and MLP).
   - Measure classification accuracy and calibration.

6. Top-k expansion study (`branch_topk` in 2-stage CSSR):
   - Run oracle CSSR with different rollout strategies when computing k-step futures:
     - `branch_topk = 1, 2, 4, full` (full = expand all tokens).
   - For each setting, measure:
     - number of recovered states,
     - predictive loss of the recovered machine on held-out data,
     - stability of the “core” state structure across k.
   - Hypothesis: most predictive performance is captured with small `branch_topk`, and a relatively small, stable set of states carries most of the predictive work; full expansion mainly adds low-mass, “tail” structure.

Deliverables:

- Extend `experiments_golden_mean.md` to a generic “single-process epsilon-machine recovery” log.
- One or two plots per process: state count vs `L_max`, ARI/graph metrics vs GT, probe accuracy vs layer.

### 2.2 Mixture and belief-state benchmarks

Design mixtures where the belief-state epsilon-machine can be computed or at least approximated analytically (small mixtures of the single-process machines above).

For each mixture:

1. Generate sequences from the mixture process.
2. Train a transformer on the mixed data.
3. Run oracle CSSR to recover the model’s belief-state machine.
4. Optionally compute the ideal belief-state machine from the known components (as in `amortized_epsilon_machines.md`, section on GT mixtures) and compare.
5. Probing:
   - As above, log `h_t` and CSSR belief-state IDs.
   - Study how hidden-state geometry reflects belief mixtures (e.g., cluster tightness near “pure” beliefs vs spread for ambiguous histories).

6. Top-k expansion in mixtures:
   - Repeat the `branch_topk` sweep for mixture models.
   - Compare how many belief states are needed to achieve a given predictive loss under different top-k expansions.
   - This tests whether, even in mixture/belief settings, a small set of high-mass states explains most of the model’s predictive behavior, with extra states reflecting rare events.

Deliverables:

- A new experiments doc (`docs/experiments_mixtures.md`) describing setups and results.
- Visuals: state diagrams, confusion matrices, and hidden-space embeddings colored by CSSR belief state.

---

## 3. Experiments: agentic traces and factorization

Goal: demonstrate that from a single joint AR model on obs–act traces we can reconstruct:

- the joint epsilon-machine,
- the environment epsilon-machine (world),
- and a hierarchical agent/policy machine.

This is already largely explored in `factorized_world_models.md`; the plan is to polish and systematize.

### 3.1 Simple controlled settings (Q2, Q3)

Reuse and clean up:

- Q2 (controlled vs random regions) → agency map / empowerment metric.
- Q3 (swap env + GoLeft/GoRight agent) → clean Env and Agent factorization.

Tasks:

- Make the pipelines reproducible (scripts under `experiments/`).
- Quantify CSSR recovery quality as in the single-process case.

### 3.2 Gridworld / MiniGrid stress tests (Q4 and beyond)

Extend Q4 to a small suite of grid-like control tasks (possibly moving into MiniGrid to align with RL literature).

For each task:

1. Train a joint AR transformer on `(obs, act, obs, ...)`.
2. Joint view: run standard oracle CSSR on the joint stream to get the closed-loop machine.
3. Env view: recover env states and compare to true topology (ARI/state count).
4. Hierarchical Agent view: local clustering per env state + cross-state mode alignment; evaluate vs known goals/intents.

Deliverables:

- Update `factorized_world_models.md` with cleaned results.
- At least one example where env-machine recovery improves a simple planning or offline RL baseline (world-model value story).

---

## 4. Optional: amortized and meta-level models

Once (1)–(3) are solid, we can revisit `amortized_epsilon_machines.md` and implement:

- Sequence→epsilon-machine models (amortized CSSR).
- Model→epsilon-machine meta-mapping (from trained transformers to their CSSR machines).

These are natural follow-ups but not required for the core thesis/paper.

---

## 5. Paper / thesis structure sketch

1. **Introduction**
   - AR models, predictive state, and causal states.
   - Central question: AR models as epsilon-machine learners.
2. **Background**
   - Epsilon-machines, CSSR, synchronization, belief states.
   - Transformers as predictive oracles (pointer to `transformers_and_causal_states.md`).
3. **Oracle CSSR and Views**
   - Two-stage oracle CSSR (definition and design).
   - Env/Agent views and controlled processes (pointer to `factorized_world_models.md`).
4. **Single-Process and Mixture Experiments**
   - Recovery of known epsilon-machines.
   - Probes from hidden states to CSSR states.
5. **Agentic Traces and Factorization**
   - Joint, Env, and Agent machines from obs–act traces.
   - MiniGrid / gridworld case studies.
6. **Discussion**
   - Map vs territory (true vs model epsilon-machines).
   - When the conjecture holds, failure modes, and future directions (amortized CSSR, larger LMs, richer RL).

This plan ties together the existing docs and code into a coherent path toward both a master’s thesis and a publishable paper. 

---

## 6. Why this is a valuable open question

The central hypothesis here is not an isolated curiosity; it speaks directly to a live question in the field:

> What do autoregressive models actually internalize beyond surface-level statistics?

By grounding the answer in epsilon-/belief-state machines, this framework:

- Provides a **precise, testable formulation**: do AR models converge (in suitable regimes) to the epsilon-machine of their data-generating processes, and can we reconstruct that machine from their predictions?
- Connects to a **mature theory** (computational mechanics), importing notions of minimal sufficient statistics, synchronization, and belief-state machines.
- Offers **mechanistic, falsifiable experiments** (single-process, mixtures, controlled RL traces) rather than high-level probing anecdotes.
- Yields **practical artifacts**—recovered epsilon-machines and factored env/agent models—that can be used for planning, transfer, and interpretability.

Even without new architectures, a careful empirical and theoretical study of AR models as epsilon-machine learners is, on its own, a substantive and timely research contribution. 
