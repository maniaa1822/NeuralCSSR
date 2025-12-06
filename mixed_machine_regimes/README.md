# Mixed machine regimes experiments

This folder collects everything related to mixed-generator datasets and the hypothesis that the transformer learns a belief-space epsilon machine over the joint processes.

## Objectives

1. **Reproduce the scenarios from Plan §1** (docs/plans/multi_generator_epsilon_machines.md) inside a single, versioned location.
2. Prototype point 2 of that plan: mixed dataset design and generation tooling.
3. Provide metadata + evaluation hooks so downstream steps (nanoGPT training, CSSR, diagnostics) can reuse the same artifacts.

## Hypothesis: belief-space epsilon machines

Working assumption: when a transformer is trained on a mixture of epsilon machines, the induced epsilon machine approximated by CSSR corresponds to a **belief state** over the latent joint process, i.e., states encode the posterior over underlying machines and their local causal states.

Expected implications:

- The discovered machine should have states that correlate with both the active generator and its internal state.
- Transitions should reflect Bayesian updates on the latent mixture, especially in regimes where multiple machines emit overlapping short histories.

## Scenario coverage (plan §1)

- **Scenario A — union of sequences**
  - Multiple sequences, each drawn from a single machine, concatenated/interleaved into one dataset.
  - Metadata: segment boundaries + machine IDs.
  - Goal: see if CSSR splits states per generator despite no explicit switching.

- **Scenario B — regime switching**
  - Single long sequence with a latent machine ID that changes according to a switching policy (e.g., geometric episode durations or a Markov chain).
  - Metadata: per-timestep machine IDs, optional durations + transition matrix.
  - Goal: evaluate whether CSSR recovers a meta-machine that effectively tracks the hidden regime.

- **Scenario C — structured mixtures (optional)**
  - Factorized or hierarchical processes where generators share substructure.
  - Useful once A/B are stable to probe shared-state behavior.

## Dataset specification (point 2 checklist)

- CLI additions (to be implemented):
  - `generate_dataset.py` gains `--mixed` mode with:
    - `--machines <list>` required (≥2).
    - `--mode {union,switch}` (default TBD).
    - `--switching_policy <json or preset>` if `mode=switch`.
    - `--length` interpreted as total symbols across all machines (with helper flags for per-machine lengths).
  - Output directory structure under `experiments/datasets/<mixed_name>/`:
    - `combined.dat`
    - `combined.machine_ids.dat`
    - `combined.meta.json` (machine list, lengths, switching params, suspicion notes).
    - Optional: `combined.state_ids.dat` (true machine states) when available.
    - Metadata also stores per-machine stats (counts, theoretical entropy/loss, structural info), the expected optimal loss for the overall mixed process (nats + bits), and a belief-space summary over `(machine, state)` pairs including their expected contribution to loss.

- Helper module (planned path `mixed_machine_regimes/mixed_dataset.py`):
  - Functions for constructing union or switching sequences using existing `machines` APIs.
  - Utilities for serializing metadata and cross-validating lengths.

- Validation targets:
  - Unit tests that ensure metadata aligns with the bit sequence (same length, consistent indices).
  - Simple analytical cases (e.g., mixture of two machines with disjoint outputs) where the resulting epsilon machine is known.

## Next steps

1. Flesh out `mixed_dataset.py` with union + switching generators backed by `machines.generate_sequence_with_states`.
2. Extend `generate_dataset.py` CLI to call the mixed generator and write metadata.
3. Add regression tests (e.g., `tests/test_mixed_machine_datasets.py`) to confirm deterministic mixing with fixed seeds.
4. Document run instructions + first experimental setup here once code lands.

Track progress by updating this README and the master plan in `docs/plans/multi_generator_epsilon_machines.md`.
