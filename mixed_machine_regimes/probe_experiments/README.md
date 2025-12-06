# Probe Experiments for Mixed Machines

This subfolder under `mixed_machine_regimes/` centralizes the ad-hoc probe scripts we use to analyze mixed-machine datasets (e.g., `gm_seven_union`). It contains:

- `analyze_sequence_overlap.py` – counts substrings per machine and reports shared/unique statistics up to a requested length.
- `probe_machine_sync.py` – fits a linear probe on the last hidden state for varying history lengths to recover `machine_id` accuracy and probability.
- `probe_rollout_belief.py` – auto-discovers machine-specific "witness" substrings and measures how often rollouts hit them (behavioral belief proxy).
- `probe_rollout_probe_belief.py` – trains the same probe and tracks the probe-estimated machine belief along a synthetic ambiguous rollout.
- `probe_rollout_probe_accuracy.py` – trains a probe and records its accuracy as we roll out a few steps from histories of varying length.

Each script is documented in `docs/gm_seven_union.md` under the "Belief Collapse, Witnesses, and Mixed Regimes" section, with the commands and outputs. Use `uv run python mixed_machine_regimes/probe_experiments/<script>.py ...` to execute them for new settings.
