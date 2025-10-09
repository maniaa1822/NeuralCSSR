# Archived Models

This directory contains model checkpoints that are preserved for reference or comparison but are not actively used in the current pipeline.

## Contents

### `out-seven-state-human-multitask/`
Multitask learning model checkpoints from MTL experiments on seven-state human machine.

- **Purpose**: Experimental MTL training with epsilon state and distance heads
- **Training**: Used multitask learning objectives
- **Checkpoints**: Contains checkpoints at various training steps (every 500 steps up to 30k)
- **Usage**: These models can be evaluated using the `evaluation/` framework

## Note

These models are kept for:
1. Reproducibility of past experiments
2. Comparison with newer training approaches
3. Historical reference

For current experiments, use models in:
- `nanoGPT/out-*` directories for active training
- `out-seven-state-human-mtl100k-*` for MTL evaluation models
