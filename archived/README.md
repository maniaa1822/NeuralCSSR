# Archived Code

This directory contains code that has been superseded by newer implementations but is preserved for reference.

## Structure

### `pre_evaluation_refactor/`
Contains code that was replaced by the unified evaluation framework in `evaluation/`.

- **entanglement_analysis/**: Original entanglement analysis pipeline
  - Replaced by: `evaluation/` unified framework
  - Reason: Consolidated with MTL/OOD evaluation and probing

- **nanoGPT_mtl_eval_ood.py**: Original OOD evaluation for MTL models
  - Replaced by: `evaluation/scripts/run_ood.py`
  - Reason: Unified OOD evaluation for all model types

- **nanoGPT_probes/**: Original state probing scripts
  - Replaced by: `evaluation/scripts/run_probes.py`
  - Reason: Unified probing infrastructure with consistent API

- **entanglement_plots/**: Plots from old entanglement analysis
  - Replaced by: New plotting in evaluation framework
  - Reason: Outputs from deprecated analysis pipeline

### `legacy_generators/`
Contains old data generation scripts.

- **pysm_generator.py**: Original generator with inline machine definitions
  - Replaced by: `generate_dataset.py` (formerly pysm_generator_v2.py)
  - Reason: New version uses unified `machines/` framework

## Migration Notes

To use the new replacements:

1. **For entanglement/probing analysis**:
   ```bash
   # Old way
   python entanglement_analysis/run_analysis.py ...
   
   # New way
   python evaluation/scripts/run_probes.py ...
   ```

2. **For OOD evaluation**:
   ```bash
   # Old way
   python nanoGPT/mtl/eval_ood.py ...
   
   # New way
   python evaluation/scripts/run_ood.py ...
   ```

3. **For dataset generation**:
   ```bash
   # Old way
   python pysm_generator.py ...
   
   # New way
   python generate_dataset.py ...
   ```

## Recovery

To restore any archived code:
```bash
# Check out the checkpoint tag
git checkout checkpoint-pre-cleanup

# Or check out the backup branch
git checkout backup/pre-cleanup-2024-01-08
```
