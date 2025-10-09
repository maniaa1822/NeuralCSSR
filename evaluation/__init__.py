"""
Unified evaluation framework for MTL/OOD/Probe analysis.

Consolidates three existing codebases:
- MTL/OOD evaluation (nanoGPT/mtl/eval_ood.py)
- State probing (nanoGPT/probes/state_probing/probe_state.py)
- Entanglement analysis (entanglement_analysis/)

Target models:
- Baseline: nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt
- ε-only: out-seven-state-human-mtl100k-mtl-eps/ckpt_final.pt
- ε+distance: out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt
"""

__version__ = "0.1.0"
