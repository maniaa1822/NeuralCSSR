# DNA Promoter Experiment - Progress Savepoint
**Date:** 2025-11-21
**Status:** Phase 1 (Learning & Validation) - In Progress

## 1. Accomplishments
*   **Data Pipeline:** 
    *   Implemented `nanoGPT/data/dna_ecoli/prepare.py`.
    *   Successfully downloaded *E. coli* K-12 genome and annotations.
    *   Tokenized data (Char-level: A, C, G, T, N) and created `train.bin`, `val.bin`.
    *   Added `expected_loss` benchmarks to metadata.
*   **Model Configuration:**
    *   Created `nanoGPT/config/train_dna_char.py`.
    *   Settings: 6 layers, 6 heads, 384 embedding, context=512.
    *   Patched `nanoGPT/train.py` to handle extra return values (KV cache) from the model.
*   **Smoke Testing:**
    *   Verified pipeline with a small model (2 layers).
    *   Achieved Loss ~1.37 (better than random 1.39) on smoke test.
*   **Validation Scripts:**
    *   Created `nanoGPT/analysis/dna_validation.py`.
    *   Implemented **Loss Drop Analysis** (Context vs No-Context).
    *   Implemented **Attention Map Visualization** (Focus on -10/-35 boxes).
    *   Fixed import paths to allow running from root.

## 2. Current Status
*   **Full Training:** 
    *   Command: `uv run python train.py config/train_dna_char.py --device=cuda --compile=True --out_dir=out-dna-char`
    *   Status: Running (Last checked: Iter ~150, Loss ~1.36).
    *   Target: Loss < 1.34 (Biological structure).
*   **Validation:**
    *   Script `dna_validation.py` is ready but hasn't been fully run on a mature checkpoint yet.
    *   Attempted to run on smoke checkpoint, but path issues were just resolved.

## 3. Next Steps (Resume Plan)
1.  **Check Training:** Verify if `out-dna-char/ckpt.pt` exists and check the loss in `out-dna-char/log.txt` (if logging enabled) or standard output.
2.  **Run Validation:**
    ```bash
    uv run python nanoGPT/analysis/dna_validation.py nanoGPT/out-dna-char/ckpt.pt
    ```
    *   Check `loss_diff.png`: Is the loss lower with context?
    *   Check `attention_map.png`: Do we see spikes at -10/-35?
3.  **Decision Point:**
    *   **If Validation Passes:** Proceed to Phase 2 (Neural CSSR Extraction).
    *   **If Validation Fails:** Tune hyperparameters (larger model? longer context?) or debug data.

## 4. Key Files
*   `nanoGPT/data/dna_ecoli/prepare.py`
*   `nanoGPT/config/train_dna_char.py`
*   `nanoGPT/analysis/dna_validation.py`
*   `nanoGPT/train.py` (Patched)
