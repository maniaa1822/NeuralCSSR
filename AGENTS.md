# AGENTS — Neural CSSR (minimal modal) guidelines

## Scope

- This file lives at the repo root and applies to the entire tree.
- Subdirectories may define their own `AGENTS.md` files; those override these rules within their directory subtree.


## Environment and dependencies

- Assume Python ≥ 3.8, managed via `uv` as in `README.md`.
- Use only dependencies already present in `pyproject.toml` unless the user explicitly requests new ones.
- Do not introduce new heavyweight ML frameworks alongside PyTorch.
- If you must add a dependency, keep it lightweight and document the change briefly in `README.md`.

## Testing and validation

- Default test command from repo root: `uv run pytest`.
- Changes touching CSSR discovery or nanoGPT internals should keep at least:
  - `tests/test_kv_cache.py`
  - `tests/test_cssr_integration.py`
  passing.
- Tests are expected to run on CPU by default; where code can use CUDA, it should gracefully fall back when a GPU is unavailable.
- Prefer adding or updating tests in `tests/` over relying only on manual scripts.
- When modifying CLI behavior, either add a small regression test or update an example invocation in `README.md` or a relevant doc in `cssr_discovery/` or `docs/`.

## Python style

- Follow PEP 8 and the prevailing style in nearby code.
- Use descriptive variable and function names; avoid single-letter names except for conventional short loops or math.
- Add type hints for new public functions and methods when doing so is straightforward and does not clutter the code.
- Keep functions focused and reasonably small; prefer extracting helpers over deeply nested logic.
- Use exceptions with clear messages rather than silent failure; keep `print`-based logging consistent with the existing style in the file (do not introduce a new logging framework unless explicitly requested).

## Directory-specific guidance

### `cssr_discovery/`

- Treat `two_stage_oracle_cssr.py` as a primary entry point for the two-stage oracle CSSR pipeline; avoid breaking its CLI or JSON outputs without explicit instruction.
- When changing algorithms, thresholds, or metrics (`two_stage_oracle_cssr.py`, `js_metrics.py`, `calibration.py`, etc.), update the most relevant docs:
  - `cssr_discovery/2-stage.md`
  - `cssr_discovery/KV_CACHE.md`
  - `cssr_discovery/README_v2.md`
  so they remain roughly consistent with the implemented behavior.
- Performance-sensitive paths (tree search, JS distances, batching) should favor vectorized NumPy/Torch operations over pure Python loops when practical, but correctness and clarity are more important than micro-optimizations.
- For KV-cache-related changes, keep `cssr_discovery/KV_CACHE.md` accurate and ensure the KV tests continue to validate equivalence with non-cached behavior.

### `nanoGPT/` (vendored)

- Treat `nanoGPT/` as vendored code from upstream nanoGPT.
- Avoid broad refactors, mass reformatting, or cosmetic-only edits in this directory.
- Only modify `nanoGPT/` when necessary for this project (e.g., KV cache hooks, auxiliary heads, small API additions) and keep behavior compatible with existing usage and tests.
- Do not remove or alter `LICENSE` or upstream attribution.
- After touching `nanoGPT/model.py` or generation code, re-run:
  - `tests/test_kv_cache.py`
  - `tests/test_cssr_integration.py`
  to confirm behavior is unchanged where expected.

### `machines/` and `generate_dataset.py`

- New machines should:
  - Subclass `Machine` from `machines/base.py`.
  - Use the `@register_machine` decorator in `machines/__init__.py` style.
  - Implement required properties and `get_gt_state`.
- Preserve the unified registry interface (`get_machine`, `list_machines`) and keep machine definitions small, explicit, and easy to reason about.
- Keep `generate_dataset.py` as the canonical dataset generator:
  - Maintain a simple, clear CLI with explicit flags.
  - Avoid breaking existing flags or changing defaults without explicit instruction.
  - Provide informative error messages on invalid machine names or paths.

### `docs/` and research notes

- Use Markdown (`.md`) for documentation and research notes.
- Avoid adding binary artifacts (figures, notebooks, exports) to `docs/`; prefer text descriptions and small code snippets.
- When core algorithms or their CLI surfaces change, update the most relevant high-level doc (often under `docs/` or `cssr_discovery/`) rather than duplicating details across many files.
- Keep docs concise and explanatory; avoid large code dumps or generated logs in documentation.

## Artifacts and large files

- Do not add large datasets, model checkpoints, or result archives to version control.
- Use `experiments/` and `nanoGPT/out-*` for local artifacts, but treat them as non-versioned outputs unless the user asks otherwise.
- Do not recreate heavy directories from other branches (e.g., notebooks, archived models, full `transCSSR`) in this minimal branch unless explicitly requested.
- If you introduce a long-running script or experiment, keep it behind an explicit CLI entry point and ensure it is not executed as part of the test suite.

## Coordination with `MODAL_MINIMAL_BRANCH.md`

- Respect the intent of `MODAL_MINIMAL_BRANCH.md`: this tree is a slim, deployable subset aimed at data generation, minimal nanoGPT training, and the two-stage oracle CSSR pipeline.
- Any change that materially increases repository size or scope should be deliberate, justified in `MODAL_MINIMAL_BRANCH.md` or `README.md`, and kept as small as feasible.

