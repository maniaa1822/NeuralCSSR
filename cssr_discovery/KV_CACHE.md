# KV Cache Implementation for CSSR Discovery

## Overview

The `two_stage_oracle_cssr.py` script now supports **Key-Value (KV) caching** for the nanoGPT model, which significantly reduces redundant computations during the tree search phase of CSSR discovery.

## What is KV Caching?

During the breadth-first search in `precompute_predictions`, the algorithm expands a tree of possible future sequences. Without KV caching, each node expansion requires re-processing the entire history from scratch. With KV caching, the model stores intermediate attention key-value pairs and only processes the new token at each step.

### Performance Impact

- **Theoretical speedup**: Scales linearly with context length `L`. For `L=100`, expect ~100x faster inference per node.
- **Practical speedup**: Depends on model size, hardware, and batch size. Smaller models may see less benefit due to Python overhead.
- **Benchmark results** (L=14, K=10, 100 histories, 0.05M param model):
  - With KV cache: 4.89s
  - Without KV cache: 4.97s
  - Speedup: ~2% (limited by small scale)

## Usage

### Enabling KV Cache (Default)

KV caching is **enabled by default**. Simply run the script as usual:

```bash
uv run cssr_discovery/two_stage_oracle_cssr.py \
  --preset seven_state_human_100k \
  --L_max 14 \
  --metrics_k 10
```

### Disabling KV Cache

To disable KV caching (e.g., for benchmarking or debugging), use the `--no_kv_cache` flag:

```bash
uv run cssr_discovery/two_stage_oracle_cssr.py \
  --preset seven_state_human_100k \
  --L_max 14 \
  --metrics_k 10 \
  --no_kv_cache
```

## Implementation Details

### Modified Files

1. **`nanoGPT/model.py`**:
   - `CausalSelfAttention.forward`: Accepts `past_key_values`, concatenates with current K/V, returns updated cache
   - `Block.forward`: Propagates cache through layers
   - `GPT.forward`: Manages layer-wise cache, returns `new_key_values`
   - `GPT.generate`: Uses cache for autoregressive generation

2. **`cssr_discovery/two_stage_oracle_cssr.py`**:
   - `precompute_predictions`: Stores KV cache state in the BFS queue, passes cache to model for incremental token processing
   - Added `--no_kv_cache` flag for disabling the optimization

### Cache Structure

The cache is a list of `(k, v)` tuples, one per transformer layer:
- `k`: Key tensor of shape `(batch_size, num_heads, seq_len, head_dim)`
- `v`: Value tensor of shape `(batch_size, num_heads, seq_len, head_dim)`

### Memory Considerations

KV caching increases memory usage proportional to:
- Number of active contexts in the search tree
- Context length `L`
- Number of layers and attention heads

For large-scale runs, consider:
- Using `--max_histories` to limit the number of unique histories
- Using `--branch_topk` to prune low-probability branches
- Reducing `--pred_batch_size` if encountering OOM errors

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Reduce batch size: `--pred_batch_size 64` (default: 512)
2. Limit histories: `--max_histories 100`
3. Use branch pruning: `--branch_topk 2`
4. Reduce context length if possible

### Verification

To verify that KV caching produces identical results:

```bash
# Run the integration test
uv run tests/test_kv_cache.py
uv run tests/test_cssr_integration.py
```

Both tests should pass, confirming that the cached implementation is numerically equivalent to the standard forward pass.

## Future Work

- **Rolling cache**: Implement cache eviction for very long contexts (L > block_size)
- **Optimized collation**: Reduce Python overhead in cache batching
- **Larger models**: Test on production-scale models where compute savings dominate overhead
