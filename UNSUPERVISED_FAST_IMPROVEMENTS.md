# Unsupervised Fast JS Analysis Improvements

This document chronicles all the improvements made to `nanoGPT/js_analysis/unsupervised_fast.py` during the development session.

## Summary of Changes

### 1. Fixed Stage B Assignment Issue
**Problem**: Stage B was only storing representatives in clusters (size=1), not assigning all group histories.

**Solution**: Replaced singleton representative storage with proper emission-based assignment:

```python
# OLD: Only store representatives
final_clusters.append([rep_cluster[0]])

# NEW: Assign all histories to nearest representative
rep_hists = [c[0] for c in rep_clusters]
rep_emits = [get_next_token_distribution(model, h, platt_params) for h in rep_hists]

buckets = [[] for _ in rep_clusters]
for h in group_histories:
    em = get_next_token_distribution(model, h, platt_params)
    js_to_rep = [js_divergence(em, re) for re in rep_emits]
    j = int(np.argmin(js_to_rep))
    buckets[j].append(h)

for bucket in buckets:
    final_clusters.append(bucket)
```

**Result**: Real cluster sizes instead of all size=1, better GT purity statistics.

### 2. Added Missing Preset Support
**Problem**: `seven_state_human_100k` preset was missing from `expected_gt_states` dictionary.

**Solution**: Added preset key to both `unsupervised_fast.py` and `state_mapping.py`:

```python
expected_gt_states = {
    'seven_state_human_100k': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
    # ... other presets
}
```

**Result**: Fixed "Ground truth state coverage: 0.000" issue.

### 3. Implemented Comprehensive JS Diagnostics
**Added Functions**:
- `_pick_histories_for_gt()`: Extract histories for specific GT states
- `conditional_js_detailed()`: Detailed conditional JS with branch analysis
- `emission_js()`: Simple emission probability JS
- `report_state_pair_js()`: Between-state JS analysis
- `report_within_state_js()`: Within-state JS analysis (consistency check)

**Features**:
- **Within-state diagnostics**: Measures how consistently the model treats same-state histories (lower = better)
- **Between-state diagnostics**: Measures how well the model discriminates different states (higher = better)
- **Branch analysis**: Shows JS breakdown by 0-branch vs 1-branch
- **Multi-k analysis**: Averages across multiple k values for robustness

```python
# Example output:
[WITHIN-STATE] bb (n=5 histories), L=4, k_refine=6
  k=1 (emission): JS mean=0.000124  min=0.000089  max=0.000156  n=10
  k=6 (conditional): JS mean=0.000891  min=0.000654  max=0.001203  n=10

[JS-REPORT] bb vs ba (reps 6 x 6), L=4, k_refine=6
  k=1 (emission): JS mean=0.045621  min=0.034567  max=0.056789  n=36
  k=6 (conditional): JS mean=0.078934  min=0.067891  max=0.089765  n=36
```

### 4. Enhanced Multi-K Conditional JS
**Added Functions**:
- `cond_js_two_bit()`: 4-branch conditional JS (more granular than 2-branch)
- `cond_js_multi_k()`: Averages conditional JS across multiple k values
- Enhanced diagnostics to show both single-k and multi-k results

```python
def cond_js_two_bit(cache, model, h1, h2, k, platt_params=None):
    # 4 branches by first two bits: 00, 01, 10, 11
    branches_p = [p[i*m:(i+1)*m] for i in range(4)]
    branches_q = [q[i*m:(i+1)*m] for i in range(4)]
    # ... compute weighted JS across all 4 branches

def cond_js_multi_k(cache, model, h1, h2, ks=(4,5,6,7,8,9,10), platt_params=None):
    vals = [cond_js_two_bit(cache, model, h1, h2, k, platt_params) for k in ks]
    return float(np.mean(vals))
```

### 5. Upgraded Stage B to Use Multi-K Conditional JS
**Problem**: Stage B used single-k conditional JS, making it brittle to horizon choice.

**Solution**: Replaced all single-k calls with multi-k averaging:

```python
# OLD: Single k conditional JS
js_cond = fast_conditional_js_with_cache(cache, model, h1, h2, k_refine, platt_params)

# NEW: Multi-k conditional JS (averaged over 4 horizons)
js_cond = cond_js_multi_k(cache, model, h1, h2,
                         ks=(k_refine, k_refine+1, k_refine+2, k_refine+3),
                         platt_params=platt_params)
```

**Applied to**:
- Representative merging decisions
- History-to-representative assignment
- Evaluation context mapping

**Result**: More robust clustering decisions, less sensitive to single-k noise.

### 6. Enhanced Stage B Assignment with Conditional JS
**Problem**: Assignment used emission JS while clustering used conditional JS (inconsistent).

**Solution**: Made assignment use the same conditional JS as clustering:

```python
# OLD: Emission JS assignment
em = get_next_token_distribution(model, h, platt_params)
js_to_rep = [js_divergence(em, re) for re in rep_emits]

# NEW: Conditional JS assignment (consistent with clustering)
js_to_rep = []
for r in rep_hists:
    js_cond = cond_js_multi_k(cache, model, h, r,
                             ks=(k_refine, k_refine+1, k_refine+2, k_refine+3),
                             platt_params=platt_params)
    js_to_rep.append(js_cond)
```

**Result**: Prevents bb/baab histories from being misassigned to BA-ish buckets.

### 7. Implemented Multi-K K-Medoids Recursive Splitting
**Problem**: Stage B might still miss mixed clusters that need further splitting.

**Solution**: Added principled recursive splitting using k-medoids:

```python
def multi_k_js(cache, model, h1, h2, k_refine, platt_params=None):
    return cond_js_multi_k(cache, model, h1, h2,
                          ks=(k_refine, k_refine+1, k_refine+2, k_refine+3),
                          platt_params=platt_params)

def pairwise_multi_k(cache, model, items, k_refine, platt_params=None):
    n = len(items)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = multi_k_js(cache, model, items[i], items[j], k_refine, platt_params)
            D[i,j] = D[j,i] = d
    return D

def kmedoids(D, k=2, max_iter=100, seed=0):
    # Standard k-medoids algorithm
    # Returns medoids and assignments

def try_recursive_split(cluster_histories, model, cache, k_refine, platt_params=None,
                        tau_split=0.02, min_cluster=8, max_depth=2, seed=0):
    # Data-driven splitting: only split if avg_intra_distance > tau_split
    # Uses k-medoids for principled cluster center finding
    # Guards against degenerate splits
```

**Features**:
- **Data-driven**: Only splits clusters with high average intra-distance
- **Same metric**: Uses identical multi-k conditional JS throughout
- **Conservative**: Guards against over-fragmentation
- **Principled**: K-medoids finds true cluster centers

### 8. Futures-Aware Evaluation Mapping
**Problem**: Loss evaluation used Hamming distance, not the same metric as clustering.

**Solution**: Made evaluation mapping use the same multi-k conditional JS:

```python
# OLD: Hamming distance to representative
distance = np.sum(suffix != repr_hist)

# NEW: Multi-k conditional JS to representative (same as clustering)
js = cond_js_multi_k(cache_eval, model, h, rep,
                    ks=(k_refine, k_refine+1, k_refine+2, k_refine+3),
                    platt_params=platt_params)
```

**Result**: Evaluation mapping now consistent with clustering decisions.

### 9. Added New Model Support
**Added**: Support for `seven_state_human_char_large` preset:
- Updated argument parser choices
- Added model path mapping
- Added GT state expectations
- Updated state_mapping.py

### 10. Performance and Usability Improvements
**Added Parameters**:
- `--skip_diagnostics`: Skip JS diagnostics for faster execution
- `--max_representatives`: Control history sampling for diagnostics
- `--tau_split`: Threshold for recursive splitting
- `--min_cluster_split`: Minimum cluster size for splitting
- `--max_split_depth`: Maximum recursion depth
- `--eval_max_symbols`: Limit evaluation to subset for speed (default: 10,000)

**Tightened Defaults**:
- `--stage_b_threshold`: 0.01 → 0.0003 (tighter for multi-k robustness)

### 11. Comprehensive Pipeline Integration
**Final Pipeline** (Stage A → B → C):
1. **Stage A**: Emission-based agglomerative clustering
2. **Stage B**: Multi-k conditional JS representative refinement
3. **Stage C**: Recursive k-medoids splitting for mixed clusters

**Consistent Metrics**: Same multi-k conditional JS used throughout:
- Representative clustering decisions
- History assignment
- Recursive splitting decisions
- Evaluation context mapping

## Usage Examples

### Fast Testing:
```bash
python nanoGPT/js_analysis/unsupervised_fast.py \
  --preset seven_state_human_char_large \
  --k_refine 6 \
  --stage_b_threshold 0.0003 \
  --skip_diagnostics
```

### Full Analysis with Diagnostics:
```bash
python nanoGPT/js_analysis/unsupervised_fast.py \
  --preset seven_state_human_char_large \
  --k_refine 6 \
  --max_representatives 15 \
  --tau_split 0.02 \
  --eval_max_symbols 50000
```

### Aggressive Splitting:
```bash
python nanoGPT/js_analysis/unsupervised_fast.py \
  --preset seven_state_human_char_large \
  --k_refine 8 \
  --tau_split 0.015 \
  --min_cluster_split 6 \
  --max_split_depth 3
```

## Expected Improvements

1. **Better State Discrimination**: Multi-k conditional JS should provide clearer separation between bb/baab and similar states
2. **Consistent Pipeline**: Same metric used throughout eliminates inconsistencies
3. **Robust Clustering**: Multi-k averaging reduces sensitivity to single-horizon artifacts
4. **Data-Driven Splitting**: Recursive k-medoids only splits when statistically justified
5. **Meaningful Cluster Sizes**: Fixed assignment gives realistic cluster populations
6. **Better Loss Estimates**: Futures-aware evaluation mapping more accurately reflects model performance

## Files Modified

1. **`nanoGPT/js_analysis/unsupervised_fast.py`**: Main clustering script (extensive changes)
2. **`nanoGPT/js_analysis/state_mapping.py`**: Added `seven_state_human_char_large` preset support
3. **`nanoGPT/config/train_seven_state_human_char_large.py`**: New training config for larger model

## Technical Notes

- **Multi-K Range**: Uses `ks=(k_refine, k_refine+1, k_refine+2, k_refine+3)` for 4-horizon averaging
- **Caching**: Extensive use of `KStepCache` to avoid recomputing k-step distributions
- **Fallback Handling**: Graceful fallback to emission JS if conditional JS fails
- **Memory Efficiency**: Recursive splitting parameters prevent excessive fragmentation
- **Speed Optimization**: Evaluation subset parameter for rapid testing iterations

The improvements transform the original 2-stage clustering into a comprehensive 3-stage pipeline with principled, consistent metrics throughout, while maintaining backwards compatibility and adding extensive diagnostic capabilities.