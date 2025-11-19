




# Oracle CSSR (2-Stage, Oracle-Driven, Sync-Length Aware)

This algorithm is a **new CSSR-style method** that explicitly exploits a predictive oracle (e.g., a neural model or ground-truth machine) to reconstruct causal states. It differs from classical CSSR in two key ways:

1. **States are defined only from long histories** (fixed length `L_max`), so we never let pre-synchronization histories mint states.
2. We **do not run a recursion / transient removal phase**. Instead:

   * Stage 1: cluster `L_max` histories by increasing prediction horizon `k` until the partition stabilizes.
   * Stage 2: discover **minimal synchronizing suffixes** for each cluster. These are the causal state labels; short (< sync-length) contexts never become states.

The result is an ε-machine–like partition of histories into states, plus minimal synchronizing suffixes per state.

---

## 0. Inputs, notation, assumptions

### Data and alphabet

* Observed sequence:
  [
  x_1, x_2, \dots, x_N \in \mathcal{A}
  ]
* Alphabet:
  [
  \mathcal{A} = {0, 1, \dots}
  ]
  (binary in your current experiments, but keep it general).

### Oracle

We assume a predictive oracle with:

```python
get_kstep_distribution(model, history: np.ndarray, k: int, platt_params: Optional[dict]) -> np.ndarray
```

* `history`: 1D array of ints, shape `(L,)`.
* Returns `p`, a probability vector over all length-`k` strings from `A^k` (ordering fixed and known, like lexicographic).
* Optionally also:

```python
get_next_token_distribution(model, history, platt_params) -> np.ndarray
```

for the `k = 1` case.

### Parameters

* `L_max`: maximum past length used to define states (fixed, **all states come from histories of this length**).
* `K_max`: maximum future roll-out length used for equivalence testing.
* `metrics_k`: list of k’s to use (e.g. `[1]`, `[1, 2]`, or `[1, 2, 3]`).
* `tolerance`: JS threshold for “same predictive behavior” (e.g. `1e-3` bits).

Assumptions (theory side):

* Underlying process has a **finite-state, synchronizable ε-machine**.
* Oracle approximates ( P(X_{t+1:t+k} \mid \text{history}) ) well for all `k` used.

---

## 1. Stage 1 — Oracle clustering on `L_max` histories

**Goal:** partition the set of length-`L_max` histories into clusters that approximate causal states, by **incrementally increasing** the future horizon `k` and refining the partition until it stops changing.

### 1.1 Collect length-`L_max` histories

From the data sequence:

For `t = L_max, …, N` define:
[
h_t = x_{t-L_{\max}+1}^t
]
Collect the set (or multiset) of histories:

```python
H_list = []  # list of np.ndarray, shape (L_max,)
for t in range(L_max, N+1):
    h = data[t-L_max : t]  # copy or view
    H_list.append(h.copy())
```

You may:

* Deduplicate histories later (keep an index mapping from unique history → list of indices where it appears).

Define:

```python
H_unique = list_of_unique_histories(H_list)
# Represent each as np.ndarray of length L_max
```

We will cluster `H_unique`.

### 1.2 Precompute oracle predictions

For efficiency, you can precompute all needed distributions:

```python
preds = {
    k: {}  # maps history tuple -> np.ndarray of probs over A^k
    for k in metrics_k
}

for h in H_unique:
    key = tuple(h.tolist())
    for k in metrics_k:
        preds[k][key] = get_kstep_distribution(model, h, k, platt_params)
```

You can also do this lazily, but precomputation keeps the implementation cleaner.

### 1.3 JS distance helper

Define a helper to check whether two histories are “k-equivalent” for a given `k`:

```python
def js_close(h1, h2, k, tolerance, preds):
    key1 = tuple(h1.tolist())
    key2 = tuple(h2.tolist())
    p = preds[k][key1]
    q = preds[k][key2]
    return js_divergence(p, q) < tolerance
```

And “equivalent up to current k_max_used”:

```python
def histories_equivalent(h1, h2, k_values_used, tolerance, preds):
    return all(js_close(h1, h2, k, tolerance, preds) for k in k_values_used)
```

### 1.4 Partition refinement by increasing k

We maintain an evolving partition `P_k` of `H_unique`.

We’ll do:

* Start with **k=1**.
* Cluster histories based on 1-step futures (`k=1`).
* Then increment k (e.g. 2,3,… up to `K_max`), **refining** the current partition (only splitting clusters, never merging across previous boundaries).
* Stop when adding a new k doesn’t change the partition, or when `k` hits `K_max`.

#### Initialization (k = 1)

```python
k_values_used = [1]
partition = []          # list of clusters; each cluster is a list of histories
cluster_rep = []        # representative history for each cluster

for h in H_unique:
    assigned = None
    for c_idx, rep in enumerate(cluster_rep):
        if histories_equivalent(h, rep, k_values_used, tolerance, preds):
            assigned = c_idx
            break
    if assigned is None:
        assigned = len(cluster_rep)
        cluster_rep.append(h.copy())
        partition.append([])
    partition[assigned].append(h.copy())
```

At this point, `partition` is the k=1 partition `P_1`.

#### Refinement for k = 2…K_max

For each new `k_new` in `metrics_k[1:]` (i.e., skipping the first element which is 1):

```python
stable = False
for k_new in metrics_k[1:]:
    prev_partition = partition
    prev_cluster_rep = [rep.copy() for rep in cluster_rep]
    k_values_used.append(k_new)

    new_partition = []
    new_cluster_rep = []

    # For each cluster from previous partition, refine it under the new k
    for old_c_idx, cluster in enumerate(prev_partition):
        # We'll split this cluster into subclusters based on equivalence at the updated k_values_used
        sub_partition = []
        sub_rep = []

        for h in cluster:
            assigned = None
            for sub_idx, rep in enumerate(sub_rep):
                if histories_equivalent(h, rep, k_values_used, tolerance, preds):
                    assigned = sub_idx
                    break
            if assigned is None:
                assigned = len(sub_rep)
                sub_rep.append(h.copy())
                sub_partition.append([])
            sub_partition[assigned].append(h.copy())

        # Append all subclusters to the global partition
        new_partition.extend(sub_partition)
        new_cluster_rep.extend(sub_rep)

    # Check if anything changed
    if len(new_partition) == len(prev_partition) and \
       all(set(map(tuple, new_partition[i])) == set(map(tuple, prev_partition[i]))
           for i in range(len(prev_partition))):
        # No change at this k → partition is stable w.r.t additional horizon
        stable = True
        partition = prev_partition
        cluster_rep = prev_cluster_rep
        break
    else:
        partition = new_partition
        cluster_rep = new_cluster_rep

# After the loop:
# - partition: list of clusters (states)
# - cluster_rep: representative history per cluster
# - k_values_used: the k’s that actually refined the partition
```

Properties:

* At each k, you **only split existing clusters**, never merge clusters that were distinct at previous k.
* When the partition stops changing at some `k`, you can stop early even if `k < K_max`.

Denote:

* Number of clusters = `M = len(partition)`.
* These `M` clusters are the **candidate causal states** at the `L_max` level.

### 1.5 State index map

Build a map from each history to its cluster index:

```python
state_of_Lmax = {}  # maps history tuple -> state index

for s_idx, cluster in enumerate(partition):
    for h in cluster:
        state_of_Lmax[tuple(h.tolist())] = s_idx
```

This is the canonical “which state is this L_max history in?” mapping.

**Important:**
Up to here, we’ve *never* created states from shorter contexts. All states are equivalence classes of **length-L_max histories**.

No transient removal is needed because:

* We’re not constructing a full Markov chain with extra “prefix states”.
* Every state is a cluster of histories that *already share* similar futures at all tested horizons.
* Transience in the Markov-chain sense (infinite-time classification) is not part of this algorithm’s state definition; we only care about **predictive equivalence** on the sampled histories.

---

## 2. Stage 2 — Synchronizing suffix discovery (sync length per state)

Now we have:

* A set of states `S = {0,…,M-1}` (clusters over `H_unique`).
* A mapping `state_of_Lmax[h]`.

We want to find, for each state, **minimal synchronizing suffixes** that uniquely identify that state. These automatically have length ≥ sync length; **shorter pre-sync contexts will not pass the test**.

### 2.1 Build suffix → state usage map

We look at all suffixes of all L_max histories in `H_unique`.

```python
from collections import defaultdict

suffix_to_states = defaultdict(set)
suffix_to_histories = defaultdict(list)

for h in H_unique:
    j = state_of_Lmax[tuple(h.tolist())]
    for ell in range(1, L_max + 1):
        u = tuple(h[-ell:].tolist())         # suffix as tuple
        suffix_to_states[(ell, u)].add(j)
        suffix_to_histories[(ell, u)].append(h)
```

After this:

* `suffix_to_states[(ell, u)]` is the set of states in which this suffix appears at the end of some L_max history.

### 2.2 Synchronizing suffixes per state

A suffix `(ell, u)` is **synchronizing for state `j`** if:

```python
suffix_to_states[(ell, u)] == {j}
```

Interpretation:

> Whenever the process has just seen suffix `u` at the end of a length-`L_max` history (in our data), it’s in state `j` and no other state.

For each state `j`:

```python
sync_suffixes_per_state = [[] for _ in range(M)]

for j in range(M):
    sync_suffixes = []
    for (ell, u), states_set in suffix_to_states.items():
        if states_set == {j}:
            sync_suffixes.append((ell, u))
    sync_suffixes_per_state[j] = sync_suffixes
```

### 2.3 Minimal synchronizing suffixes

We care about **minimal** synchronizing suffixes (no shorter suffix with the same property).

For each state `j`:

```python
minimal_suffixes_per_state = [[] for _ in range(M)]

for j in range(M):
    sync_suffixes = sync_suffixes_per_state[j]
    # sort by length ascending, then lexicographically for determinism
    sync_suffixes.sort(key=lambda item: (item[0], item[1]))

    minimal_for_j = []
    for ell, u in sync_suffixes:
        u_list = list(u)
        drop = False
        for ell_m, u_m in minimal_for_j:
            u_m_list = list(u_m)
            # if an already-kept minimal suffix is a suffix of u → u is non-minimal
            if ell_m <= ell and u_list[-ell_m:] == u_m_list:
                drop = True
                break
        if not drop:
            minimal_for_j.append((ell, u))

    minimal_suffixes_per_state[j] = minimal_for_j
```

These `minimal_suffixes_per_state[j]` are the **state labels** (analog of CSSR’s minimal predictive suffixes).

### 2.4 Why pre-sync histories are not states

Key property:

* For any suffix `u` shorter than the true sync length:

  * In a synchronizable ε-machine, there exist at least two causal states that share that suffix in their pasts.
  * Therefore, in the limit of enough data, `suffix_to_states[(ell, u)]` will contain multiple state indices.
  * So `u` **never** satisfies `suffix_to_states[(ell, u)] == {j}` for any single `j`.

Conclusion:

* Pre-sync suffixes **cannot** be synchronizing for a single state.
* They **never** appear in `minimal_suffixes_per_state`.
* Only suffixes with effective length ≥ sync length (in practice, those that appear exclusively in one cluster) survive.

So your algorithm **automatically “removes” < sync-length histories from the causal state set.** They remain ambiguous contexts (mixtures) but are not ε-states.

---

## 3. Optional: Build an explicit ε-machine (transitions)

Strictly for the 2-stage algorithm you proposed, we *don’t need* to explicitly build transitions or remove transients — the state partition + minimal synchronizing suffixes are the main outputs.

However, if you want a concrete ε-machine (for simulation, likelihood, etc.), you can:

* Define states `S = {0,…,M-1}`.
* For each state `j` and symbol `a ∈ A`:

  * Pick any `h ∈ partition[j]`.
  * Form `h_ext = (h + a)[-L_max:]`.
  * If `h_ext` is in `state_of_Lmax`, define a transition:
    [
    j \xrightarrow{a} state_of_Lmax[h_ext]
    ]
* Optionally estimate per-state emission probabilities from the oracle or empirical frequencies over `partition[j]`.

This gives you a unifilar-ish HMM over the recovered states. You *could* run a determinization pass here if needed, but it’s conceptually separate from the **2-stage partition + sync-suffix scheme**.

---

## 4. Summary / algorithm sketch

**Inputs:** `data`, `model`, `L_max`, `metrics_k`, `tolerance`, `platt_params`.

**Stage 1 – Oracle clustering on L_max pasts**

1. Extract all length-`L_max` histories `H_unique` from data.
2. For each `h ∈ H_unique` and each `k ∈ metrics_k`, compute `p_h^{(k)} = get_kstep_distribution(...)`.
3. Initialize partition `P_1` by clustering using `k=1` and JS tests (`< tolerance`).
4. For each new `k` in `metrics_k[1:]`:

   * Refine partition `P_{k-1}`: split clusters where members differ at horizon `k`.
   * If `P_k` = `P_{k-1}`, stop early (partition stabilized).
5. Output:

   * `partition` = final clusters (states).
   * `state_of_Lmax`: mapping from each L_max history to its state index.

**Stage 2 – Synchronizing suffixes**

6. For all `h ∈ H_unique`, for all suffix lengths `1..L_max`:

   * Record which states each suffix `(ell, u)` appears in.
7. For each state `j`:

   * Collect suffixes used only by that state: `suffix_to_states[(ell, u)] == {j}`.
   * Among these, keep those that are **minimal** under the suffix relation.
8. Output:

   * `minimal_suffixes_per_state[j]` as the ε-machine’s state labels.

No explicit transient removal step is needed because:

* States are constructed exclusively from long histories,
* Transience in the Markov-chain sense is not part of the state definition here,
* Pre-sync histories are naturally excluded from becoming state labels by the synchronizing suffix criterion.

---
