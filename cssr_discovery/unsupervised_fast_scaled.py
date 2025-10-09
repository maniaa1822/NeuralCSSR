# unsupervised_fast_scaled.py
# Fast, scalable unsupervised epsilon-machine recovery with global minimal-suffixing,
# ANN-like Stage A/B using LSH + small-k neighbor checks, and graph-based Stage C remerge.
#
# Drop-in replacement for the previous script; relies only on numpy/torch + your repo helpers.

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import json
from dataclasses import dataclass

# ===== Repo-local deps (unchanged) =====
from js_metrics import (
    get_next_token_distribution,
    js_divergence,
    js_divergence_conditional_k,
    get_kstep_distribution,
    extract_subsequences
)
from state_mapping import get_gt_state
from calibration import fit_platt_params


# =========================
# Utilities & data classes
# =========================

@dataclass
class FastClusteringResult:
    clusters: List[List[np.ndarray]]
    cluster_representatives: List[np.ndarray]
    cluster_emission_sigs: List[str]
    total_histories_sampled: int
    cache_hits: int
    cache_misses: int


def js_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JS divergence in bits (base-2)."""
    import math
    m = (p + q) / 2.0

    def kl_bits(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 1e-12:
                s += ai * math.log(ai / max(bi, 1e-12), 2)
        return s

    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def compute_emission_signature(probs: np.ndarray, precision: int = 2) -> str:
    p0, p1 = float(probs[0]), float(probs[1])
    p0_rounded = round(p0, precision)
    p1_rounded = round(p1, precision)
    return f"{p0_rounded:.{precision}f}_{p1_rounded:.{precision}f}"


# =========================
# Global minimal-suffixing
# =========================

class SuffixTrieNode:
    __slots__ = ("children", "suffix_array", "depth", "emission_ready", "emission")
    def __init__(self, depth=0):
        self.children: Dict[int, "SuffixTrieNode"] = {}
        self.suffix_array: Optional[np.ndarray] = None  # store the concrete suffix (np array)
        self.depth: int = depth
        self.emission_ready: bool = False
        self.emission: Optional[np.ndarray] = None


class SuffixTrie:
    def __init__(self, min_suffix_len: int):
        self.root = SuffixTrieNode(depth=0)
        self.min_suffix_len = min_suffix_len

    def insert_suffix(self, suffix: np.ndarray):
        node = self.root
        d0 = max(self.min_suffix_len, 1)
        # suffix is assumed length >= min_suffix_len
        for i, bit in enumerate(suffix):
            b = int(bit)
            if b not in node.children:
                node.children[b] = SuffixTrieNode(depth=node.depth + 1)
            node = node.children[b]
            # only store concrete arrays starting when we reach min_suffix_len
            if node.depth >= self.min_suffix_len and node.suffix_array is None:
                node.suffix_array = suffix[: i + 1].copy()

    def build_from_histories(self, histories: List[np.ndarray]):
        for h in histories:
            L = len(h)
            for d in range(self.min_suffix_len, L + 1):
                self.insert_suffix(h[-d:])

    def nodes_at_depth(self, depth: int) -> List[SuffixTrieNode]:
        out = []
        q = deque([(self.root, 0)])
        while q:
            node, d = q.popleft()
            if d == depth:
                out.append(node)
            elif d < depth:
                for child in node.children.values():
                    q.append((child, d + 1))
        return out

    def iter_nodes_with_suffix(self) -> List[SuffixTrieNode]:
        """Iterate all nodes that actually have a concrete suffix array (depth >= min_suffix_len)."""
        nodes = []
        q = deque([self.root])
        while q:
            node = q.popleft()
            if node.suffix_array is not None:
                nodes.append(node)
            for child in node.children.values():
                q.append(child)
        return nodes


def batch_emissions_for_nodes(model, nodes: List[SuffixTrieNode], platt_params: Optional[dict]) -> None:
    """Compute next-token emission probs for all given trie nodes in a single batch."""
    if not nodes:
        return
    # Gather suffixes
    arrs = [n.suffix_array for n in nodes if (n.suffix_array is not None and not n.emission_ready)]
    if not arrs:
        return
    # Stack with the same dtype
    Lmax = max(len(a) for a in arrs)
    # Pad left to equal length for the batch; model/get_next_token_distribution should handle varying lens,
    # but if your helper expects exact lengths, we can just feed lists (the provided helper supports list of arrays).
    # Here we pass as list.
    probs_batch = get_next_token_distribution(model, arrs, platt_params)  # returns list/ndarray of Bx2
    # Assign back
    k = 0
    for n in nodes:
        if n.suffix_array is not None and not n.emission_ready:
            n.emission = probs_batch[k]
            n.emission_ready = True
            k += 1


def minimalize_histories(histories: List[np.ndarray],
                         model,
                         platt_params: Optional[dict],
                         tolerance_bits: float,
                         min_suffix_len: int) -> Tuple[List[np.ndarray], Dict[Tuple[int, ...], np.ndarray]]:
    """
    Build a suffix trie once, batch-compute emissions for all unique suffixes,
    then reduce each history to its shortest stable suffix under JS<=tolerance_bits.
    Returns the list of minimal suffixes (one per history, preserving order) and a memo dict.
    """
    if not histories:
        return [], {}

    trie = SuffixTrie(min_suffix_len=min_suffix_len)
    trie.build_from_histories(histories)

    # Batch evaluate emissions for all nodes that carry suffixes (single sweep)
    nodes = trie.iter_nodes_with_suffix()
    # You can also batch in chunks if memory is a concern:
    BATCH = 4096
    for i in range(0, len(nodes), BATCH):
        batch_emissions_for_nodes(model, nodes[i:i + BATCH], platt_params)

    # Helper: JS between full-history emission and suffix emission
    def js_full_vs_suffix(full: np.ndarray, suffix_node: SuffixTrieNode) -> float:
        return js_bits(full, suffix_node.emission)

    # Map suffix tuple to emission for reuse in Stage A
    suffix_to_emission: Dict[Tuple[int, ...], np.ndarray] = {}

    minimal_suffixes: List[np.ndarray] = []
    # We also batch the "full" emissions for speed
    BATCH2 = 2048
    for i in range(0, len(histories), BATCH2):
        batch = histories[i:i + BATCH2]
        full_probs_batch = get_next_token_distribution(model, batch, platt_params)
        for h, full in zip(batch, full_probs_batch):
            L = len(h)
            # Walk from shorter suffix to longer? We want the shortest suffix that is "stable" wrt full
            # Start from min_suffix_len up to L, and stop at the first suffix that yields JS<=tol for all longer ones.
            # Efficiently: binary search could be used; here we go ascending since L is typically small-ish per history.
            chosen = h[-min_suffix_len:]
            # Start from min_suffix_len and expand until we exceed tolerance, then take previous.
            # But we want the SHORTEST that preserves. We'll test ascending; if small suffix already within tol, keep it.
            lo = min_suffix_len
            hi = L
            # test lo immediately
            node = trie.root
            for bit in h[-lo:]:
                node = node.children[int(bit)]
            js0 = js_full_vs_suffix(full, node)
            if js0 <= tolerance_bits:
                chosen = h[-lo:]
            else:
                # Expand gradually until stable
                stable_found = False
                for d in range(lo + 1, hi + 1):
                    node = trie.root
                    for bit in h[-d:]:
                        node = node.children[int(bit)]
                    jsd = js_full_vs_suffix(full, node)
                    if jsd <= tolerance_bits:
                        chosen = h[-d:]
                        stable_found = True
                        break
                if not stable_found:
                    chosen = h  # fall back to full history if nothing is stable

            minimal_suffixes.append(chosen)
            suffix_to_emission[tuple(int(x) for x in chosen)] = trie.root  # placeholder; fixed below

    # Fill suffix_to_emission with actual vectors
    for n in nodes:
        if n.suffix_array is not None and n.emission_ready and n.emission is not None:
            key = tuple(int(x) for x in n.suffix_array)
            suffix_to_emission[key] = n.emission

    return minimal_suffixes, suffix_to_emission


# =========================
# Lightweight ANN building
# =========================

class RandomHyperplaneLSH:
    """Very small LSH for cosine-similarity partitioning to reduce pairwise checks."""
    def __init__(self, dim: int, n_planes: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((n_planes, dim)).astype(np.float32)
        self.tables: Dict[int, List[int]] = defaultdict(list)  # signature int -> list of indices

    def _signature(self, v: np.ndarray) -> int:
        # v: (dim,)
        bits = (self.proj @ v) >= 0.0  # (n_planes,)
        sig = 0
        for b in bits:
            sig = (sig << 1) | int(b)
        return sig

    def add(self, idx: int, v: np.ndarray):
        self.tables[self._signature(v)].append(idx)

    def candidates(self, v: np.ndarray) -> List[int]:
        return self.tables.get(self._signature(v), [])


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)
    if du < 1e-12 or dv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (du * dv))


def stage_a_ann_threshold_clustering(
    vectors: np.ndarray,
    js_check,
    js_threshold: float,
    seed: int = 42
) -> List[List[int]]:
    """
    Online threshold clustering using LSH to get a tiny candidate set per point,
    then attach to the first cluster whose centroid passes JS<th; else start a cluster.

    - vectors: NxD array of emission vectors (P(next|suffix))
    - js_check: callable(u, v_centroid) -> float JS divergence
    - returns: list of clusters (list of index lists)
    """
    N, D = vectors.shape
    lsh = RandomHyperplaneLSH(dim=D, n_planes=16, seed=seed)
    clusters: List[List[int]] = []
    centroids: List[np.ndarray] = []
    counts: List[int] = []

    for i in range(N):
        v = vectors[i]
        # LSH candidates plus a small random backoff
        cands = lsh.candidates(v)
        # Try attaching to an existing cluster among candidates first
        attached = False
        for ci in set([c for c in cands if c < len(clusters)]):  # ci indexes are not the same as clusters; fix below
            pass  # We can't map buckets to clusters directly; we’ll check all clusters but early-exit on cosine screen.

        # cosine screen to prune most clusters quickly; then JS check
        # In practice, #clusters << N. Keep it simple/robust.
        best_ci = -1
        for ci, cen in enumerate(centroids):
            # quick cosine pre-check
            if cosine(v, cen) < 0.95:  # coarse gate, tuneable
                continue
            jsd = js_check(v, cen)
            if jsd <= js_threshold:
                best_ci = ci
                break

        if best_ci >= 0:
            # attach & update centroid
            clusters[best_ci].append(i)
            counts[best_ci] += 1
            centroids[best_ci] = (centroids[best_ci] * (counts[best_ci] - 1) + v) / counts[best_ci]
            attached = True
        else:
            # start new
            clusters.append([i])
            centroids.append(v.copy())
            counts.append(1)

        # Add to LSH
        lsh.add(i, v)

    return clusters


# =========================
# Stage B: rollout embeddings + ANN + selective exact JS
# =========================

def rollout_embedding(model, reps: List[np.ndarray], k_refine: int, platt_params: Optional[dict]) -> np.ndarray:
    """Build a compact embedding that approximates conditional behavior."""
    if not reps:
        return np.zeros((0, 2 + (2 ** k_refine)), dtype=np.float32)
    # Batch
    P1 = get_next_token_distribution(model, reps, platt_params)             # Rx2
    Pk = get_kstep_distribution(model, reps, k_refine, platt_params)        # Rx(2^k)
    Phi = np.concatenate([P1, Pk], axis=1).astype(np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(Phi, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return Phi / norms


def ann_union_find(Phi: np.ndarray,
                   exact_js_edge,
                   cutoff_cos: float,
                   confirm_margin: float) -> List[List[int]]:
    """
    Build clusters via an ANN-like pass over normalized Phi with cosine similarity.
    - cutoff_cos: neighbors above this are connected
    - confirm_margin: edges within (cutoff_cos - confirm_margin, cutoff_cos] are confirmed with exact JS
    """
    N = Phi.shape[0]
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    # LSH for Phi
    lsh = RandomHyperplaneLSH(dim=Phi.shape[1], n_planes=18, seed=123)
    for i in range(N):
        lsh.add(i, Phi[i])

    for i in range(N):
        vi = Phi[i]
        cands = lsh.candidates(vi)
        if not cands:
            continue
        # Dedup and include a few random probes
        seen = set([i])
        for j in cands:
            if j in seen:
                continue
            seen.add(j)
            if j == i:
                continue
            cosv = cosine(vi, Phi[j])
            if cosv < cutoff_cos - confirm_margin:
                continue
            if cosv >= cutoff_cos:
                union(i, j)
            else:
                # borderline: confirm with exact JS edge predicate
                if exact_js_edge(i, j):
                    union(i, j)

    # collect components
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(N):
        clusters[find(i)].append(i)
    return list(clusters.values())


# =========================
# Stage C: graph-based remerge
# =========================

def stage_c_remerge_states(
    clusters: List[List[np.ndarray]],
    representatives: List[np.ndarray],
    model,
    platt_params: Optional[dict],
    k_refine: int,
    emission_threshold: float,
    rollout_threshold: float
) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[str]]:
    if not clusters:
        return clusters, representatives, []

    R = len(representatives)
    # Build graph over reps
    parent = list(range(R))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Precompute P1 for reps to cheap-check emission JS
    P1 = get_next_token_distribution(model, representatives, platt_params)
    # Remerge by gates
    for i in range(R):
        for j in range(i + 1, R):
            ejs = js_divergence(P1[i], P1[j])
            if ejs >= emission_threshold:
                continue
            rjs = js_divergence_conditional_k(model, representatives[i], representatives[j], k_refine, platt_params)
            if rjs < rollout_threshold:
                union(i, j)

    # Collect components
    comp_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i in range(R):
        comp_to_indices[find(i)].append(i)

    # Merge clusters by components
    new_clusters: List[List[np.ndarray]] = []
    new_reps: List[np.ndarray] = []
    new_sigs: List[str] = []

    for comp, idxs in comp_to_indices.items():
        merged_histories: List[np.ndarray] = []
        for idx in idxs:
            merged_histories.extend(clusters[idx])
        # Choose rep = one with largest cluster size
        sizes = [len(clusters[idx]) for idx in idxs]
        best_idx = idxs[int(np.argmax(sizes))]
        new_clusters.append(merged_histories)
        new_reps.append(representatives[best_idx])
        sig_probs = get_next_token_distribution(model, [representatives[best_idx]], platt_params)[0]
        new_sigs.append(compute_emission_signature(sig_probs, 2))

    return new_clusters, new_reps, new_sigs


# =========================
# Sampling strategies
# =========================

def random_history_sampling(data: np.ndarray, L: int, n_samples: int,
                           seed: Optional[int] = None) -> List[np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    all_histories = extract_subsequences(data, L)
    if len(all_histories) <= n_samples:
        return all_histories
    indices = np.random.choice(len(all_histories), size=n_samples, replace=False)
    return [all_histories[i] for i in indices]


def emission_stratified_sampling(data: np.ndarray, L: int, model, n_samples: int,
                                platt_params: Optional[dict] = None, n_strata: int = 5,
                                seed: Optional[int] = None) -> List[np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    all_histories = extract_subsequences(data, L)
    sample_size = min(len(all_histories), 4000)
    sample_indices = np.random.choice(len(all_histories), size=sample_size, replace=False)

    emissions = []
    sampled_histories = []
    # Batch eval for speed
    batch = [all_histories[idx] for idx in sample_indices]
    probs_batch = get_next_token_distribution(model, batch, platt_params)
    for hist, probs in zip(batch, probs_batch):
        emissions.append(float(probs[0]))
        sampled_histories.append(hist)

    emissions = np.array(emissions)
    quantiles = np.linspace(0, 1, n_strata + 1)
    boundaries = np.quantile(emissions, quantiles)

    selected_histories = []
    per_stratum = n_samples // n_strata
    extra = n_samples % n_strata

    for i in range(n_strata):
        lower = boundaries[i]
        upper = boundaries[i + 1] if i < n_strata - 1 else 1.0
        in_stratum = [(hist, em) for hist, em in zip(sampled_histories, emissions) if lower <= em <= upper]
        stratum_size = per_stratum + (1 if i < extra else 0)

        unique_contexts = {}
        for hist, em in in_stratum:
            hist_key = tuple(hist)
            if hist_key not in unique_contexts:
                unique_contexts[hist_key] = (hist, em)

        unique_list = list(unique_contexts.values())
        if len(unique_list) >= stratum_size:
            stratum_indices = np.random.choice(len(unique_list), size=stratum_size, replace=False)
            selected_histories.extend([unique_list[idx][0] for idx in stratum_indices])
        else:
            selected_histories.extend([hist for hist, _ in unique_list])
            remaining = stratum_size - len(unique_list)
            if remaining > 0 and unique_list:
                repeat_indices = np.random.choice(len(unique_list), size=remaining, replace=True)
                selected_histories.extend([unique_list[idx][0] for idx in repeat_indices])

    return selected_histories


# =========================
# Main scalable pipeline
# =========================

def efficient_scaled_clustering(
    histories: List[np.ndarray],
    model,
    k_refine: int = 4,
    platt_params: Optional[dict] = None,
    stage_a_threshold: float = 0.001,
    stage_b_threshold: float = 0.001,
    emission_precision: int = 2,
    max_representatives: int = 10,
    tolerance_bits: float = 1e-3,
    min_suffix_len: int = 2,
    enable_remerging: bool = True,
    remerge_emission_threshold: float = 1e-4,
    remerge_rollout_threshold: float = 5e-4,
    seed: int = 42
) -> FastClusteringResult:
    """
    Scalable variant:
    1) Global minimal-suffixing with trie (single batched pass).
    2) Stage A: ANN-like online threshold clustering on emission vectors of minimal suffixes.
    3) Stage B: reps → rollout-embeddings → ANN+UF, confirm borderline with exact conditional JS.
    4) Stage C: graph-based remerge.
    """
    if not histories:
        return FastClusteringResult([], [], [], 0, 0, 0)

    print("\n[Stage 0] Global minimal-suffixing (backward stability)")
    # 1) Minimal-suffixing (global)
    minimal_suffixes, suffix_to_emission = minimalize_histories(
        histories, model, platt_params, tolerance_bits, min_suffix_len
    )
    orig_lengths = [len(h) for h in histories]
    minimal_lengths = [len(h) for h in minimal_suffixes]
    tokens_dropped = sum(max(0, o - m) for o, m in zip(orig_lengths, minimal_lengths))
    avg_min_len = sum(minimal_lengths) / max(1, len(minimal_lengths))
    avg_drop = tokens_dropped / max(1, len(minimal_lengths))
    print(f"[Stage 0] Reduced contexts to minimal suffixes: avg len={avg_min_len:.2f}, avg tokens dropped={avg_drop:.2f}")

    # Build dedup of minimal suffixes (to avoid clustering duplicates exhaustively)
    uniq_suffixes: List[Tuple[int, ...]] = []
    uniq_map: Dict[Tuple[int, ...], int] = {}
    original_to_uniq: List[int] = []
    for arr in minimal_suffixes:
        key = tuple(int(x) for x in arr)
        if key not in uniq_map:
            uniq_map[key] = len(uniq_suffixes)
            uniq_suffixes.append(key)
        original_to_uniq.append(uniq_map[key])
    print(f"[Stage 0] Unique minimal suffixes: {len(uniq_suffixes)}")

    # Emission vectors for uniq minimal suffixes
    uniq_arrays = [np.array(k, dtype=np.int64) for k in uniq_suffixes]
    # Prefer cached from suffix_to_emission; fall back to model
    emis = []
    missing = []
    for k in uniq_suffixes:
        if k in suffix_to_emission and suffix_to_emission[k] is not None:
            emis.append(suffix_to_emission[k])
        else:
            missing.append(np.array(k, dtype=np.int64))
            emis.append(None)
    if missing:
        probs_missing = get_next_token_distribution(model, missing, platt_params)
        it = iter(probs_missing)
        for i, v in enumerate(emis):
            if v is None:
                emis[i] = next(it)
    emission_mat = np.vstack(emis).astype(np.float32)  # Ux2

    print("\n[Stage 1] Emission clustering (Stage A)")
    # 2) Stage A: ANN online threshold clustering on emission vectors
    js_check = lambda u, cen: js_divergence(u, cen)
    a_clusters_idx = stage_a_ann_threshold_clustering(
        emission_mat, js_check, js_threshold=stage_a_threshold, seed=seed
    )
    # Build Stage A groups as lists of minimal-suffix arrays
    stage_a_groups: List[List[np.ndarray]] = []
    for idxs in a_clusters_idx:
        group = [uniq_arrays[j] for j in idxs]
        stage_a_groups.append(group)
    num_groups = len(stage_a_groups)
    avg_group = sum(len(g) for g in stage_a_groups) / max(1, num_groups)
    print(f"[Stage 1] Formed {num_groups} emission groups (avg size={avg_group:.1f}, threshold={stage_a_threshold})")

    # 3) Stage B: refine within each Stage A group using rollout-embeddings over representatives
    print("\n[Stage 2] Conditional refinement (Stage B)")
    print(f"[Stage 2] Parameters: k={k_refine}, stage_b_threshold={stage_b_threshold}, max_reps={max_representatives}")
    final_clusters: List[List[np.ndarray]] = []
    rng = np.random.default_rng(seed)

    for group_idx, group in enumerate(stage_a_groups, start=1):
        # If group small or singleton, keep as-is
        # Choose up to max_representatives unique patterns as reps
        # Ensure uniqueness by tuple keys
        start_cluster_count = len(final_clusters)
        seen = set()
        reps = []
        for h in group:
            t = tuple(int(x) for x in h)
            if t not in seen:
                seen.add(t)
                reps.append(h)
                if len(reps) >= max_representatives:
                    break
        if len(reps) <= 1:
            # Just keep the group as a single cluster
            final_clusters.append(group)
            print(f"  Group {group_idx}: {len(group)} contexts → 1 cluster (singleton or uniform)")
            continue

        Phi = rollout_embedding(model, reps, k_refine, platt_params)  # RxD
        # Define exact JS edge check for borderline neighbors
        def exact_js_edge(i, j):
            try:
                return js_divergence_conditional_k(model, reps[i], reps[j], k_refine, platt_params) <= stage_b_threshold
            except Exception:
                return False

        # Cosine cutoff heuristic (tunable). Start high, confirm near-boundary.
        rep_components = ann_union_find(Phi, exact_js_edge, cutoff_cos=0.995, confirm_margin=0.01)

        # Map each component to a cluster. Start by placing the reps.
        # Build mapping rep_index -> cluster_index
        rep_to_cluster = {}
        for comp in rep_components:
            new_cluster: List[np.ndarray] = []
            for ridx in comp:
                rep_to_cluster[ridx] = len(final_clusters)
                new_cluster.append(reps[ridx])
            final_clusters.append(new_cluster)

        # Assign the remaining non-rep members of the Stage A group to the nearest rep-component.
        remaining = [h for h in group if tuple(int(x) for x in h) not in set(tuple(int(x) for x in r) for r in reps)]
        if remaining:
            # Cheap assignment: to the rep whose emission vector is closest in JS
            # Precompute rep emissions
            rep_emissions = get_next_token_distribution(model, reps, platt_params)
            for h in remaining:
                eh = get_next_token_distribution(model, [h], platt_params)[0]
                best = None
                best_js = 1e9
                for ridx, er in enumerate(rep_emissions):
                    jsd = js_divergence(eh, er)
                    if jsd < best_js:
                        best_js = jsd
                        best = ridx
                if best is not None:
                    final_clusters[rep_to_cluster[best]].append(h)
                else:
                    # fallback: put into the first cluster
                    final_clusters[rep_to_cluster[0]].append(h)

        new_clusters = len(final_clusters) - start_cluster_count
        print(f"  Group {group_idx}: {len(group)} contexts → {new_clusters} clusters (reps={len(reps)})")

    # Representatives & signatures
    cluster_representatives = [c[0] for c in final_clusters]
    cluster_emission_sigs = []
    for h in cluster_representatives:
        probs = get_next_token_distribution(model, [h], platt_params)[0]
        cluster_emission_sigs.append(compute_emission_signature(probs, emission_precision))

    print(f"[Stage 2] Total clusters after refinement: {len(final_clusters)}")

    # 4) Stage C: optional remerge
    if enable_remerging:
        print("\n[Stage 3] State remerging (Stage C)")
        before_remerge = len(final_clusters)
        final_clusters, cluster_representatives, cluster_emission_sigs = stage_c_remerge_states(
            final_clusters, cluster_representatives, model, platt_params, k_refine,
            emission_threshold=remerge_emission_threshold,
            rollout_threshold=remerge_rollout_threshold
        )
        print(f"[Stage 3] Clusters remerged: {before_remerge} → {len(final_clusters)}")
    else:
        print("\n[Stage 3] State remerging skipped (--disable_remerging)")

    # cache stats are not explicitly tracked here (we used batch eval everywhere)
    return FastClusteringResult(
        clusters=final_clusters,
        cluster_representatives=cluster_representatives,
        cluster_emission_sigs=cluster_emission_sigs,
        total_histories_sampled=len(histories),
        cache_hits=0,
        cache_misses=0
    )


# =========================
# Loss & evaluation (unchanged API)
# =========================

def compute_epsilon_machine_loss(clustering_result: FastClusteringResult, model, data: np.ndarray,
                                L: int, platt_params: Optional[dict] = None) -> Dict:
    clusters = clustering_result.clusters
    if not clusters:
        return {'total_loss': float('inf'), 'avg_loss_per_symbol': float('inf'), 'num_predictions': 0, 'num_states': 0}

    # Step 1: emissions per discovered state via representative
    state_emissions = {}
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        repr_hist = cluster[0]
        probs = get_next_token_distribution(model, [repr_hist], platt_params)[0]
        state_emissions[i] = probs

    # Step 2: map context -> state via exact suffix match or 1-bit fuzzy on shortest reps first
    def get_discovered_state(history: np.ndarray) -> Optional[int]:
        if len(history) < 2:
            return None
        # exact suffix
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            repr_hist = cluster[0]
            rL = len(repr_hist)
            if len(history) >= rL and np.array_equal(history[-rL:], repr_hist):
                return i
        # fuzzy 1-bit distance, shortest reps first
        order = sorted(range(len(clusters)), key=lambda idx: len(clusters[idx][0]) if clusters[idx] else 1 << 30)
        best, min_d = None, 1 << 30
        for i in order:
            cluster = clusters[i]
            if not cluster:
                continue
            repr_hist = cluster[0]
            rL = len(repr_hist)
            if len(history) < rL:
                continue
            hs = history[-rL:]
            d = int(np.sum(hs != repr_hist))
            if d < min_d:
                min_d = d
                best = i
        return best if min_d <= 1 else None

    total_loss = 0.0
    num_predictions = 0
    for i in range(L, len(data)):
        ctx = data[i - L:i]
        nxt = int(data[i])
        st = get_discovered_state(ctx)
        if st is None or st not in state_emissions:
            continue
        p = float(state_emissions[st][nxt])
        if p > 1e-12:
            total_loss -= np.log(p)
            num_predictions += 1

    avg_loss = total_loss / num_predictions if num_predictions > 0 else float('inf')
    return {
        'total_loss': total_loss,
        'avg_loss_per_symbol': avg_loss,
        'avg_loss_per_symbol_bits': (avg_loss / np.log(2)) if np.isfinite(avg_loss) else float('inf'),
        'num_predictions': num_predictions,
        'num_states': len([c for c in clusters if c])
    }


def evaluate_clustering_against_gt(clustering_result: FastClusteringResult, preset: str) -> Dict:
    clusters = clustering_result.clusters
    cluster_analysis = []

    for i, cluster in enumerate(clusters):
        gt_states = []
        hist_strs = []
        for hist in cluster:
            gt_state = get_gt_state(hist, preset)
            if gt_state:
                gt_states.append(gt_state)
            hist_strs.append(''.join(map(str, hist)))

        gt_counts = {}
        for gt in gt_states:
            gt_counts[gt] = gt_counts.get(gt, 0) + 1

        purity = max(gt_counts.values()) / len(gt_states) if gt_states else 0.0
        dominant_gt = max(gt_counts.keys(), key=gt_counts.get) if gt_counts else None

        cluster_info = {
            'cluster_id': i,
            'size': len(cluster),
            'gt_state_counts': gt_counts,
            'purity': purity,
            'dominant_gt_state': dominant_gt,
            'emission_signature': clustering_result.cluster_emission_sigs[i] if i < len(clustering_result.cluster_emission_sigs) else "",
            'sample_histories': hist_strs[:5]
        }
        cluster_analysis.append(cluster_info)

    total_histories = sum(len(cluster) for cluster in clusters)
    weighted_purity = sum(info['purity'] * info['size'] for info in cluster_analysis) / total_histories if total_histories > 0 else 0.0

    all_gt_states = set()
    for info in cluster_analysis:
        all_gt_states.update(info['gt_state_counts'].keys())

    expected_gt_states = {
        'seven_state_human': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'seven_state_human_large': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'seven_state_human_char_large': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'sevestateold': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'golden_mean': {'A', 'B'},
        'even_process': {'E', 'O'}
    }
    expected_states = expected_gt_states.get(preset, set())
    gt_coverage = len(all_gt_states & expected_states) / len(expected_states) if expected_states else 0.0

    return {
        'num_clusters_discovered': len(clusters),
        'total_histories': total_histories,
        'weighted_purity': weighted_purity,
        'gt_state_coverage': gt_coverage,
        'discovered_gt_states': sorted(list(all_gt_states)),
        'expected_gt_states': sorted(list(expected_states)),
        'cluster_analysis': cluster_analysis,
        'cache_hits': 0,
        'cache_misses': 0
    }


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description='Scaled unsupervised epsilon machine discovery')
    parser.add_argument('--preset', type=str,
                       choices=['seven_state_human', 'seven_state_human_100k',
                               'seven_state_human_large', 'seven_state_human_char_large', 'sevestateold', 'even_process'],
                       default='seven_state_human_large')
    parser.add_argument('--model_ckpt', type=str, help='Override model checkpoint path')
    parser.add_argument('--data', type=str, help='Override dataset path')
    parser.add_argument('--L', type=int, default=5, help='History length')
    parser.add_argument('--k_refine', type=int, default=4, help='k for conditional JS in Stage B')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of histories to sample')
    parser.add_argument('--sampling_strategy', type=str, choices=['random', 'emission_stratified', 'exhaustive'],
                       default='emission_stratified', help='History sampling strategy')
    parser.add_argument('--stage_a_threshold', type=float, default=0.001, help='Stage A JS threshold')
    parser.add_argument('--stage_b_threshold', type=float, default=0.001, help='Stage B conditional JS threshold')
    parser.add_argument('--emission_precision', type=int, default=2, help='Decimal places for emission signatures')
    parser.add_argument('--max_representatives', type=int, default=10, help='Max representatives per emission group')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_json', type=str, help='Save results to JSON')
    parser.add_argument('--skip_platt', action='store_true', help='Skip Platt calibration and use raw model probabilities')

    # Backward stability (now global, always on through minimal-suffix stage)
    parser.add_argument('--tolerance_bits', type=float, default=1e-3, help='JS tolerance (bits) for minimal suffix')
    parser.add_argument('--min_suffix_len', type=int, default=2, help='Minimum suffix length')

    # State remerging
    parser.add_argument('--enable_remerging', action='store_true', default=True, help='Enable Stage C remerging')
    parser.add_argument('--disable_remerging', action='store_true', help='Disable Stage C remerging')
    parser.add_argument('--remerge_emission_threshold', type=float, default=1e-4, help='Emission JS gate for remerge')
    parser.add_argument('--remerge_rollout_threshold', type=float, default=5e-4, help='Rollout JS gate for remerge')

    args = parser.parse_args()
    if args.disable_remerging:
        args.enable_remerging = False

    # Paths: resolve relative to repository root (one level up from this file)
    # Previously anchored under cssr_discovery/, which broke checkpoint discovery.
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char_50k' / 'ckpt.pt'
    default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'

    if args.preset == 'seven_state_human_100k':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char_100k' / 'ckpt.pt'
    elif args.preset == 'seven_state_human_large':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-char_large' / 'ckpt.pt'
    elif args.preset == 'seven_state_human_char_large':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char-large' / 'ckpt.pt'
    elif args.preset == 'sevestateold':
        default_model = repo_root / 'nanoGPT' / 'out-sevestateold-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'sevestateold' / 'sevestateold.dat'
    elif args.preset == 'even_process':
        default_model = repo_root / 'nanoGPT' / 'out-even-process-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'even_process' / 'even_process.dat'

    model_ckpt = Path(args.model_ckpt) if args.model_ckpt else default_model
    data_path = Path(args.data) if args.data else default_data

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from transcssr_baseline.transcssr_neural_runner import _load_nano_gpt_model, load_binary_string

    print("Scaled Unsupervised Epsilon Machine Discovery")
    print(f"Model: {model_ckpt}")
    print(f"Sampling: {args.sampling_strategy} (n={args.n_samples}, L={args.L})")
    print(f"Refinement: k={args.k_refine}, max_reps={args.max_representatives}")

    # Load model/data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, block_size = _load_nano_gpt_model(model_ckpt, device)
    s = load_binary_string(data_path)
    data = np.array([int(c) for c in s], dtype=np.int64)
    print(f"Data length: {len(data)}")

    # Calibrate (batched)
    if args.skip_platt:
        print("\nSkipping Platt calibration (--skip_platt enabled).")
        platt = None
    else:
        print("\nFitting Platt calibration...")
        platt = fit_platt_params(data, model, L_max=args.L, min_count=5)
        if platt:
            print(f"Platt params: {platt}")

    # Sample histories
    print(f"\nSampling histories using {args.sampling_strategy}...")
    if args.sampling_strategy == 'random':
        sampled_histories = random_history_sampling(data, args.L, args.n_samples, args.seed)
    elif args.sampling_strategy == 'emission_stratified':
        sampled_histories = emission_stratified_sampling(data, args.L, model, args.n_samples, platt, n_strata=5, seed=args.seed)
    elif args.sampling_strategy == 'exhaustive':
        sampled_histories = [np.array([int(c) for c in format(i, f'0{args.L}b')], dtype=np.int64) for i in range(2 ** args.L)]
        print(f"Generated all {len(sampled_histories)} possible length-{args.L} histories")
    print(f"Sampled {len(sampled_histories)} histories")

    print("\n" + "=" * 70)
    print("FAST SCALED UNSUPERVISED EPSILON MACHINE DISCOVERY")
    print("=" * 70)

    clustering_result = efficient_scaled_clustering(
        sampled_histories, model, args.k_refine, platt,
        args.stage_a_threshold, args.stage_b_threshold,
        args.emission_precision, args.max_representatives,
        args.tolerance_bits, args.min_suffix_len,
        args.enable_remerging, args.remerge_emission_threshold, args.remerge_rollout_threshold,
        seed=args.seed
    )

    print("\n" + "=" * 70)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("=" * 70)
    evaluation = evaluate_clustering_against_gt(clustering_result, args.preset)
    print(f"Discovered {evaluation['num_clusters_discovered']} epsilon machine states")
    print(f"Overall weighted purity: {evaluation['weighted_purity']:.3f}")
    print(f"Ground truth state coverage: {evaluation['gt_state_coverage']:.3f}")
    print(f"Expected GT states: {evaluation['expected_gt_states']}")
    print(f"Discovered GT states: {evaluation['discovered_gt_states']}")
    print(f"Cache performance: {evaluation['cache_hits']} hits, {evaluation['cache_misses']} misses")

    for cluster_info in evaluation['cluster_analysis']:
        cid = cluster_info['cluster_id']
        size = cluster_info['size']
        purity = cluster_info['purity']
        dominant = cluster_info['dominant_gt_state']
        emission_sig = cluster_info['emission_signature']
        gt_counts = cluster_info['gt_state_counts']
        print(f"\nCluster {cid}: {size} histories (purity={purity:.3f})")
        print(f" Emission: {emission_sig}, Dominant GT: {dominant}")
        print(f" GT distribution: {gt_counts}")
        print(f" Sample: {cluster_info['sample_histories']}")

    expected_states = {
        'seven_state_human': 7,
        'seven_state_human_100k': 7,
        'seven_state_human_large': 7,
        'seven_state_human_char_large': 7,
        'sevestateold': 7,
        'golden_mean': 2,
        'even_process': 2
    }
    expected = expected_states.get(args.preset, 'unknown')
    print(f"\nExpected: {expected} states → Discovered: {len(clustering_result.clusters)} states")

    print("\n" + "=" * 70)
    print("EPSILON MACHINE LOSS EVALUATION")
    print("=" * 70)
    loss_results = compute_epsilon_machine_loss(clustering_result, model, data, args.L, platt)
    print(f" States: {loss_results['num_states']}")
    print(f" Average loss: {loss_results['avg_loss_per_symbol_bits']:.4f} bits/symbol")
    print(f" Total predictions: {loss_results['num_predictions']}")

    if args.output_json:
        results = {
            'parameters': vars(args),
            'platt_params': platt,
            'clustering_result': {
                'num_clusters': len(clustering_result.clusters),
                'clusters': [
                    {
                        'cluster_id': i,
                        'size': len(cluster),
                        'histories': [h.tolist() for h in cluster],
                        'history_strings': [''.join(map(str, h)) for h in cluster],
                        'emission_signature': clustering_result.cluster_emission_sigs[i] if i < len(clustering_result.cluster_emission_sigs) else ""
                    }
                    for i, cluster in enumerate(clustering_result.clusters)
                ],
                'cache_hits': clustering_result.cache_hits,
                'cache_misses': clustering_result.cache_misses
            },
            'evaluation': evaluation,
            'loss_evaluation': loss_results
        }
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
