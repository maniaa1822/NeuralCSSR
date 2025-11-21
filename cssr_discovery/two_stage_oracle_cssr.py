"""
Oracle-driven two-stage CSSR implementation.

Stage 1 clusters fixed-length histories using oracle rollouts at increasing
prediction horizons. Stage 2 discovers minimal synchronizing suffixes (state
labels) directly from the clustered histories. All states therefore correspond
to histories of length L_max and short pre-sync contexts never become states.

Notes on future speedups:
 - We currently batch rollouts across histories and reuse results across k
   expansions. For long contexts or large k, a transformer KV cache could help,
   but nanoGPT here does not expose one; adding it would require a model
   refactor. The rollout/tree caching implemented below is the main lever
   without touching the model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
import time
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
NANOGPT_DIR = REPO_ROOT / "nanoGPT"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from model import GPT, GPTConfig

from calibration import fit_platt_params
from js_metrics import (
    extract_subsequences,
    get_kstep_distribution,
    get_next_token_distribution,
    js_divergence,
)


def load_binary_string(path: Path) -> str:
    """Load a binary dataset from disk, filtering to '0'/'1' characters."""
    text = Path(path).read_text()
    filtered = "".join(ch for ch in text if ch in ("0", "1"))
    if not filtered:
        raise ValueError(f"{path} is empty after filtering to binary symbols.")
    return filtered


def _load_nano_gpt_model(ckpt_path: Path, device: torch.device):
    """Load a nanoGPT checkpoint and return (model, block_size)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_args" not in ckpt or "model" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} missing 'model_args' or 'model'.")

    model_args = ckpt["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)

    # Handle DDP or torch.compile prefixes that may appear in saved state dicts.
    state_dict = {}
    for k, v in ckpt["model"].items():
        cleaned = k
        if cleaned.startswith("module."):
            cleaned = cleaned[len("module.") :]
        if cleaned.startswith("_orig_mod."):
            cleaned = cleaned[len("_orig_mod.") :]
        state_dict[cleaned] = v

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model.config.block_size


History = np.ndarray
HistoryKey = Tuple[int, ...]


@dataclass
class StageOneResult:
    partition: List[List[History]]
    representatives: List[History]
    k_values_used: List[int]
    state_of_history: Dict[HistoryKey, int]


@dataclass
class StageTwoResult:
    minimal_suffixes: List[List[Tuple[int, Tuple[int, ...]]]]
    suffix_to_states: Dict[Tuple[int, HistoryKey], set]


@dataclass
class TwoStageResult:
    stage_one: StageOneResult
    stage_two: StageTwoResult
    history_counts: Dict[HistoryKey, int]


def history_to_key(hist: History) -> HistoryKey:
    return tuple(int(x) for x in hist.tolist())


def unique_histories(histories: Sequence[History]) -> Tuple[List[History], Dict[HistoryKey, int]]:
    """Deduplicate histories while tracking their multiplicities."""
    if len(histories) == 0:
        return [], {}

    arr = np.asarray(histories)
    if arr.ndim != 2:
        # Fallback to original Python loop if we could not coerce to 2D.
        unique: Dict[HistoryKey, History] = {}
        counts: Dict[HistoryKey, int] = {}
        ordered_keys: List[HistoryKey] = []
        for hist in histories:
            key = history_to_key(hist)
            counts[key] = counts.get(key, 0) + 1
            if key not in unique:
                unique[key] = np.asarray(hist, dtype=np.int64).copy()
                ordered_keys.append(key)
        uniques = [unique[k] for k in ordered_keys]
        return uniques, counts

    L = arr.shape[1]
    # One-pass encoding to integers whenever possible for fast unique/counts.
    packed_keys: Optional[np.ndarray] = None
    if L <= 63 and np.array_equal(arr, arr.astype(bool)):
        # Pack bits into a uint64: dot with powers of two.
        powers = (1 << np.arange(L, dtype=np.uint64))
        packed_keys = (arr.astype(np.uint64) * powers).sum(axis=1)

    if packed_keys is not None:
        uniq_codes, first_idx, counts = np.unique(
            packed_keys, return_index=True, return_counts=True
        )
        uniques = [arr[idx].astype(np.int64, copy=True) for idx in first_idx]
        counts_dict: Dict[HistoryKey, int] = {}
        for code, idx, cnt in zip(uniq_codes.tolist(), first_idx.tolist(), counts.tolist()):
            key = history_to_key(arr[idx])
            counts_dict[key] = int(cnt)
        return uniques, counts_dict

    # Fallback: use tuple keys, still with numpy to avoid repeated conversions.
    unique_map: Dict[HistoryKey, History] = {}
    counts_dict: Dict[HistoryKey, int] = {}
    ordered: List[HistoryKey] = []
    for row in arr:
        key = history_to_key(row)
        counts_dict[key] = counts_dict.get(key, 0) + 1
        if key not in unique_map:
            unique_map[key] = row.astype(np.int64, copy=True)
            ordered.append(key)
    uniques = [unique_map[k] for k in ordered]
    return uniques, counts_dict


def partition_equal(a: Sequence[Sequence[History]], b: Sequence[Sequence[History]]) -> bool:
    """Compare partitions ignoring ordering inside clusters."""
    if len(a) != len(b):
        return False

    def normalize(partition: Sequence[Sequence[History]]) -> List[List[HistoryKey]]:
        normalized: List[List[HistoryKey]] = []
        for cluster in partition:
            cluster_keys = sorted(history_to_key(h) for h in cluster)
            normalized.append(cluster_keys)
        normalized.sort()
        return normalized

    return normalize(a) == normalize(b)


def cluster_histories(
    histories: Sequence[History],
    metrics_k: Sequence[int],
    tolerance: float,
    preds: Dict[int, Dict[HistoryKey, np.ndarray]],
    cluster_device: torch.device,
    cluster_dtype: torch.dtype,
    cluster_chunk_size: Optional[int] = None,
) -> StageOneResult:
    if not metrics_k:
        raise ValueError("metrics_k must contain at least one horizon.")

    keys = [history_to_key(h) for h in histories]
    # Cache stacked predictions per k for vectorized divergence
    stacked: Dict[int, torch.Tensor] = {}

    def adjacency_for(k_values: Sequence[int]) -> np.ndarray:
        """Return NxN boolean adjacency where True means JS<divergence tolerance for all k."""
        nonlocal stacked
        N = len(histories)
        device = cluster_device
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        # Helper to ensure preds are stacked on device
        def _ensure_stacked(k: int) -> torch.Tensor:
            if k not in stacked:
                arr = np.stack([preds[k][key] for key in keys], axis=0)
                stacked[k] = torch.from_numpy(arr).to(device=device, dtype=cluster_dtype)
            return torch.clamp(stacked[k], min=1e-12)

        # Fast path: no chunking, use full matrix (for small N)
        if not cluster_chunk_size or cluster_chunk_size <= 0:
            adj: Optional[torch.Tensor] = None
            for k in k_values:
                p = _ensure_stacked(k)  # (N, D)
                p_i = p[:, None, :]  # (N,1,D)
                p_j = p[None, :, :]  # (1,N,D)
                m = torch.clamp(0.5 * (p_i + p_j), min=1e-12)
                js = 0.5 * (
                    (p_i * torch.log(p_i / m)).sum(dim=-1)
                    + (p_j * torch.log(p_j / m)).sum(dim=-1)
                )
                adj_k = js < tolerance
                adj = adj_k if adj is None else (adj & adj_k)
            if adj is None:
                return np.ones((N, N), dtype=bool)
            adj = torch.tril(adj) | torch.tril(adj).T
            adj = adj | torch.eye(adj.shape[0], dtype=torch.bool, device=adj.device)
            return adj.cpu().numpy()

        # Chunked path: process blocks to limit peak memory
        chunk = int(cluster_chunk_size)
        adj_np = np.ones((N, N), dtype=bool)

        def _update_adj_for_k(p_full: torch.Tensor) -> None:
            for i in range(0, N, chunk):
                p_i = p_full[i : i + chunk]
                for j in range(i, N, chunk):
                    p_j = p_full[j : j + chunk]
                    m = torch.clamp(0.5 * (p_i[:, None, :] + p_j[None, :, :]), min=1e-12)
                    js = 0.5 * (
                        (p_i[:, None, :] * torch.log(p_i[:, None, :] / m)).sum(dim=-1)
                        + (p_j[None, :, :] * torch.log(p_j[None, :, :] / m)).sum(dim=-1)
                    )
                    block_ok = (js < tolerance).cpu().numpy()
                    i_end = i + p_i.shape[0]
                    j_end = j + p_j.shape[0]
                    adj_np[i:i_end, j:j_end] &= block_ok
                    if j != i:
                        adj_np[j:j_end, i:i_end] &= block_ok.T

        for k in k_values:
            p_full = _ensure_stacked(k)
            _update_adj_for_k(p_full)

        np.fill_diagonal(adj_np, True)
        return adj_np

    def components_from_adj(adj: np.ndarray) -> Tuple[List[List[History]], List[History]]:
        N = adj.shape[0]
        visited = [False] * N
        partition: List[List[History]] = []
        representatives: List[History] = []
        for i in range(N):
            if visited[i]:
                continue
            stack = [i]
            comp_idx: List[int] = []
            while stack:
                j = stack.pop()
                if visited[j]:
                    continue
                visited[j] = True
                comp_idx.append(j)
                neighbors = np.nonzero(adj[j])[0]
                for n in neighbors:
                    if not visited[int(n)]:
                        stack.append(int(n))
            comp_hist = [histories[idx].copy() for idx in comp_idx]
            partition.append(comp_hist)
            representatives.append(comp_hist[0])
        return partition, representatives

    k_values_used = [metrics_k[0]]
    adj = adjacency_for(k_values_used)
    partition, representatives = components_from_adj(adj)

    for k_new in metrics_k[1:]:
        trial_k_values = k_values_used + [k_new]
        adj_trial = adjacency_for(trial_k_values)
        new_partition, new_reps = components_from_adj(adj_trial)
        if partition_equal(partition, new_partition):
            break
        partition = new_partition
        representatives = new_reps
        k_values_used.append(k_new)

    state_map: Dict[HistoryKey, int] = {}
    for idx, cluster in enumerate(partition):
        for hist in cluster:
            state_map[history_to_key(hist)] = idx

    return StageOneResult(
        partition=partition,
        representatives=representatives,
        k_values_used=k_values_used,
        state_of_history=state_map,
    )


def collect_suffix_maps(
    histories: Sequence[History],
    state_map: Dict[HistoryKey, int],
) -> Dict[Tuple[int, HistoryKey], set]:
    suffix_to_states: Dict[Tuple[int, HistoryKey], set] = defaultdict(set)
    for hist in histories:
        key = history_to_key(hist)
        state = state_map[key]
        L = len(hist)
        for length in range(1, L + 1):
            suffix = tuple(int(x) for x in hist[-length:].tolist())
            suffix_to_states[(length, suffix)].add(state)
    return suffix_to_states


def minimal_suffixes_per_state(
    num_states: int,
    suffix_to_states: Dict[Tuple[int, HistoryKey], set],
) -> List[List[Tuple[int, Tuple[int, ...]]]]:
    per_state: List[List[Tuple[int, Tuple[int, ...]]]] = [[] for _ in range(num_states)]
    for (length, suffix), states in suffix_to_states.items():
        if len(states) == 1:
            state = next(iter(states))
            per_state[state].append((length, suffix))

    minimal: List[List[Tuple[int, Tuple[int, ...]]]] = [[] for _ in range(num_states)]
    for state, suffixes in enumerate(per_state):
        suffixes.sort(key=lambda item: (item[0], item[1]))
        kept: List[Tuple[int, Tuple[int, ...]]] = []
        for length, suffix in suffixes:
            is_subsumed = False
            suffix_list = list(suffix)
            for kept_len, kept_suffix in kept:
                if kept_len <= length and suffix_list[-kept_len:] == list(kept_suffix):
                    is_subsumed = True
                    break
            if not is_subsumed:
                kept.append((length, suffix))
        minimal[state] = kept
    return minimal


def stage_two(
    histories: Sequence[History],
    stage_one: StageOneResult,
) -> StageTwoResult:
    suffix_map = collect_suffix_maps(histories, stage_one.state_of_history)
    minimal = minimal_suffixes_per_state(len(stage_one.partition), suffix_map)
    return StageTwoResult(minimal_suffixes=minimal, suffix_to_states=suffix_map)


def apply_platt_scaling(p_vec: np.ndarray, platt_params: Optional[dict]) -> np.ndarray:
    if platt_params is None or 'a' not in platt_params or 'b' not in platt_params:
        return p_vec
    a = float(platt_params['a'])
    b = float(platt_params['b'])
    eps = 1e-12
    p1 = float(p_vec[1])
    p1c = max(eps, min(1.0 - eps, p1))
    logit = np.log(p1c / (1.0 - p1c))
    s = a * logit + b
    p1_adj = float(1.0 / (1.0 + np.exp(-s)))
    p1_adj = max(1e-6, min(1.0 - 1e-6, p1_adj))
    p0_adj = 1.0 - p1_adj
    return np.array([p0_adj, p1_adj], dtype=np.float64)


def precompute_predictions(
    histories: Sequence[History],
    metrics_k: Sequence[int],
    model,
    platt_params: Optional[dict],
    pred_batch_size: Optional[int],
    cache: Optional[Dict[int, Dict[HistoryKey, np.ndarray]]] = None,
    branch_topk: Optional[int] = None,
    use_kv_cache: bool = True,
) -> Dict[int, Dict[HistoryKey, np.ndarray]]:
    """Compute k-step predictions, optionally filling missing entries in a cache.

    For multiple k, we roll out once up to max(k) and reuse the branching tree, which
    avoids redundant model calls during auto-expansion. When branch_topk is set (<2),
    we only expand the top branch(es) and distribute the remaining probability mass
    uniformly over the unexpanded subtree to keep work bounded (top-k + noise).
    """

    preds = cache if cache is not None else {}
    if branch_topk is not None:
        branch_topk = max(1, int(branch_topk))
    missing_ks = sorted(set(k for k in metrics_k if k not in preds))
    if not missing_ks:
        return preds
    for k in metrics_k:
        preds.setdefault(k, {})

    max_k = max(missing_ks)
    # Initialize buffers for each missing k per history
    buffers: Dict[int, List[np.ndarray]] = {
        k: [np.zeros(1 << k, dtype=np.float64) for _ in histories] for k in missing_ks
    }

    # Helper to distribute mass for an unexpanded branch uniformly over its subtree leaves
    def spread_noise(hist_idx: int, prefix_idx: int, depth_now: int, prob_mass: float):
        for k in missing_ks:
            if k <= depth_now:
                continue
            remaining = k - depth_now
            width = 1 << remaining
            start = prefix_idx << remaining
            arr = buffers[k][hist_idx]
            arr[start:start + width] += prob_mass / width

    # Context tuple: (hist_idx, context_arr, prefix_idx, prob_mass, depth, past_key_values)
    # past_key_values is None for depth=0, and a list of tensors for depth>0
    contexts: List[Tuple[int, np.ndarray, int, float, int, Optional[List[torch.Tensor]]]] = []
    for idx, hist in enumerate(histories):
        contexts.append((idx, np.array(hist, dtype=np.int64), 0, 1.0, 0, None))

    try:
        model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        model_device = torch.device('cpu')

    ctx_val = getattr(getattr(model, 'config', None), 'block_size', None)
    ctx_len = int(ctx_val) if ctx_val is not None else None

    while contexts:
        # group by depth
        depth_now = contexts[0][4]
        batch_ctx_all = [c for c in contexts if c[4] == depth_now]
        contexts = [c for c in contexts if c[4] != depth_now]



        # To unify, we need to get `probs` for all `batch_ctx_all` first?
        # But KV cache path processes in chunks to manage memory/batch size.
        # Legacy path `get_next_token_distribution` handles batching internally.
        
        # Let's split the logic.
        
        if not use_kv_cache:
            batch = [ctx[1] for ctx in batch_ctx_all]
            probs_all = get_next_token_distribution(model, batch, platt_params=platt_params, batch_size=pred_batch_size)
            if probs_all.ndim == 1:
                probs_all = probs_all.reshape(1, -1)
            
            # Iterate and expand
            for (hist_idx, ctx_arr, prefix_idx, prob_mass, depth_item, _), p in zip(batch_ctx_all, probs_all):
                # Apply Platt (get_next_token_distribution already applies it if passed!)
                # Wait, get_next_token_distribution applies platt.
                # My new KV path applies platt manually.
                # So if use_kv_cache is False, p is already calibrated.
                # If True, I apply it.
                
                # To avoid double application, I should check.
                # get_next_token_distribution applies it.
                
                tokens = [(i, float(pi)) for i, pi in enumerate(p)]
                tokens.sort(key=lambda x: x[1], reverse=True)
                keep_n = len(tokens) if branch_topk is None else max(0, min(len(tokens), int(branch_topk)))
                keep_tokens = tokens[:keep_n]
                drop_tokens = tokens[keep_n:]

                for tok_id, tok_p in keep_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx << 1) | (tok_id % 2)

                    if next_depth in buffers:
                        buffers[next_depth][hist_idx][next_prefix] += branch_mass

                    if next_depth < max_k:
                        ctx_next = np.concatenate([ctx_arr, np.array([tok_id], dtype=np.int64)])
                        contexts.append((hist_idx, ctx_next, next_prefix, branch_mass, next_depth, None))

                for tok_id, tok_p in drop_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx << 1) | (tok_id % 2)
                    spread_noise(hist_idx, next_prefix, next_depth, branch_mass)
            
            continue # Done with this depth

        # Process in chunks to respect pred_batch_size (KV Path)
        chunk_size = pred_batch_size if pred_batch_size is not None else len(batch_ctx_all)
        
        for i in range(0, len(batch_ctx_all), chunk_size):
            batch_ctx = batch_ctx_all[i : i + chunk_size]
            
            # Prepare inputs
            input_ids_list = []
            past_kv_list = []
            
            for ctx in batch_ctx:
                hist_arr = ctx[1]
                kv = ctx[5]
                
                if kv is None:
                    # First step: full history
                    # Crop to block_size if needed
                    if ctx_len is not None and len(hist_arr) > ctx_len:
                        input_ids_list.append(hist_arr[-ctx_len:])
                    else:
                        input_ids_list.append(hist_arr)
                    past_kv_list.append(None)
                else:
                    # Subsequent steps: only the last token
                    input_ids_list.append(hist_arr[-1:])
                    past_kv_list.append(kv)

            # Stack inputs
            # input_ids_list is list of arrays. 
            # If depth=0, they might have different lengths? 
            # histories are fixed length L_max in this script (extract_subsequences).
            # So they should be same length.
            input_ids_np = np.vstack(input_ids_list).astype(np.int64)
            input_tensor = torch.from_numpy(input_ids_np).to(model_device)
            
            # Collate KV cache
            # past_kv_list is list of (List[(k, v)] or None).
            # If depth > 0, they are List[(k, v)].
            # Structure of KV: List[Layer], each Layer is (k, v) tuple.
            # k, v are (1, nh, T, hs).
            
            batched_kv = None
            if past_kv_list[0] is not None:
                # Transpose list of lists -> list of stacked tensors
                # past_kv_list: [ [L0_kv, L1_kv...], [L0_kv, L1_kv...] ... ]
                # L0_kv is (k0, v0)
                # We want: [ (cat(k0s), cat(v0s)), (cat(k1s), cat(v1s))... ]
                n_layers = len(past_kv_list[0])
                batched_kv = []
                for l in range(n_layers):
                    # layer_kvs is list of (k, v) tuples for layer l
                    layer_kvs = [item[l] for item in past_kv_list]
                    # unzip
                    ks = [kv[0] for kv in layer_kvs]
                    vs = [kv[1] for kv in layer_kvs]
                    # stack along batch dim (dim 0)
                    batched_k = torch.cat(ks, dim=0)
                    batched_v = torch.cat(vs, dim=0)
                    batched_kv.append((batched_k, batched_v))
            
            with torch.no_grad():
                logits, _, new_kv = model(input_tensor, past_key_values=batched_kv)
                # logits: (B, T_seq, V). We want last token.
                # If depth > 0, T_seq=1.
                next_token_logits = logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1).detach().cpu().numpy()

            # De-collate new_kv
            # new_kv is List[(k, v)] where k, v are (B, nh, T_new, hs)
            # We want to split back to B lists of Layers.
            # Each ctx needs its own slice.
            
            B_local = len(batch_ctx)
            new_kv_per_ctx = []
            for b in range(B_local):
                # Extract slice b for each layer
                ctx_kv = []
                for layer_idx, (k, v) in enumerate(new_kv):
                    # slice(b, b+1) keeps the dimension 0 as 1.
                    k_slice = k[b:b+1]
                    v_slice = v[b:b+1]
                    ctx_kv.append((k_slice, v_slice))
                new_kv_per_ctx.append(ctx_kv)

            if probs.shape[-1] < 2:
                probs = np.tile(np.array([[0.5, 0.5]], dtype=np.float64), (probs.shape[0], 1))

            for b, (ctx, p) in enumerate(zip(batch_ctx, probs)):
                hist_idx, ctx_arr, prefix_idx, prob_mass, depth_item, _ = ctx
                
                # Apply Platt
                p_adj = apply_platt_scaling(p, platt_params)
                
                tokens = [(i, float(pi)) for i, pi in enumerate(p_adj)]
                tokens.sort(key=lambda x: x[1], reverse=True)
                keep_n = len(tokens) if branch_topk is None else max(0, min(len(tokens), int(branch_topk)))
                keep_tokens = tokens[:keep_n]
                drop_tokens = tokens[keep_n:]

                # Distribute kept tokens
                for tok_id, tok_p in keep_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx << 1) | (tok_id % 2)

                    if next_depth in buffers:
                        buffers[next_depth][hist_idx][next_prefix] += branch_mass

                    if next_depth < max_k:
                        # extend context for next token
                        ctx_next = np.concatenate([ctx_arr, np.array([tok_id], dtype=np.int64)])
                        # Pass the NEW KV cache
                        contexts.append((hist_idx, ctx_next, next_prefix, branch_mass, next_depth, new_kv_per_ctx[b]))

                # Distribute dropped tokens as uniform noise over their subtrees
                for tok_id, tok_p in drop_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx << 1) | (tok_id % 2)  # binary index
                    spread_noise(hist_idx, next_prefix, next_depth, branch_mass)

    # Normalize and stash into preds
    for k in missing_ks:
        for hist_idx, hist in enumerate(histories):
            arr = buffers[k][hist_idx]
            s = arr.sum()
            if not np.isfinite(s) or s <= 0:
                arr = np.full_like(arr, 1.0 / len(arr))
            else:
                arr = arr / s
            preds[k][history_to_key(hist)] = arr

    print(
        f"  computed k-step distributions up to max k={max_k} for {len(histories)} histories (missing ks={missing_ks}, topk={branch_topk})",
        flush=True,
    )
    return preds


def run_two_stage(
    data: np.ndarray,
    model,
    L_max: int,
    metrics_k: Sequence[int],
    tolerance_bits: float,
    platt_params: Optional[dict] = None,
    auto_expand_k: bool = False,
    max_k: Optional[int] = None,
    branch_topk: Optional[int] = None,
    cluster_device: Optional[torch.device] = None,
    cluster_dtype: torch.dtype = torch.float32,
    max_histories: Optional[int] = None,
    cluster_chunk_size: Optional[int] = None,
    pred_batch_size: Optional[int] = 512,
    use_kv_cache: bool = True,
) -> Tuple[TwoStageResult, Dict[str, float]]:
    """
    Run the full two-stage CSSR process.
    """
    timings: Dict[str, float] = defaultdict(float)
    t0 = time.perf_counter()

    histories = extract_subsequences(data, L_max)
    timings["extract_histories"] = time.perf_counter() - t0
    print(f"Collected {len(histories)} length-{L_max} histories.")

    t0 = time.perf_counter()
    unique_hist, hist_counts = unique_histories(histories)
    timings["dedupe_histories"] = time.perf_counter() - t0
    print(f"Unique histories: {len(unique_hist)}")
    if max_histories is not None and max_histories > 0 and len(unique_hist) > max_histories:
        # Keep the most frequent histories to cap clustering size at the cost of approximation.
        sorted_items = sorted(hist_counts.items(), key=lambda kv: kv[1], reverse=True)
        keep_keys = set(k for k, _ in sorted_items[:max_histories])
        filtered_hist = []
        for h in unique_hist:
            if history_to_key(h) in keep_keys:
                filtered_hist.append(h)
        unique_hist = filtered_hist
        hist_counts = {k: v for k, v in hist_counts.items() if k in keep_keys}
        print(f"Truncated to top-{max_histories} histories by frequency (remaining {len(unique_hist)}).")

    tolerance = tolerance_bits * math.log(2.0)
    k_list: List[int] = sorted(set(metrics_k))
    if not k_list:
        raise ValueError("metrics_k must contain at least one horizon.")
    cluster_device = cluster_device or torch.device("cpu")

    k_cap = max_k if max_k is not None else L_max
    stage_one: Optional[StageOneResult] = None
    preds_cache: Dict[int, Dict[HistoryKey, np.ndarray]] = {}

    while True:
        t0 = time.perf_counter()
        preds_cache = precompute_predictions(
            unique_hist, k_list, model, platt_params, pred_batch_size, cache=preds_cache, branch_topk=branch_topk
        )
        timings["precompute_predictions"] += time.perf_counter() - t0

        print("\n=== Stage 1: oracle clustering ===")
        t0 = time.perf_counter()
        stage_one = cluster_histories(
            unique_hist,
            k_list,
            tolerance,
            preds_cache,
            cluster_device,
            cluster_dtype,
            cluster_chunk_size,
        )
        timings["stage_one_cluster"] += time.perf_counter() - t0
        print(f"k horizons used: {stage_one.k_values_used}")
        print(f"States discovered: {len(stage_one.partition)}")
        for idx, cluster in enumerate(stage_one.partition):
            example = "".join(str(int(b)) for b in cluster[0].tolist())
            print(f"  State {idx}: {len(cluster)} histories (example suffix {example})")

        if not auto_expand_k:
            break

        next_k = stage_one.k_values_used[-1] + 1
        if next_k > k_cap:
            break

        # Probe next_k to see if it creates new splits
        probe_k_list = stage_one.k_values_used + [next_k]
        print(f"Auto-expanding k to {next_k} for probe...")
        t0 = time.perf_counter()
        preds_cache = precompute_predictions(
            unique_hist, probe_k_list, model, platt_params, pred_batch_size, cache=preds_cache, branch_topk=branch_topk
        )
        timings["precompute_predictions"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        probe_stage = cluster_histories(
            unique_hist,
            probe_k_list,
            tolerance,
            preds_cache,
            cluster_device,
            cluster_dtype,
            cluster_chunk_size,
        )
        timings["stage_one_cluster"] += time.perf_counter() - t0

        if probe_stage.k_values_used == stage_one.k_values_used:
            print(f"No further splits at k={next_k}; stopping expansion.")
            break

        k_list = probe_stage.k_values_used
        stage_one = probe_stage
        print(f"Splits found at k={next_k}; continuing expansion.")

    print("\n=== Stage 2: synchronizing suffixes ===")
    t0 = time.perf_counter()
    stage_two_res = stage_two(unique_hist, stage_one)
    timings["stage_two_suffixes"] = time.perf_counter() - t0
    for idx, suffixes in enumerate(stage_two_res.minimal_suffixes):
        rendered = [f"{length}:{''.join(str(x) for x in suffix)}" for length, suffix in suffixes]
        label = rendered if rendered else ["<none>"]
        print(f"  State {idx}: minimal suffixes {label}")

    return TwoStageResult(stage_one=stage_one, stage_two=stage_two_res, history_counts=hist_counts), timings


def compute_epsilon_machine_loss(
    result: TwoStageResult,
    model,
    data: np.ndarray,
    L: int,
    platt_params: Optional[dict] = None,
    pred_batch_size: Optional[int] = None,
) -> Tuple[Dict[str, float], float]:
    """Compute negative log-likelihood of data under the discovered epsilon machine."""
    clusters = result.stage_one.partition
    if not clusters:
        return {
            "total_loss": float("inf"),
            "avg_loss_per_symbol": float("inf"),
            "avg_loss_per_symbol_bits": float("inf"),
            "num_predictions": 0,
            "num_states": 0,
        }, 0.0

    print("\n=== Computing Epsilon Machine Loss ===")

    t0 = time.perf_counter()
    state_emissions: Dict[int, np.ndarray] = {}
    suffix_lookup: Dict[int, Dict[Tuple[int, ...], int]] = defaultdict(dict)
    lengths: List[int] = []
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        repr_hist = cluster[0]
        probs = get_next_token_distribution(model, repr_hist, platt_params, batch_size=pred_batch_size)
        state_emissions[i] = probs
        print(f"State {i}: P(0)={probs[0]:.4f}, P(1)={probs[1]:.4f} (from {len(cluster)} histories)")

        for length, suffix in result.stage_two.minimal_suffixes[i]:
            if suffix not in suffix_lookup[length]:
                suffix_lookup[length][suffix] = i
                lengths.append(length)

    lengths = sorted(set(lengths), reverse=True)

    def get_state_from_context(history: np.ndarray) -> Optional[int]:
        """Map a context to discovered state using minimal synchronizing suffixes."""
        for length in lengths:
            if len(history) < length:
                continue
            key = tuple(int(x) for x in history[-length:].tolist())
            state = suffix_lookup[length].get(key)
            if state is not None:
                return state
        return None

    total_loss = 0.0
    num_predictions = 0
    print(f"Evaluating epsilon machine on {len(data)} symbols...")

    for i in range(L, len(data)):
        context = data[i - L : i]
        next_symbol = int(data[i])
        state = get_state_from_context(context)
        if state is None or state not in state_emissions:
            continue
        emission_probs = state_emissions[state]
        prob_next = float(emission_probs[next_symbol])
        if prob_next > 1e-12:
            total_loss -= math.log(prob_next)
            num_predictions += 1

    if num_predictions == 0:
        avg_loss = float("inf")
        avg_bits = float("inf")
    else:
        avg_loss = total_loss / num_predictions
        avg_bits = avg_loss / math.log(2.0)

    print(f"Total predictions: {num_predictions}")
    print(f"Average loss: {avg_loss:.4f} nats ({avg_bits:.4f} bits)")

    return {
        "total_loss": total_loss,
        "avg_loss_per_symbol": avg_loss,
        "avg_loss_per_symbol_bits": avg_bits,
        "num_predictions": num_predictions,
        "num_states": len([c for c in clusters if c]),
    }, time.perf_counter() - t0


def save_result_json(
    path: Path,
    result: TwoStageResult,
    args,
    loss_stats: Optional[Dict[str, float]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> None:
    payload = {
        "preset": args.preset,
        "L_max": args.L_max,
        "metrics_k": result.stage_one.k_values_used,
        "tolerance_bits": args.tolerance_bits,
        "num_states": len(result.stage_one.partition),
        "state_sizes": [len(cluster) for cluster in result.stage_one.partition],
        "minimal_suffixes": [
            [{"length": length, "suffix": ''.join(str(x) for x in suffix)} for length, suffix in suffixes]
            for suffixes in result.stage_two.minimal_suffixes
        ],
        "histories_considered": len(result.history_counts),
        "total_histories": sum(result.history_counts.values()),
    }
    if loss_stats is not None:
        payload["loss"] = loss_stats
    if timings is not None:
        payload["timings_sec"] = timings
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-driven two-stage CSSR.")
    parser.add_argument("--preset", type=str, default="seven_state_human_char_large")
    parser.add_argument("--model_ckpt", type=str, help="Path to nanoGPT checkpoint.")
    parser.add_argument("--data", type=str, help="Path to binary .dat file.")
    parser.add_argument("--L_max", type=int, default=5, help="History length used to define states.")
    parser.add_argument(
        "--metrics_k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Sequence of k horizons for refining the partition.",
    )
    parser.add_argument(
        "--fixed_k",
        type=int,
        help="Use a single k horizon (overrides --metrics_k).",
    )
    parser.add_argument(
        "--auto_expand_k",
        action="store_true",
        help="Incrementally increase k by 1 until no further splits (capped by --max_k or L_max).",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        help="Maximum k to consider when auto-expanding (defaults to L_max).",
    )
    parser.add_argument(
        "--branch_topk",
        type=int,
        help="Expand only the top-k next tokens per step and allocate the rest as uniform noise (binary indexing assumed). Defaults to full expansion.",
    )
    parser.add_argument(
        "--cluster_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for clustering JS computations (auto=match model device).",
    )
    parser.add_argument(
        "--cluster_dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Dtype for clustering JS computations (float32 saves memory; float64 is more precise).",
    )
    parser.add_argument(
        "--max_histories",
        type=int,
        help="Optional cap on number of unique histories to cluster (keeps most frequent).",
    )
    parser.add_argument(
        "--cluster_chunk_size",
        type=int,
        help="Optional chunk size for pairwise JS; use to avoid GPU OOM at large N (defaults to none).",
    )
    parser.add_argument(
        "--pred_batch_size",
        type=int,
        default=512,
        help="Max batch size for model forward passes when computing predictions (default: 512).",
    )
    parser.add_argument("--tolerance_bits", type=float, default=1e-3, help="JS tolerance in bits.")
    parser.add_argument(
        "--disable_platt",
        action="store_true",
        help="Disable Platt calibration before computing predictions.",
    )
    parser.add_argument(
        "--platt_sample",
        type=int,
        help="Max number of histories to fit Platt params (subsampled; defaults to all).",
    )
    parser.add_argument(
        "--platt_min_count",
        type=int,
        default=5,
        help="Minimum count per history when fitting Platt params (default: 5).",
    )
    parser.add_argument(
        "--no_platt",
        action="store_true",
        help="Alias for --disable_platt (kept for convenience).",
    )
    parser.add_argument(
        "--compute_loss",
        action="store_true",
        help="Compute epsilon-machine negative log-likelihood on the dataset.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Optional path to save JSON summary.",
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV cache (for benchmarking).",
    )
    return parser.parse_args()


def resolve_paths(preset: str, model_override: Optional[str], data_override: Optional[str]) -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "nanoGPT" / "out-seven-state-human-char-large" / "ckpt.pt"
    default_data = repo_root / "experiments" / "datasets" / "seven_state_human" / "seven_state_human.dat"

    if preset == "seven_state_human_100k":
        default_model = repo_root / "nanoGPT" / "out-seven-state-human-char_100k" / "ckpt.pt"
    elif preset == "seven_state_human_large":
        default_model = repo_root / "nanoGPT" / "out-seven-state-char_large" / "ckpt.pt"
    elif preset == "sevestateold":
        default_model = repo_root / "nanoGPT" / "out-sevestateold-char" / "ckpt.pt"
        default_data = repo_root / "experiments" / "datasets" / "sevestateold" / "sevestateold.dat"
    elif preset == "even_process":
        default_model = repo_root / "nanoGPT" / "out-even-process-char" / "ckpt.pt"
        default_data = repo_root / "experiments" / "datasets" / "even_process" / "even_process.dat"

    model_path = Path(model_override) if model_override else default_model
    data_path = Path(data_override) if data_override else default_data
    return model_path, data_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    t_start = time.perf_counter()
    timing_summary: Dict[str, float] = {}

    metrics_k = args.metrics_k
    if args.fixed_k is not None:
        if args.fixed_k <= 0:
            raise ValueError("--fixed_k must be positive.")
        metrics_k = [args.fixed_k]
    # Back-compat alias: --no_platt behaves like --disable_platt
    if getattr(args, "no_platt", False):
        args.disable_platt = True

    model_path, data_path = resolve_paths(args.preset, args.model_ckpt, args.data)
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, block_size = _load_nano_gpt_model(model_path, device)
    timing_summary["load_model"] = time.perf_counter() - t0
    if args.cluster_device == "auto":
        cluster_device = device
    elif args.cluster_device == "cuda":
        if torch.cuda.is_available():
            cluster_device = torch.device("cuda")
        else:
            print("Requested --cluster_device=cuda but CUDA is unavailable; falling back to CPU.")
            cluster_device = torch.device("cpu")
    else:
        cluster_device = torch.device("cpu")
    cluster_dtype = torch.float64 if args.cluster_dtype == "float64" else torch.float32

    dataset_str = load_binary_string(data_path)
    data = np.array([int(c) for c in dataset_str], dtype=np.int64)
    if args.L_max > block_size:
        print(f"Warning: L_max={args.L_max} exceeds model block size {block_size}. Contexts will be truncated.")

    platt_params = None
    if not args.disable_platt and not args.no_platt:
        print("Fitting Platt calibration parameters...")
        t0 = time.perf_counter()
        platt_params = fit_platt_params(
            data,
            model,
            L_max=min(args.L_max, 6),
            min_count=args.platt_min_count,
            sample_size=args.platt_sample,
        )
        timing_summary["platt_calibration"] = time.perf_counter() - t0
        if platt_params:
            print(f"  Platt parameters: a={platt_params['a']:.3f}, b={platt_params['b']:.3f}")
        else:
            print("  Could not fit Platt parameters; proceeding without calibration.")

    print(
        f"Running 2-stage CSSR with L_max={args.L_max}, k horizons={metrics_k}, tolerance={args.tolerance_bits} bits"
    )
    result, timings_two_stage = run_two_stage(
        data=data,
        model=model,
        L_max=args.L_max,
        metrics_k=metrics_k,
        tolerance_bits=args.tolerance_bits,
        platt_params=platt_params,
        auto_expand_k=args.auto_expand_k,
        max_k=args.max_k,
        branch_topk=args.branch_topk,
        cluster_device=cluster_device,
        cluster_dtype=cluster_dtype,
        max_histories=args.max_histories,
        cluster_chunk_size=args.cluster_chunk_size,
        pred_batch_size=args.pred_batch_size,
        use_kv_cache=not args.no_kv_cache,
    )
    timing_summary.update(timings_two_stage)

    loss_stats: Optional[Dict[str, float]] = None
    if args.compute_loss:
        loss_stats, t_loss = compute_epsilon_machine_loss(
            result, model, data, L=args.L_max, platt_params=platt_params, pred_batch_size=args.pred_batch_size
        )
        timing_summary["compute_loss"] = t_loss

    if args.output_json:
        save_result_json(Path(args.output_json), result, args, loss_stats, timing_summary)

    elapsed = time.perf_counter() - t_start
    timing_summary["total_runtime"] = elapsed
    print("\nTiming (s):")
    for name, val in timing_summary.items():
        print(f"  {name:20}: {val:.2f}")
    print(f"Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
