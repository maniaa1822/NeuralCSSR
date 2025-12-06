"""
Generalized Oracle-driven two-stage CSSR implementation.

Supports arbitrary alphabet sizes (base-N).
Stage 1 clusters fixed-length histories using oracle rollouts.
Stage 2 discovers minimal synchronizing suffixes.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
import time
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
REPOROOT = Path(__file__).resolve().parents[1]
if str(REPOROOT) not in sys.path:
    sys.path.insert(0, str(REPOROOT))
NANOGPT_DIR = REPOROOT / "nanoGPT"
if str(NANOGPT_DIR) not in sys.path:
    sys.path.insert(0, str(NANOGPT_DIR))

from model import GPT, GPTConfig

from js_metrics import (
    extract_subsequences,
    get_next_token_distribution,
    js_divergence,
)


def load_dataset_and_meta(data_dir: Path) -> Tuple[str, int]:
    """Load dataset string and metadata to determine alphabet size."""
    # Try to find .dat file
    dat_files = list(data_dir.glob("*.dat"))
    if not dat_files:
        raise ValueError(f"No .dat file found in {data_dir}")
    dat_path = dat_files[0]
    
    # Try to find meta.pkl
    meta_path = data_dir / "meta.pkl"
    if not meta_path.exists():
        raise ValueError(f"No meta.pkl found in {data_dir}")
        
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        
    alphabet_size = meta.get('vocab_size')
    if alphabet_size is None:
        raise ValueError("meta.pkl missing 'vocab_size'")
        
    print(f"Loaded dataset from {dat_path} (Alphabet Size: {alphabet_size})")
    return dat_path.read_text().strip(), alphabet_size


def _load_nano_gpt_model(ckpt_path: Path, device: torch.device):
    """Load a nanoGPT checkpoint and return (model, block_size)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_args" not in ckpt or "model" not in ckpt:
        raise ValueError(f"Checkpoint at {ckpt_path} missing 'model_args' or 'model'.")

    model_args = ckpt["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)

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
    """Deduplicate histories."""
    if len(histories) == 0:
        return [], {}

    arr = np.asarray(histories)
    # Generic fallback is safest for arbitrary alphabet
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
    if len(a) != len(b):
        return False
    
    def normalize(partition):
        norm = []
        for cluster in partition:
            keys = sorted(history_to_key(h) for h in cluster)
            norm.append(keys)
        norm.sort()
        return norm

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
    stacked: Dict[int, torch.Tensor] = {}

    def adjacency_for(k_values: Sequence[int]) -> np.ndarray:
        N = len(histories)
        device = cluster_device
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        def _ensure_stacked(k: int) -> torch.Tensor:
            if k not in stacked:
                arr = np.stack([preds[k][key] for key in keys], axis=0)
                stacked[k] = torch.from_numpy(arr).to(device=device, dtype=cluster_dtype)
            return torch.clamp(stacked[k], min=1e-12)

        chunk = int(cluster_chunk_size) if cluster_chunk_size and cluster_chunk_size > 0 else N
        adj_np = np.ones((N, N), dtype=bool)

        def _update_adj_for_k(p_full: torch.Tensor) -> None:
            # Full or chunked self-similarity
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
            # Only use chunking if explicitly requested, otherwise compute all at once
            if chunk == N:
                 p_i = p_full[:, None, :]
                 p_j = p_full[None, :, :]
                 m = torch.clamp(0.5 * (p_i + p_j), min=1e-12)
                 js = 0.5 * (
                    (p_i * torch.log(p_i / m)).sum(dim=-1)
                    + (p_j * torch.log(p_j / m)).sum(dim=-1)
                 )
                 adj_k = (js < tolerance).cpu().numpy()
                 adj_np &= adj_k
            else:    
                _update_adj_for_k(p_full)

        np.fill_diagonal(adj_np, True)
        return adj_np

    def components_from_adj(adj: np.ndarray) -> Tuple[List[List[History]], List[History]]:
        N = adj.shape[0]
        visited = [False] * N
        partition: List[List[History]] = []
        representatives: List[History] = []
        for i in range(N):
            if visited[i]: continue
            stack = [i]
            comp_idx = []
            while stack:
                j = stack.pop()
                if visited[j]: continue
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
        kept = []
        for length, suffix in suffixes:
            is_subsumed = False
            suffix_list = list(suffix)
            for kept_len, kept_suffix in kept:
                # If existing kept suffix is a suffix of this new one, then new one is redundant
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


def precompute_predictions(
    histories: Sequence[History],
    metrics_k: Sequence[int],
    model,
    alphabet_size: int,
    pred_batch_size: Optional[int],
    cache: Optional[Dict[int, Dict[HistoryKey, np.ndarray]]] = None,
    branch_topk: Optional[int] = None,
    branch_nucleus: Optional[float] = None,
    use_kv_cache: bool = True,
) -> Dict[int, Dict[HistoryKey, np.ndarray]]:
    
    preds = cache if cache is not None else {}
    if branch_topk is not None:
        branch_topk = max(1, int(branch_topk))
    if branch_nucleus is not None:
        branch_nucleus = max(0.0, min(1.0, float(branch_nucleus)))
    
    missing_ks = sorted(set(k for k in metrics_k if k not in preds))
    if not missing_ks:
        return preds
    for k in metrics_k:
        preds.setdefault(k, {})

    max_k = max(missing_ks)
    
    # Initialize buffers: size = alphabet_size**k
    buffers: Dict[int, List[np.ndarray]] = {
        k: [np.zeros(alphabet_size ** k, dtype=np.float64) for _ in histories] for k in missing_ks
    }

    def spread_noise(hist_idx: int, prefix_idx: int, depth_now: int, prob_mass: float):
        for k in missing_ks:
            if k <= depth_now: continue
            remaining = k - depth_now
            width = alphabet_size ** remaining
            start = prefix_idx * width # Base-N indexing
            arr = buffers[k][hist_idx]
            arr[start:start + width] += prob_mass / width

    # Tuple: (hist_idx, context_arr, prefix_idx, prob_mass, depth, past_key_values)
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
        depth_now = contexts[0][4]
        batch_ctx_all = [c for c in contexts if c[4] == depth_now]
        contexts = [c for c in contexts if c[4] != depth_now]

        if not use_kv_cache:
            # Fallback for no cache (slower)
            batch = [ctx[1] for ctx in batch_ctx_all]
            probs_all = get_next_token_distribution(model, batch, platt_params=None, batch_size=pred_batch_size)
            if probs_all.ndim == 1: probs_all = probs_all.reshape(1, -1)
            
            for (hist_idx, ctx_arr, prefix_idx, prob_mass, depth_item, _), p in zip(batch_ctx_all, probs_all):
                tokens = [(i, float(pi)) for i, pi in enumerate(p)]
                tokens.sort(key=lambda x: x[1], reverse=True)
                
                # Determine how many tokens to keep (Intersection of Top-K and Nucleus)
                keep_count = len(tokens)
                if branch_topk is not None:
                    keep_count = min(keep_count, int(branch_topk))
                
                if branch_nucleus is not None:
                    cumsum = 0.0
                    nucleus_count = 0
                    for _, p_val in tokens:
                        cumsum += p_val
                        nucleus_count += 1
                        if cumsum >= branch_nucleus:
                            break
                    keep_count = min(keep_count, nucleus_count)

                # Ensure at least 1 token is kept
                keep_count = max(1, keep_count)

                keep_tokens = tokens[:keep_count]
                drop_tokens = tokens[keep_count:]

                for tok_id, tok_p in keep_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx * alphabet_size) + tok_id

                    if next_depth in buffers:
                        buffers[next_depth][hist_idx][next_prefix] += branch_mass

                    if next_depth < max_k:
                        ctx_next = np.concatenate([ctx_arr, np.array([tok_id], dtype=np.int64)])
                        contexts.append((hist_idx, ctx_next, next_prefix, branch_mass, next_depth, None))

                for tok_id, tok_p in drop_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx * alphabet_size) + tok_id
                    spread_noise(hist_idx, next_prefix, next_depth, branch_mass)
            continue

        # Chunked processing
        chunk_size = pred_batch_size if pred_batch_size is not None else len(batch_ctx_all)
        for i in range(0, len(batch_ctx_all), chunk_size):
            batch_ctx = batch_ctx_all[i : i + chunk_size]
            input_ids_list = []
            past_kv_list = []
            
            for ctx in batch_ctx:
                hist_arr = ctx[1]
                kv = ctx[5]
                if kv is None:
                    if ctx_len is not None and len(hist_arr) > ctx_len:
                        input_ids_list.append(hist_arr[-ctx_len:])
                    else:
                        input_ids_list.append(hist_arr)
                    past_kv_list.append(None)
                else:
                    input_ids_list.append(hist_arr[-1:])
                    past_kv_list.append(kv)

            input_ids_np = np.vstack(input_ids_list).astype(np.int64)
            input_tensor = torch.from_numpy(input_ids_np).to(model_device)
            
            batched_kv = None
            if past_kv_list[0] is not None:
                n_layers = len(past_kv_list[0])
                batched_kv = []
                for l in range(n_layers):
                    layer_kvs = [item[l] for item in past_kv_list]
                    ks = [kv[0] for kv in layer_kvs]
                    vs = [kv[1] for kv in layer_kvs]
                    batched_k = torch.cat(ks, dim=0)
                    batched_v = torch.cat(vs, dim=0)
                    batched_kv.append((batched_k, batched_v))
            
            with torch.no_grad():
                logits, _, new_kv = model(input_tensor, past_key_values=batched_kv)
                next_token_logits = logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1).detach().cpu().numpy()

            B_local = len(batch_ctx)
            new_kv_per_ctx = []
            for b in range(B_local):
                ctx_kv = []
                for layer_idx, (k, v) in enumerate(new_kv):
                    k_slice = k[b:b+1]
                    v_slice = v[b:b+1]
                    ctx_kv.append((k_slice, v_slice))
                new_kv_per_ctx.append(ctx_kv)

            # Pad probabilities if smaller than alphabet_size (e.g. if model output size mismatch)
            # But normally we trust model. 
            
            for b, (ctx, p) in enumerate(zip(batch_ctx, probs)):
                hist_idx, ctx_arr, prefix_idx, prob_mass, depth_item, _ = ctx
                
                # Filter to valid alphabet if model is larger (optional safety)
                p = p[:alphabet_size]
                p = p / p.sum() # Normalize
                
                tokens = [(i, float(pi)) for i, pi in enumerate(p)]
                tokens.sort(key=lambda x: x[1], reverse=True)
                
                # Determine how many tokens to keep (Intersection of Top-K and Nucleus)
                keep_count = len(tokens)
                if branch_topk is not None:
                    keep_count = min(keep_count, int(branch_topk))
                
                if branch_nucleus is not None:
                    cumsum = 0.0
                    nucleus_count = 0
                    for _, p_val in tokens:
                        cumsum += p_val
                        nucleus_count += 1
                        if cumsum >= branch_nucleus:
                            break
                    keep_count = min(keep_count, nucleus_count)

                # Ensure at least 1 token is kept
                keep_count = max(1, keep_count)

                keep_tokens = tokens[:keep_count]
                drop_tokens = tokens[keep_count:]

                for tok_id, tok_p in keep_tokens:
                    # branch_mass = prob_mass * p_transition
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    # Base-N Indexing
                    next_prefix = (prefix_idx * alphabet_size) + tok_id

                    if next_depth in buffers:
                        buffers[next_depth][hist_idx][next_prefix] += branch_mass

                    if next_depth < max_k:
                        ctx_next = np.concatenate([ctx_arr, np.array([tok_id], dtype=np.int64)])
                        contexts.append((hist_idx, ctx_next, next_prefix, branch_mass, next_depth, new_kv_per_ctx[b]))

                for tok_id, tok_p in drop_tokens:
                    branch_mass = prob_mass * max(tok_p, 1e-12)
                    next_depth = depth_item + 1
                    next_prefix = (prefix_idx * alphabet_size) + tok_id
                    spread_noise(hist_idx, next_prefix, next_depth, branch_mass)

    for k in missing_ks:
        for hist_idx, hist in enumerate(histories):
            arr = buffers[k][hist_idx]
            s = arr.sum()
            if not np.isfinite(s) or s <= 0:
                arr = np.full_like(arr, 1.0 / len(arr))
            else:
                arr = arr / s
            preds[k][history_to_key(hist)] = arr

    return preds


def run_two_stage(
    data: str,
    alphabet_size: int,
    model,
    L_max: int,
    metrics_k: Sequence[int],
    tolerance_bits: float,
    auto_expand_k: bool = False,
    max_k: Optional[int] = None,
    branch_topk: Optional[int] = None,
    branch_nucleus: Optional[float] = None,
    cluster_device: Optional[torch.device] = None,
    pred_batch_size: Optional[int] = 512,
    use_kv_cache: bool = True,
) -> TwoStageResult:
    
    # Convert string data to int array
    # Assume '0'...'N-1' chars or map them?
    # The prepare script mapped chars to ints.
    # Generalized version should ideally take int tokens or char string.
    # For now, let's assume `data` is a string of chars that map to ints directly or via meta?
    # Actually, simpler to pass pre-tokenized numpy array if we want generality.
    # But for CLI usage, we often load text.
    # Let's support digits 0-9 as direct ints.
    
    # Actually, let's just make it robust.
    token_list = []
    for ch in data:
        if ch.isdigit():
             token_list.append(int(ch))
        else:
             # Just map to code point for now? or error?
             # Butterfly is 0-7.
             pass
    data_arr = np.array(token_list, dtype=np.int64)

    print(f"Tokenized data length: {len(data_arr)}")
    
    histories = extract_subsequences(data_arr, L_max)
    print(f"Collected {len(histories)} length-{L_max} histories.")

    unique_hist, hist_counts = unique_histories(histories)
    print(f"Unique histories: {len(unique_hist)}")

    tolerance = tolerance_bits * math.log(2.0)
    k_list: List[int] = sorted(set(metrics_k))
    cluster_device = cluster_device or torch.device("cpu")
    k_cap = max_k if max_k is not None else L_max
    
    stage_one: Optional[StageOneResult] = None
    preds_cache: Dict[int, Dict[HistoryKey, np.ndarray]] = {}

    while True:
        preds_cache = precompute_predictions(
            unique_hist, k_list, model, alphabet_size, pred_batch_size, cache=preds_cache, branch_topk=branch_topk, branch_nucleus=branch_nucleus, use_kv_cache=use_kv_cache
        )

        print("\n=== Stage 1: oracle clustering ===")
        stage_one = cluster_histories(
            unique_hist, k_list, tolerance, preds_cache, cluster_device, torch.float32
        )
        print(f"k horizons used: {stage_one.k_values_used}")
        print(f"States discovered: {len(stage_one.partition)}")
        for idx, cluster in enumerate(stage_one.partition):
            # ex = "".join(str(x) for x in cluster[0])
            print(f"  State {idx}: {len(cluster)} histories")

        if not auto_expand_k:
            break
        
        next_k = stage_one.k_values_used[-1] + 1
        if next_k > k_cap:
            break
            
        # Probe
        probe_k_list = stage_one.k_values_used + [next_k]
        print(f"Auto-expanding k to {next_k}...")
        preds_cache = precompute_predictions(
            unique_hist, probe_k_list, model, alphabet_size, pred_batch_size, cache=preds_cache, branch_topk=branch_topk, branch_nucleus=branch_nucleus, use_kv_cache=use_kv_cache
        )
        probe_stage = cluster_histories(
            unique_hist, probe_k_list, tolerance, preds_cache, cluster_device, torch.float32
        )
        
        if probe_stage.k_values_used == stage_one.k_values_used:
             print("No further splits.")
             break
        
        stage_one = probe_stage
        k_list = probe_stage.k_values_used

    print("\n=== Stage 2: synchronizing suffixes ===")
    stage_two_res = stage_two(unique_hist, stage_one)
    for idx, suffixes in enumerate(stage_two_res.minimal_suffixes):
        rendered = [f"len{length}:{''.join(str(x) for x in suffix)}" for length, suffix in suffixes]
        print(f"  State {idx}: {rendered}")

    return TwoStageResult(stage_one, stage_two_res, hist_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized 2-Stage CSSR")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to simple model ckpt")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing .dat and meta.pkl")
    parser.add_argument("--L", type=int, default=4, help="History length")
    parser.add_argument("--k", type=int, nargs="+", default=[1], help="Prediction horizons")
    parser.add_argument("--tol", type=float, default=0.05, help="JS divergence tolerance (bits)")
    parser.add_argument("--nucleus", type=float, default=None, help="Nucleus sampling probability (e.g. 0.95)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data & Meta
    data_str, alphabet_size = load_dataset_and_meta(Path(args.dataset_dir))
    
    # Load Model
    model, block_size = _load_nano_gpt_model(Path(args.ckpt), device)
    print(f"Model loaded (block_size={block_size})")

    run_two_stage(
        data=data_str,
        alphabet_size=alphabet_size,
        model=model,
        L_max=args.L,
        metrics_k=args.k,
        tolerance_bits=args.tol,
        auto_expand_k=True,
        max_k=3,
        branch_topk=4,
        branch_nucleus=args.nucleus,
        cluster_device=device
    )
