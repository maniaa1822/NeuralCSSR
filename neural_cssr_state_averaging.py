#!/usr/bin/env python3
"""
neural_cssr_single_file.py

Single-file implementation of Neural CSSR (CSSR with a neural probability provider).

Features:
- Follows classical CSSR procedure but replaces empirical next-step counts with
  transformer/model probability estimates.
- Implements explicit state-averaging (p_state) as in the Neural CSSR design.
- Uses a G-test (likelihood ratio test) or chi-square test on pseudo-counts
  constructed from model probabilities and optional empirical counts.
- Supports Platt / temperature calibration and an option to mix empirical counts
  with model probabilities.
- Optional self-supervised refinement loop: sample synthetic trajectories from
  the discovered epsilon-machine and fine-tune the model (hook points provided).

Usage (example):
  python neural_cssr_single_file.py --data path/to/data.dat --model_ckpt path/to/model.pt \
    --L_max 8 --alpha 0.01 --backend model --pseudo_count_scale 100

Dependencies:
- python >= 3.8
- torch
- numpy
- scipy

This file intentionally keeps the neural model interface minimal: the script
expects a PyTorch `model` with either:
  - method `generate_probabilities(x_tensor)` returning probabilities [B, T, V]
  - or `model(x_tensor)` returning logits ([B,T,V] or [B,V])

Tune the `model_next_probs` function if your model API differs.

Author: generated for user request
"""

import argparse
import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import chi2
import sys

# ----------------------------- Utility functions -----------------------------

def load_binary_tokens(dat_path: Path) -> List[int]:
    s = dat_path.read_text().strip()
    toks = [int(c) for c in s if c in "01"]
    if not toks:
        raise ValueError(f"No binary tokens in {dat_path}")
    return toks


def make_history_string(seq: List[int]) -> str:
    return ''.join(str(x) for x in seq)


# ----------------------------- Model wrapper -------------------------------

def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    """Load EBM/AR binary LM from checkpoint using experiments.ebm.models definitions.

    Returns (model, context_window, model_type).
    Mirrors probability_analysis/compare_model_empirical.py behavior.
    """
    # Ensure repository root on path for experiments.ebm import
    repo_root = Path(__file__).resolve().parents[0]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from experiments.ebm.models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM  # type: ignore
    except Exception:
        # Fallback to direct directory
        ebm_dir = repo_root / 'experiments' / 'ebm'
        if str(ebm_dir) not in sys.path:
            sys.path.insert(0, str(ebm_dir))
        from models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM  # type: ignore

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: Dict = ckpt.get('config', {})
    model_type = cfg.get('model_type', 'ar_binary')  # Variable context model is AR
    context_window = int(cfg.get('context_window', cfg.get('max_len', 32)))

    if model_type == 'ar_binary':
        model = AutoRegressiveBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(cfg.get('d_model', 128)),
            nhead=int(cfg.get('heads', 8)),
            num_layers=int(cfg.get('layers', 4)),
            max_len=context_window,
            dropout=float(cfg.get('dropout', 0.0)),
        ).to(device)
    else:
        model = EnergyBasedBinaryLM(
            vocab_size=3,
            output_vocab_size=2,
            d_model=int(cfg.get('d_model', 128)),
            nhead=int(cfg.get('heads', 8)),
            num_layers=int(cfg.get('layers', 4)),
            max_len=context_window,
            dropout=float(cfg.get('dropout', 0.0)),
        ).to(device)

    if 'state_dict' not in ckpt:
        raise RuntimeError('Checkpoint missing state_dict. Please provide EBM/AR checkpoint with config and state_dict.')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, context_window, model_type

class ModelProvider:
    """Wraps a PyTorch model to provide P(next | history) with options.

    The model must accept integer token tensors and return either logits or
    probabilities. We support two common paths:
      - model.generate_probabilities(x) -> probs [B,T,V]
      - model(x) -> logits ([B,T,V] or [B,V])

    The wrapper also supports temperature scaling, clipping, and optional
    Platt calibration (sigmoid(a*logit + b) applied to logit(p1)).
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, context_window: int = 64,
                 temperature: float = 1.0, prob_clip: float = 1e-6, platt_params: Optional[Dict] = None):
        self.model = model
        self.device = device
        self.context_window = context_window
        self.temperature = float(temperature)
        self.prob_clip = float(prob_clip)
        self.platt_params = platt_params
        self.model.eval()

    @torch.no_grad()
    def next_probs(self, history_ids: List[int]) -> Tuple[float, float]:
        """Return (p0,p1) for the next token given history_ids (list of ints).

        History is truncated to the rightmost self.context_window tokens.
        """
        if not history_ids:
            return 0.5, 0.5
        ids = history_ids[-self.context_window:]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Path 1: generate_probabilities
        if hasattr(self.model, 'generate_probabilities'):
            probs = self.model.generate_probabilities(x)  # [B, T, V] or [B,1,V]
            if probs.dim() == 3:
                probs = probs[:, -1, :]
            probs = probs.squeeze(0).cpu().numpy()
            p0, p1 = float(probs[0]), float(probs[1])
        else:
            out = self.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # logits may be [B,T,V] or [B,V]
            if out.dim() == 3:
                logits = out[:, -1, :]
            else:
                logits = out
            logits = logits.squeeze(0).detach()
            if self.temperature != 1.0:
                logits = logits / float(self.temperature)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            p0, p1 = float(probs[0]), float(probs[1])

        # Platt calibration on p1 (optional)
        if self.platt_params is not None:
            a = float(self.platt_params.get('a', 1.0))
            b = float(self.platt_params.get('b', 0.0))
            # avoid extreme values
            eps = 1e-12
            p1c = max(eps, min(1.0 - eps, p1))
            logit = math.log(p1c / (1.0 - p1c))
            s = a * logit + b
            p1_new = 1.0 / (1.0 + math.exp(-s))
            p1 = float(max(self.prob_clip, min(1.0 - self.prob_clip, p1_new)))
            p0 = 1.0 - p1

        # clip
        p0 = max(self.prob_clip, min(1.0 - self.prob_clip, p0))
        p1 = max(self.prob_clip, min(1.0 - self.prob_clip, p1))
        s = p0 + p1
        return p0 / s, p1 / s


# ----------------------------- Statistical tests ---------------------------

def g_test_pvalue(counts_a: np.ndarray, counts_b: np.ndarray, eps: float = 1e-12) -> float:
    """G-test (likelihood ratio) for two discrete distributions given counts.

    counts_a and counts_b are arrays of nonnegative counts for k categories.
    We compute G = 2 * sum( a_i * log(a_i / m_i) + b_i * log(b_i / m_i) )
    where m_i = (a_i + b_i)/2. Under H0, G ~ chi2(df=k-1).
    Returns p-value.
    """
    a = counts_a.astype(float)
    b = counts_b.astype(float)
    m = (a + b) / 2.0
    # avoid zeros
    a_safe = np.maximum(a, eps)
    b_safe = np.maximum(b, eps)
    m_safe = np.maximum(m, eps)
    term = 2.0 * (np.sum(a_safe * np.log(a_safe / m_safe)) + np.sum(b_safe * np.log(b_safe / m_safe)))
    df = max(1, len(a) - 1)
    p = 1.0 - chi2.cdf(term, df)
    return float(max(0.0, min(1.0, p)))


def chi2_test_pvalue(counts_a: np.ndarray, counts_b: np.ndarray, eps: float = 1e-12) -> float:
    """Pearson chi-square test for two binned distributions (counts_a vs counts_b).

    We use the standard statistic: sum((a - e)^2 / e) where e are expected counts.
    For two-sample test use pooled proportions. This function returns p-value.
    """
    a = counts_a.astype(float)
    b = counts_b.astype(float)
    n_a = a.sum(); n_b = b.sum()
    if n_a <= 0 or n_b <= 0:
        return 1.0
    p_pool = (a + b) / (n_a + n_b)
    # expected
    exp_a = n_a * p_pool
    exp_b = n_b * p_pool
    # avoid zero expected
    mask = exp_a > 0
    chi = 0.0
    chi += np.sum((a[mask] - exp_a[mask]) ** 2 / exp_a[mask])
    chi += np.sum((b[mask] - exp_b[mask]) ** 2 / exp_b[mask])
    df = max(1, len(a) - 1)
    p = 1.0 - chi2.cdf(chi, df)
    return float(max(0.0, min(1.0, p)))


def test_same_distribution(p_ext: List[float], p_state: List[float], method: str = 'g',
                           effective_count_ext: float = 10.0, effective_count_state: float = 100.0,
                           alpha: float = 0.01) -> Tuple[bool, float]:
    """Test whether two probability vectors are the same using pseudo-counts.

    We convert probabilities to pseudo-counts by multiplying by effective counts.
    Returns (same_bool, p_value).
    """
    p_ext = np.array(p_ext, dtype=float)
    p_state = np.array(p_state, dtype=float)
    counts_ext = p_ext * float(effective_count_ext)
    counts_state = p_state * float(effective_count_state)
    if method == 'g':
        pval = g_test_pvalue(counts_ext, counts_state)
    else:
        pval = chi2_test_pvalue(counts_ext, counts_state)
    # same if pval > alpha (can't reject H0 that they're equal)
    return (pval > alpha), pval


# ----------------------------- CSSR machinery -------------------------------

class NeuralCSSR:
    """Implements Neural CSSR with explicit p_state averaging.

    Public attributes after run:
      - states: list of dicts; each has 'histories' set and 'vec' distribution and 'weight'
      - epsilon_map: dict mapping (L_history) tuples to state id
    """

    def __init__(self, tokens: List[int], provider: ModelProvider, L_max: int = 8,
                 alpha: float = 0.01, test_method: str = 'g', pseudo_count_scale: float = 100.0,
                 min_count: int = 1, mix_empirical: float = 0.0, backend: str = 'neural'):
        self.tokens = tokens
        self.provider = provider
        self.L_max = L_max
        self.alpha = alpha
        self.test_method = test_method
        self.states: List[Dict] = []
        self.pseudo_count_scale = float(pseudo_count_scale)
        self.min_count = int(min_count)
        self.mix_empirical = float(mix_empirical)
        self.backend = backend

    def observed_history_counts(self, L: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in range(L, len(self.tokens)):
            h = ''.join(str(x) for x in self.tokens[t-L:t])
            counts[h] = counts.get(h, 0) + 1
        return counts

    def history_occurrences(self, history: str) -> int:
        L = len(history)
        if L == 0:
            return max(1, len(self.tokens))
        cnt = 0
        for t in range(L, len(self.tokens)):
            h = ''.join(str(x) for x in self.tokens[t-L:t])
            if h == history:
                cnt += 1
        return cnt

    def compute_state_vector(self, histories: List[str]) -> List[float]:
        # counts-weighted average morph of member histories
        num = np.zeros(2, dtype=float)
        den = 0.0
        for h in histories:
            if self.backend == 'empirical':
                emp = self.empirical_morph(h)
                if emp is None:
                    p = [0.5, 0.5]
                else:
                    p = emp
            else:
                ids = [int(c) for c in h]
                p0, p1 = self.provider.next_probs(ids)
                p = [p0, p1]
            c = self.history_occurrences(h)
            num += c * np.array(p)
            den += c
        if den > 0:
            acc = num / den
        elif len(histories) > 0:
            acc = num / max(1.0, float(len(histories)))
        else:
            acc = np.array([0.5, 0.5])
        return [float(acc[0]), float(acc[1])]

    def empirical_morph(self, history: str) -> Optional[List[float]]:
        # compute empirical next-symbol distribution for a history if count >= min_count
        cnt = 0
        ones = 0
        L = len(history)
        for t in range(L, len(self.tokens)):
            h = ''.join(str(x) for x in self.tokens[t-L:t])
            if h == history:
                cnt += 1
                if self.tokens[t] == 1:
                    ones += 1
        if cnt < self.min_count:
            return None
        return [float(cnt - ones) / cnt, float(ones) / cnt]

    def empirical_counts(self, history: str) -> Optional[List[int]]:
        # counts of next 0 and 1 following history
        cnt0 = 0
        cnt1 = 0
        L = len(history)
        for t in range(L, len(self.tokens)):
            h = ''.join(str(x) for x in self.tokens[t-L:t])
            if h == history:
                if self.tokens[t] == 1:
                    cnt1 += 1
                else:
                    cnt0 += 1
        if (cnt0 + cnt1) == 0:
            return None
        return [cnt0, cnt1]

    def run(self) -> Dict:
        # Initialize states with null history and its neural morph
        self.states = []
        if self.backend == 'empirical':
            p_empty = [0.5, 0.5]
        else:
            p_empty = list(self.provider.next_probs([]))
        state0 = {'histories': set(['']), 'vec': p_empty[:], 'weight': 0, '_centroid_count': 0}
        if self.backend == 'empirical':
            state0['counts'] = [0, 0]
        self.states.append(state0)

        # Epsilon mapping (history -> state id)
        epsilon: Dict[str, int] = {'': 0}

        # Phase II: grow histories by increasing L, extending each state's histories
        for L in range(1, self.L_max + 1):
            hist_counts_L = self.observed_history_counts(L)
            # Snapshot of current states to iterate over when extending
            state_indices = list(range(len(self.states)))
            for sid in state_indices:
                st = self.states[sid]
                # Compute current state's morph vector
                if len(st['histories']) > 0:
                    st['vec'] = self.compute_state_vector(list(st['histories']))
                p_state = st['vec']
                # Extend only histories of length L-1
                for h in list(st['histories']):
                    if len(h) != L - 1:
                        continue
                    # Track whether any child stays in the parent state
                    any_child_in_parent = False
                    for a in ['0', '1']:
                        new_hist = a + h
                        if self.backend == 'empirical':
                            # Only skip if never observed
                            if hist_counts_L.get(new_hist, 0) == 0:
                                continue
                        else:
                            if hist_counts_L.get(new_hist, 0) < self.min_count:
                                continue
                        if new_hist in epsilon:
                            continue
                        # Probability for the extended history
                        if self.backend == 'empirical':
                            counts_ext_list = self.empirical_counts(new_hist)
                            if counts_ext_list is None:
                                p_ext = [0.5, 0.5]
                            else:
                                s = float(sum(counts_ext_list))
                                p_ext = [counts_ext_list[0]/s, counts_ext_list[1]/s]
                        else:
                            ids = [int(c) for c in new_hist][-self.provider.context_window:]
                            p_ext = list(self.provider.next_probs(ids))
                            # Optional empirical mix
                            if self.mix_empirical > 0.0:
                                emp = self.empirical_morph(new_hist)
                                if emp is not None:
                                    p_ext = [
                                        (1.0 - self.mix_empirical) * p_ext[i] + self.mix_empirical * emp[i]
                                        for i in range(2)
                                    ]
                        cnt = float(hist_counts_L.get(new_hist, 1))
                        # First test against parent state
                        if self.backend == 'empirical':
                            # Compare raw counts to state's accumulated counts
                            counts_state = np.array(st.get('counts', [0,0]), dtype=float)
                            counts_ext = np.array(counts_ext_list if counts_ext_list is not None else [1,1], dtype=float)
                            pval_parent = g_test_pvalue(counts_ext, counts_state) if self.test_method == 'g' else chi2_test_pvalue(counts_ext, counts_state)
                            same_parent = pval_parent > self.alpha
                        else:
                            eff_state = max(self.pseudo_count_scale, st.get('weight', 1))
                            eff_ext = max(1.0, min(cnt, self.pseudo_count_scale))
                            same_parent, pval_parent = test_same_distribution(
                                p_ext, p_state, method=self.test_method,
                                effective_count_ext=eff_ext, effective_count_state=eff_state, alpha=self.alpha
                            )
                        assign_sid = None
                        best_pval = -1.0
                        if same_parent and pval_parent > best_pval:
                            assign_sid = sid
                            best_pval = pval_parent
                        else:
                            # Try alternate states
                            for sid2, st2 in enumerate(self.states):
                                if sid2 == sid:
                                    continue
                                if self.backend == 'empirical':
                                    counts_state2 = np.array(st2.get('counts', [0,0]), dtype=float)
                                    pval2 = g_test_pvalue(counts_ext, counts_state2) if self.test_method == 'g' else chi2_test_pvalue(counts_ext, counts_state2)
                                    same2 = pval2 > self.alpha
                                else:
                                    p_state2 = st2['vec']
                                    eff_state2 = max(self.pseudo_count_scale, st2.get('weight', 1))
                                    same2, pval2 = test_same_distribution(
                                        p_ext, p_state2, method=self.test_method,
                                        effective_count_ext=eff_ext, effective_count_state=eff_state2, alpha=self.alpha
                                    )
                                if same2 and pval2 > best_pval:
                                    assign_sid = sid2
                                    best_pval = pval2
                        if assign_sid is None:
                            # Create a new state for this history
                            new_state = {
                                'histories': set([new_hist]),
                                'vec': p_ext[:],
                                'weight': int(cnt),
                                '_centroid_count': int(cnt),
                            }
                            if self.backend == 'empirical':
                                new_state['counts'] = counts_ext_list if counts_ext_list is not None else [0,0]
                            self.states.append(new_state)
                            epsilon[new_hist] = len(self.states) - 1
                        else:
                            # Assign to existing state
                            self.states[assign_sid]['histories'].add(new_hist)
                            self.states[assign_sid]['weight'] = self.states[assign_sid].get('weight', 0) + int(cnt)
                            old_w = self.states[assign_sid].get('_centroid_count', 0)
                            if old_w == 0:
                                self.states[assign_sid]['vec'] = p_ext[:]
                                self.states[assign_sid]['_centroid_count'] = int(cnt)
                            else:
                                w_old = float(old_w)
                                w_new = w_old + float(cnt)
                                self.states[assign_sid]['vec'] = [
                                    (w_old * self.states[assign_sid]['vec'][i] + cnt * p_ext[i]) / w_new
                                    for i in range(2)
                                ]
                                self.states[assign_sid]['_centroid_count'] = int(w_new)
                            if self.backend == 'empirical' and counts_ext_list is not None:
                                cs = self.states[assign_sid].get('counts', [0,0])
                                self.states[assign_sid]['counts'] = [cs[0] + counts_ext_list[0], cs[1] + counts_ext_list[1]]
                            epsilon[new_hist] = assign_sid
                            if assign_sid == sid:
                                any_child_in_parent = True
                    # If all generated histories belong to different states than the parent, remove parent
                    if not any_child_in_parent:
                        if h in st['histories']:
                            st['histories'].remove(h)
                            if self.backend == 'empirical':
                                counts_h = self.empirical_counts(h)
                                if counts_h is not None:
                                    cs = st.get('counts', [0,0])
                                    st['counts'] = [max(0, cs[0] - counts_h[0]), max(0, cs[1] - counts_h[1])]
                            if h in epsilon:
                                epsilon.pop(h)

        # Finalize state vectors and clean helper keys
        for st in self.states:
            if isinstance(st.get('histories'), set) and len(st['histories']) > 0:
                if self.backend == 'empirical':
                    cs = st.get('counts', [0,0])
                    s = float(cs[0] + cs[1])
                    st['vec'] = [cs[0]/s if s>0 else 0.5, cs[1]/s if s>0 else 0.5]
                else:
                    st['vec'] = self.compute_state_vector(list(st['histories']))
            st.pop('_centroid_count', None)

        # Helper: compute extension with shift if needed
        def extend_history(h: str, symbol: str) -> str:
            if len(h) >= self.L_max:
                return (h[1:] + symbol)
            else:
                return (h + symbol)

        # Helper: rebuild epsilon map from current states (prefer L_max histories)
        def rebuild_epsilon() -> Dict[str, int]:
            eps = {}
            for sid_local, st_local in enumerate(self.states):
                # Insert longer histories after shorter to let longer override
                for h_local in sorted(st_local['histories'], key=lambda x: len(x)):
                    eps[h_local] = sid_local
            return eps

        # Helper: build adjacency using epsilon-only mapping
        def build_adjacency(eps_map: Dict[str, int]) -> Dict[int, set]:
            adj: Dict[int, set] = {i: set() for i in range(len(self.states))}
            for sid_local, st_local in enumerate(self.states):
                for h_local in st_local['histories']:
                    for sym in ['0', '1']:
                        ext = extend_history(h_local, sym)
                        to_sid = eps_map.get(ext, None)
                        if to_sid is not None:
                            adj[sid_local].add(to_sid)
            return adj

        # Helper: remove transient states by keeping largest SCC
        def remove_transients(eps_map: Dict[str, int]) -> Dict[str, int]:
            adj = build_adjacency(eps_map)
            n = len(self.states)
            # Tarjan SCC
            index = 0
            indices = [-1] * n
            lowlink = [0] * n
            stack = []
            onstack = [False] * n
            sccs: List[List[int]] = []

            def strongconnect(v: int):
                nonlocal index
                indices[v] = index
                lowlink[v] = index
                index += 1
                stack.append(v)
                onstack[v] = True
                for w in adj.get(v, []):
                    if indices[w] == -1:
                        strongconnect(w)
                        lowlink[v] = min(lowlink[v], lowlink[w])
                    elif onstack[w]:
                        lowlink[v] = min(lowlink[v], indices[w])
                if lowlink[v] == indices[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        onstack[w] = False
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(comp)

            for v in range(n):
                if indices[v] == -1:
                    strongconnect(v)

            if not sccs:
                return eps_map
            # pick largest SCC
            largest = max(sccs, key=lambda c: len(c))
            keep = set(largest)
            if len(keep) == n:
                return eps_map
            # filter states and rebuild indices
            old_to_new: Dict[int, int] = {}
            new_states: List[Dict] = []
            for old_sid, st_local in enumerate(self.states):
                if old_sid in keep:
                    old_to_new[old_sid] = len(new_states)
                    new_states.append({'histories': set(st_local['histories']), 'vec': st_local['vec'], 'weight': st_local.get('weight', 0)})
            self.states = new_states
            # rebuild epsilon
            new_eps: Dict[str, int] = {}
            for old_sid, st_local in enumerate(new_states):
                for h_local in st_local['histories']:
                    new_eps[h_local] = old_sid
            return new_eps

        # Rebuild epsilon, remove short histories (< L_max-1), remove empties, remove transients (pre-determinization)
        epsilon = rebuild_epsilon()
        if self.L_max >= 2:
            for st in self.states:
                st['histories'] = set(h for h in st['histories'] if len(h) >= max(0, self.L_max - 1))
            # drop empty states
            self.states = [st for st in self.states if len(st['histories']) > 0]
            # Recompute counts and vec for empirical backend after pruning
            if self.backend == 'empirical':
                for st in self.states:
                    c0, c1 = 0, 0
                    for h in st['histories']:
                        ch = self.empirical_counts(h)
                        if ch is not None:
                            c0 += ch[0]; c1 += ch[1]
                    st['counts'] = [c0, c1]
                    s = float(c0 + c1)
                    st['vec'] = [c0/s if s>0 else 0.5, c1/s if s>0 else 0.5]
            epsilon = rebuild_epsilon()
        if len(self.states) > 1:
            epsilon = remove_transients(epsilon)

        # Phase III: determinize by splitting states on conflicting transitions (epsilon-only mapping)
        changed = True
        while changed:
            changed = False
            epsilon = rebuild_epsilon()
            for sid_idx in range(len(self.states)):
                st = self.states[sid_idx]
                hist_list = list(st['histories'])
                if not hist_list:
                    continue
                for symbol in ['0', '1']:
                    # Find a pilot history that transitions somewhere
                    pilot_to = None
                    for h in sorted(hist_list, key=lambda x: -len(x)):
                        ext = extend_history(h, symbol)
                        to_sid = epsilon.get(ext, None)
                        if to_sid is not None:
                            pilot_to = to_sid
                            break
                    if pilot_to is None:
                        continue
                    # Find histories that transition differently
                    to_split = []
                    for h in hist_list:
                        ext = extend_history(h, symbol)
                        to_sid = epsilon.get(ext, None)
                        if to_sid is None:
                            continue
                        if to_sid != pilot_to:
                            to_split.append(h)
                    if len(to_split) > 0:
                        # Move these histories to a new state
                        new_hist_set = set(to_split)
                        for h in to_split:
                            st['histories'].remove(h)
                        new_state = {
                            'histories': new_hist_set,
                            'vec': self.compute_state_vector(list(new_hist_set)),
                            'weight': sum(self.observed_history_counts(len(h)).get(h, 1) for h in new_hist_set),
                        }
                        if self.backend == 'empirical':
                            # move counts
                            cs_old = st.get('counts', [0,0])
                            move_c0 = 0; move_c1 = 0
                            for h in new_hist_set:
                                ch = self.empirical_counts(h)
                                if ch is not None:
                                    move_c0 += ch[0]; move_c1 += ch[1]
                            st['counts'] = [max(0, cs_old[0] - move_c0), max(0, cs_old[1] - move_c1)]
                            new_state['counts'] = [move_c0, move_c1]
                        self.states.append(new_state)
                        # Recompute current state's vector
                        if self.backend == 'empirical':
                            cs = st.get('counts', [0,0])
                            s = float(cs[0]+cs[1])
                            st['vec'] = [cs[0]/s if s>0 else 0.5, cs[1]/s if s>0 else 0.5]
                        else:
                            st['vec'] = self.compute_state_vector(list(st['histories']))
                        changed = True
                        break
                if changed:
                    break

        # Build transitions after determinization using epsilon-only mapping (reference-like logic)
        epsilon = rebuild_epsilon()
        transitions: Dict[int, Dict[int, int]] = {i: {} for i in range(len(self.states))}
        for sid_idx, st in enumerate(self.states):
            need_Lmax = True
            for h in st['histories']:
                if len(h) == self.L_max - 1:
                    need_Lmax = False
            for symbol, sym_int in [('0', 0), ('1', 1)]:
                to_sid_found = None
                if need_Lmax:
                    # use L_max extensions if all histories shorter than L_max-1
                    for h in st['histories']:
                        if len(h) == self.L_max:
                            ext = extend_history(h, symbol)
                            to_sid = epsilon.get(ext, None)
                            if to_sid is not None:
                                to_sid_found = to_sid
                                break
                else:
                    # use existing histories
                    for h in st['histories']:
                        ext = extend_history(h, symbol) if len(h) == self.L_max - 1 else (h + symbol)
                        to_sid = epsilon.get(ext, None)
                        if to_sid is not None:
                            to_sid_found = to_sid
                            break
                if to_sid_found is not None:
                    transitions[sid_idx][sym_int] = to_sid_found

        for sid in range(len(self.states)):
            self.states[sid]['transitions'] = transitions.get(sid, {})

        # Remove transients after determinization
        if len(self.states) > 1:
            epsilon = remove_transients(rebuild_epsilon())
            # transitions may be stale; rebuild once more
            transitions = {i: {} for i in range(len(self.states))}
            epsilon = rebuild_epsilon()
            for sid_idx, st in enumerate(self.states):
                for symbol, sym_int in [('0', 0), ('1', 1)]:
                    to_sid_found = None
                    for h in st['histories']:
                        ext = extend_history(h, symbol)
                        to_sid = epsilon.get(ext, None)
                        if to_sid is not None:
                            to_sid_found = to_sid
                            break
                    if to_sid_found is not None:
                        transitions[sid_idx][sym_int] = to_sid_found
            for sid in range(len(self.states)):
                self.states[sid]['transitions'] = transitions.get(sid, {})

        # Save epsilon_map
        self.epsilon_map = rebuild_epsilon()

        result = {
            'num_states': len(self.states),
            'states': [
                {
                    'id': i,
                    'histories': sorted(list(st['histories'])),
                    'distribution_vector': st['vec'],
                    'weight': st.get('weight', 0),
                    'transitions': st.get('transitions', {})
                }
                for i, st in enumerate(self.states)
            ],
            'epsilon_map': epsilon,
        }
        return result


# ----------------------------- Helpers -------------------------------------

def js_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    m = 0.5 * (p + q)
    def kl(a, b):
        a_safe = np.maximum(a, eps); b_safe = np.maximum(b, eps)
        return float(np.sum(a_safe * np.log(a_safe / b_safe)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ----------------------------- Synthetic sampling --------------------------

def sample_from_epsilon(states_def: Dict, length: int, seed: Optional[int] = None) -> List[int]:
    """Sample a binary sequence of given length from the discovered epsilon-machine.

    states_def should be dict with keys 'states' containing per-state transitions and morphs.
    Each state entry must include 'distribution_vector' and 'transitions'.
    """
    if seed is not None:
        random.seed(seed)
    states = states_def['states']
    if not states:
        return []
    cur = 0
    out = []
    for _ in range(length):
        p0, p1 = states[cur]['distribution_vector']
        r = random.random()
        if r < p0:
            symbol = 0
        else:
            symbol = 1
        out.append(symbol)
        # next state via transitions; if missing, stay
        nxt = states[cur].get('transitions', {}).get(symbol, cur)
        cur = nxt
    return out


# ----------------------------- CLI entrypoint ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model_ckpt', type=Path)
    p.add_argument('--L_max', type=int, default=8)
    p.add_argument('--alpha', type=float, default=0.001)
    p.add_argument('--test_method', type=str, choices=['g', 'chi2'], default='chi2')
    p.add_argument('--context_window', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--pseudo_count_scale', type=float, default=20.0)
    p.add_argument('--min_count', type=int, default=5)
    p.add_argument('--mix_empirical', type=float, default=0.0)
    p.add_argument('--backend', type=str, choices=['neural', 'empirical'], default='neural', help='Use neural probabilities or empirical counts for morphs/tests')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--prob_clip', type=float, default=1e-6)
    p.add_argument('--platt_fit', action='store_true')
    p.add_argument('--output_json', type=Path, default=Path('neural_cssr_result.json'))
    # Probability comparison output (empirical vs neural)
    p.add_argument('--dump_prob_diff', type=Path, help='If set, dump CSV of empirical vs neural probabilities over histories')
    p.add_argument('--diff_min_count', type=int, default=5, help='Min count threshold for histories in prob diff report')
    args = p.parse_args()

    device = torch.device(args.device)
    tokens = load_binary_tokens(args.data)

    provider = None
    ctx_win = args.context_window
    if args.backend != 'empirical':
        if args.model_ckpt is None:
            raise RuntimeError('--model_ckpt is required when backend=neural')
        # Load model checkpoint using EBM/AR loader (mirrors compare_model_empirical)
        model, ctx_win_loaded, model_type = load_model_from_ckpt(args.model_ckpt, device)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
        # Prefer model-configured context window unless overridden
        context_window = args.context_window if args.context_window else ctx_win_loaded
        provider = ModelProvider(model, device=device, context_window=context_window,
                                 temperature=args.temperature, prob_clip=args.prob_clip)

    # Optional Platt fit: fit against empirical morphs to produce platt_params
    platt_params = None
    if args.platt_fit:
        # build empirical morphs for histories up to L_max
        counts = {}
        ones = {}
        for L in range(1, args.L_max + 1):
            for t in range(L, len(tokens)):
                h = ''.join(str(x) for x in tokens[t-L:t])
                counts[h] = counts.get(h, 0) + 1
                if tokens[t] == 1:
                    ones[h] = ones.get(h, 0) + 1
        histories = [h for h, c in counts.items() if c >= args.min_count]
        if histories:
            margins = []
            targets = []
            weights = []
            for h in histories:
                p0, p1 = provider.next_probs([int(c) for c in h][-args.context_window:])
                p1c = max(1e-12, min(1.0 - 1e-12, p1))
                margins.append(math.log(p1c / (1.0 - p1c)))
                targets.append(ones.get(h, 0) / float(counts[h]))
                weights.append(counts[h])
            # simple weighted logistic regression (scipy or torch LBFGS)
            margins_t = torch.tensor(margins, dtype=torch.float64, device=device)
            targets_t = torch.tensor(targets, dtype=torch.float64, device=device)
            weights_t = torch.tensor(weights, dtype=torch.float64, device=device)
            a = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)
            b = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
            opt = torch.optim.LBFGS([a, b], lr=0.25, max_iter=200)
            def closure():
                opt.zero_grad()
                s = a * margins_t + b
                loss = F.binary_cross_entropy_with_logits(s, targets_t, weight=weights_t)
                loss.backward()
                return loss
            opt.step(closure)
            platt_params = {'a': float(a.detach().cpu().item()), 'b': float(b.detach().cpu().item())}
            print(f'Fitted Platt params: {platt_params}')
            provider.platt_params = platt_params

    # Optional: dump empirical vs neural probability differences before CSSR
    if args.dump_prob_diff is not None and args.backend != 'empirical':
        # Collect histories across 1..L_max with counts
        hist_stats: Dict[str, Dict[str, float]] = {}
        for L in range(1, args.L_max + 1):
            for t in range(L, len(tokens)):
                h = ''.join(str(x) for x in tokens[t-L:t])
                if h not in hist_stats:
                    hist_stats[h] = {'count': 0.0, 'count1': 0.0, 'L': L}
                s = hist_stats[h]
                s['count'] += 1.0
                if tokens[t] == 1:
                    s['count1'] += 1.0
                if L < s['L']:
                    s['L'] = L
        # Write CSV
        import csv
        args.dump_prob_diff.parent.mkdir(parents=True, exist_ok=True)
        wmse_num = 0.0; wmse_den = 0.0
        with args.dump_prob_diff.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['history', 'L', 'count', 'p_emp', 'p_neural'])
            for h, d in sorted(hist_stats.items(), key=lambda kv: (-kv[1]['count'], kv[0])):
                c = int(d['count'])
                if c < args.diff_min_count:
                    continue
                p_emp = float(d['count1'] / d['count'])
                ids = [int(cch) for cch in h][-provider.context_window:]
                p0, p1 = provider.next_probs(ids)
                writer.writerow([h, int(d['L']), c, p_emp, p1])
                wmse_num += c * (p_emp - p1) * (p_emp - p1)
                wmse_den += c
        if wmse_den > 0:
            print(f"Weighted MSE (neural vs empirical): {wmse_num / wmse_den:.6f}")

    # Run Neural CSSR
    cssr = NeuralCSSR(tokens, provider, L_max=args.L_max, alpha=args.alpha,
                      test_method=args.test_method, pseudo_count_scale=args.pseudo_count_scale,
                      min_count=args.min_count, mix_empirical=args.mix_empirical,
                      backend=args.backend)
    result = cssr.run()

    # Attach configuration summary and optional metrics
    cfg = {
        'data': str(args.data),
        'backend': str(args.backend),
        'L_max': int(args.L_max),
        'alpha': float(args.alpha),
        'test_method': str(args.test_method),
        'device': str(args.device),
        'pseudo_count_scale': float(args.pseudo_count_scale),
        'min_count': int(args.min_count),
        'mix_empirical': float(args.mix_empirical),
        'context_window': int(args.context_window),
        'temperature': float(args.temperature),
        'prob_clip': float(args.prob_clip),
        'model_ckpt': (str(args.model_ckpt) if args.model_ckpt is not None else None),
        'platt_fit': bool(args.platt_fit),
    }

    if isinstance(result, dict):
        result['config'] = cfg

    # Save
    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {args.output_json}")


if __name__ == '__main__':
    main()
