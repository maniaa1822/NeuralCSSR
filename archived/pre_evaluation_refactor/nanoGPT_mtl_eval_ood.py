#!/usr/bin/env python3
"""Evaluate nanoGPT checkpoints on controlled out-of-distribution regimes."""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machines import get_machine
from machines.base import Machine

from nanoGPT.model import GPT, GPTConfig
from nanoGPT.mtl.model_wrapper import MultitaskConfig, MultitaskGPT


# ---------------------------------------------------------------------------
# Data generation utilities


@dataclass
class DistanceInfo:
    matrix: np.ndarray
    num_classes: int
    unreachable_class: int


def compute_state_distances(machine: Machine) -> DistanceInfo:
    """Compute shortest-hop distances between states (used for geometry head)."""

    states = machine.states
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    alphabet = machine.alphabet
    transitions = machine.transitions

    n_states = len(states)
    distances = np.full((n_states, n_states), fill_value=-1, dtype=np.int16)

    for state_idx, state in enumerate(states):
        distances[state_idx, state_idx] = 0
        frontier = [state]
        visited = {state}
        depth = 0

        while frontier:
            depth += 1
            next_frontier: List[str] = []
            for current in frontier:
                for symbol in alphabet:
                    nxt = transitions.get((current, symbol))
                    if nxt is None or nxt in visited:
                        continue
                    visited.add(nxt)
                    next_frontier.append(nxt)
                    tgt_idx = state_to_idx[nxt]
                    if distances[state_idx, tgt_idx] == -1 or depth < distances[state_idx, tgt_idx]:
                        distances[state_idx, tgt_idx] = depth
            frontier = next_frontier

    finite_mask = distances >= 0
    max_distance = int(distances[finite_mask].max()) if finite_mask.any() else 0
    unreachable_class = max_distance + 1
    distances[~finite_mask] = unreachable_class
    num_classes = unreachable_class + 1

    return DistanceInfo(distances.astype(np.int64), num_classes, unreachable_class)


class OODGenerator:
    """Generate token/state traces under an OOD regime."""

    def __init__(
        self,
        machine: Machine,
        regime: str,
        severity: float,
        seed: int,
        mixed_regimes: Optional[List[str]] = None,
        mixed_severities: Optional[List[float]] = None,
    ):
        self.machine = machine
        self.regime = regime
        self.severity = severity
        self.rng = np.random.default_rng(seed)
        self.states = machine.states
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.alphabet = machine.alphabet
        self.mixed_regimes = mixed_regimes or []
        self.mixed_severities = mixed_severities or []
        self._state_swap_set: Optional[set] = None

        if regime == "emission_mix" and not (0.0 <= severity <= 1.0):
            raise ValueError("emission_mix severity must be between 0 and 1")
        if regime == "transition_noise" and not (0.0 <= severity <= 1.0):
            raise ValueError("transition_noise severity must be between 0 and 1")
        if regime == "emission_bias" and not (0.0 <= severity <= 1.0):
            raise ValueError("emission_bias severity must be between 0 and 1")
        if regime == "temporal_swap" and not (0.0 <= severity <= 1.0):
            raise ValueError("temporal_swap severity must be between 0 and 1 (per-step swap probability)")
        if regime == "state_dependent_swap":
            if not (0.0 <= severity <= 1.0):
                raise ValueError("state_dependent_swap severity must be between 0 and 1 (fraction of states swapped)")
            # Choose a subset of states to swap emissions for
            n_states = len(self.states)
            n_swap = max(1, int(round(severity * n_states))) if n_states > 0 else 0
            chosen = self.rng.choice(self.states, size=n_swap, replace=False) if n_swap > 0 else []
            self._state_swap_set = set(chosen)
        if regime == "mixed_regime":
            if not mixed_regimes:
                raise ValueError("mixed_regime requires --mixed-regimes")
            if not mixed_severities or len(mixed_severities) != len(mixed_regimes):
                raise ValueError("mixed_regime requires --mixed-severities with same length as --mixed-regimes")

    def generate_sequence(self, length: int) -> Tuple[List[str], List[int]]:
        """Generate symbols and epsilon-state indices of given length."""

        tokens: List[str] = []
        eps_states: List[int] = []
        state = self._initial_state()

        for _ in range(length):
            symbol, next_state = self._sample_next(state)
            tokens.append(symbol)
            eps_states.append(self.state_to_idx[next_state])
            state = next_state

        return tokens, eps_states

    def _initial_state(self) -> str:
        if self.regime == "start_state_uniform":
            return self.rng.choice(self.states)
        return self.machine.start_state

    def _sample_next(self, state: str) -> Tuple[str, str]:
        emission_probs = self.machine.emissions[state]
        base_probs = np.array([emission_probs[sym] for sym in self.alphabet], dtype=np.float64)

        # Apply emission perturbations
        active_regime = self.regime
        active_severity = self.severity

        # For mixed regime, apply all emission-based perturbations
        if self.regime == "mixed_regime":
            probs = base_probs.copy()
            for reg, sev in zip(self.mixed_regimes, self.mixed_severities):
                probs = self._apply_emission_perturbation(probs, reg, sev)
        else:
            probs = self._apply_emission_perturbation(base_probs, active_regime, active_severity)

        # State-dependent swap applied after generic perturbation
        if self.regime == "state_dependent_swap" and self._state_swap_set is not None:
            if state in self._state_swap_set:
                probs = probs[::-1]
        # Temporal swap: with probability p, flip emissions at this step
        if self.regime == "temporal_swap" and self.rng.random() < self.severity:
            probs = probs[::-1]

        probs = probs / probs.sum()
        idx = self.rng.choice(len(self.alphabet), p=probs)
        symbol = self.alphabet[idx]

        next_state = self.machine.transitions.get((state, symbol))
        if next_state is None:
            raise RuntimeError(f"Transition undefined for state={state}, symbol={symbol}")

        # Apply transition perturbations
        if self.regime == "transition_noise" and self.rng.random() < self.severity:
            next_state = self.rng.choice(self.states)
        elif self.regime == "mixed_regime":
            for reg, sev in zip(self.mixed_regimes, self.mixed_severities):
                if reg == "transition_noise" and self.rng.random() < sev:
                    next_state = self.rng.choice(self.states)

        return symbol, next_state

    def _apply_emission_perturbation(self, probs: np.ndarray, regime: str, severity: float) -> np.ndarray:
        """Apply a single emission-based perturbation to probability distribution."""
        if regime == "emission_mix":
            uniform = np.full_like(probs, 1.0 / len(self.alphabet))
            return (1.0 - severity) * probs + severity * uniform
        elif regime == "alphabet_swap":
            return probs[::-1]
        elif regime == "emission_bias":
            # Push probabilities toward extremes (0.9 or 0.1)
            biased = np.zeros_like(probs)
            for i, p in enumerate(probs):
                target = 0.9 if p > 0.5 else 0.1
                biased[i] = (1.0 - severity) * p + severity * target
            return biased
        else:
            return probs


def build_eval_tensors(
    generator: OODGenerator,
    num_sequences: int,
    block_size: int,
    stoi: Dict[str, int],
    distance_info: DistanceInfo,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate non-overlapping evaluation blocks."""

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    eps_list: List[np.ndarray] = []
    dist_list: List[np.ndarray] = []

    for _ in range(num_sequences):
        tokens, eps_states = generator.generate_sequence(block_size + 1)
        token_ids = np.array([stoi[sym] for sym in tokens], dtype=np.int64)
        eps_arr = np.array(eps_states[:-1], dtype=np.int64)
        x_list.append(token_ids[:-1])
        y_list.append(token_ids[1:])
        eps_list.append(eps_arr)
        dist_list.append(distance_info.matrix[eps_arr])

    x = torch.tensor(np.stack(x_list), dtype=torch.long)
    y = torch.tensor(np.stack(y_list), dtype=torch.long)
    eps = torch.tensor(np.stack(eps_list), dtype=torch.long)
    dist = torch.tensor(np.stack(dist_list), dtype=torch.long)

    return x, y, eps, dist


# ---------------------------------------------------------------------------
# Model loading helpers


@dataclass
class ModelHandle:
    name: str
    model: torch.nn.Module
    kind: str
    tap_preference: Optional[str]
    has_distance_head: bool
    block_size: int


def parse_model_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Model spec must be name=path format (got '{spec}')")
    name, path = spec.split("=", 1)
    return name, Path(path)


def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model(
    name: str,
    ckpt_path: Path,
    device: torch.device,
    vocab_size: int,
    distance_info: DistanceInfo,
) -> ModelHandle:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    block_size = int(config.get("block_size", 0))
    n_layer = int(config.get("n_layer", 0))
    n_head = int(config.get("n_head", 0))
    n_embd = int(config.get("n_embd", 0))
    dropout = float(config.get("dropout", 0.0))
    bias = bool(config.get("bias", False))

    state_dict = strip_ddp_prefix(checkpoint["model"])

    weight_key = None
    for candidate in ("gpt.transformer.wte.weight", "transformer.wte.weight"):
        if candidate in state_dict:
            weight_key = candidate
            break
    if weight_key is None:
        raise KeyError("Could not locate embedding weights in checkpoint")
    vocab_size_ckpt = state_dict[weight_key].shape[0]
    if vocab_size != vocab_size_ckpt:
        vocab_size = vocab_size_ckpt

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )

    multitask_layers = config.get("multitask_tap_layers")
    if multitask_layers:
        tap_layers = list(multitask_layers)
        multitask_config = MultitaskConfig(
            tap_layers=tap_layers,
            bottleneck_dim=config.get("multitask_bottleneck_dim"),
            epsilon_classes=int(config.get("multitask_epsilon_classes", 0)),
            epsilon_loss_weight=float(config.get("multitask_epsilon_loss_weight", 1.0)),
            num_states=len(distance_info.matrix),
            distance_num_classes=distance_info.num_classes,
            distance_loss_weight=float(config.get("multitask_distance_loss_weight", 0.0)),
        )
        model = MultitaskGPT(gpt_config, multitask_config)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        distance_weight = float(config.get("multitask_distance_loss_weight", 0.0) or 0.0)
        has_distance = distance_weight > 0 and any(key.startswith("distance_heads") for key in state_dict)
        tap_preference = tap_layers[-1] if tap_layers else None
        return ModelHandle(
            name=name,
            model=model,
            kind="mtl",
            tap_preference=tap_preference,
            has_distance_head=has_distance,
            block_size=block_size,
        )

    model = GPT(gpt_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return ModelHandle(name=name, model=model, kind="baseline", tap_preference=None, has_distance_head=False, block_size=block_size)


# ---------------------------------------------------------------------------
# Evaluation logic


@dataclass
class MetricsAccumulator:
    tokens: int = 0
    lm_loss_sum: float = 0.0
    epsilon_loss_sum: float = 0.0
    epsilon_correct: int = 0
    epsilon_count: int = 0
    distance_loss_sum: float = 0.0
    distance_correct: int = 0
    distance_count: int = 0

    def add_lm(self, loss: float, count: int) -> None:
        self.lm_loss_sum += loss * count
        self.tokens += count

    def add_epsilon(self, loss: float, correct: int, count: int) -> None:
        self.epsilon_loss_sum += loss * count
        self.epsilon_correct += correct
        self.epsilon_count += count

    def add_distance(self, loss: float, correct: int, count: int) -> None:
        self.distance_loss_sum += loss * count
        self.distance_correct += correct
        self.distance_count += count


def evaluate_multistep(
    handle: ModelHandle, x: torch.Tensor, y: torch.Tensor, k: int, device: torch.device
) -> float:
    """Evaluate k-step ahead prediction loss by autoregressive rollout."""
    B, T = x.shape
    total_loss = 0.0
    count = 0

    # For each position, predict k steps ahead
    for t in range(T - k + 1):
        context = x[:, : t + 1].to(device)  # Context up to position t

        # Autoregressively generate k steps
        current = context
        for step in range(k):
            if handle.kind == "mtl":
                # For multitask models, just use the base GPT forward
                out = handle.model.gpt(current)
                logits = out if isinstance(out, torch.Tensor) else out[0]
            else:
                logits, _ = handle.model(current)

            # Get prediction for next token
            next_logits = logits[:, -1, :]  # [B, vocab]

            if step == k - 1:
                # Final step: compute loss against ground truth k steps ahead
                target = y[:, t + k - 1]
                loss = F.cross_entropy(next_logits, target, reduction="sum")
                total_loss += float(loss)
                count += B
            else:
                # Intermediate step: sample and append
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
                current = torch.cat([current, next_token], dim=1)

    return total_loss / count if count > 0 else 0.0


def _apply_prob_remap(
    logits: torch.Tensor,
    remap: Optional[torch.Tensor],
    targets: torch.Tensor,
) -> Optional[float]:
    """Optionally apply a 2x2 probability remap and compute CE loss.

    - Expects vocab size 2. Returns None if remap is None or vocab != 2.
    - Computes cross-entropy on remapped probabilities without converting back to logits.
    """
    if remap is None:
        return None
    if logits.size(-1) != 2:
        return None
    # probabilities after softmax
    probs = torch.softmax(logits, dim=-1)  # [B, T, 2]
    # right-multiply by 2x2 matrix to remap class probabilities
    # p' = p @ M, where swap M=[[0,1],[1,0]] swaps classes
    remapped = torch.einsum("bti,ij->btj", probs, remap)
    # renormalize to guard against numerical drift
    remapped = torch.clamp(remapped, min=1e-12)
    remapped = remapped / remapped.sum(dim=-1, keepdim=True)
    # negative log-likelihood for true targets
    true_p = remapped.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = -torch.log(torch.clamp(true_p, min=1e-12))
    return float(nll.mean())


def _apply_state_prob_remap(
    logits: torch.Tensor,
    remap_per_state: Optional[torch.Tensor],
    targets: torch.Tensor,
    state_ids: torch.Tensor,
) -> Optional[float]:
    if remap_per_state is None:
        return None
    probs = torch.softmax(logits, dim=-1)
    num_classes = probs.size(-1)
    # remap_per_state expected shape (num_states, C, C)
    if remap_per_state.dim() != 3 or remap_per_state.size(1) != num_classes:
        return None
    # Gather per-position matrices
    mats = remap_per_state[state_ids]
    remapped = torch.einsum("bti,btij->btj", probs, mats)
    remapped = torch.clamp(remapped, min=1e-12)
    remapped = remapped / remapped.sum(dim=-1, keepdim=True)
    true_p = remapped.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = -torch.log(torch.clamp(true_p, min=1e-12))
    return float(nll.mean())


def _fit_global_remap(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> Optional[torch.Tensor]:
    # probs: [B, T_ctx, C]
    if probs.numel() == 0:
        return None
    B, T, C = probs.shape
    probs_flat = probs.reshape(-1, C).double()
    if probs_flat.numel() == 0:
        return None
    targets_flat = targets.reshape(-1)
    Y = torch.nn.functional.one_hot(targets_flat, num_classes=C).double()
    # Solve least squares for matrix M: probs_flat @ M ≈ Y
    try:
        solution = torch.linalg.lstsq(probs_flat, Y).solution
    except RuntimeError:
        return None
    return solution.to(probs.dtype)


def _fit_per_state_remap(
    probs: torch.Tensor,
    targets: torch.Tensor,
    state_ids: torch.Tensor,
    num_states: int,
) -> Optional[torch.Tensor]:
    if probs.numel() == 0:
        return None
    B, T, C = probs.shape
    mats = torch.eye(C, dtype=probs.dtype, device=probs.device).unsqueeze(0).repeat(num_states, 1, 1)
    probs_flat = probs.reshape(-1, C).double().cpu()
    targets_flat = targets.reshape(-1).cpu()
    states_flat = state_ids.reshape(-1).cpu()
    Y_full = torch.nn.functional.one_hot(targets_flat, num_classes=C).double()

    for state in range(num_states):
        mask = states_flat == state
        if not mask.any():
            continue
        p_state = probs_flat[mask]
        y_state = Y_full[mask]
        if p_state.shape[0] < C:
            continue
        try:
            solution = torch.linalg.lstsq(p_state, y_state).solution
        except RuntimeError:
            continue
        mats[state] = solution.to(probs.dtype).to(probs.device)

    return mats


def evaluate_model(
    handle: ModelHandle,
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    device: torch.device,
    tap_preference_order: Optional[Sequence[str]] = None,
    k_step: int = 1,
    remap_matrix: Optional[torch.Tensor] = None,
    fit_remap_mode: str = "none",
    num_states: int = 0,
    context_tokens: int = 0,
) -> Dict[str, Optional[float]]:
    x, y, eps, dist = data
    total_sequences = x.size(0)
    acc = MetricsAccumulator()

    with torch.no_grad():
        for start in range(0, total_sequences, batch_size):
            end = min(start + batch_size, total_sequences)
            xb = x[start:end].to(device)
            yb = y[start:end].to(device)
            epsb = eps[start:end].to(device)
            distb = dist[start:end].to(device)

            if k_step > 1:
                if k_step < 1:
                    raise ValueError("k_step must be >= 1")
                if k_step > yb.size(1):
                    raise ValueError(
                        f"k_step={k_step} exceeds available prediction horizon {yb.size(1)}"
                    )
                # Multi-step prediction: evaluate k-step ahead predictions
                lm_loss_val = evaluate_multistep(handle, xb, yb, k_step, device)
                token_count = yb[:, k_step - 1 :].numel()  # Only valid predictions
                acc.add_lm(lm_loss_val, token_count)
                # Skip auxiliary tasks for multi-step (not well-defined)
                continue

            if handle.kind == "mtl":
                out = handle.model(xb, yb, epsb, distb)
                lm_logits = out.get("lm_logits")
                if lm_logits is None:
                    raise RuntimeError("Multitask model did not return lm_logits")

                batch_remap = remap_matrix
                batch_state_remap = None
                if fit_remap_mode != "none":
                    if context_tokens <= 0:
                        raise ValueError("fit_remap requires --context-tokens > 0")
                    ctx_logits = lm_logits[:, :context_tokens, :]
                    ctx_targets = yb[:, :context_tokens]
                    ctx_probs = torch.softmax(ctx_logits, dim=-1)
                    if fit_remap_mode == "global":
                        fitted = _fit_global_remap(ctx_probs, ctx_targets)
                        if fitted is not None:
                            batch_remap = fitted.to(device)
                    elif fit_remap_mode == "state":
                        ctx_states = epsb[:, :context_tokens]
                        fitted_state = _fit_per_state_remap(ctx_probs, ctx_targets, ctx_states, num_states)
                        if fitted_state is not None:
                            batch_state_remap = fitted_state.to(device)

                logits_tail = lm_logits[:, context_tokens:, :]
                targets_tail = yb[:, context_tokens:]
                token_count = targets_tail.numel()
                if token_count == 0:
                    raise ValueError("context_tokens consumes entire window; no tokens left to score")

                remapped_loss = None
                if fit_remap_mode == "state":
                    remapped_loss = _apply_state_prob_remap(
                        logits_tail,
                        batch_state_remap,
                        targets_tail,
                        epsb[:, context_tokens:],
                    )
                if remapped_loss is None:
                    remapped_loss = _apply_prob_remap(logits_tail, batch_remap, targets_tail)
                if remapped_loss is not None:
                    acc.add_lm(remapped_loss, token_count)
                else:
                    loss = F.cross_entropy(
                        logits_tail.reshape(-1, logits_tail.size(-1)),
                        targets_tail.reshape(-1),
                        reduction="mean",
                    )
                    acc.add_lm(float(loss), token_count)

                epsilon_logits = out.get("epsilon_logits", {})
                if epsilon_logits:
                    tap_name = _select_tap(epsilon_logits.keys(), handle.tap_preference, tap_preference_order)
                    logits = epsilon_logits[tap_name][:, context_tokens:, :]
                    eps_targets = epsb[:, context_tokens:]
                    eps_count = eps_targets.numel()
                    if eps_count > 0:
                        eps_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            eps_targets.reshape(-1),
                            reduction="mean",
                        )
                        preds = logits.argmax(dim=-1)
                        correct = int((preds == eps_targets).sum().item())
                        acc.add_epsilon(float(eps_loss), correct, eps_count)

                distance_logits = out.get("distance_logits", {})
                if distance_logits and handle.has_distance_head:
                    tap_name = _select_tap(distance_logits.keys(), handle.tap_preference, tap_preference_order)
                    logits = distance_logits[tap_name][:, context_tokens:, :, :]
                    dist_targets = distb[:, context_tokens:, :]
                    if dist_targets.numel() > 0:
                        B, T, S, C = logits.shape
                        logits_flat = logits.reshape(B * T * S, C)
                        targets_flat = dist_targets.reshape(-1)
                        dist_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
                        preds = logits.argmax(dim=-1)
                        correct = int((preds == dist_targets).sum().item())
                        acc.add_distance(float(dist_loss), correct, dist_targets.numel())
            else:
                logits, _ = handle.model(xb, yb)

                batch_remap = remap_matrix
                batch_state_remap = None
                if fit_remap_mode != "none":
                    if context_tokens <= 0:
                        raise ValueError("fit_remap requires --context-tokens > 0")
                    ctx_logits = logits[:, :context_tokens, :]
                    ctx_targets = yb[:, :context_tokens]
                    ctx_probs = torch.softmax(ctx_logits, dim=-1)
                    if fit_remap_mode == "global":
                        fitted = _fit_global_remap(ctx_probs, ctx_targets)
                        if fitted is not None:
                            batch_remap = fitted.to(device)
                    elif fit_remap_mode == "state":
                        ctx_states = epsb[:, :context_tokens]
                        fitted_state = _fit_per_state_remap(ctx_probs, ctx_targets, ctx_states, num_states)
                        if fitted_state is not None:
                            batch_state_remap = fitted_state.to(device)

                logits_tail = logits[:, context_tokens:, :]
                targets_tail = yb[:, context_tokens:]
                token_count = targets_tail.numel()
                if token_count == 0:
                    raise ValueError("context_tokens consumes entire window; no tokens left to score")

                remapped_loss = None
                if fit_remap_mode == "state":
                    remapped_loss = _apply_state_prob_remap(
                        logits_tail,
                        batch_state_remap,
                        targets_tail,
                        epsb[:, context_tokens:],
                    )
                if remapped_loss is None:
                    remapped_loss = _apply_prob_remap(logits_tail, batch_remap, targets_tail)
                if remapped_loss is not None:
                    acc.add_lm(remapped_loss, token_count)
                else:
                    loss = F.cross_entropy(
                        logits_tail.reshape(-1, logits_tail.size(-1)),
                        targets_tail.reshape(-1),
                        reduction="mean",
                    )
                    acc.add_lm(float(loss), token_count)

    metrics: Dict[str, Optional[float]] = {
        "lm_loss": acc.lm_loss_sum / acc.tokens if acc.tokens else None,
        "bits_per_token": (acc.lm_loss_sum / acc.tokens) / math.log(2) if acc.tokens else None,
        "token_count": acc.tokens,
    }

    if acc.epsilon_count:
        metrics.update(
            {
                "epsilon_loss": acc.epsilon_loss_sum / acc.epsilon_count,
                "epsilon_accuracy": acc.epsilon_correct / acc.epsilon_count,
            }
        )
    else:
        metrics.update({"epsilon_loss": None, "epsilon_accuracy": None})

    if acc.distance_count:
        metrics.update(
            {
                "distance_loss": acc.distance_loss_sum / acc.distance_count,
                "distance_accuracy": acc.distance_correct / acc.distance_count,
            }
        )
    else:
        metrics.update({"distance_loss": None, "distance_accuracy": None})

    return metrics


def _select_tap(
    available: Iterable[str],
    preferred: Optional[str],
    fallback_order: Optional[Sequence[str]] = None,
) -> str:
    available_set = list(available)
    if preferred and preferred in available_set:
        return preferred
    if fallback_order:
        for candidate in fallback_order:
            if candidate in available_set:
                return candidate
    return available_set[0]


# ---------------------------------------------------------------------------
# CLI


def load_meta(dataset: str, repo_root: Path) -> Tuple[Dict[str, int], int]:
    meta_path = repo_root / "nanoGPT" / "data" / dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    stoi = meta["stoi"]
    vocab_size = int(meta["vocab_size"])
    return stoi, vocab_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model spec in the form name=path/to/ckpt.pt (repeatable)",
    )
    parser.add_argument(
        "--regime",
        choices=[
            "emission_mix",
            "start_state_uniform",
            "transition_noise",
            "alphabet_swap",
            "emission_bias",
            "multistep_prediction",
            "mixed_regime",
            "state_dependent_swap",
            "temporal_swap",
        ],
        required=True,
        help="OOD perturbation to apply",
    )
    parser.add_argument("--severity", type=float, default=0.0, help="Regime severity parameter")
    parser.add_argument("--tokens", type=int, default=65536, help="Evaluation tokens per model")
    parser.add_argument("--block-size", type=int, default=64, help="Block size for evaluation windows")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for forward passes")
    parser.add_argument("--dataset", type=str, default="seven_state_human_mtl100k", help="Dataset meta to borrow vocab from")
    parser.add_argument("--machine", type=str, default="seven_state_human", help="Machine name for ground-truth labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--k-step", type=int, default=2, help="Prediction horizon for multistep_prediction regime")
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="Number of initial tokens per window treated as context (ignored for metrics)",
    )
    parser.add_argument(
        "--fit-remap",
        choices=["none", "global", "state"],
        default="none",
        help="Fit a probability remap from context tokens (requires --context-tokens > 0)",
    )
    # Optional 2x2 probability remap for vocab=2 (post-hoc head calibration)
    parser.add_argument("--remap-swap", action="store_true", help="Apply fixed swap matrix [[0,1],[1,0]] on softmax probs (vocab=2 only)")
    parser.add_argument(
        "--remap-matrix",
        type=str,
        default=None,
        help="Custom 2x2 matrix a,b,c,d applied to probs as p' = p @ [[a,b],[c,d]] (vocab=2 only)",
    )
    parser.add_argument("--mixed-regimes", type=str, default=None, help="Comma-separated regime names for mixed_regime (e.g., 'emission_mix,transition_noise')")
    parser.add_argument("--mixed-severities", type=str, default=None, help="Comma-separated severity values for mixed_regime (e.g., '0.3,0.05')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    stoi, vocab_size = load_meta(args.dataset, repo_root)

    machine = get_machine(args.machine)
    distance_info = compute_state_distances(machine)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_specs = [parse_model_spec(spec) for spec in args.models]
    models = [load_model(name, repo_root / path if not path.is_absolute() else path, device, vocab_size, distance_info) for name, path in model_specs]

    model_block_sizes = {handle.block_size for handle in models if handle.block_size}
    if len(model_block_sizes) > 1:
        raise ValueError(f"Loaded models disagree on block_size: {model_block_sizes}")
    block_size = args.block_size
    if model_block_sizes and block_size not in model_block_sizes:
        block_size = next(iter(model_block_sizes))

    if args.context_tokens < 0:
        raise ValueError("--context-tokens must be >= 0")
    if args.context_tokens >= block_size:
        raise ValueError("--context-tokens must be smaller than the evaluation block size")
    if args.regime == "multistep_prediction" and args.context_tokens:
        raise ValueError("--context-tokens is not supported with multistep_prediction regime")
    if args.fit_remap != "none" and args.context_tokens <= 0:
        raise ValueError("--fit-remap requires --context-tokens > 0")

    # Parse mixed regime parameters
    mixed_regimes = None
    mixed_severities = None
    if args.regime == "mixed_regime":
        if not args.mixed_regimes or not args.mixed_severities:
            raise ValueError("mixed_regime requires both --mixed-regimes and --mixed-severities")
        mixed_regimes = [r.strip() for r in args.mixed_regimes.split(",")]
        mixed_severities = [float(s.strip()) for s in args.mixed_severities.split(",")]

    num_sequences = max(1, args.tokens // block_size)
    generator = OODGenerator(
        machine, args.regime, args.severity, args.seed, mixed_regimes, mixed_severities
    )
    data = build_eval_tensors(generator, num_sequences, block_size, stoi, distance_info)

    tap_order: List[str] = []
    for handle in models:
        if handle.tap_preference and handle.tap_preference not in tap_order:
            tap_order.append(handle.tap_preference)

    # Determine k_step for evaluation
    k_step = args.k_step if args.regime == "multistep_prediction" else 1

    # Build optional remap matrix tensor
    remap_matrix: Optional[torch.Tensor] = None
    if args.remap_swap or args.remap_matrix:
        if args.fit_remap != "none":
            raise ValueError("Cannot combine fitted remap with --remap-swap/--remap-matrix")
        if args.remap_swap and args.remap_matrix:
            raise ValueError("Specify only one of --remap-swap or --remap-matrix")
        if args.remap_swap:
            M = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        else:
            try:
                parts = [float(x.strip()) for x in args.remap_matrix.split(",")]
                if len(parts) != 4:
                    raise ValueError
                M = np.array([[parts[0], parts[1]], [parts[2], parts[3]]], dtype=np.float32)
            except Exception:
                raise ValueError("--remap-matrix must be four comma-separated floats: a,b,c,d")
        # No constraint enforcement here; normalization is applied per-example downstream
        remap_matrix = torch.tensor(M, device=device)

    results = []
    for handle in models:
        metrics = evaluate_model(
            handle,
            data,
            args.batch_size,
            device,
            tap_order,
            k_step,
            remap_matrix,
            args.fit_remap,
            distance_info.matrix.shape[0],
            args.context_tokens,
        )
        record = {
            "model": handle.name,
            "kind": handle.kind,
            "regime": args.regime,
            "severity": args.severity,
        }
        record.update(metrics)
        results.append(record)

    headers = [
        "model",
        "kind",
        "regime",
        "severity",
        "token_count",
        "lm_loss",
        "bits_per_token",
        "epsilon_loss",
        "epsilon_accuracy",
        "distance_loss",
        "distance_accuracy",
    ]

    for record in results:
        line_parts = [
            f"{record['model']}",
            record['kind'],
            args.regime,
            f"{args.severity:.3f}",
            _fmt_optional(record.get('token_count')), 
            _fmt_optional(record.get('lm_loss')),
            _fmt_optional(record.get('bits_per_token')),
            _fmt_optional(record.get('epsilon_loss')),
            _fmt_optional(record.get('epsilon_accuracy')), 
            _fmt_optional(record.get('distance_loss')),
            _fmt_optional(record.get('distance_accuracy')),
        ]
        print("\t".join(line_parts))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            fh.write(",".join(headers) + "\n")
            for record in results:
                row = [
                    record['model'],
                    record['kind'],
                    args.regime,
                    f"{args.severity:.6f}",
                    _fmt_optional(record.get('token_count'), csv=True),
                    _fmt_optional(record.get('lm_loss'), csv=True),
                    _fmt_optional(record.get('bits_per_token'), csv=True),
                    _fmt_optional(record.get('epsilon_loss'), csv=True),
                    _fmt_optional(record.get('epsilon_accuracy'), csv=True),
                    _fmt_optional(record.get('distance_loss'), csv=True),
                    _fmt_optional(record.get('distance_accuracy'), csv=True),
                ]
                fh.write(",".join(row) + "\n")


def _fmt_optional(value: Optional[float], csv: bool = False) -> str:
    if value is None:
        return "" if csv else "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    numeric = float(value)
    if math.isnan(numeric):
        return "" if csv else "NA"
    return f"{numeric:.6f}"


if __name__ == "__main__":
    main()
