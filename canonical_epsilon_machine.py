"""Canonical (unifilar) epsilon-machine construction utilities.

This module provides a drop-in builder that converts the (possibly non-unifilar,
partially specified) structure produced by the current CSSREnhancedExtractor
into a canonical CSSR-style epsilon machine:

Structure returned (dictionary):
{
  'num_states': int,
  'alphabet': List[str],
  'states': List[int],
  'start_state': int | None,
  'transitions': { state_id: { symbol: next_state_id } },      # deterministic
  'emissions':   { state_id: { symbol: P(symbol | state) } },  # probability dist
  'causal_state_info': { state_id: { 'suffixes': [...], 'count': int, 'size': int,
                                     'future_probabilities': {...} } },
  'validation': { 'unifilar': bool, 'complete': bool }
}

Assumptions about the extractor instance passed in:
 - extractor.causal_states: iterable of dicts each with keys:
       'id', 'suffixes' (iterable of strings), 'count', 'size', 'future_probabilities'
 - extractor.suffix_futures: dict mapping suffix -> Counter/dict of future symbol counts
 - (optionally) extractor.alphabet or will be inferred from suffix_futures keys.

The builder enforces:
 - Unifilarity: for each (state, symbol) choose the modal destination state across all
   extended suffix observations; ties broken by highest cumulative count then lowest id.
 - Full symbol coverage: if a state has zero observations for a symbol, it creates a
   self-loop (configurable) and assigns a smoothed emission probability.
 - Separation of emissions (symbol probabilities) and transitions (deterministic mapping).

This does NOT perform additional causal state splitting; it operates strictly on the
provided causal state partition.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Any


@dataclass
class CanonicalBuildConfig:
    alphabet: Optional[List[str]] = None          # If None, infer from future counts
    ensure_full_alphabet: bool = True             # Enforce coverage for each symbol
    missing_symbol_self_loop: bool = True         # If True, fallback missing edges to self
    smoothing: float = 0.0                        # Additive smoothing for emissions
    choose_start: str = "largest_count"           # 'largest_count' or 'first'
    verbose: bool = False


class CanonicalEpsilonMachineBuilder:
    """Builder to derive a canonical unifilar epsilon machine from an extractor."""

    def __init__(self, config: CanonicalBuildConfig | None = None):
        self.config = config or CanonicalBuildConfig()

    def from_extractor(self, extractor: Any) -> Dict[str, Any]:
        """Construct canonical epsilon machine from a CSSREnhancedExtractor instance.

        Parameters
        ----------
        extractor : object
            Instance exposing attributes causal_states, suffix_futures, optionally alphabet.

        Returns
        -------
        Dict[str, Any]
            Canonical epsilon machine description (see module docstring).
        """
        cs = getattr(extractor, 'causal_states', None)
        if not cs:
            raise ValueError("Extractor has no causal_states (build them first).")

        suffix_futures = getattr(extractor, 'suffix_futures', None)
        if suffix_futures is None:
            raise ValueError("Extractor missing suffix_futures; cannot derive transitions.")

        # Infer alphabet if not provided
        alphabet = (self.config.alphabet or getattr(extractor, 'alphabet', None) or
                    self._infer_alphabet(suffix_futures))
        alphabet = list(sorted(alphabet))

        # Map suffix -> state id
        suffix_to_state: Dict[str, int] = {}
        for state in cs:
            sid = state['id']
            for suf in state.get('suffixes', []):
                suffix_to_state[suf] = sid

        # Emission counts and extension destination counts
        emission_counts: Dict[int, Counter] = {s['id']: Counter() for s in cs}
        extension_dest_counts: Dict[int, Dict[str, Counter]] = {
            s['id']: {sym: Counter() for sym in alphabet} for s in cs
        }

        for state in cs:
            sid = state['id']
            for suf in state.get('suffixes', []):
                futures = suffix_futures.get(suf, {})
                # futures: symbol -> count
                for symbol, cnt in futures.items():
                    if symbol not in alphabet:
                        continue  # ignore out-of-alphabet symbols
                    emission_counts[sid][symbol] += cnt
                    extended = suf + symbol
                    dest_state = suffix_to_state.get(extended)
                    if dest_state is not None:
                        extension_dest_counts[sid][symbol][dest_state] += cnt

        # Build deterministic transitions by selecting modal destination
        transitions: Dict[int, Dict[str, int]] = {}
        for state in cs:
            sid = state['id']
            transitions[sid] = {}
            for symbol in alphabet:
                dest_ctr = extension_dest_counts[sid][symbol]
                if dest_ctr:
                    # Choose modal destination; break ties by: larger count, then lower id
                    max_count = max(dest_ctr.values())
                    candidates = [d for d, c in dest_ctr.items() if c == max_count]
                    dest = min(candidates)
                    transitions[sid][symbol] = dest
                else:
                    if self.config.ensure_full_alphabet:
                        transitions[sid][symbol] = sid if self.config.missing_symbol_self_loop else None
                    # else: leave symbol absent

        # Compute emission probabilities per state with optional smoothing.
        emissions: Dict[int, Dict[str, float]] = {}
        alpha = self.config.smoothing
        for state in cs:
            sid = state['id']
            counts = emission_counts[sid]
            if alpha > 0:
                # Apply additive smoothing over full alphabet including missing symbols.
                for sym in alphabet:
                    counts[sym] += alpha
            total = sum(counts.values())
            if total == 0:
                # Uniform fallback
                prob = 1.0 / len(alphabet) if alphabet else 1.0
                emissions[sid] = {sym: prob for sym in alphabet}
            else:
                emissions[sid] = {sym: counts.get(sym, 0.0) / total for sym in alphabet}

        # Choose start state
        if self.config.choose_start == 'first':
            start_state = cs[0]['id']
        else:  # largest_count
            start_state = max(cs, key=lambda s: s.get('count', 0))['id']

        # Validation flags
        unifilar = self._validate_unifilar(transitions)
        complete = self._validate_complete(transitions, alphabet) if self.config.ensure_full_alphabet else True

        machine = {
            'num_states': len(cs),
            'alphabet': alphabet,
            'states': [s['id'] for s in cs],
            'start_state': start_state,
            'transitions': transitions,
            'emissions': emissions,
            'causal_state_info': {
                s['id']: {
                    'suffixes': list(s.get('suffixes', [])),
                    'future_probabilities': s.get('future_probabilities', {}),
                    'count': s.get('count', 0),
                    'size': s.get('size', len(s.get('suffixes', [])))
                } for s in cs
            },
            'validation': {
                'unifilar': unifilar,
                'complete': complete
            }
        }

        if self.config.verbose:
            print(f"Canonical machine: {machine['num_states']} states | alphabet={alphabet} | unifilar={unifilar} | complete={complete}")

        return machine

    # ------------------------ Helpers ------------------------

    @staticmethod
    def _infer_alphabet(suffix_futures: Dict[str, Dict[str, int]]) -> List[str]:
        symbols = set()
        for fut in suffix_futures.values():
            symbols.update(fut.keys())
        if not symbols:
            raise ValueError("Cannot infer alphabet; no future symbols recorded.")
        return list(symbols)

    @staticmethod
    def _validate_unifilar(transitions: Dict[int, Dict[str, int]]) -> bool:
        # transitions already deterministic (symbol->single state); always True if symbol appears once.
        for sid, mapping in transitions.items():
            for sym, dest in mapping.items():
                if isinstance(dest, (list, tuple, set)):
                    return False
        return True

    @staticmethod
    def _validate_complete(transitions: Dict[int, Dict[str, int]], alphabet: Iterable[str]) -> bool:
        alpha = set(alphabet)
        for sid, mapping in transitions.items():
            if not alpha.issubset(mapping.keys()):
                return False
        return True


def build_canonical_epsilon_machine(extractor: Any, config: Optional[CanonicalBuildConfig] = None) -> Dict[str, Any]:
    """Convenience function to build the canonical epsilon machine.

    Example
    -------
    >>> from canonical_epsilon_machine import build_canonical_epsilon_machine
    >>> machine = build_canonical_epsilon_machine(extractor)
    >>> print(machine['transitions'])
    """
    builder = CanonicalEpsilonMachineBuilder(config=config)
    return builder.from_extractor(extractor)

__all__ = [
    'CanonicalBuildConfig',
    'CanonicalEpsilonMachineBuilder',
    'build_canonical_epsilon_machine'
]


if __name__ == "__main__":  # Simple smoke test placeholder
    class _DummyExtractor:
        def __init__(self):
            self.causal_states = [
                {'id': 0, 'suffixes': ['0', '10'], 'count': 15, 'size': 2, 'future_probabilities': {'0': 0.6, '1': 0.4}},
                {'id': 1, 'suffixes': ['1', '01'], 'count': 12, 'size': 2, 'future_probabilities': {'0': 0.3, '1': 0.7}},
            ]
            self.suffix_futures = {
                '0': {'0': 6, '1': 4},
                '10': {'0': 5, '1': 0},
                '1': {'0': 2, '1': 5},
                '01': {'0': 1, '1': 4},
            }

    dummy = _DummyExtractor()
    m = build_canonical_epsilon_machine(dummy)
    print("Canonical transitions:", m['transitions'])
    print("Emissions:", m['emissions'])
