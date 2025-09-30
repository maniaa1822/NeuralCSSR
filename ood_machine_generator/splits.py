from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


_EMISSION_HOLDOUT = {5, 11}


def categorize_state_count(num_states: int) -> str:
    if 2 <= num_states <= 6:
        return "train"
    if num_states == 7:
        return "val"
    if 8 <= num_states <= 9:
        return "test"
    return "extra"


def categorize_emission(emission_info: Sequence[Mapping[str, Any]]) -> str:
    holdout = any(item.get("grid_index") in _EMISSION_HOLDOUT for item in emission_info if item.get("mode") == "grid+jitter")
    if holdout:
        return "test"
    if all(item.get("mode") == "beta" for item in emission_info):
        return "continuous"
    if any(item.get("mode") == "grid+jitter" for item in emission_info):
        return "train"
    return "mixed"


def categorize_topology(metadata: Mapping[str, Any]) -> str:
    family = metadata.get("family")
    if family in {"even_process_like", "forbidden_substring", "lollipop"}:
        return "test"
    return "train"
