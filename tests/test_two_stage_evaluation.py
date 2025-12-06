import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cssr_discovery"))

from cssr_discovery.two_stage_oracle_cssr import (
    StageOneResult,
    StageTwoResult,
    TwoStageResult,
    evaluate_against_state_labels,
)


def build_simple_result():
    hist0 = np.array([0], dtype=np.int64)
    hist1 = np.array([1], dtype=np.int64)
    stage_one = StageOneResult(
        partition=[[hist0], [hist1]],
        representatives=[hist0, hist1],
        k_values_used=[1],
        state_of_history={tuple(hist0.tolist()): 0, tuple(hist1.tolist()): 1},
    )
    stage_two = StageTwoResult(
        minimal_suffixes=[
            [(1, (0,))],
            [(1, (1,))],
        ],
        suffix_to_states={},
    )
    return TwoStageResult(stage_one=stage_one, stage_two=stage_two, history_counts={})


def test_evaluate_against_state_labels_counts():
    result = build_simple_result()
    data = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    # labels align with next-symbol contexts; length matches data.
    state_labels = ["gm:A", "gm:A", "gm:B", "gm:B", "gm:A", "gm:B"]
    summary = evaluate_against_state_labels(result, state_labels, data, L=1)
    assert summary["positions_evaluated"] == len(data) - 1
    assert summary["positions_with_state"] == len(data) - 1
    assert 0 <= summary["coverage_ratio"] <= 1
    states = {entry["state"]: entry for entry in summary["state_alignment"]}
    assert states[0]["assignments"] > 0
    assert states[1]["assignments"] > 0
