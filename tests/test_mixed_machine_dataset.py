import json
import math
from collections import Counter
from pathlib import Path

from mixed_machine_regimes.mixed_dataset import (
    generate_switching_dataset,
    generate_union_dataset,
    resolve_machines,
    save_mixed_dataset,
)
from machines import get_machine


def test_union_dataset_lengths(tmp_path):
    machines = resolve_machines(["golden_mean", "even_process"])
    lengths = [10, 6]
    specs = list(zip(machines, lengths))
    result = generate_union_dataset(specs, seed=123)
    assert result.length == sum(lengths)
    assert result.machine_ids.count("golden_mean") == 10
    assert result.machine_ids.count("even_process") == 6
    assert len(result.state_labels) == result.length
    paths = save_mixed_dataset(result, output_dir=tmp_path, dataset_name="union_test")
    assert (tmp_path / "union_test" / "combined.dat").is_file()
    meta = json.loads(Path(paths["meta"]).read_text())
    assert meta["mode"] == "union"
    assert "expected_optimal_loss" in meta
    assert "machine_stats" in meta
    assert set(meta["machine_stats"].keys()) == {"golden_mean", "even_process"}
    assert "expected_belief_loss" in meta
    assert "joint_state_stats" in meta
    assert "belief_machine" in meta
    assert "states" in meta["belief_machine"]


def test_switching_dataset_segments(tmp_path):
    machines = resolve_machines(["golden_mean", "even_process"])
    total_length = 12
    switch_interval = 3
    result = generate_switching_dataset(
        machines,
        total_length=total_length,
        switch_interval=switch_interval,
        seed=7,
    )
    assert result.length == total_length
    assert len(result.machine_ids) == total_length
    segment_lengths = [seg.length for seg in result.segments]
    assert sum(segment_lengths) == total_length
    # With two machines and total_length divisible by switch_interval, expect alternating segments.
    assert segment_lengths == [switch_interval] * (total_length // switch_interval)
    paths = save_mixed_dataset(result, output_dir=tmp_path, dataset_name="switch_test")
    meta = json.loads(Path(paths["meta"]).read_text())
    assert meta["mode"] == "switch"
    assert meta["expected_optimal_loss"]["bits"] > 0
    assert set(meta["machine_counts"].keys()) == {"golden_mean", "even_process"}
    assert "expected_belief_loss" in meta
    assert meta["expected_belief_loss"]["bits"] > 0
    assert "belief_machine" in meta


def test_expected_loss_matches_gt(tmp_path):
    machines = resolve_machines(["golden_mean", "seven_state_human"])
    lengths = [20, 10]
    result = generate_union_dataset(list(zip(machines, lengths)), seed=99)
    paths = save_mixed_dataset(result, output_dir=tmp_path, dataset_name="gm_seven")
    meta = json.loads(Path(paths["meta"]).read_text())
    expected_bits = meta["expected_optimal_loss"]["bits"]

    gm_entropy = get_machine("golden_mean").compute_theoretical_entropy() / math.log(2)
    seven_entropy = get_machine("seven_state_human").compute_theoretical_entropy() / math.log(2)
    total = sum(lengths)
    manual_bits = (lengths[0] / total) * gm_entropy + (lengths[1] / total) * seven_entropy
    assert abs(expected_bits - manual_bits) < 1e-9
    manual_belief_bits = compute_manual_joint_bits(result.state_labels)
    assert abs(meta["expected_belief_loss"]["bits"] - manual_belief_bits) < 1e-9


def compute_manual_joint_bits(state_labels):
    counts = Counter(state_labels)
    total = len(state_labels)
    total = max(total, 1)
    bits = 0.0
    for label, count in counts.items():
        machine_name, state_name = label.split(":", 1)
        machine = get_machine(machine_name)
        emission_probs = machine.emissions[state_name]
        state_entropy_bits = 0.0
        for prob in emission_probs.values():
            if prob > 0:
                state_entropy_bits -= prob * math.log(prob, 2)
        bits += (count / total) * state_entropy_bits
    return bits
