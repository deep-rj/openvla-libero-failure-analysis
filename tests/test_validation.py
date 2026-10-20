import numpy as np
import pytest

from harness import validation


def test_identical_action_sequences_are_reported_as_bitwise_identical():
    a = np.random.default_rng(0).normal(size=(20, 7))
    result = validation.compare_actions(a, a.copy())
    assert result == {"identical": True, "first_divergent_step": None, "max_abs_diff": 0.0}


def test_the_first_divergent_step_and_size_of_the_difference_are_reported():
    a = np.zeros((20, 7))
    b = a.copy()
    b[7, 2] = 1e-3
    b[12, 0] = 5e-3
    result = validation.compare_actions(a, b)
    assert result["identical"] is False
    assert result["first_divergent_step"] == 7
    assert result["max_abs_diff"] == pytest.approx(5e-3)


def test_a_shorter_episode_diverges_where_it_ends():
    a = np.zeros((20, 7))
    result = validation.compare_actions(a, a[:15])
    assert result["identical"] is False
    assert result["first_divergent_step"] == 15


def test_top2_margin_is_the_gap_between_the_two_largest_logits_per_dim():
    logits = np.zeros((7, 256), dtype=np.float16)
    logits[:, 10] = 5.0
    logits[:, 20] = 4.0
    logits[3, 30] = 4.75
    margins = validation.top2_margin(logits)
    assert margins.shape == (7,)
    assert margins[0] == pytest.approx(1.0)
    assert margins[3] == pytest.approx(0.25)


def _rows(n=4, steps=300, s_per_step=0.27):
    return [{"num_steps": steps, "seconds_per_step": s_per_step} for _ in range(n)]


def test_cost_projection_scales_measured_episode_time_to_the_baseline():
    report = validation.cost_report(
        _rows(), episode_bytes=[2.0e6] * 4, baseline_episodes=250, dollars_per_hour=0.75
    )
    per_episode = 300 * 0.27
    assert report["seconds_per_step"] == pytest.approx(0.27)
    assert report["projected_gpu_hours"] == pytest.approx(250 * per_episode / 3600)
    assert report["projected_dollars"] == pytest.approx(250 * per_episode / 3600 * 0.75)
    assert report["mean_episode_bytes"] == pytest.approx(2.0e6)
    assert report["size_far_from_budget"] is False


@pytest.mark.parametrize("size", [0.5e6, 6e6])
def test_episode_sizes_far_from_the_two_megabyte_budget_are_flagged(size):
    report = validation.cost_report(_rows(), [size] * 4, 250, 0.75)
    assert report["size_far_from_budget"] is True


def test_the_markdown_report_states_the_numbers_a_reader_needs():
    report = validation.cost_report(_rows(), [2.0e6] * 4, 250, 0.75)
    text = validation.render_markdown(
        {"t1": {"identical": True, "first_divergent_step": None, "max_abs_diff": 0.0}},
        {"t2": []},
        report,
    )
    assert "0.27" in text
    assert "T1" in text
    assert "T2" in text
    assert f"{report['projected_dollars']:.2f}" in text


def test_cli_defaults_match_the_spec_validation_run():
    cfg = validation.parse_args(["--out", "runs/a"])
    assert cfg.tasks == [0, 7, 3, 9]
    assert cfg.episodes_per_task == 2
    assert cfg.determinism_task == 0
    assert cfg.truncation_steps == 60
    assert cfg.baseline_episodes == 250
    assert cfg.policy == "stub"
    assert cfg.parity is False
    assert cfg.max_steps is None


def test_cli_flags_select_the_real_policy_with_parity_checking():
    cfg = validation.parse_args(
        ["--out", "runs/a", "--policy", "openvla", "--parity", "--dollars-per-hour", "0.69"]
    )
    assert cfg.policy == "openvla"
    assert cfg.parity is True
    assert cfg.dollars_per_hour == pytest.approx(0.69)
