import pytest

from harness import summary


def _row(task, success, *, seed=0, steps=300, sps=0.27, terminated=None):
    return {
        "episode_id": f"e-{task}-{success}-{seed}-{steps}",
        "task_id": task,
        "seed": seed,
        "success": success,
        "num_steps": steps,
        "seconds_per_step": sps,
        "terminated_by": terminated or ("success" if success else "max_steps"),
    }


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0)],
)
def test_wilson_interval_matches_reference_values(k, n, low, high):
    got_low, got_high = summary.wilson_interval(k, n)
    assert got_low == pytest.approx(low, abs=1e-3)
    assert got_high == pytest.approx(high, abs=1e-3)


def test_wilson_interval_of_an_empty_sample_is_uninformative():
    assert summary.wilson_interval(0, 0) == (0.0, 1.0)


def test_suite_rate_reports_n_and_an_interval():
    rows = [_row(0, True)] * 6 + [_row(0, False)] * 4
    result = summary.summarize(rows)
    assert result["n"] == 10
    assert result["successes"] == 6
    assert result["success_rate"] == pytest.approx(0.6)
    low, high = result["ci95"]
    assert low < 0.6 < high


def test_per_task_rates_carry_their_own_n_and_interval():
    rows = [_row(0, True)] * 3 + [_row(0, False)] + [_row(7, False)] * 4
    result = summary.summarize(rows)
    assert result["per_task"][0]["n"] == 4
    assert result["per_task"][0]["success_rate"] == pytest.approx(0.75)
    assert result["per_task"][7]["success_rate"] == 0.0
    assert set(result["per_task"]) == {0, 7}


def test_throughput_is_seconds_per_episode_and_per_policy_step():
    rows = [_row(0, True, steps=100, sps=0.5), _row(0, False, steps=300, sps=0.3)]
    result = summary.summarize(rows)
    assert result["seconds_per_episode"] == pytest.approx((50 + 90) / 2)
    assert result["seconds_per_policy_step"] == pytest.approx((50 + 90) / 400)
    assert result["mean_policy_steps"] == pytest.approx(200)


def test_termination_reasons_are_counted():
    rows = [_row(0, True), _row(0, False), _row(1, False)]
    assert summary.summarize(rows)["terminated_by"] == {"success": 1, "max_steps": 2}


def test_a_single_seed_has_no_per_seed_breakdown():
    assert "per_seed" not in summary.summarize([_row(0, True)])


def test_multiple_seeds_report_a_rate_per_seed_and_their_spread():
    rows = [_row(0, True, seed=0)] * 4 + [_row(0, True, seed=1)] * 2 + [_row(0, False, seed=1)] * 2
    result = summary.summarize(rows)
    assert result["per_seed"][0]["success_rate"] == 1.0
    assert result["per_seed"][1]["success_rate"] == 0.5
    assert result["seed_rate_range"] == (0.5, 1.0)


def test_an_empty_manifest_is_rejected():
    with pytest.raises(ValueError, match="no episodes"):
        summary.summarize([])


def test_markdown_states_the_headline_number_with_n_and_interval():
    text = summary.render_markdown(summary.summarize([_row(0, True)] * 6 + [_row(0, False)] * 4))
    assert "6/10" in text
    assert "60.0%" in text
    assert "95%" in text
    assert "seconds per episode" in text


def test_write_outputs_reads_the_manifest_and_writes_markdown_and_json(tmp_path):
    import json

    from harness import manifest

    row = {
        "episode_id": "a", "task_suite": "libero_10", "task_id": 3, "task_instruction": "t",
        "init_state_idx": 0, "seed": 0, "success": True, "num_steps": 100,
        "terminated_by": "success", "episode_file": "e.h5", "video_file": "",
        "git_sha": "x", "seconds_per_step": 0.25,
    }  # fmt: skip
    manifest.append_row(tmp_path / "manifest.csv", row)
    manifest.append_row(tmp_path / "manifest.csv", {**row, "episode_id": "b", "success": False})

    summary.write_outputs(tmp_path)

    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["n"] == 2
    assert data["per_task"]["3"]["n"] == 2
    assert "1/2" in (tmp_path / "summary.md").read_text()
