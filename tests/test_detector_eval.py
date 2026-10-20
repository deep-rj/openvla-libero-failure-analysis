import pytest
from harness.detector_eval import AttemptScore

from harness import detector_eval


def _score(i: int, failed: bool, v1: float, v2: float, width_delay: int | None = None):
    return AttemptScore(
        episode_id=f"t00-i{i:03d}",
        task_id=0,
        object="bowl",
        episode_success=str(not failed),
        failed=failed,
        v1=v1,
        v2=v2,
        width_delay=width_delay,
    )


def _separated(n: int = 6) -> list[AttemptScore]:
    fails = [
        _score(i, True, 1.5 + i / 100, 0.9, width_delay=20 if i % 2 else None) for i in range(n)
    ]
    succs = [_score(n + i, False, 1.0 + i / 100, 0.1) for i in range(n)]
    return fails + succs


THRESHOLDS = {"v1": 1.2, "v2": 0.5}


def test_fully_separated_scores_pass_with_every_failure_caught():
    r = detector_eval.evaluate(_separated(), THRESHOLDS, level=0.95, n_boot=50, seed=0)
    assert (r["n_failure"], r["n_success"]) == (6, 6)
    assert r["v1"]["auroc"] == 1.0
    assert (r["v1"]["caught"], r["v1"]["false_alarms"]) == (6, 0)
    assert r["v1"]["excess_over_chance"] == 1.0
    assert r["v1"]["passes"]


def test_width_overlap_counts_caught_failures_it_also_fires_on():
    r = detector_eval.evaluate(_separated(), THRESHOLDS, level=0.95, n_boot=50, seed=0)
    assert r["width"] == {"caught": 3, "false_alarms": 0}
    assert (r["v1"]["width_also_catches"], r["v1"]["width_never_catches"]) == (3, 3)
    assert r["v1"]["width_delay_median"] == 20.0


def test_alarms_on_successes_are_false_alarms_against_the_random_baseline():
    scores = _separated()
    r = detector_eval.evaluate(scores, {"v1": 1.025, "v2": 0.5}, level=0.95, n_boot=50, seed=0)
    assert r["v1"]["false_alarms"] == 3
    assert r["v1"]["false_alarm_rate"] == 0.5
    assert r["v1"]["random_alarm_expected"] == 3.0
    assert r["v1"]["excess_over_chance"] == pytest.approx(0.5)


def test_scores_round_trip_through_csv_exactly(tmp_path):
    scores = [_score(0, True, 1.0 + 1e-9 / 3, 0.123456789012345), _score(1, False, 1.0, 0.5, 7)]
    path = tmp_path / "scores.csv"
    detector_eval.write_scores(path, scores)
    assert detector_eval.read_scores(path) == scores
