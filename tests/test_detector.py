import numpy as np

from harness import detector

PARAMS = detector.WidthBaseline(threshold_m=0.003, reopen_m=0.002, match_tol=10)


def _closing(n: int, command: int, settle: float, steps: int = 20) -> np.ndarray:
    """Width open at 80 mm until `command`, then a linear close to `settle` over `steps`."""
    w = np.full(n, 0.08)
    w[command : command + steps] = np.linspace(0.08, settle, steps)
    w[command + steps :] = settle
    return w


def test_each_open_to_close_switch_is_one_command():
    cmd = np.array([-1, -1, 1, 1, 1, -1, 1, -1], dtype=float)
    assert detector.close_commands(cmd) == [2, 6]


def test_a_close_held_from_the_first_step_is_not_a_command():
    assert detector.close_commands(np.array([1, 1, -1, 1], dtype=float)) == [3]


def test_the_command_nearest_the_anchor_within_tolerance_is_matched():
    assert detector.nearest_command([40, 63, 75], anchor=60, tol=10) == 63


def test_no_command_within_tolerance_matches_nothing():
    assert detector.nearest_command([40, 80], anchor=60, tol=10) is None


def test_an_empty_close_fires_at_the_first_step_below_the_threshold():
    w = _closing(200, command=50, settle=0.001)
    # linspace(0.08, 0.001, 20) first drops below 3 mm at index 19
    assert detector.empty_close_step(w, command=50, params=PARAMS) == 69


def test_a_close_on_an_object_never_fires():
    w = _closing(200, command=50, settle=0.03)
    assert detector.empty_close_step(w, command=50, params=PARAMS) is None


def test_a_slow_close_with_no_reopening_fires_however_late():
    w = _closing(300, command=50, settle=0.0005, steps=80)
    # linspace(0.08, 0.0005, 80) first drops below 3 mm at index 77
    assert detector.empty_close_step(w, command=50, params=PARAMS) == 127


def test_an_empty_close_after_the_fingers_reopen_belongs_to_another_attempt():
    w = np.full(200, 0.08)
    w[50:60] = np.linspace(0.08, 0.005, 10)
    w[60:80] = np.linspace(0.005, 0.08, 20)
    w[80:100] = np.linspace(0.08, 0.001, 20)
    w[100:] = 0.001
    assert detector.empty_close_step(w, command=50, params=PARAMS) is None


def test_an_attempt_ends_when_the_fingers_open_past_their_narrowest_by_the_margin():
    w = _closing(200, command=50, settle=0.03)
    w[100:] = 0.03 + 0.001 * np.arange(1, 101)
    assert detector.attempt_end(w, command=50, reopen_m=0.002) == 102


def test_finger_jitter_within_the_margin_does_not_end_an_attempt():
    w = _closing(200, command=50, settle=0.03)
    w[100:110] = 0.0315
    assert detector.attempt_end(w, command=50, reopen_m=0.002) == 200


def test_a_narrow_width_before_the_command_does_not_fire():
    w = np.full(200, 0.04)
    w[:50] = 0.001
    assert detector.empty_close_step(w, command=50, params=PARAMS) is None


def test_the_window_mean_covers_the_policy_steps_before_the_command():
    values = np.arange(20, dtype=float)
    is_wait = np.zeros(20, dtype=bool)
    is_wait[:3] = True
    # steps 7..9 before command 10
    assert detector.window_mean(values, is_wait, command=10, length=3) == 8.0
    # wait steps 0..2 are skipped: steps 3..4 remain
    assert detector.window_mean(values, is_wait, command=5, length=5, min_window=2) == 3.5


def test_a_window_shorter_than_the_minimum_has_no_mean():
    is_wait = np.zeros(20, dtype=bool)
    assert detector.window_mean(np.ones(20), is_wait, command=4, length=30, min_window=10) is None


def test_the_window_mean_can_keep_a_subset_of_steps():
    values = np.arange(10, dtype=float)
    keep = np.array([True, False] * 5)
    assert detector.window_mean(values, np.zeros(10, bool), 10, 4, keep=keep, min_window=1) == 7.0


def test_the_percentile_rank_is_the_share_of_reference_values_at_or_below():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert [detector.percentile_rank(ref, x) for x in (0.5, 2.0, 2.5, 9.0)] == [0, 0.5, 0.5, 1]


def test_the_budget_threshold_lets_at_most_the_budget_of_successes_alarm():
    successes = np.arange(10, dtype=float)
    t = detector.budget_threshold(successes, budget=0.2)
    assert (successes > t).sum() == 2


def test_the_v2_score_averages_the_two_percentile_ranks():
    g_ref, r_ref = np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0])
    assert detector.v2_score(3.0, 15.0, g_ref, r_ref) == (0.75 + 0.25) / 2


def test_the_v2_score_falls_back_to_the_gripper_when_roll_has_no_window():
    g_ref, r_ref = np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0])
    assert detector.v2_score(3.0, None, g_ref, r_ref) == 0.75


def test_excess_over_chance_is_the_catch_rate_minus_the_false_alarm_rate():
    alarm = np.array([True, True, False, False, True, False, False, False])
    failed = np.array([True, True, True, True, False, False, False, False])
    assert detector.excess_over_chance(alarm, failed) == 0.5 - 0.25
