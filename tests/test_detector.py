import numpy as np

from harness import detector

PARAMS = detector.WidthBaseline(threshold_m=0.003, horizon=60, match_tol=10)


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


def test_width_is_the_sum_of_both_finger_travels():
    q = np.array([[0.04, -0.039], [0.002, -0.001]])
    np.testing.assert_allclose(detector.finger_width(q), [0.079, 0.003])


def test_an_empty_close_fires_at_the_first_step_below_the_threshold():
    w = _closing(200, command=50, settle=0.001)
    # linspace(0.08, 0.001, 20) first drops below 3 mm at index 19
    assert detector.empty_close_step(w, command=50, params=PARAMS) == 69


def test_a_close_on_an_object_never_fires():
    w = _closing(200, command=50, settle=0.03)
    assert detector.empty_close_step(w, command=50, params=PARAMS) is None


def test_a_close_that_finishes_after_the_horizon_does_not_fire():
    w = _closing(200, command=50, settle=0.001, steps=80)
    assert detector.empty_close_step(w, command=50, params=PARAMS) is None


def test_a_narrow_width_before_the_command_does_not_fire():
    w = np.full(200, 0.04)
    w[:50] = 0.001
    assert detector.empty_close_step(w, command=50, params=PARAMS) is None
