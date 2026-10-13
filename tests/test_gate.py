import numpy as np
import pytest
from harness.gate import Descent

from harness import gate
from test_mechanism import _close, _lift, _reach, _scene, _sync

WAIT = 10


def _open(x, t0):
    pads = x["finger_pad_pos"]
    for t in range(t0, len(pads)):
        mid = pads[t, :, 0].mean()
        pads[t, 0, 0], pads[t, 1, 0] = mid - 0.04, mid + 0.04


def _grasp_and_carry(x, t0=20, t_bottom=60):
    _reach(x, t0, t_bottom, at_xy=(0.0, 0.0))
    _close(x, t_bottom + 1, to_half_gap=0.016)
    _lift(x, t_bottom + 20, carry=True)


def test_a_working_grasp_is_anchored_at_the_bottom_of_its_descent():
    x, info, _ = _scene()
    _grasp_and_carry(x)
    x = _sync(x)
    anchor = gate.success_anchor(x, info, gate.grasp_descents(x, info), g=0)
    assert anchor is not None and abs(anchor.bottom - 60) <= 1


def test_a_success_after_a_miss_is_anchored_on_the_grasp_that_worked():
    x, info, _ = _scene(n=300)
    _reach(x, 20, 60, at_xy=(0.06, 0.0))
    _close(x, 61)
    _lift(x, 80, carry=False)
    _open(x, 110)
    _grasp_and_carry(x, t0=120, t_bottom=160)
    x = _sync(x)
    descents = gate.grasp_descents(x, info)
    assert [abs(d.bottom - b) <= 1 for d, b in zip(descents, (60, 160), strict=True)] == [True] * 2
    assert abs(gate.success_anchor(x, info, descents, g=0).bottom - 160) <= 1


def test_a_success_with_no_grasp_of_the_object_has_no_anchor():
    x, info, _ = _scene()
    x = _sync(x)
    assert gate.success_anchor(x, info, gate.grasp_descents(x, info), g=0) is None


def test_a_failure_is_anchored_on_the_descent_nearest_its_labeled_onset():
    x, info, _ = _scene()
    x = _sync(x)
    descents = [Descent(60, 0, 61), Descent(160, 0, None)]
    assert gate.failure_anchor(x, info, descents, onset=150, tol=60) == descents[1]
    assert gate.failure_anchor(x, info, descents, onset=20, tol=60) == descents[0]


def test_a_failure_with_no_descent_within_tolerance_has_no_anchor():
    x, info, _ = _scene(n=400)
    x = _sync(x)
    assert gate.failure_anchor(x, info, [Descent(60, 0, 61)], onset=300, tol=60) is None


def test_a_slip_is_anchored_on_the_grasp_that_slipped_not_a_nearer_reattempt():
    x, info, _ = _scene(n=300)
    _grasp_and_carry(x)
    obj = x["object_pos"]
    obj[120:] = obj[119]
    obj[120:, 0, 2] = 0.03
    _open(x, 130)
    _reach(x, 135, 150, at_xy=(0.0, 0.0))
    _close(x, 151, to_half_gap=0.006)
    x = _sync(x)
    descents = gate.grasp_descents(x, info)
    anchor = gate.failure_anchor(x, info, descents, onset=120, tol=60)
    assert abs(anchor.bottom - 60) <= 1


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [(100, list(range(50, 100))), (40, list(range(WAIT, 40))), (15, None)],
)
def test_the_pre_window_skips_wait_steps_and_needs_a_minimum_length(anchor, expected):
    is_wait = np.arange(200) < WAIT
    window = gate.pre_window(is_wait, anchor, length=50, min_length=10)
    assert (None if window is None else window.tolist()) == expected


def test_near_tie_rate_is_the_share_of_window_steps_within_the_margin():
    margins = np.array([0.0, 3.0, 4.0, 5.0, 100.0, 1.0])
    assert gate.near_tie_rate(margins, np.array([1, 2, 3, 4]), max_ulps=4) == 0.5


def test_auroc_counts_ties_as_half():
    assert gate.auroc([3.0, 2.0], [1.0, 2.0]) == pytest.approx((2 + 1.5) / 4)
    assert gate.auroc([0.0], [1.0]) == 0.0


def _unit(ep, task, obj, group, value):
    return gate.Unit(ep, task, obj, group, value)


def test_the_same_task_auroc_only_compares_units_in_the_same_stratum():
    units = [
        _unit("a", 0, "can", "failure", 5.0),
        _unit("b", 0, "can", "success", 1.0),
        _unit("c", 1, "mug", "failure", 0.0),
        _unit("d", 1, "mug", "success", 2.0),
    ]
    assert gate.same_task_auroc(units) == 0.5
    assert gate.auroc([5.0, 0.0], [1.0, 2.0]) == 0.5
    units[2] = _unit("c", 1, "mug", "failure", 3.0)
    assert gate.same_task_auroc(units) == 1.0


def test_the_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    units = [_unit(f"f{i}", i % 2, "can", "failure", v) for i, v in enumerate(rng.normal(1, 1, 40))]
    units += [_unit(f"s{i}", i % 2, "can", "success", v) for i, v in enumerate(rng.normal(0, 1, 60))]
    lo, hi = gate.bootstrap_ci(units, gate.same_task_auroc, level=0.95, n_boot=500, seed=1)
    assert (lo, hi) == gate.bootstrap_ci(units, gate.same_task_auroc, 0.95, n_boot=500, seed=1)
    assert lo < gate.same_task_auroc(units) < hi
    assert lo > 0.5


def test_the_bootstrap_resamples_whole_episodes():
    units = [
        _unit("s", 0, "can", "success", 0.0),
        _unit("s", 0, "mug", "success", 0.0),
        _unit("f", 0, "can", "failure", 1.0),
        _unit("g", 0, "mug", "failure", 1.0),
    ]
    for seed in range(5):
        draw = gate.resample_episodes(units, np.random.default_rng(seed))
        assert sum(u.episode_id == "s" for u in draw) == 2
        assert sum(u.group == "failure" for u in draw) == 2


def test_window_speed_is_the_mean_step_displacement_times_the_rate():
    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10) * 0.01
    assert gate.window_speed(pos, np.array([2, 3, 4]), hz=20) == pytest.approx(0.2)


def test_window_rotation_speed_measures_the_angle_between_steps():
    angles = np.arange(10) * 0.05
    quats = np.stack([np.zeros(10), np.zeros(10), np.sin(angles / 2), np.cos(angles / 2)], axis=1)
    assert gate.window_rotation_speed(quats, np.array([1, 2]), hz=20) == pytest.approx(1.0)


def test_residuals_remove_the_linear_part_explained_by_the_covariate():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2 * x + np.array([0.1, -0.1, 0.1, -0.1])
    assert gate.residualize(y, x) == pytest.approx(y - np.polyval(np.polyfit(x, y, 1), x))
    assert abs(np.corrcoef(gate.residualize(y, x), x)[0, 1]) < 1e-9


def test_stop_rank_is_where_the_no_motion_bin_sits_among_the_top_two():
    probs = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]])
    assert gate.stop_rank(probs, zero_bin=1).tolist() == [0, 2, 1]


def test_window_mean_over_a_subset_skips_units_with_no_qualifying_step():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    window = np.array([0, 1, 2])
    assert gate.subset_mean(values, window, np.array([True, False, True, True])) == 2.0
    assert gate.subset_mean(values, window, np.array([False, False, False, True])) is None


def test_far_mass_is_the_probability_at_least_k_bins_from_the_top_bin():
    probs = np.zeros((2, 30))
    probs[0, [5, 6, 20]] = [0.6, 0.3, 0.1]
    probs[1, [5, 14, 15]] = [0.5, 0.2, 0.3]
    assert gate.far_mass(probs, k=10) == pytest.approx([0.1, 0.3])


def test_the_anchor_is_the_bottom_of_its_own_descent_not_a_lower_point_left_by_a_miss():
    x, info, _ = _scene(n=300)
    _reach(x, 20, 60, at_xy=(0.06, 0.0))
    _close(x, 61)
    pads = x["finger_pad_pos"]
    pads[61:90, :, 2] = 0.03
    pads[90:110, :, 2] = np.linspace(0.03, 0.10, 20)[:, None]
    _open(x, 95)
    _reach(x, 110, 130, at_xy=(0.0, 0.0), bottom_z=0.05)
    _close(x, 131, to_half_gap=0.016)
    _lift(x, 150, carry=True)
    x = _sync(x)
    anchor = gate.success_anchor(x, info, gate.grasp_descents(x, info), g=0)
    assert abs(anchor.bottom - 130) <= 1
