import numpy as np
import pytest

from harness import milestones as ms

WAIT = 10


def _holds(n_steps, n_goals, spans):
    """`spans` maps subgoal -> list of (first, last) inclusive steps where the predicate holds."""
    holds = np.zeros((n_steps, n_goals), dtype=bool)
    for g, ranges in spans.items():
        for first, last in ranges:
            holds[first : last + 1, g] = True
    return holds


def _k(holds, deps, n):
    return ms.max_simultaneous(ms.counting(ms.debounced(holds, n, WAIT), deps))


def test_single_step_blip_does_not_count_under_debounce():
    holds = _holds(60, 1, {0: [(30, 30)]})
    assert _k(holds, [()], n=5)[0] == 0
    assert _k(holds, [()], n=1)[0] == 1


def test_window_includes_the_current_step():
    holds = _holds(60, 1, {0: [(30, 34)]})
    stable = ms.debounced(holds, 5, WAIT)
    assert stable[34, 0]
    assert not stable[33, 0]
    assert not ms.debounced(_holds(60, 1, {0: [(30, 33)]}), 5, WAIT).any()


def test_debounce_runs_along_time_not_across_subgoals():
    holds = _holds(60, 2, {0: [(20, 40)], 1: [(30, 30)]})
    stable = ms.debounced(holds, 5, WAIT)
    assert stable[:, 0].any()
    assert not stable[:, 1].any()


def test_window_never_reaches_into_wait_steps():
    holds = _holds(60, 1, {0: [(0, 59)]})
    stable = ms.debounced(holds, 5, WAIT)
    assert not stable[: WAIT + 4].any()
    assert stable[WAIT + 4, 0]


def test_dependent_subgoal_needs_its_own_predicate():
    # Drawer (subgoal 0) depends on the bowl (subgoal 1); bowl in, drawer still open.
    holds = _holds(60, 2, {1: [(20, 59)]})
    count = ms.counting(ms.debounced(holds, 1, WAIT), [(1,), ()])
    assert not count[:, 0].any()
    assert ms.max_simultaneous(count)[0] == 1


def test_dependent_subgoal_needs_its_prerequisite():
    # Drawer closed while the bowl is outside: closing first does not count.
    holds = _holds(60, 2, {0: [(20, 59)]})
    assert _k(holds, [(1,), ()], n=1)[0] == 0


def test_dependency_is_checked_against_its_own_column():
    # Bowl in only at 20-25; drawer closed from 40. They never hold together.
    holds = _holds(60, 2, {0: [(40, 59)], 1: [(20, 25)]})
    assert _k(holds, [(1,), ()], n=1)[0] == 1


def test_k_is_most_subgoals_at_once_not_ever_true():
    # A placed then knocked off, then B placed: never both at once.
    holds = _holds(60, 2, {0: [(20, 30)], 1: [(40, 59)]})
    k, step = _k(holds, [(), ()], n=1)
    assert (k, step) == (1, 20)


def test_k_reports_first_step_reaching_the_maximum():
    holds = _holds(60, 2, {0: [(20, 59)], 1: [(35, 59)]})
    assert _k(holds, [(), ()], n=1) == (2, 35)


def test_dependency_cycle_is_rejected():
    holds = _holds(20, 2, {})
    with pytest.raises(ValueError, match="cycle"):
        ms.counting(holds, [(1,), (0,)])


def _carry_trajectory(n_steps=60, lift_at=30):
    eef = np.zeros((n_steps, 3))
    eef[:, 2] = 1.0
    obj = np.zeros((n_steps, 3))
    obj[:, 2] = 0.9
    width = np.full(n_steps, 0.08)
    for t in range(lift_at, n_steps):
        rise = 0.01 * (t - lift_at + 1)
        eef[t, 2] = 0.95 + rise
        obj[t, 2] = 0.9 + rise
        width[t] = 0.04
    return obj, eef, width


def test_object_moving_with_closed_fingers_is_grasped():
    obj, eef, width = _carry_trajectory()
    held = ms.grasped(obj, eef, width, settled_z=0.9)
    assert not held[:30].any()
    assert held[45:].all()


def test_lifted_object_with_open_fingers_is_not_grasped():
    obj, eef, width = _carry_trajectory()
    width[:] = 0.08
    assert not ms.grasped(obj, eef, width, settled_z=0.9).any()


def test_object_not_following_the_end_effector_is_not_grasped():
    obj, eef, width = _carry_trajectory()
    obj[30:, 2] = 0.95  # lifted once, then left behind while the EEF rises
    assert not ms.grasped(obj, eef, width, settled_z=0.9)[45:].any()


def test_still_object_next_to_a_still_gripper_is_not_grasped():
    # An elevated object resting near a motionless gripper keeps a fixed offset trivially.
    n_steps = 40
    obj = np.tile([0.0, 0.0, 0.95], (n_steps, 1))
    eef = np.tile([0.0, 0.05, 1.0], (n_steps, 1))
    width = np.full(n_steps, 0.04)
    assert not ms.grasped(obj, eef, width, settled_z=0.9).any()


def test_still_object_next_to_a_slowly_moving_gripper_is_not_grasped():
    n_steps = 40
    obj = np.tile([0.0, 0.0, 1.05], (n_steps, 1))
    eef = np.tile([0.0, 0.10, 1.0], (n_steps, 1))
    eef[:, 2] -= 0.003 * np.arange(n_steps)
    width = np.full(n_steps, 0.04)
    assert not ms.grasped(obj, eef, width, settled_z=0.9).any()


def test_grasp_persists_while_the_end_effector_pauses():
    obj, eef, width = _carry_trajectory()
    obj[45:] = obj[45]
    eef[45:] = eef[45]
    held = ms.grasped(obj, eef, width, settled_z=0.9)
    assert held[45:].all()


def test_opening_the_fingers_ends_the_grasp():
    obj, eef, width = _carry_trajectory()
    obj[45:] = obj[45]
    eef[45:] = eef[45]
    width[50:] = 0.08
    held = ms.grasped(obj, eef, width, settled_z=0.9)
    assert held[45:50].all()
    assert not held[50:].any()


def test_travel_fraction_measures_from_the_settled_value():
    q = np.array([-0.20] * 10 + [-0.15, -0.10, -0.004, 0.001])
    frac = ms.travel_fraction(q, threshold=0.0, start=10)
    assert frac[10] == 0.0
    assert frac[12] == pytest.approx(0.146 / 0.15)
    assert frac[13] == 1.0


def test_travel_fraction_is_nan_when_already_at_threshold():
    q = np.full(20, 1.0)
    assert np.isnan(ms.travel_fraction(q, threshold=1.0, start=10)).all()


def _drawer_episode(bowl_in, drawer_q, success=False, n_steps=60):
    """Task-3 shape: subgoal 0 closes the drawer and depends on subgoal 1, the bowl in it."""
    holds = np.zeros((n_steps, 2), dtype=bool)
    for first, last in bowl_in:
        holds[first : last + 1, 1] = True
    holds[:, 0] = drawer_q >= 0.0
    extracted = {
        "goal_holds": holds,
        "over_target": np.zeros((n_steps, 2), dtype=bool),
        "fixture_q": np.stack([drawer_q, np.full(n_steps, np.nan)], axis=1),
        "object_pos": np.tile([0.0, 0.0, 0.9], (n_steps, 1, 1)),
        "eef_pos": np.tile([0.0, 0.0, 1.2], (n_steps, 1)),
        "gripper_qpos": np.tile([0.04, -0.04], (n_steps, 1)),
    }
    info = {
        "final_success": success,
        "num_wait_steps": WAIT,
        "subgoals": [
            {"kind": "fixture", "threshold": 0.0, "depends_on": [1], "object_index": -1},
            {"kind": "place", "threshold": float("nan"), "depends_on": [], "object_index": 0},
        ],
    }
    return extracted, info


def test_drawer_pushed_most_of_the_way_is_k1_partial():
    q = np.concatenate([np.full(40, -0.15), np.linspace(-0.15, -0.004, 20)])
    extracted, info = _drawer_episode([(20, 59)], q)
    m = ms.compute_milestone(extracted, info)
    assert (m.k, m.phase) == (1, "partial")
    assert ms.closure_fraction(extracted, info, 0) == pytest.approx(0.146 / 0.15, abs=1e-3)


def test_drawer_nudged_is_engaged():
    q = np.concatenate([np.full(40, -0.15), np.full(20, -0.14)])
    extracted, info = _drawer_episode([(20, 59)], q)
    assert ms.compute_milestone(extracted, info).phase == "engaged"


def test_drawer_motion_before_the_bowl_is_in_does_not_set_the_phase():
    q = np.concatenate([np.full(15, -0.15), np.full(45, -0.05)])
    extracted, info = _drawer_episode([], q)
    m = ms.compute_milestone(extracted, info)
    assert (m.k, m.phase) == (0, "none")


def test_success_sets_k_to_all_subgoals():
    q = np.full(60, -0.15)
    extracted, info = _drawer_episode([(20, 59)], q, success=True)
    assert ms.compute_milestone(extracted, info).k == 2


def _place_episode(holds_from, holds_to, release_at, n_steps=80):
    """One place subgoal: carried from step 30, released at `release_at`; `On` holds in a span."""
    obj, eef, width = _carry_trajectory(n_steps=n_steps, lift_at=30)
    obj[release_at:] = obj[release_at - 1]
    eef[release_at:] = eef[release_at - 1] + [0.0, 0.0, 0.05]
    width[release_at:] = 0.08
    holds = np.zeros((n_steps, 1), dtype=bool)
    holds[holds_from : holds_to + 1, 0] = True
    extracted = {
        "goal_holds": holds,
        "over_target": holds.copy(),
        "fixture_q": np.full((n_steps, 1), np.nan),
        "object_pos": obj[:, None, :],
        "eef_pos": eef,
        "gripper_qpos": np.stack([width / 2, -width / 2], axis=1),
    }
    info = {
        "final_success": False,
        "num_wait_steps": WAIT,
        "subgoals": [
            {"kind": "place", "threshold": float("nan"), "depends_on": [], "object_index": 0}
        ],
    }
    return extracted, info


def test_touchdown_while_gripped_that_slides_off_is_not_placed():
    extracted, info = _place_episode(holds_from=40, holds_to=60, release_at=61)
    m = ms.compute_milestone(extracted, info, ms.MilestoneParams(debounce=10))
    assert m.k == 0


def test_released_object_that_stays_is_placed_after_n_steps():
    extracted, info = _place_episode(holds_from=40, holds_to=79, release_at=45)
    m = ms.compute_milestone(extracted, info, ms.MilestoneParams(debounce=10))
    assert (m.k, m.k_step) == (1, 54)


def test_released_object_that_leaves_within_n_steps_is_not_placed():
    extracted, info = _place_episode(holds_from=40, holds_to=50, release_at=45)
    m = ms.compute_milestone(extracted, info, ms.MilestoneParams(debounce=10))
    assert m.k == 0
