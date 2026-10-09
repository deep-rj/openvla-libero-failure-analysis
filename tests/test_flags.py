import numpy as np

from harness import flags

WAIT = 10
HZ = 20


def _still(n_steps, xyz=(0.0, 0.0, 0.9)):
    return np.tile(xyz, (n_steps, 1)).astype(float)


def test_moved_more_than_d_from_settled_pose_not_spawn_pose():
    pos = _still(40)
    pos[:WAIT, 2] = 0.97  # spawn height, settles during the wait
    assert flags.displaced(pos, WAIT, 0.02).sum() == 0
    pos[30:, 0] = 0.03
    assert flags.displaced(pos, WAIT, 0.02)[30:].all()


def test_wrong_object_needs_a_grasp_of_a_non_goal_object():
    held = np.zeros((40, 2), dtype=bool)
    moved = np.zeros((40, 2), dtype=bool)
    moved[25:, 1] = True
    assert not flags.wrong_object(held, moved, goal_objects=[0])
    held[20:30, 1] = True
    assert flags.wrong_object(held, moved, goal_objects=[0])
    assert not flags.wrong_object(held, moved, goal_objects=[0, 1])


def test_collision_disturbance_ignores_carried_objects_and_settling_after_release():
    pos = _still(80)
    held = np.zeros(80, dtype=bool)
    held[20:40] = True
    pos[20:40, 2] = np.linspace(0.9, 1.0, 20)  # carried up
    pos[40:45, 2] = np.linspace(1.0, 0.9, 5)  # dropped into place after release
    pos[45:, 2] = 0.9
    in_goal = np.zeros(80, dtype=bool)
    assert not flags.collision_disturbance(pos, held, in_goal, WAIT, d=0.02, settle=10)


def test_collision_disturbance_ignores_the_squeeze_just_before_a_grasp_onset():
    pos = _still(80)
    pos[32:40, 1] = np.linspace(0.0, 0.025, 8)  # fingers close and drag it before lift
    pos[40:, 1] = 0.025
    held = np.zeros(80, dtype=bool)
    held[40:60] = True
    in_goal = np.zeros(80, dtype=bool)
    assert not flags.collision_disturbance(pos, held, in_goal, WAIT, d=0.02, settle=10)


def _creep_then_lift():
    """Fingers close on a handle at step 20; the object creeps 3 cm with the gripper on the
    table, then lifts at step 120."""
    pos = _still(160)
    pos[20:120, 0] = np.linspace(0.0, 0.03, 100)
    pos[120:, 0] = 0.03
    pos[120:, 2] = 0.95
    held = np.zeros(160, dtype=bool)
    held[120:] = True
    contact = np.zeros(160, dtype=bool)
    contact[20:] = True
    return pos, held, contact


def test_collision_disturbance_ignores_creep_in_closed_fingers_that_leads_to_a_grasp():
    pos, held, contact = _creep_then_lift()
    in_goal = np.zeros(160, dtype=bool)
    assert not flags.collision_disturbance(
        pos, held, in_goal, WAIT, d=0.02, settle=10, contact=contact
    )


def test_collision_disturbance_flags_a_push_in_closed_fingers_that_never_becomes_a_grasp():
    pos, held, contact = _creep_then_lift()
    held[:] = False
    in_goal = np.zeros(160, dtype=bool)
    assert flags.collision_disturbance(pos, held, in_goal, WAIT, d=0.02, settle=10, contact=contact)


def test_collision_disturbance_flags_an_object_knocked_while_not_held():
    pos = _still(80)
    pos[50:, 0] = np.minimum(0.003 * np.arange(30), 0.05)  # pushed 5 cm, never grasped
    held = np.zeros(80, dtype=bool)
    in_goal = np.zeros(80, dtype=bool)
    assert flags.collision_disturbance(pos, held, in_goal, WAIT, d=0.02, settle=10)
    # from passing 2 cm until the object comes to rest
    assert flags.collision_disturbance(pos, held, in_goal, WAIT, d=0.02, settle=10) == [(57, 68)]


def test_collision_disturbance_ignores_motion_inside_its_goal_container():
    pos = _still(80)
    pos[50:, 1] = np.minimum(0.007 * np.arange(30), 0.15)  # bowl rides the closing drawer
    held = np.zeros(80, dtype=bool)
    in_goal = np.zeros(80, dtype=bool)
    in_goal[40:] = True
    assert not flags.collision_disturbance(pos, held, in_goal, WAIT, d=0.02, settle=10)


def test_wrong_target_needs_release_and_rest_on_another_receptacle():
    n = 60
    cand = np.zeros((n, 1), dtype=bool)
    cand[30:, 0] = True
    held = np.zeros(n, dtype=bool)
    held[:35] = True
    speed = np.zeros(n)
    speed[:40] = 0.05
    assert flags.wrong_target(cand, held, speed, WAIT, rest_speed=0.01, dwell=10)
    assert not flags.wrong_target(cand[:45], held[:45], speed[:45], WAIT, 0.01, 10)


def test_carrying_over_another_receptacle_is_not_wrong_target():
    n = 60
    cand = np.zeros((n, 1), dtype=bool)
    cand[20:30, 0] = True
    held = np.ones(n, dtype=bool)
    speed = np.full(n, 0.05)
    assert not flags.wrong_target(cand, held, speed, WAIT, rest_speed=0.01, dwell=10)


def test_regression_needs_a_counted_subgoal_to_stop_counting_for_n_steps():
    count = np.zeros((60, 1), dtype=bool)
    count[20:40, 0] = True
    assert flags.regression(count, n=10)
    count[40:45, 0] = True
    count[45:52, 0] = False
    count[52:, 0] = True
    assert not flags.regression(count, n=10)


def test_order_violation_when_a_dependent_holds_before_its_prerequisite():
    n = 60
    stable = np.zeros((n, 2), dtype=bool)
    stable[20:, 0] = True  # drawer closed
    count = np.zeros((n, 2), dtype=bool)
    assert flags.order_violation(stable, count, deps=[(1,), ()])
    count[15:, 1] = True  # bowl already placed
    assert not flags.order_violation(stable, count, deps=[(1,), ()])


def test_dropped_is_a_release_followed_by_a_fall_outside_the_target():
    z = np.full(60, 1.0)
    held = np.zeros(60, dtype=bool)
    held[20:30] = True
    z[30:36] = np.linspace(1.0, 0.9, 6)
    z[36:] = 0.9
    over = np.zeros(60, dtype=bool)
    assert flags.dropped(z, held, over, fall_m=0.03, window=10)
    over[25:] = True
    assert not flags.dropped(z, held, over, fall_m=0.03, window=10)


def test_setting_an_object_down_is_not_dropped():
    z = np.full(60, 0.9)
    held = np.zeros(60, dtype=bool)
    held[20:30] = True
    assert not flags.dropped(z, held, np.zeros(60, dtype=bool), fall_m=0.03, window=10)


def test_regrasps_are_the_grasp_onsets_after_the_first():
    held = np.zeros(60, dtype=bool)
    assert flags.regrasps(held, WAIT) == []
    held[10:20] = True
    held[30:35] = True
    held[40:45] = True
    assert flags.regrasps(held, WAIT) == [(30, 31), (40, 41)]


def test_intervals_format_as_python_slices_and_points():
    assert flags.format_intervals([]) == ""
    assert flags.format_intervals([(145, 175), (265, 266)]) == "145:175;265"


def test_intervals_parse_back_to_the_same_list():
    for intervals in ([], [(145, 175), (265, 266)], [(10, 530)]):
        assert flags.parse_intervals(flags.format_intervals(intervals)) == intervals


def test_stalled_reports_each_still_run_end_exclusive():
    eef = _still(120, (0.0, 0.0, 1.0))
    eef[WAIT:60, 0] = 0.002 * np.arange(50)
    eef[60:, 0] = eef[59, 0]
    assert flags.stalled(eef, HZ, WAIT, v_min=0.005, seconds=2.0) == [(60, 120)]


def test_dropped_reports_the_release_step():
    z = np.full(60, 1.0)
    held = np.zeros(60, dtype=bool)
    held[20:30] = True
    z[30:36] = np.linspace(1.0, 0.9, 6)
    z[36:] = 0.9
    assert flags.dropped(z, held, np.zeros(60, dtype=bool), fall_m=0.03, window=10) == [(30, 31)]


def test_stalled_needs_the_end_effector_still_for_t_seconds():
    eef = _still(120, (0.0, 0.0, 1.0))
    eef[WAIT:60, 0] = 0.002 * np.arange(50)
    eef[60:, 0] = eef[59, 0]
    assert flags.stalled(eef, HZ, WAIT, v_min=0.005, seconds=2.0)
    assert not flags.stalled(eef[:90], HZ, WAIT, v_min=0.005, seconds=2.0)


def test_oscillation_is_repeated_reversals_within_the_window():
    t = np.arange(100)
    eef = _still(100, (0.0, 0.0, 1.0))
    eef[:, 0] = 0.03 * np.sin(t / 2.0)  # a reversal about every 6 steps
    assert flags.oscillation(eef, WAIT, window=80, reversals=6, min_travel=0.02)


def test_reaching_down_and_lifting_is_not_oscillation():
    eef = _still(100, (0.0, 0.0, 1.0))
    eef[20:40, 2] = np.linspace(1.0, 0.9, 20)
    eef[40:60, 2] = np.linspace(0.9, 1.0, 20)
    assert not flags.oscillation(eef, WAIT, window=80, reversals=6, min_travel=0.02)


def test_reversals_below_the_travel_threshold_are_not_oscillation():
    eef = _still(100, (0.0, 0.0, 1.0))
    eef[:, 0] = 0.005 * np.sin(np.arange(100) / 2.0)
    assert not flags.oscillation(eef, WAIT, window=80, reversals=6, min_travel=0.02)


def test_gripper_chatter_counts_empty_open_close_cycles():
    width = np.full(80, 0.08)
    for start in (20, 30, 40):
        width[start : start + 5] = 0.0
    held = np.zeros(80, dtype=bool)
    assert flags.gripper_chatter(width, held, WAIT, window=40, cycles=3)
    held[20:50] = True
    assert not flags.gripper_chatter(width, held, WAIT, window=40, cycles=3)


def test_object_off_table_is_below_its_settled_height():
    z = np.full(60, 0.9)
    assert not flags.object_off_table(z, settled_z=0.9, drop_m=0.05)
    z[40:] = 0.3
    assert flags.object_off_table(z, settled_z=0.9, drop_m=0.05)


def test_workspace_limit_uses_a_fraction_of_each_joint_range():
    ranges = [(-3.0, 3.0), (-3.07, -0.07)]
    q = np.zeros((60, 2))
    q[:, 1] = -1.5
    assert not flags.workspace_limit(q, ranges, WAIT, fraction=0.02, n=10)
    q[30:45, 1] = -0.10  # within 2% of 3.0 rad (0.06) of the -0.07 limit
    assert flags.workspace_limit(q, ranges, WAIT, fraction=0.02, n=10)
    q[30:45, 1] = -0.20
    assert not flags.workspace_limit(q, ranges, WAIT, fraction=0.02, n=10)


def test_truncated_mid_action_needs_a_grasp_or_motion_at_the_cap():
    assert not flags.truncated_mid_action(False, held_last=False, moving_last=False, last_step=519)
    assert flags.truncated_mid_action(False, held_last=True, moving_last=False, last_step=519)
    assert flags.truncated_mid_action(False, held_last=False, moving_last=True, last_step=519)
    assert not flags.truncated_mid_action(True, held_last=True, moving_last=True, last_step=519)
    assert flags.truncated_mid_action(False, True, False, last_step=519) == [(519, 520)]


def _clean_place_episode(n_steps=120):
    """Object lifted from step 30, carried over the target, released at 60, rests there."""
    eef = _still(n_steps, (0.0, 0.0, 1.0))
    obj = _still(n_steps, (0.0, 0.0, 0.9))
    width = np.full(n_steps, 0.08)
    for t in range(30, 60):
        eef[t] = [0.003 * (t - 30), 0.0, 0.95 + 0.002 * (t - 30)]
        obj[t] = eef[t] - [0.0, 0.0, 0.05]
        width[t] = 0.04
    obj[60:] = obj[59]
    eef[60:] = eef[59] + [0.0, 0.0, 0.05]
    holds = np.zeros((n_steps, 1), dtype=bool)
    holds[58:, 0] = True
    extracted = {
        "goal_holds": holds,
        "over_target": holds.copy(),
        "candidate_holds": np.zeros((n_steps, 1, 1), dtype=bool),
        "fixture_q": np.full((n_steps, 1), np.nan),
        "object_pos": obj[:, None, :],
        "eef_pos": eef,
        "gripper_qpos": np.stack([width / 2, -width / 2], axis=1),
        "arm_q": np.zeros((n_steps, 1)),
    }
    info = {
        "final_success": False,
        "num_wait_steps": WAIT,
        "control_freq_hz": HZ,
        "subgoals": [
            {"kind": "place", "threshold": float("nan"), "depends_on": [], "object_index": 0,
             "candidates": ["other_region"]}
        ],
        "arm_joints": [{"name": "j", "qpos_addr": 0, "range": [-3.0, 3.0]}],
    }  # fmt: skip
    return extracted, info


def test_clean_pick_and_place_raises_no_flags():
    extracted, info = _clean_place_episode()
    result = flags.compute_flags(extracted, info)
    assert "oscillation" in result
    assert result.pop("regrasp_count") == []
    assert result.pop("stalled")  # the arm rests for the last 3 s
    assert not any(result.values()), result


def test_engaged_within_needs_closed_fingers_near_the_object_in_the_last_steps():
    n_steps = 200
    obj = _still(n_steps)
    eef = _still(n_steps, (0.0, 0.0, 1.0))
    width = np.full(n_steps, 0.08)
    assert not flags.engaged_within(obj, eef, width, steps=100)
    width[99] = 0.03  # 100 steps before the last step
    assert flags.engaged_within(obj, eef, width, steps=100)
    assert not flags.engaged_within(obj, eef, width, steps=99)


def test_engaged_within_ignores_closed_fingers_far_from_the_object():
    n_steps = 50
    obj = _still(n_steps)
    eef = _still(n_steps, (0.0, 0.3, 1.0))
    width = np.full(n_steps, 0.03)
    assert not flags.engaged_within(obj, eef, width, steps=40)
