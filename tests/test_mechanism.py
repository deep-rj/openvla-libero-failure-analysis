import numpy as np
import pytest

from harness import mechanism

HALF = [0.004, 0.008, 0.008]
IDENTITY = [0.0, 0.0, 0.0, 1.0]
QZ90 = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]


def _pads(half_gap=0.02, axis=0):
    pads = np.zeros((2, 3))
    pads[0, axis], pads[1, axis] = -half_gap, half_gap
    return pads


def _box(half, pos=(0.0, 0.0, 0.0), quat=IDENTITY):
    return {"pos": list(pos), "quat_xyzw": list(quat), "half_size": list(half)}


def _cover(boxes, pads=None, grip_q=IDENTITY, obj_p=(0.0, 0.0, 0.0), obj_q=IDENTITY):
    pads = _pads() if pads is None else pads
    return mechanism.pad_face_cover(pads, grip_q, HALF, np.array(obj_p), obj_q, boxes)


def test_quat_to_mat_rotates_x_onto_y_for_90_degrees_about_z():
    assert mechanism.quat_to_mat(QZ90) @ [1, 0, 0] == pytest.approx([0, 1, 0])


def test_an_object_larger_than_the_pad_face_covers_all_of_it():
    assert _cover([_box([0.005, 0.02, 0.02])]) == 1.0


def test_an_object_over_half_the_pad_face_covers_about_half():
    assert _cover([_box([0.005, 0.008, 0.02], pos=(0, 0.008, 0))]) == pytest.approx(0.5, abs=0.07)


def test_an_object_beside_the_gap_or_past_the_pad_edge_covers_nothing():
    assert _cover([_box([0.005, 0.02, 0.02], pos=(0.05, 0, 0))]) == 0.0
    assert _cover([_box([0.005, 0.005, 0.02], pos=(0, 0.03, 0))]) == 0.0


def test_cover_does_not_depend_on_how_open_the_fingers_are():
    box = [_box([0.005, 0.005, 0.02])]
    assert _cover(box, pads=_pads(0.02)) == _cover(box, pads=_pads(0.04))


def test_the_object_pose_places_its_boxes():
    thin_in_x = [_box([0.02, 0.005, 0.02])]
    assert _cover(thin_in_x, obj_q=QZ90) == 1.0
    assert _cover(thin_in_x, obj_p=(0, 0.1, 0)) == 0.0


def test_the_grip_frame_sets_the_closing_axis():
    thin_in_y = [_box([0.02, 0.005, 0.02])]
    assert _cover(thin_in_y) == pytest.approx(5 / 9)
    assert _cover(thin_in_y, pads=_pads(axis=1), grip_q=QZ90) == 1.0


def test_closed_fingers_sample_the_line_at_the_pad_midpoint():
    assert _cover([_box([0.005, 0.02, 0.02])], pads=_pads(0.004)) == 1.0


P = mechanism.DEFAULT_PARAMS
WAIT = 10


def test_close_onsets_are_the_first_step_of_each_shrinking_run():
    gap = np.full(60, 0.08)
    gap[20:26] = np.linspace(0.08, 0.02, 6)
    gap[26:40] = 0.02
    gap[40:46] = np.linspace(0.02, 0.08, 6)
    gap[46:] = 0.08
    assert mechanism.close_onsets(gap, WAIT, P.close_rate_m) == [19]


def test_close_end_is_where_the_gap_stops_shrinking():
    gap = np.full(60, 0.08)
    gap[20:26] = np.linspace(0.08, 0.02, 6)
    gap[26:] = 0.02
    assert mechanism.close_end(gap, 19, P) in (24, 25)


def test_open_descents_need_depth_rise_and_no_close():
    z = np.full(100, 0.30)
    z[20:40] = np.linspace(0.30, 0.25, 20)
    z[40:50] = np.linspace(0.25, 0.28, 10)
    z[50:] = 0.28
    assert mechanism.open_descents(z, [], WAIT, P) == [(20, 39)]
    assert mechanism.open_descents(z, [42], WAIT, P) == []
    shallow = np.where(z < 0.29, 0.29, z)
    assert mechanism.open_descents(shallow, [], WAIT, P) == []


def test_held_runs_are_end_exclusive():
    held = np.zeros(20, dtype=bool)
    held[5:9] = held[12:20] = True
    assert mechanism.held_runs(held) == [(5, 9), (12, 20)]


def test_a_slip_leaves_the_hand_before_the_fingers_open():
    n = 60
    dist = np.full(n, 0.01)
    gap = np.full(n, 0.03)
    dist[40:] = np.linspace(0.01, 0.20, 20)  # object falls away from 40, 1 cm per step
    gap[45:] = 0.08  # fingers open later
    assert mechanism.leave_and_open(dist, gap, 10, 41, P) == (41, 45)
    gap[38:] = 0.08  # fingers open first: a release
    leave, opened = mechanism.leave_and_open(dist, gap, 10, 41, P)
    assert opened < leave


def test_fell_needs_a_drop_within_the_window():
    z = np.full(40, 0.20)
    z[21:] = 0.15
    assert mechanism.fell(z, 20, 0.03, 10)
    assert not mechanism.fell(np.full(40, 0.20), 20, 0.03, 10)


def test_first_shift_ignores_the_settle_after_onset_and_over_target_steps():
    off = np.zeros((50, 3))
    off[12:14, 0] = 0.008  # squeeze while settling
    off[14:, 0] = 0.0
    over = np.zeros(50, dtype=bool)
    assert mechanism.first_shift(off, 10, 50, over, P) is None
    off[30:, 0] = 0.01
    assert mechanism.first_shift(off, 10, 50, over, P) == 30
    over[25:] = True
    assert mechanism.first_shift(off, 10, 50, over, P) is None


def test_final_descent_start_walks_back_through_a_plateau():
    z = np.concatenate([np.linspace(0.2, 0.3, 10), np.linspace(0.3, 0.22, 10), np.full(10, 0.22)])
    assert mechanism.final_descent_start(z, 29) == 9


def test_tilt_is_the_angle_between_settled_and_current_up_axes():
    tilt_x30 = [np.sin(np.pi / 12), 0, 0, np.cos(np.pi / 12)]
    assert mechanism.tilt_deg(IDENTITY, tilt_x30) == pytest.approx(30.0)
    assert mechanism.tilt_deg(QZ90, QZ90) == pytest.approx(0.0, abs=1e-6)


def test_progress_rises_with_count_then_with_the_open_subgoal_level():
    count = np.zeros((6, 2), dtype=bool)
    count[4:, 0] = True
    levels = np.array([[0, 0], [1, 0], [2, 0], [2, 0], [0, 0], [0, 1]])
    assert mechanism.progress(count, levels, [(), ()]).tolist() == [0, 1, 2, 2, 3, 4]


def test_progress_ignores_a_subgoal_whose_prerequisite_does_not_count():
    count = np.zeros((3, 2), dtype=bool)
    levels = np.array([[0, 0], [0, 1], [0, 2]])  # drawer closing before the bowl is in
    assert mechanism.progress(count, levels, [(), (0,)]).tolist() == [0, 0, 0]


def test_a_slow_descent_with_level_stretches_is_one_descent():
    z = np.full(120, 0.30)
    z[20:100] = np.repeat(np.linspace(0.30, 0.24, 8), 10)  # steps down with level stretches
    z[100:] = 0.24
    assert mechanism.open_descents(z, [], WAIT, P) == [(29, 90)]


def test_an_open_descent_that_parks_until_the_end_counts():
    z = np.full(100, 0.30)
    z[20:40] = np.linspace(0.30, 0.25, 20)
    z[40:] = 0.25
    assert mechanism.open_descents(z, [], WAIT, P) == [(20, 39)]


CAN = [_box([0.01, 0.01, 0.03])]
NO_FLAGS = {"wrong_object": [], "wrong_target": []}


def _scene(n=200, name="can_1"):
    """One goal object resting at the origin; pads open, 30 cm above it, closing along x."""
    pads = np.zeros((n, 2, 3))
    pads[:, :, 2] = 0.30
    pads[:, 0, 0], pads[:, 1, 0] = -0.04, 0.04
    obj = np.zeros((n, 1, 3))
    obj[:, 0, 2] = 0.03
    x = {
        "object_pos": obj,
        "object_quat": np.tile(IDENTITY, (n, 1, 1)),
        "finger_pad_pos": pads,
        "grip_quat": np.tile(IDENTITY, (n, 1)),
        "goal_holds": np.zeros((n, 1), dtype=bool),
        "over_target": np.zeros((n, 1), dtype=bool),
        "fixture_q": np.full((n, 1), np.nan),
        "candidate_holds": np.zeros((n, 1, 1), dtype=bool),
        "arm_q": np.zeros((n, 7)),
    }
    info = {
        "num_wait_steps": WAIT,
        "control_freq_hz": 20,
        "task_id": 0,
        "final_success": False,
        "object_names": [name],
        "finger_pad_half_size": HALF,
        "subgoals": [
            {
                "kind": "place",
                "object_index": 0,
                "target": "basket_1_contain_region",
                "candidates": [],
                "threshold": float("nan"),
                "depends_on": [],
            }
        ],
    }
    return x, info, {name: CAN}


def _sync(x):
    """eef and finger qpos from the pads, as the extraction would see them."""
    pads = x["finger_pad_pos"]
    x["eef_pos"] = pads.mean(axis=1)
    half_width = (mechanism.pad_gap(pads) - 0.007) / 2
    x["gripper_qpos"] = np.stack([half_width, -half_width], axis=1)
    return x


def _reach(x, t0, t_bottom, at_xy, bottom_z=0.03):
    pads = x["finger_pad_pos"]
    for t in range(t0, len(pads)):
        frac = min(1.0, (t - t0) / max(t_bottom - t0, 1))
        pads[t, :, 2] = 0.30 + frac * (bottom_z - 0.30)
        pads[t, :, 1] = at_xy[1]
        half = (pads[t, 1, 0] - pads[t, 0, 0]) / 2
        pads[t, 0, 0], pads[t, 1, 0] = at_xy[0] - half, at_xy[0] + half


def _close(x, t0, to_half_gap=0.006):
    pads = x["finger_pad_pos"]
    for t in range(t0, len(pads)):
        half = max(to_half_gap, 0.04 - 0.008 * (t - t0))
        mid = pads[t, :, 0].mean()
        pads[t, 0, 0], pads[t, 1, 0] = mid - half, mid + half


def _lift(x, t0, carry=True, dz=0.004):
    pads, obj = x["finger_pad_pos"], x["object_pos"]
    for t in range(t0, len(pads)):
        pads[t, :, 2] = pads[t0 - 1, :, 2] + dz * (t - t0 + 1)
        if carry:
            obj[t, 0] = pads[t].mean(axis=0)


def _classify(x, info, boxes, flags=NO_FLAGS):
    return mechanism.classify(_sync(x), info, flags, boxes, reference={})


def test_an_episode_with_no_error_abstains():
    x, info, boxes = _scene()
    pred = _classify(x, info, boxes)
    assert pred.abstained and pred.mechanism == ""


def test_grounding_flags_come_first():
    x, info, boxes = _scene()
    pred = _classify(x, info, boxes, {"wrong_object": [(50, 60)], "wrong_target": []})
    assert (pred.mechanism, pred.onset_step) == ("grounding", 50)


def test_a_close_beside_the_object_is_perception_at_the_bottom():
    x, info, boxes = _scene()
    _reach(x, 20, 60, at_xy=(0.06, 0.0))
    _close(x, 61)
    _lift(x, 80, carry=False)
    pred = _classify(x, info, boxes)
    assert pred.mechanism == "perception"
    assert abs(pred.onset_step - 60) <= 1


def test_a_close_around_the_object_that_does_not_lift_it_is_control_grasp():
    x, info, boxes = _scene()
    _reach(x, 20, 60, at_xy=(0.0, 0.0))
    _close(x, 61, to_half_gap=0.016)
    _lift(x, 80, carry=False)
    assert _classify(x, info, boxes).mechanism == "control: grasp"


def _edge_close(name):
    x, info, _ = _scene(name=name)
    _reach(x, 20, 60, at_xy=(0.0, 0.0))
    _close(x, 61, to_half_gap=0.016)
    _lift(x, 80, carry=False)
    sliver = {name: [_box([0.01, 0.001, 0.03], pos=(0, 0.0075, 0))]}
    return _classify(x, info, sliver)


def test_thin_handle_objects_count_any_cover_as_between():
    assert _edge_close("moka_pot_1").mechanism == "control: grasp"
    assert _edge_close("can_1").mechanism == "perception"


def test_a_recovered_miss_is_not_the_initiator():
    x, info, boxes = _scene(n=260)
    _reach(x, 20, 60, at_xy=(0.06, 0.0))
    _close(x, 61)
    pads = x["finger_pad_pos"]
    pads[80:, :, 2] = 0.30
    pads[80:, 0, 0], pads[80:, 1, 0] = -0.04, 0.04
    _reach(x, 100, 140, at_xy=(0.0, 0.0))
    _close(x, 141, to_half_gap=0.016)
    _lift(x, 150)
    assert _classify(x, info, boxes).mechanism != "perception"


def test_a_slip_in_transport_is_control_grasp_at_the_step_it_leaves_the_hand():
    x, info, boxes = _scene(n=260)
    _reach(x, 20, 60, at_xy=(0.0, 0.0))
    _close(x, 61, to_half_gap=0.016)
    _lift(x, 70)
    obj = x["object_pos"]
    obj[150:, 0, 2] = np.maximum(obj[149, 0, 2] - 0.02 * np.arange(1, 111), 0.03)
    obj[150:, 0, :2] = obj[149, 0, :2]
    pred = _classify(x, info, boxes)
    assert pred.mechanism == "control: grasp"
    assert 149 <= pred.onset_step <= 151


def test_a_shift_in_the_hand_that_ends_in_a_slip_is_the_slip():
    x, info, boxes = _scene(n=260)
    _reach(x, 20, 60, at_xy=(0.0, 0.0))
    _close(x, 61, to_half_gap=0.016)
    _lift(x, 70)
    obj = x["object_pos"]
    obj[70:, 0, 0] -= 0.01  # held 1 cm off the pad midpoint
    obj[144:150, 0, 0] += 0.006  # slides toward the midpoint (offset moves, distance shrinks)
    obj[150:, 0, 2] = np.maximum(obj[149, 0, 2] - 0.02 * np.arange(1, 111), 0.03)
    obj[150:, 0, :2] = obj[149, 0, :2]
    assert _classify(x, info, boxes).mechanism == "control: grasp"


def _two_objects_second_placed():
    """Object 0 is carried over its target and released, never counting; object 1 is then
    placed and counts."""
    x, info, boxes = _scene(n=300)
    n = 300
    for key in ("object_pos", "object_quat"):
        x[key] = np.concatenate([x[key], x[key]], axis=1)
    x["object_pos"][:, 1, 0] = 0.3
    for key in ("goal_holds", "over_target"):
        x[key] = np.zeros((n, 2), dtype=bool)
    x["fixture_q"] = np.full((n, 2), np.nan)
    x["candidate_holds"] = np.zeros((n, 2, 1), dtype=bool)
    info["object_names"] = ["can_1", "can_2"]
    info["subgoals"] = [dict(info["subgoals"][0]), dict(info["subgoals"][0], object_index=1)]
    boxes = {"can_1": CAN, "can_2": CAN}
    _reach(x, 20, 60, at_xy=(0.0, 0.0))
    _close(x, 61, to_half_gap=0.016)
    _lift(x, 70)
    x["over_target"][90:120, 0] = True
    pads, obj = x["finger_pad_pos"], x["object_pos"]
    pads[120:, 0, 0], pads[120:, 1, 0] = -0.04, 0.04  # release over the target; it stays put
    obj[120:, 0] = obj[119, 0]
    x["goal_holds"][200:, 1] = True  # the other object counts later
    return x, info, boxes


def test_a_descent_stopped_by_another_object_is_a_collision_not_a_grasp_attempt():
    x, info, boxes = _scene(n=200)
    for key in ("object_pos", "object_quat"):
        x[key] = np.concatenate([x[key], x[key]], axis=1)
    info["object_names"] = ["can_1", "milk_1"]
    boxes = {"can_1": CAN, "milk_1": CAN}
    obj = x["object_pos"]
    obj[:, 1, 0] = 0.06  # the milk stands beside the can
    _reach(x, 20, 60, at_xy=(0.06, 0.0), bottom_z=0.07)
    obj[62:80, 1, 0] = np.linspace(0.06, 0.12, 18)  # pushed over by the hand
    obj[80:, 1, 0] = 0.12
    pred = _classify(x, info, boxes)
    assert pred.mechanism == "control: grasp"
    assert "collision" in pred.reason


def _miss_then_aligned_close():
    x, info, boxes = _scene(n=260)
    _reach(x, 20, 60, at_xy=(0.06, 0.0))
    pads = x["finger_pad_pos"]
    pads[70:, :, 2] = 0.30
    pads[70:, 0, 0], pads[70:, 1, 0] = -0.04, 0.04
    _reach(x, 100, 140, at_xy=(0.0, 0.0))
    _close(x, 141, to_half_gap=0.016)
    _lift(x, 160, carry=False)
    return _sync(x), info, boxes


def test_a_miss_corrected_by_a_later_aligned_attempt_is_recovered_per_object():
    x, info, boxes = _miss_then_aligned_close()
    by_k = mechanism.classify(x, info, NO_FLAGS, boxes, reference={})
    params = mechanism.MechanismParams(per_object_recovery=True)
    per_object = mechanism.classify(x, info, NO_FLAGS, boxes, {}, params)
    assert by_k.mechanism == "perception"
    assert per_object.mechanism == "control: grasp"


def test_a_later_open_descent_over_the_object_does_not_recover_a_miss():
    x, info, boxes = _scene(n=260)
    _reach(x, 20, 60, at_xy=(0.06, 0.0))
    _close(x, 61)
    pads = x["finger_pad_pos"]
    pads[80:, :, 2] = 0.30
    pads[80:, 0, 0], pads[80:, 1, 0] = -0.04, 0.04
    _reach(x, 120, 160, at_xy=(0.0, 0.0))  # hovers down over it, never closes, parks
    params = mechanism.MechanismParams(per_object_recovery=True)
    pred = mechanism.classify(_sync(x), info, NO_FLAGS, boxes, {}, params)
    assert (pred.mechanism, pred.onset_step) == ("perception", 60)


def test_recovery_by_another_subgoal_counting_depends_on_the_recovery_mode():
    x, info, boxes = _two_objects_second_placed()
    _sync(x)
    by_k = mechanism.classify(x, info, NO_FLAGS, boxes, reference={})
    per_object = mechanism.classify(
        x, info, NO_FLAGS, boxes, {}, mechanism.MechanismParams(per_object_recovery=True)
    )
    assert by_k.abstained
    assert per_object.mechanism == "control: place"
