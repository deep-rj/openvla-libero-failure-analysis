import numpy as np
import pytest

pytest.importorskip("libero.libero")

from envs import build_env

from harness import world, world_extract

pytestmark = pytest.mark.env


@pytest.fixture(scope="module")
def subgoals():
    cache = {}

    def get(task_id):
        if task_id not in cache:
            env, _, _, _ = build_env(task_id)
            try:
                names = world.scene_layout(env).object_names
                cache[task_id] = world_extract.build_subgoals(env.env, names)
            finally:
                env.close()
        return cache[task_id]

    return get


def _by_verb(subs):
    return {s.predicate[0]: s for s in subs}


@pytest.mark.parametrize("task_id", [3, 9])
def test_closing_a_container_depends_on_placing_into_it(subgoals, task_id):
    subs = subgoals(task_id)
    close, place = _by_verb(subs)["close"], _by_verb(subs)["in"]
    assert close.depends_on == (subs.index(place),)
    assert place.depends_on == ()


@pytest.mark.parametrize("task_id", [0, 2, 4, 6, 8])
def test_unordered_tasks_have_no_dependencies(subgoals, task_id):
    assert all(s.depends_on == () for s in subgoals(task_id))


def test_fixture_thresholds_match_libero(subgoals):
    assert _by_verb(subgoals(2))["turnon"].threshold == 0.5
    assert _by_verb(subgoals(3))["close"].threshold == 0.0
    assert _by_verb(subgoals(9))["close"].threshold == -0.005


def test_caddy_compartments_are_wrong_target_candidates(subgoals):
    (book,) = subgoals(5)
    assert "desk_caddy_1_front_contain_region" in book.candidates
    assert "desk_caddy_1_back_contain_region" not in book.candidates
    assert not any("study_table" in c for c in book.candidates)


def test_other_plate_is_a_wrong_target_candidate(subgoals):
    white_mug = next(s for s in subgoals(4) if s.predicate[1] == "porcelain_mug_1")
    assert "plate_2" in white_mug.candidates
    assert "plate_1" not in white_mug.candidates


def test_fixtures_are_moved_to_the_logged_poses():
    env, _, _, _ = build_env(5)
    try:
        logged = world.scene_layout(env).static_fixture_poses
        env.reset()
        assert world.scene_layout(env).static_fixture_poses != logged
        world_extract.place_fixtures(env, logged)
        placed = world.scene_layout(env).static_fixture_poses
        for name, pose in logged.items():
            assert placed[name] == pytest.approx(pose, abs=1e-6)
    finally:
        env.close()


def test_arm_joints_are_the_seven_hinges_with_their_ranges():
    env, _, _, _ = build_env(0)
    try:
        joints = world_extract.arm_joints(env.env)
    finally:
        env.close()
    assert [j["name"] for j in joints] == [f"robot0_joint{i}" for i in range(1, 8)]
    joint4 = joints[3]
    assert joint4["range"] == pytest.approx([-3.0718, -0.0698], abs=1e-3)


@pytest.fixture(scope="module")
def extracted_stub_episode(tmp_path_factory):
    from harness import policy, runner, schema

    out = tmp_path_factory.mktemp("run")
    cfg = runner.RunConfig(
        out_dir=out, tasks=[0], init_states=[0], max_steps=6, video=False, camera_size=64
    )
    rows = runner.run(cfg, policy_factory=lambda c: (policy.StubPolicy(seed=1), {}))
    path = out / rows[0]["episode_file"]
    x, info = world_extract.extract_episode(path)
    arrays, _ = schema.read_world_columns(path)
    return x, info, arrays


def test_extracted_object_quats_equal_the_free_joint_quats_in_sim_state(
    extracted_stub_episode, reset_env
):
    x, info, arrays = extracted_stub_episode
    model = reset_env[0].sim.model
    for i, name in enumerate(info["object_names"]):
        addr = model.get_joint_qpos_addr(f"{name}_joint0")[0]
        # sim_state[0] is the simulation time; the free joint stores position then wxyz.
        wxyz = arrays["sim_state"][:, 1 + addr + 3 : 1 + addr + 7]
        expected = wxyz[:, [1, 2, 3, 0]] / np.linalg.norm(wxyz, axis=1, keepdims=True)
        assert x["object_quat"][:, i] == pytest.approx(expected, abs=1e-12)


def test_extracted_finger_pads_have_one_position_per_finger_per_step(extracted_stub_episode):
    x, _, _ = extracted_stub_episode
    assert x["finger_pad_pos"].shape == (len(x["step_idx"]), 2, 3)
    assert x["grip_quat"].shape == (len(x["step_idx"]), 4)


def test_extracted_info_records_the_pad_half_size(extracted_stub_episode):
    _, info, _ = extracted_stub_episode
    assert info["finger_pad_half_size"] == pytest.approx([0.004, 0.008, 0.008])


@pytest.fixture(scope="module")
def reset_env():
    env, obs, _, _ = build_env(0)
    yield env, obs
    env.close()


def _set_fingers(env, q_left, q_right):
    robot = env.env.robots[0]
    qpos = env.sim.data.qpos
    for addr, q in zip(robot._ref_gripper_joint_pos_indexes, (q_left, q_right), strict=True):
        qpos[addr] = q
    env.sim.forward()


def _grip_frame(env):
    from robosuite.utils.transform_utils import quat2mat

    return quat2mat(world_extract.grip_quat(env.env))


def _gap(pads):
    return np.linalg.norm(pads[1] - pads[0])


def test_finger_pad_separation_tracks_the_finger_opening(reset_env):
    env, _ = reset_env
    _set_fingers(env, 0.01, -0.01)
    narrow = world_extract.finger_pad_positions(env.env)
    _set_fingers(env, 0.03, -0.03)
    wide = world_extract.finger_pad_positions(env.env)
    assert _gap(wide) - _gap(narrow) == pytest.approx(0.04, abs=1e-6)


def test_finger_pads_differ_only_along_the_grip_frame_x_axis(reset_env):
    env, _ = reset_env
    _set_fingers(env, 0.02, -0.02)
    pads = world_extract.finger_pad_positions(env.env)
    in_grip = _grip_frame(env).T @ (pads[1] - pads[0])
    assert in_grip[[1, 2]] == pytest.approx([0, 0], abs=1e-6)


def test_grip_frame_z_axis_points_down_at_reset(reset_env):
    env, _ = reset_env
    assert _grip_frame(env)[:, 2] @ [0, 0, -1] > 0.99


def test_object_quats_match_the_env_observation_convention(reset_env):
    env, obs = reset_env
    names = world.scene_layout(env).object_names
    quats = world_extract.object_quats(env.env, names)
    expected = np.array([obs[f"{name}_quat"] for name in names])
    assert quats == pytest.approx(expected, abs=1e-6)


def test_finger_pad_half_size_is_thinnest_along_the_closing_axis(reset_env):
    env, _ = reset_env
    assert world_extract.finger_pad_half_size(env.env) == pytest.approx([0.004, 0.008, 0.008])


def test_collision_boxes_placed_by_the_object_pose_match_the_sim(reset_env):
    from robosuite.utils.transform_utils import quat2mat

    env, _ = reset_env
    inner, model, data = env.env, env.sim.model, env.sim.data
    names = world.scene_layout(env).object_names
    boxes = world_extract.object_collision_boxes(inner, names)
    for name, quat in zip(names, world_extract.object_quats(inner, names), strict=True):
        body = inner.obj_body_id[name]
        geoms = [
            g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]
        ]
        assert len(boxes[name]) == len(geoms)
        # robosuite's quat2mat computes in float32, hence the 1e-6 tolerances.
        r_obj, p_obj = quat2mat(quat), data.body_xpos[body]
        for box, g in zip(boxes[name], geoms, strict=True):
            assert p_obj + r_obj @ box["pos"] == pytest.approx(data.geom_xpos[g], abs=1e-6)
            r_box = r_obj @ quat2mat(box["quat_xyzw"])
            assert r_box == pytest.approx(data.geom_xmat[g].reshape(3, 3), abs=1e-6)
            assert box["half_size"] == pytest.approx(model.geom_size[g])
