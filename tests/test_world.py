import numpy as np
import pytest

pytest.importorskip("libero.libero")

from envs import build_env

from harness import world

pytestmark = pytest.mark.env

SUBSTEP_TOL = 5e-3


def _build(task_id):
    env, obs, _, _ = build_env(task_id)
    return env, obs


@pytest.fixture(scope="module")
def scene():
    cache = {}

    def get(task_id):
        if task_id not in cache:
            cache[task_id] = _build(task_id)
        return cache[task_id]

    yield get
    for env, _ in cache.values():
        env.close()


def test_basket_task_lists_its_objects_and_has_no_fixture_joints(scene):
    env, _ = scene(7)
    layout = world.scene_layout(env)
    assert layout.object_names == tuple(env.env.objects_dict)
    assert len(layout.object_names) == 5
    assert layout.fixture_joints == ()


def test_cabinet_task_exposes_three_slide_drawer_joints(scene):
    env, _ = scene(3)
    joints = world.scene_layout(env).fixture_joints
    assert len(joints) == 3
    assert {j.type for j in joints} == {"slide"}
    assert all("cabinet" in j.name for j in joints)
    assert all(j.range == pytest.approx((-0.16, 0.01)) for j in joints)


@pytest.mark.parametrize(("task_id", "fixture"), [(9, "microwave"), (2, "stove")])
def test_microwave_and_stove_tasks_expose_one_hinge_joint(scene, task_id, fixture):
    env, _ = scene(task_id)
    joints = world.scene_layout(env).fixture_joints
    assert len(joints) == 1
    assert joints[0].type == "hinge"
    assert fixture in joints[0].name


@pytest.mark.parametrize("task_id", [5, 8])
def test_static_fixtures_are_recorded_once_as_poses(scene, task_id):
    env, _ = scene(task_id)
    layout = world.scene_layout(env)
    assert set(layout.static_fixture_poses) == set(env.env.fixtures_dict)
    assert all(len(pose) == 7 for pose in layout.static_fixture_poses.values())


def test_world_state_agrees_with_obs_and_the_simulator(scene):
    env, obs = scene(3)
    layout = world.scene_layout(env)
    state = world.read_world_state(env, obs, layout)

    for row, name in zip(state.object_poses, layout.object_names, strict=True):
        assert np.array_equal(row[:3], obs[f"{name}_pos"].astype(np.float32))
        assert np.array_equal(row[3:], obs[f"{name}_quat"].astype(np.float32))
    qpos = env.sim.data.qpos
    expected = [qpos[j.qpos_addr] for j in layout.fixture_joints]
    assert np.array_equal(state.fixture_joint_qpos, np.array(expected, dtype=np.float32))
    assert np.array_equal(state.eef_pos, obs["robot0_eef_pos"])
    assert np.array_equal(state.eef_quat_raw, obs["robot0_eef_quat"])
    assert np.array_equal(state.gripper_qpos, obs["robot0_gripper_qpos"])
    assert state.sim_state.shape == (1 + env.sim.model.nq + env.sim.model.nv,)


def test_object_poses_equal_the_free_joint_slice_of_sim_state(scene):
    env, obs = scene(7)
    layout = world.scene_layout(env)
    state = world.read_world_state(env, obs, layout)

    for row, name in zip(state.object_poses, layout.object_names, strict=True):
        addr = env.sim.model.get_joint_qpos_addr(f"{name}_joint0")[0]
        pos_wxyz = state.sim_state[1 + addr : 1 + addr + 7]
        assert row[:3] == pytest.approx(pos_wxyz[:3], abs=1e-6)
        assert row[3:] == pytest.approx(pos_wxyz[[4, 5, 6, 3]], abs=1e-6)


def test_restoring_a_mid_episode_state_in_a_fresh_env_reproduces_the_logged_state():
    env, obs = _build(3)
    layout = world.scene_layout(env)
    rng = np.random.default_rng(0)
    for _ in range(15):
        obs, *_ = env.step(np.append(rng.uniform(-0.5, 0.5, 6), -1.0).tolist())
    logged = world.read_world_state(env, obs, layout)
    env.close()

    fresh, _ = _build(3)
    restored_obs = fresh.set_init_state(logged.sim_state)
    restored = world.read_world_state(fresh, restored_obs, world.scene_layout(fresh))
    fresh.close()

    assert restored.sim_state == pytest.approx(logged.sim_state, abs=1e-12)
    assert restored.fixture_joint_qpos == pytest.approx(logged.fixture_joint_qpos, abs=1e-9)
    # Obs-derived poses lag qpos by one 2 ms physics substep; restore recomputes them exactly.
    assert restored.eef_pos == pytest.approx(logged.eef_pos, abs=SUBSTEP_TOL)
    assert restored.eef_quat_raw == pytest.approx(logged.eef_quat_raw, abs=SUBSTEP_TOL)
    assert restored.object_poses == pytest.approx(logged.object_poses, abs=SUBSTEP_TOL)
