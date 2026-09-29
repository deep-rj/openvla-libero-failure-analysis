import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("libero.libero")

from envs import build_env

from harness import checks, policy, rollout, schema, video

TASK_ID = 9  # microwave: has an articulated fixture joint
MAX_STEPS = 6


class SpyEnv:
    """Delegates to a real env and records every action passed to `step`."""

    def __init__(self, env):
        self._env = env
        self.sent = []

    def step(self, action):
        self.sent.append(np.array(action, dtype=np.float64))
        return self._env.step(action)

    def __getattr__(self, name):
        return getattr(self._env, name)


@pytest.fixture(scope="module")
def scene():
    env, _, label, init_state = build_env(TASK_ID)
    yield env, label, init_state
    env.close()


def _run(scene, *, mode="logged", seed=3, spy=None, **kwargs):
    env, label, init_state = scene
    return rollout.run_episode(
        spy or env,
        policy.StubPolicy(seed=seed, out_of_slice_rate=0.2),
        label,
        init_state,
        max_steps=MAX_STEPS,
        image_fn=lambda obs: obs["agentview_image"],
        mode=mode,
        **kwargs,
    )


def test_logging_does_not_change_the_physics(scene):
    env = scene[0]
    first = _run(scene, mode="reference")
    state_first = env.get_sim_state().copy()
    second = _run(scene, mode="reference")
    assert np.array_equal(first.executed_actions, second.executed_actions)
    assert np.array_equal(state_first, env.get_sim_state()), "reference runs must be repeatable"

    logged = _run(scene, mode="logged")
    assert np.array_equal(logged.executed_actions, first.executed_actions)
    assert np.array_equal(env.get_sim_state(), state_first)


def test_a_real_env_episode_passes_the_self_consistency_checks(scene, tmp_path):
    spy = SpyEnv(scene[0])
    result = _run(scene, spy=spy)
    schema.write_episode(tmp_path / "ep.h5", result.steps, {})
    arrays, _ = schema.read_episode(tmp_path / "ep.h5")

    stats = policy.LIBERO_10_ACTION_STATS
    assert checks.check_self_consistency(arrays, stats, spy.sent) == []
    assert checks.check_wait_alignment(arrays, rollout.NUM_STEPS_WAIT) == []
    assert arrays["fixture_joint_qpos"].shape == (16, 1)
    assert arrays["object_poses"].shape == (16, 2, 7)
    assert arrays["sim_state"].shape == (16, 47)


def test_the_first_logged_state_is_the_freshly_initialised_scene(scene):
    env, _, init_state = scene
    env.reset()
    env.set_init_state(init_state)
    expected = env.get_sim_state().copy()

    result = _run(scene)
    assert np.array_equal(result.steps[0].world.sim_state, expected)


def test_video_frames_line_up_with_logged_steps(scene, tmp_path):
    result = _run(scene, record_frames=True)
    video.write_video(
        tmp_path / "ep.mp4", result.agent_frames, result.wrist_frames, result.is_wait_steps
    )
    n_frames = sum(1 for _ in imageio.get_reader(tmp_path / "ep.mp4"))
    assert n_frames == len(result.steps) == rollout.NUM_STEPS_WAIT + MAX_STEPS
