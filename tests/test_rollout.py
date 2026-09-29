import numpy as np
import pytest

from factories import DriftedPolicy, PureFakePolicy

from harness import checks, policy, rollout, schema
from harness.schema import WorldState

LABEL = "put the yellow and white mug in the microwave and close it"
INIT_STATE = np.zeros(3)


class FakeEnv:
    """Steps to `done` after `done_after` policy steps; records every action it is sent."""

    def __init__(self, done_after=None):
        self.done_after = done_after
        self.sent = []
        self.resets = 0
        self.init_states = []

    def reset(self):
        self.resets += 1

    def set_init_state(self, state):
        self.init_states.append(state)
        return self._obs()

    def _obs(self):
        frame = np.full((8, 8, 3), len(self.sent), dtype=np.uint8)
        return {"agentview_image": frame, "robot0_eye_in_hand_image": frame + 1}

    def step(self, action):
        self.sent.append(np.array(action, dtype=np.float64))
        policy_steps = len(self.sent) - rollout.NUM_STEPS_WAIT
        done = self.done_after is not None and policy_steps >= self.done_after
        return self._obs(), 0.0, done, {}


def _fake_world(env, obs, layout):
    k = len(env.sent)
    return WorldState(
        sim_state=np.full(5, float(k)),
        object_poses=np.zeros((1, 7), dtype=np.float32),
        fixture_joint_qpos=np.zeros(0, dtype=np.float32),
        eef_pos=np.zeros(3),
        eef_quat_raw=np.zeros(4),
        gripper_qpos=np.zeros(2),
    )


def _run(env, *, seed=0, max_steps=5, mode="logged", record_frames=False, rate=0.0):
    return rollout.run_episode(
        env,
        policy.StubPolicy(seed=seed, out_of_slice_rate=rate),
        LABEL,
        INIT_STATE,
        max_steps=max_steps,
        image_fn=lambda obs: obs["agentview_image"],
        mode=mode,
        layout=object(),
        world_reader=_fake_world,
        record_frames=record_frames,
    )


def test_wait_steps_come_first_then_policy_steps_up_to_the_budget():
    env = FakeEnv()
    result = _run(env, max_steps=5)

    assert len(result.steps) == rollout.NUM_STEPS_WAIT + 5
    assert [s.is_wait_step for s in result.steps] == [True] * 10 + [False] * 5
    assert [s.step_idx for s in result.steps] == list(range(15))
    assert result.num_policy_steps == 5
    assert result.terminated_by == "max_steps"
    assert result.success is False
    assert env.resets == 1
    assert env.init_states[0] is INIT_STATE


def test_wait_steps_send_the_no_op_action_with_the_gripper_open():
    env = FakeEnv()
    _run(env)
    assert all(a.tolist() == [0, 0, 0, 0, 0, 0, -1.0] for a in env.sent[:10])


def test_done_ends_the_episode_as_a_success_on_that_step():
    env = FakeEnv(done_after=3)
    result = _run(env, max_steps=50)

    assert result.success is True
    assert result.terminated_by == "success"
    assert result.num_policy_steps == 3
    assert len(result.steps) == 13
    assert result.steps[-1].done and result.steps[-1].success
    assert not any(s.done for s in result.steps[:-1])


def test_logged_steps_pass_the_self_consistency_checks_against_what_the_env_received(tmp_path):
    env = FakeEnv()
    result = _run(env, seed=1, rate=0.3)
    schema.write_episode(tmp_path / "ep.h5", result.steps, {})
    arrays, _ = schema.read_episode(tmp_path / "ep.h5")

    stats = policy.LIBERO_10_ACTION_STATS
    assert checks.check_self_consistency(arrays, stats, env.sent) == []
    assert checks.check_wait_alignment(arrays, rollout.NUM_STEPS_WAIT) == []


def test_state_and_action_are_logged_before_the_step_that_they_precede():
    env = FakeEnv()
    result = _run(env)
    assert [s.world.sim_state[0] for s in result.steps] == [float(k) for k in range(15)]
    pairs = zip(result.steps, env.sent, strict=True)
    assert all(np.array_equal(s.executed_action, a) for s, a in pairs)


def test_reference_mode_sends_the_same_actions_and_logs_nothing():
    logged_env, reference_env = FakeEnv(), FakeEnv()
    logged = _run(logged_env, seed=4, rate=0.3)
    reference = _run(reference_env, seed=4, rate=0.3, mode="reference")

    assert reference.steps is None
    assert np.array_equal(logged.executed_actions, reference.executed_actions)
    assert np.array_equal(np.array(logged_env.sent), np.array(reference_env.sent))


def test_frames_are_recorded_for_every_step_including_wait_steps():
    result = _run(FakeEnv(), record_frames=True)
    assert len(result.agent_frames) == len(result.wrist_frames) == 15
    assert result.agent_frames[0].max() == 0
    assert result.agent_frames[12].max() == 12
    assert result.is_wait_steps == [True] * 10 + [False] * 5


def test_frames_are_not_kept_unless_requested():
    result = _run(FakeEnv())
    assert result.agent_frames is None


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        _run(FakeEnv(), mode="fast")


def _run_parity(policy_obj):
    return rollout.run_episode(
        FakeEnv(),
        policy_obj,
        LABEL,
        INIT_STATE,
        max_steps=4,
        image_fn=lambda obs: obs["agentview_image"],
        mode="parity",
        layout=object(),
        world_reader=_fake_world,
    )


def test_parity_mode_finds_no_mismatch_when_the_two_paths_agree():
    result = _run_parity(PureFakePolicy())
    assert result.parity_mismatches == []
    assert len(result.steps) == rollout.NUM_STEPS_WAIT + 4


def test_parity_mode_lists_every_policy_step_where_the_vendored_action_differs():
    result = _run_parity(DriftedPolicy())
    assert result.parity_mismatches == [10, 11, 12, 13]


def test_other_modes_report_no_parity_result():
    assert _run(FakeEnv()).parity_mismatches is None
