import numpy as np
from harness.schema import ModelOutput, StepRecord, WorldState

from harness import decode

Q01 = np.array([-0.63, -0.77, -0.76, -0.097, -0.148, -0.274, 0.0])
Q99 = np.array([0.77, 0.846, 0.9375, 0.139, 0.159, 0.324, 1.0])
MASK = np.array([True] * 6 + [False])
WAIT_ACTION = np.array([0, 0, 0, 0, 0, 0, -1.0])


def make_world(rng, n_obj=3, n_joint=1, sim_state_len=60):
    return WorldState(
        sim_state=rng.normal(size=sim_state_len),
        object_poses=rng.normal(size=(n_obj, 7)).astype(np.float32),
        fixture_joint_qpos=rng.normal(size=n_joint).astype(np.float32),
        eef_pos=rng.normal(size=3),
        eef_quat_raw=rng.normal(size=4),
        gripper_qpos=rng.normal(size=2),
    )


def make_model_output(rng):
    logits = rng.normal(scale=5, size=(7, 256)).astype(np.float16)
    tokens = 31744 + logits.astype(np.float32).argmax(-1)
    return ModelOutput(
        slice_logits=logits,
        lse_full=rng.normal(size=7).astype(np.float32),
        argmax_full_id=tokens.astype(np.int32),
        chosen_token_id=tokens.astype(np.int32),
        inference_latency_ms=270.0,
    )


def make_step(rng, step_idx, *, wait=False, **world_kwargs):
    world = make_world(rng, **world_kwargs)
    if wait:
        return StepRecord(
            step_idx=step_idx,
            model=None,
            decoded=None,
            executed_action=WAIT_ACTION.copy(),
            wall_timestamp=1.0 + step_idx,
            world=world,
            done=False,
            success=False,
        )
    model = make_model_output(rng)
    decoded = decode.decode_action(model.chosen_token_id, Q01, Q99, MASK)
    return StepRecord(
        step_idx=step_idx,
        model=model,
        decoded=decoded,
        executed_action=decoded.executed_action,
        wall_timestamp=1.0 + step_idx,
        world=world,
        done=False,
        success=False,
    )


def make_episode(rng, n_wait=10, n_act=20, **world_kwargs):
    return [make_step(rng, i, wait=i < n_wait, **world_kwargs) for i in range(n_wait + n_act)]
