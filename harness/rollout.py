"""Closed-loop LIBERO rollout that records a full per-step trace.

Step `k` records the observation and world state before action `k`; `done`/`success` are
the result of executing it. Wait steps send a no-op and run no inference.
"""

import time
from dataclasses import dataclass

import numpy as np

from harness import decode
from harness.schema import StepRecord
from harness.world import read_world_state, scene_layout

NUM_STEPS_WAIT = 10
# Per-suite step budgets, from openvla's run_libero_eval.py.
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
WAIT_ACTION = np.array([0, 0, 0, 0, 0, 0, -1.0])
IMAGE_SIZE = 224
_MODES = ("logged", "reference", "parity")


@dataclass(frozen=True)
class EpisodeSpec:
    suite: str
    task_id: int
    init_state_idx: int
    seed: int = 0
    perturbation_config_id: str = "none"

    @property
    def episode_id(self) -> str:
        return (
            f"{self.suite}-t{self.task_id:02d}-i{self.init_state_idx:03d}"
            f"-s{self.seed}-{self.perturbation_config_id}"
        )


@dataclass
class EpisodeResult:
    executed_actions: np.ndarray  # [steps, 7], wait steps included
    success: bool
    terminated_by: str  # "success" or "max_steps"
    num_policy_steps: int
    wall_s: float
    steps: list[StepRecord] | None  # None in reference mode
    agent_frames: list[np.ndarray] | None
    wrist_frames: list[np.ndarray] | None
    is_wait_steps: list[bool]
    parity_mismatches: list[int] | None = None


def _default_image_fn(obs):
    from experiments.robot.libero.libero_utils import get_libero_image

    return get_libero_image(obs, IMAGE_SIZE)


def run_episode(
    env,
    policy,
    task_label: str,
    init_state,
    *,
    max_steps: int,
    image_fn=None,
    mode: str = "logged",
    layout=None,
    world_reader=read_world_state,
    record_frames: bool = False,
) -> EpisodeResult:
    """Runs one episode.

    `logged` records every field through `policy.predict`; `reference` drives the env
    through `policy.reference_action` (the vendored path) and records only the actions;
    `parity` is `logged` plus a per-step comparison of the two paths, at twice the
    inference cost.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    logged = mode in ("logged", "parity")
    image_fn = image_fn or _default_image_fn

    t0 = time.perf_counter()
    env.reset()
    obs = env.set_init_state(init_state)
    if logged:
        layout = layout if layout is not None else scene_layout(env)
        stats = policy.action_stats

    steps, executed, agent_frames, wrist_frames, wait_flags = [], [], [], [], []
    success, terminated_by, policy_steps = False, "max_steps", 0
    mismatches = []
    for t in range(max_steps + NUM_STEPS_WAIT):
        is_wait = t < NUM_STEPS_WAIT
        world = world_reader(env, obs, layout) if logged else None
        if record_frames:
            agent_frames.append(np.array(obs["agentview_image"]))
            wrist_frames.append(np.array(obs["robot0_eye_in_hand_image"]))
        wait_flags.append(is_wait)

        model = decoded = None
        wall = time.time()
        if is_wait:
            action = WAIT_ACTION.copy()
        elif logged:
            img = image_fn(obs)
            model = policy.predict(img, task_label)
            decoded = decode.decode_action(
                model.chosen_token_id, stats["q01"], stats["q99"], stats["mask"]
            )
            action = decoded.executed_action
            if mode == "parity":
                reference = policy.reference_action(img, task_label)
                if not np.array_equal(reference, decoded.unnormalized_action):
                    mismatches.append(t)
        else:
            action = decode.to_env_action(policy.reference_action(image_fn(obs), task_label))

        obs, _reward, done, _info = env.step(action.tolist())
        executed.append(action)
        policy_steps += 0 if is_wait else 1
        if logged:
            steps.append(
                StepRecord(
                    step_idx=t,
                    model=model,
                    decoded=decoded,
                    executed_action=action,
                    wall_timestamp=wall,
                    world=world,
                    done=bool(done),
                    success=bool(done) and not is_wait,
                )
            )
        if done and not is_wait:
            success, terminated_by = True, "success"
            break

    return EpisodeResult(
        executed_actions=np.stack(executed),
        success=success,
        terminated_by=terminated_by,
        num_policy_steps=policy_steps,
        wall_s=time.perf_counter() - t0,
        steps=steps if logged else None,
        agent_frames=agent_frames if record_frames else None,
        wrist_frames=wrist_frames if record_frames else None,
        is_wait_steps=wait_flags,
        parity_mismatches=mismatches if mode == "parity" else None,
    )
