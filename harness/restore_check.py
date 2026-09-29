"""T4: restore a logged mid-episode `sim_state` into a fresh env and compare named state."""

import numpy as np

from harness import runner, schema, world
from harness.rollout import NUM_STEPS_WAIT

# Observation-derived poses lag qpos by one 2 ms physics substep.
SUBSTEP_TOL = 5e-3


def _differs(name, restored, logged, tol, step):
    if np.allclose(restored, logged, atol=tol, rtol=0):
        return []
    worst = float(np.abs(np.asarray(restored) - np.asarray(logged)).max())
    return [f"step {step}: {name} differs after restore (max abs {worst:.3g}, tol {tol:g})"]


def check_restore(episode_path, step: int | None = None, camera_size: int = 64) -> list[str]:
    """Returns one message per violation; an empty list means the state restored cleanly."""
    from libero.libero import benchmark

    arrays, meta = schema.read_episode(episode_path)
    n_steps = len(arrays["step_idx"])
    if step is None:
        step = NUM_STEPS_WAIT + (n_steps - NUM_STEPS_WAIT) // 2
    if not 0 <= step < n_steps:
        raise ValueError(f"step {step} outside the episode's {n_steps} steps")

    suite = benchmark.get_benchmark_dict()[meta["suite"]]()
    env, _ = runner.build_env(suite, meta["task_id"], camera_size, meta["seed"])
    try:
        env.reset()
        obs = env.set_init_state(arrays["sim_state"][step])
        layout = world.scene_layout(env)
        restored = world.read_world_state(env, obs, layout)

        problems = []
        if list(layout.object_names) != meta["object_names"]:
            problems.append("object names differ from the episode metadata")
        if [j.name for j in layout.fixture_joints] != [j["name"] for j in meta["fixture_joints"]]:
            problems.append("fixture joints differ from the episode metadata")
        problems += _differs(
            "sim_state", restored.sim_state, arrays["sim_state"][step], 1e-12, step
        )
        problems += _differs(
            "fixture_joint_qpos",
            restored.fixture_joint_qpos,
            arrays["fixture_joint_qpos"][step],
            1e-9,
            step,
        )
        for name in ("eef_pos", "eef_quat_raw", "object_poses"):
            problems += _differs(
                name, getattr(restored, name), arrays[name][step], SUBSTEP_TOL, step
            )
        problems += _object_pose_cross_check(env, layout, arrays, step)
        return problems
    finally:
        env.close()


def _object_pose_cross_check(env, layout, arrays, step) -> list[str]:
    """Logged object poses must equal the free-joint slice of the logged sim_state."""
    state = arrays["sim_state"][step]
    poses = arrays["object_poses"][step]
    problems = []
    for row, name in zip(poses, layout.object_names, strict=True):
        addr = env.sim.model.get_joint_qpos_addr(f"{name}_joint0")[0]
        pos_wxyz = state[1 + addr : 1 + addr + 7]  # state[0] is the simulation time
        expected = np.concatenate([pos_wxyz[:3], pos_wxyz[[4, 5, 6, 3]]])
        problems += _differs(f"object_poses[{name}] vs sim_state", row, expected, SUBSTEP_TOL, step)
    return problems
