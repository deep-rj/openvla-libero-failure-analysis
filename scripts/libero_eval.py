# %%
import csv
import os
import time
from dataclasses import dataclass
from typing import cast

import h5py
import numpy as np
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_processor,
    get_vla,
    get_vla_action,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

REPO_DIR = os.environ.get(
    "REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

IMG_RESOLUTION = 256  # sim camera resolution

# Per-suite step budgets, from openvla's run_libero_eval.py.
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10  # steps to let dropped objects settle before acting


# %%
@dataclass
class EvalConfig:
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    model_family: str = "openvla"
    task_suite_name: str = "libero_10"
    task_id: int = 0
    num_trials: int = 5
    unnorm_key: str | None = None  # defaults to task_suite_name below
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    def __post_init__(self):
        if self.unnorm_key is None:
            self.unnorm_key = self.task_suite_name


# %%
def load_model(cfg: EvalConfig):
    """Loads the VLA + processor and validates the action un-normalization key.

    Raises with the available keys if `unnorm_key` isn't in the checkpoint's
    norm_stats, rather than letting the model silently apply the wrong scale.
    """
    vla = get_vla(cfg)
    processor = get_processor(cfg)

    if cfg.unnorm_key not in vla.norm_stats:
        available = sorted(vla.norm_stats.keys())
        raise KeyError(
            f"unnorm_key {cfg.unnorm_key!r} not found in the loaded checkpoint's "
            f"norm_stats. Available keys: {available}. This usually means the "
            "checkpoint was never fine-tuned on this task suite -- point "
            "pretrained_checkpoint at a suite-specific checkpoint (e.g. "
            "openvla/openvla-7b-finetuned-libero-10) or a LoRA checkpoint fine-tuned "
            "on it."
        )
    return vla, processor


# %%
def build_env(task_suite, task_id: int, resolution: int = IMG_RESOLUTION):
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # seed affects object positions even with a fixed init state
    return env, task.language


# %%
def run_trial(
    env,
    vla,
    processor,
    cfg: EvalConfig,
    task_description,
    init_state,
    max_steps,
    resize_size,
):
    """Runs one closed-loop rollout and returns its full per-step trajectory.

    Frame/state at index i is the observation the model was shown before producing
    actions[i]; reward/done at index i is the result of executing that action.
    """
    env.reset()
    obs = env.set_init_state(init_state)

    agent_frames, wrist_frames, proprio_states = [], [], []
    actions_log, rewards, dones = [], [], []
    success = False

    t = 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, _reward, _done, _info = env.step(
                get_libero_dummy_action(cfg.model_family)
            )
            t += 1
            continue

        img = get_libero_image(obs, resize_size)
        # Not fed to the model; recorded for trajectory analysis.
        proprio_state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
        agent_frame = obs["agentview_image"][::-1, ::-1]
        wrist_frame = obs["robot0_eye_in_hand_image"][::-1, ::-1]

        action = get_vla_action(
            vla,
            processor,
            cfg.pretrained_checkpoint,
            {"full_image": img},
            task_description,
            cfg.unnorm_key,
            center_crop=cfg.center_crop,
        )
        action = normalize_gripper_action(action, binarize=True)  # [0,1] -> [-1,1]
        if cfg.model_family == "openvla":
            action = invert_gripper_action(
                action
            )  # align gripper sign with sim convention

        obs, reward, done, _info = env.step(action.tolist())

        agent_frames.append(agent_frame)
        wrist_frames.append(wrist_frame)
        proprio_states.append(proprio_state)
        actions_log.append(np.asarray(action, dtype=np.float64))
        rewards.append(reward)
        dones.append(done)

        if done:
            success = True
            break
        t += 1

    return {
        "agent_frames": np.array(agent_frames, dtype=np.uint8),
        "wrist_frames": np.array(wrist_frames, dtype=np.uint8),
        "proprio_states": np.array(proprio_states, dtype=np.float64),
        "actions": np.array(actions_log, dtype=np.float64),
        "rewards": np.array(rewards, dtype=np.float64),
        "dones": np.array(dones, dtype=bool),
        "success": success,
        "num_steps": len(actions_log),
    }


# %%
cfg = EvalConfig()

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[cfg.task_suite_name]()
init_states = task_suite.get_task_init_states(cfg.task_id)
max_steps = MAX_STEPS_BY_SUITE[cfg.task_suite_name]
resize_size = get_image_resize_size(cfg)

# %%
vla, processor = load_model(cfg)

# %%
env, task_description = build_env(task_suite, cfg.task_id)
print(f"Task: {task_description}")

# %%
run_id = time.strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(REPO_DIR, "outputs", f"libero_eval_{run_id}.h5")
manifest_path = os.path.join(REPO_DIR, "outputs", "libero_eval_manifest.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

MANIFEST_FIELDS = [
    "run_id",
    "task_suite_name",
    "task_id",
    "task_instruction",
    "trial_id",
    "init_state_idx",
    "success",
    "num_steps",
    "pretrained_checkpoint",
    "unnorm_key",
]

trial_results = []
write_header = not os.path.exists(manifest_path)
with (
    h5py.File(output_path, "w") as f,
    open(manifest_path, "a", newline="") as manifest_file,
):
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    if write_header:
        manifest_writer.writeheader()

    f.attrs["run_id"] = run_id
    f.attrs["task_suite_name"] = cfg.task_suite_name
    f.attrs["task_id"] = cfg.task_id
    f.attrs["task_instruction"] = task_description
    f.attrs["pretrained_checkpoint"] = cfg.pretrained_checkpoint
    f.attrs["unnorm_key"] = cfg.unnorm_key
    f.attrs["num_trials"] = cfg.num_trials

    task_grp = f.create_group(f"task_{cfg.task_id}")

    for trial_idx in range(cfg.num_trials):
        init_state_idx = trial_idx % len(init_states)
        print(
            f"Running trial {trial_idx + 1}/{cfg.num_trials} (init_state {init_state_idx})..."
        )
        result = run_trial(
            env,
            vla,
            processor,
            cfg,
            task_description,
            init_states[init_state_idx],
            max_steps,
            resize_size,
        )
        trial_results.append(result)
        print(f"  success={result['success']} num_steps={result['num_steps']}")

        grp = task_grp.create_group(f"trial_{trial_idx}")
        grp.create_dataset(
            "agentview_images", data=result["agent_frames"], compression="gzip"
        )
        grp.create_dataset(
            "wrist_images", data=result["wrist_frames"], compression="gzip"
        )
        grp.create_dataset("proprio_states", data=result["proprio_states"])
        grp.create_dataset("actions", data=result["actions"])
        grp.create_dataset("rewards", data=result["rewards"])
        grp.create_dataset("dones", data=result["dones"])
        grp.attrs["success"] = bool(result["success"])
        grp.attrs["num_steps"] = result["num_steps"]
        grp.attrs["init_state_idx"] = init_state_idx

        manifest_writer.writerow(
            {
                "run_id": run_id,
                "task_suite_name": cfg.task_suite_name,
                "task_id": cfg.task_id,
                "task_instruction": task_description,
                "trial_id": trial_idx,
                "init_state_idx": init_state_idx,
                "success": bool(result["success"]),
                "num_steps": result["num_steps"],
                "pretrained_checkpoint": cfg.pretrained_checkpoint,
                "unnorm_key": cfg.unnorm_key,
            }
        )
        manifest_file.flush()

    num_successes = sum(r["success"] for r in trial_results)
    f.attrs["num_successes"] = num_successes
    f.attrs["success_rate"] = num_successes / cfg.num_trials

env.close()
print(
    f"Success rate: {num_successes}/{cfg.num_trials} ({num_successes / cfg.num_trials * 100:.1f}%)"
)

# %%
# Post-write validation
with h5py.File(output_path, "r") as f:
    assert f.attrs["num_trials"] == cfg.num_trials
    assert f.attrs["num_successes"] == num_successes

    for trial_idx in range(cfg.num_trials):
        grp = cast(h5py.Group, f[f"task_{cfg.task_id}/trial_{trial_idx}"])
        n_steps = cast(int, grp.attrs["num_steps"])

        actions_ds = cast(h5py.Dataset, grp["actions"])
        assert actions_ds.shape == (n_steps, env.env.action_dim), (
            f"trial {trial_idx}: unexpected action shape {actions_ds.shape}"
        )

        for key in (
            "agentview_images",
            "wrist_images",
            "proprio_states",
            "rewards",
            "dones",
        ):
            ds = cast(h5py.Dataset, grp[key])
            assert ds.shape[0] == n_steps, f"trial {trial_idx}: {key} length mismatch"

        agent_imgs = cast(h5py.Dataset, grp["agentview_images"])
        wrist_imgs = cast(h5py.Dataset, grp["wrist_images"])
        if n_steps > 0:
            assert agent_imgs.dtype == np.uint8 and wrist_imgs.dtype == np.uint8
            assert agent_imgs.shape[1:] == wrist_imgs.shape[1:]

        rewards = cast(h5py.Dataset, grp["rewards"])[:]
        assert np.all(np.isfinite(rewards)), (
            f"trial {trial_idx}: non-finite reward encountered"
        )

print("Eval script completed successfully!")
