# %%
# Thread pinning must be applied before numpy/TF/torch import. Their pools size
# themselves to core count, which oversubscribes once N env workers exist.
# Run with PIN_THREADS=1 to A/B against library defaults.
import os

if os.environ.get("PIN_THREADS") == "1":
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_var] = "1"

import csv
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import h5py
import numpy as np
import tensorflow as tf
import torch
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    DEVICE,
    crop_and_resize,
    get_processor,
    get_vla,
)
from PIL import Image
from experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

if os.environ.get("PIN_THREADS") == "1":
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1)

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
class StageTimer:
    """Accumulates wall time per named stage across a run.

    CUDA work is asynchronous, so a stage that touches the GPU must be timed with
    `sync=True`; otherwise its cost is attributed to whichever later call happens
    to synchronize.
    """

    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    @contextmanager
    def stage(self, name, sync=False):
        sync = sync and torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if sync:
            torch.cuda.synchronize()
        self.total[name] += time.perf_counter() - t0
        self.count[name] += 1

    def report(self, title="Timing breakdown", wall=None, num_steps=None):
        grand = sum(self.total.values())
        print(f"\n=== {title} ===")
        print(f"{'stage':<40}{'calls':>7}{'total s':>10}{'mean ms':>10}{'share':>8}")
        for name, tot in sorted(self.total.items(), key=lambda kv: -kv[1]):
            n = self.count[name]
            print(
                f"{name:<40}{n:>7}{tot:>10.2f}{tot / n * 1e3:>10.2f}"
                f"{tot / grand * 100:>7.1f}%"
            )
        print("-" * 75)
        print(f"{'SUM OF TIMED STAGES':<40}{'':>7}{grand:>10.2f}")
        if num_steps:
            print(f"per-step (timed stages only): {grand / num_steps * 1e3:.1f} ms")
        if wall:
            # Coverage below ~90% means real cost is hiding between the timers.
            print(f"measured wall time:           {wall:.2f} s")
            print(f"coverage:                     {grand / wall * 100:.1f}%")


# %%
def get_vla_action_timed(
    vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop, timer
):
    """Mirror of openvla's `get_vla_action` with per-stage timers.

    Faithful to openvla/experiments/robot/openvla_utils.py:127-169. The vendored
    version is a single opaque call, which hides the CPU/GPU split being measured.
    """
    assert "openvla-v01" not in base_vla_name, "v01 prompt format is not mirrored here"

    with timer.stage("action 1: PIL convert"):
        image = Image.fromarray(obs["full_image"]).convert("RGB")

    if center_crop:
        with timer.stage("action 2: center crop (6 TF ops)"):
            image_t = tf.convert_to_tensor(np.array(image))
            orig_dtype = image_t.dtype
            image_t = tf.image.convert_image_dtype(image_t, tf.float32)
            image_t = crop_and_resize(image_t, 0.9, 1)
            image_t = tf.clip_by_value(image_t, 0, 1)
            image_t = tf.image.convert_image_dtype(image_t, orig_dtype, saturate=True)
            image = Image.fromarray(image_t.numpy()).convert("RGB")

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    with timer.stage("action 3: HF processor (2 tv stacks)"):
        inputs = processor(prompt, image)
    with timer.stage("action 4: H2D transfer", sync=True):
        inputs = inputs.to(DEVICE, dtype=torch.bfloat16)
    with timer.stage("action 5: GPU generate", sync=True):
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


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
    timer,
):
    """Runs one closed-loop rollout and returns its full per-step trajectory.

    Frame/state at index i is the observation the model was shown before producing
    actions[i]; reward/done at index i is the result of executing that action.
    """
    with timer.stage("trial: env.reset (hard_reset=True)"):
        env.reset()
    with timer.stage("trial: set_init_state"):
        obs = env.set_init_state(init_state)

    agent_frames, wrist_frames, proprio_states = [], [], []
    actions_log, rewards, dones = [], [], []
    success = False

    t = 0
    while t < max_steps + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            with timer.stage("settle: env.step (dummy action)"):
                obs, _reward, _done, _info = env.step(
                    get_libero_dummy_action(cfg.model_family)
                )
            t += 1
            continue

        with timer.stage("obs 1: get_libero_image (jpeg+lanczos3)"):
            img = get_libero_image(obs, resize_size)
        with timer.stage("obs 2: proprio + frame copies (logging)"):
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

        action = get_vla_action_timed(
            vla,
            processor,
            cfg.pretrained_checkpoint,
            {"full_image": img},
            task_description,
            cfg.unnorm_key,
            cfg.center_crop,
            timer,
        )
        with timer.stage("action 6: gripper post-process"):
            action = normalize_gripper_action(action, binarize=True)  # [0,1] -> [-1,1]
            if cfg.model_family == "openvla":
                action = invert_gripper_action(
                    action
                )  # align gripper sign with sim convention

        with timer.stage("env.step (physics + 2 renders)"):
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
timer = StageTimer()

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
run_t0 = time.perf_counter()
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
            timer,
        )
        trial_results.append(result)
        print(f"  success={result['success']} num_steps={result['num_steps']}")

        grp = task_grp.create_group(f"trial_{trial_idx}")
        with timer.stage("io: hdf5 write (gzip)"):
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

run_wall = time.perf_counter() - run_t0
env.close()
print(
    f"Success rate: {num_successes}/{cfg.num_trials} ({num_successes / cfg.num_trials * 100:.1f}%)"
)

timer.report(
    title="Per-stage timing",
    wall=run_wall,
    num_steps=sum(r["num_steps"] for r in trial_results),
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
