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
import hashlib
import inspect
import json
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Protocol, cast

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
from experiments.robot.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

if os.environ.get("PIN_THREADS") == "1":
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1)

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    policy: str = "openvla"  # "openvla" or "stub"
    stub_seed: int = 0
    max_steps: int | None = None  # overrides the suite budget; None uses MAX_STEPS_BY_SUITE
    hard_reset: bool = True
    h5_compression: str | None = "gzip"
    tf_device: str | None = None  # e.g. "/CPU:0"; None leaves placement to TF

    def __post_init__(self):
        if self.unnorm_key is None:
            self.unnorm_key = self.task_suite_name
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {sorted(POLICIES)}, got {self.policy!r}")
        if self.h5_compression not in (None, "gzip", "lzf"):
            raise ValueError(f"unsupported h5_compression {self.h5_compression!r}")

    @classmethod
    def from_env(cls):
        """Builds a config from defaults plus any EVAL_* environment overrides."""
        return cls(
            **{
                field: parse(os.environ[var])
                for var, (field, parse) in ENV_OVERRIDES.items()
                if var in os.environ
            }
        )


ENV_OVERRIDES = {
    "EVAL_POLICY": ("policy", str),
    "EVAL_TASK_ID": ("task_id", int),
    "EVAL_NUM_TRIALS": ("num_trials", int),
    "EVAL_MAX_STEPS": ("max_steps", int),
    "EVAL_HARD_RESET": ("hard_reset", lambda v: v == "1"),
    "EVAL_H5_COMPRESSION": ("h5_compression", lambda v: None if v == "none" else v),
    "EVAL_TF_DEVICE": ("tf_device", str),
}


def tf_placement(tf_device):
    return tf.device(tf_device) if tf_device else nullcontext()


# %%
# Profiling instrumentation. Synchronizing around GPU stages defeats CPU/GPU overlap, so
# timings from it are per-stage costs, not throughput. The PIN_THREADS block above is separate.
class StageTimer:
    """Records per-call wall time for named stages, and for substages nested in them.

    CUDA work is asynchronous, so a stage that touches the GPU must be timed with
    `sync=True`; otherwise its cost is attributed to whichever later call happens
    to synchronize.
    """

    def __init__(self):
        self.samples = defaultdict(list)
        # (parent, substage) -> substage total per parent call
        self.sub_samples = defaultdict(list)
        self._active = None
        self._pending = defaultdict(float)

    @contextmanager
    def stage(self, name, sync=False):
        sync = sync and torch.cuda.is_available()
        if sync:
            torch.cuda.synchronize()
        self._active = name
        t0 = time.perf_counter()
        yield
        if sync:
            torch.cuda.synchronize()
        self.samples[name].append(time.perf_counter() - t0)
        for sub, total in self._pending.items():
            self.sub_samples[(name, sub)].append(total)
        self._pending.clear()
        self._active = None

    @contextmanager
    def substage(self, name):
        t0 = time.perf_counter()
        yield
        if self._active is not None:
            self._pending[name] += time.perf_counter() - t0

    @staticmethod
    def _row(name, samples, denom):
        s = np.asarray(samples)
        return {
            "stage": name,
            "calls": len(s),
            "total_s": float(s.sum()),
            "mean_ms": float(s.mean() * 1e3),
            "p50_ms": float(np.percentile(s, 50) * 1e3),
            "p90_ms": float(np.percentile(s, 90) * 1e3),
            "share_pct": float(s.sum() / denom * 100),
        }

    def report(self, title="Timing breakdown", wall=None, num_steps=None):
        grand = sum(sum(s) for s in self.samples.values())
        rows = []
        for name, samples in sorted(self.samples.items(), key=lambda kv: -sum(kv[1])):
            rows.append(self._row(name, samples, grand))
            for (parent, sub), sub_samples in self.sub_samples.items():
                if parent == name:
                    rows.append(self._row(f"  > {sub}", sub_samples, sum(samples)))

        print(f"\n=== {title} ===")
        print(
            f"{'stage':<46}{'calls':>7}{'total s':>10}{'mean ms':>10}"
            f"{'p50 ms':>10}{'p90 ms':>10}{'share':>8}"
        )
        for r in rows:
            print(
                f"{r['stage']:<46}{r['calls']:>7}{r['total_s']:>10.2f}{r['mean_ms']:>10.2f}"
                f"{r['p50_ms']:>10.2f}{r['p90_ms']:>10.2f}{r['share_pct']:>7.1f}%"
            )
        print("-" * 101)
        print("('  >' rows are substages; their calls and share are relative to the parent)")
        print(f"{'SUM OF TIMED STAGES':<46}{'':>7}{grand:>10.2f}")
        if num_steps:
            print(f"per-step (timed stages only): {grand / num_steps * 1e3:.1f} ms")
        if wall:
            # Coverage below ~90% means real cost is hiding between the timers.
            print(f"measured wall time:           {wall:.2f} s")
            print(f"coverage:                     {grand / wall * 100:.1f}%")
        return {
            "rows": rows,
            "timed_total_s": grand,
            "wall_s": wall,
            "coverage_pct": grand / wall * 100 if wall else None,
            "num_steps": num_steps,
        }


def instrument_env_substages(env, timer):
    """Times observable updates (camera renders) and the OSC controller inside robosuite's step.

    Patches instance attributes on the robosuite env, which persist across hard resets.
    """
    inner = env.env
    for attr, name in (
        ("_update_observables", "observables (incl. camera renders)"),
        ("_pre_action", "controller (_pre_action)"),
    ):
        orig = getattr(inner, attr)

        def timed(*args, _orig=orig, _name=name, **kwargs):
            with timer.substage(_name):
                return _orig(*args, **kwargs)

        setattr(inner, attr, timed)


def _read_tokens(path):
    try:
        with open(path) as f:
            return f.read().split()
    except OSError:
        return None


def cgroup_limits():
    """Container CPU and memory limits; os.cpu_count() and /proc/meminfo report the host's."""
    cpu_limit = None
    cpu_max = _read_tokens("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota|max> <period>"
    if cpu_max and cpu_max[0] != "max":
        cpu_limit = int(cpu_max[0]) / int(cpu_max[1])
    elif cpu_max is None:  # cgroup v1: quota of -1 means unlimited
        quota = _read_tokens("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_tokens("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period and int(quota[0]) > 0:
            cpu_limit = int(quota[0]) / int(period[0])

    memory = _read_tokens("/sys/fs/cgroup/memory.max") or _read_tokens(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    # v1 reports "unlimited" as a near-2**63 byte count.
    memory_gb = (
        int(memory[0]) / 1e9 if memory and memory[0].isdigit() and int(memory[0]) < 2**60 else None
    )
    return {"cgroup_cpu_limit": cpu_limit, "cgroup_memory_limit_gb": memory_gb}


def environment_banner(cfg):
    """Collects the runtime facts a timing table can't be interpreted without."""
    # Runs a real TF op, so call it after the model is loaded: TF reserves most
    # free GPU memory on first GPU use.
    with tf_placement(cfg.tf_device):
        tf_resize_device = tf.image.resize(tf.zeros([1, 8, 8, 3]), [4, 4]).device
    info = {
        "policy": cfg.policy,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "cpu_count": os.cpu_count(),
        "cpus_usable": len(os.sched_getaffinity(0)),
        **cgroup_limits(),
        "PIN_THREADS": os.environ.get("PIN_THREADS", "0"),
        **{
            var: os.environ.get(var)
            for var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # 0 means TF sizes the pool itself (typically to core count).
        "tf_intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "tf_inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "tf_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "tf_device_override": cfg.tf_device,
        "tf_resize_placed_on": tf_resize_device,
    }
    print("\n=== Environment ===")
    for key, value in info.items():
        print(f"{key:<28}{value}")
    return info


# %%
class Policy(Protocol):
    def predict(self, img: np.ndarray, task_label: str, timer: StageTimer) -> np.ndarray:
        """Returns a 7-dim action in OpenVLA's output convention (gripper in [0, 1], 1 = open)."""
        ...


def preprocess_timed(processor, img, task_label, center_crop, tf_device, timer):
    """Mirror of the CPU half of openvla's `get_vla_action`, with per-stage timers.

    Faithful to openvla/experiments/robot/openvla_utils.py:127-165. The vendored
    version is a single opaque call, which hides the CPU/GPU split being measured.
    """
    with timer.stage("action 1: PIL convert"):
        image = Image.fromarray(img).convert("RGB")

    if center_crop:
        with timer.stage("action 2: center crop (6 TF ops)"), tf_placement(tf_device):
            image_t = tf.convert_to_tensor(np.array(image))
            orig_dtype = image_t.dtype
            image_t = tf.image.convert_image_dtype(image_t, tf.float32)
            image_t = crop_and_resize(image_t, 0.9, 1)
            image_t = tf.clip_by_value(image_t, 0, 1)
            image_t = tf.image.convert_image_dtype(image_t, orig_dtype, saturate=True)
            image = Image.fromarray(image_t.numpy()).convert("RGB")

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    with timer.stage("action 3: HF processor (2 tv stacks)"):
        return processor(prompt, image)


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


class OpenVLAPolicy:
    def __init__(self, cfg: EvalConfig):
        assert "openvla-v01" not in cfg.pretrained_checkpoint, (
            "v01 prompt format is not mirrored here"
        )
        self.cfg = cfg
        self.vla, self.processor = load_model(cfg)

    def predict(self, img, task_label, timer):
        inputs = preprocess_timed(
            self.processor, img, task_label, self.cfg.center_crop, self.cfg.tf_device, timer
        )
        with timer.stage("action 4: H2D transfer", sync=True):
            inputs = inputs.to(DEVICE, dtype=torch.bfloat16)
        with timer.stage("action 5: GPU generate", sync=True):
            return self.vla.predict_action(
                **inputs, unnorm_key=self.cfg.unnorm_key, do_sample=False
            )


class StubPolicy:
    """Seeded uniform-random actions in OpenVLA's output convention; needs no GPU.

    Runs the real CPU preprocessing and discards the result, so those stages are
    still measured.
    """

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.processor = get_processor(cfg)
        self.rng = np.random.default_rng(cfg.stub_seed)

    def predict(self, img, task_label, timer):
        preprocess_timed(
            self.processor, img, task_label, self.cfg.center_crop, self.cfg.tf_device, timer
        )
        return np.append(self.rng.uniform(-1.0, 1.0, size=6), float(self.rng.integers(0, 2)))


POLICIES = {"openvla": OpenVLAPolicy, "stub": StubPolicy}


# %%
def build_env(task_suite, task_id: int, hard_reset: bool, resolution: int = IMG_RESOLUTION):
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": hard_reset,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # seed affects object positions even with a fixed init state
    return env, task.language


# %%
def run_trial(
    env,
    policy: Policy,
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
    with timer.stage(f"trial: env.reset (hard_reset={cfg.hard_reset})"):
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
                obs, _reward, _done, _info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        with (
            timer.stage("obs 1: get_libero_image (jpeg+lanczos3)"),
            tf_placement(cfg.tf_device),
        ):
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

        action = policy.predict(img, task_description, timer)
        with timer.stage("action 6: gripper post-process"):
            action = normalize_gripper_action(action, binarize=True)  # [0,1] -> [-1,1]
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)  # align gripper sign with sim convention

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


def git_state(path):
    """HEAD SHA of the checkout containing path, and whether tracked files differ from it."""

    def git(*args):
        return subprocess.run(  # noqa: S603
            ["git", "-C", path, *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        return {
            "sha": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def code_versions():
    """Versions of all code a rollout executes: this repo, the imported siblings, the pip env."""
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
    ).stdout
    freeze_lines = sorted(freeze.lower().splitlines())
    versions = {
        "harness": git_state(REPO_DIR),
        # Located via the imported modules, so these record what actually ran.
        "openvla": git_state(os.path.dirname(inspect.getfile(get_vla))),
        "LIBERO": git_state(os.path.dirname(inspect.getfile(benchmark))),
        "pip_freeze_sha256": hashlib.sha256("\n".join(freeze_lines).encode()).hexdigest()
        if freeze_lines
        else None,
    }
    print("\n=== Code versions ===")
    for key, value in versions.items():
        print(f"{key:<28}{value}")
    return versions


# %%
cfg = EvalConfig.from_env()
timer = StageTimer()

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[cfg.task_suite_name]()
init_states = task_suite.get_task_init_states(cfg.task_id)
max_steps = cfg.max_steps or MAX_STEPS_BY_SUITE[cfg.task_suite_name]
resize_size = get_image_resize_size(cfg)

# %%
policy = POLICIES[cfg.policy](cfg)
environment = environment_banner(cfg)
versions = code_versions()

# %%
env, task_description = build_env(task_suite, cfg.task_id, cfg.hard_reset)
instrument_env_substages(env, timer)
print(f"Task: {task_description}")

# %%
run_id = time.strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(REPO_DIR, "outputs", f"libero_eval_{run_id}.h5")
profile_path = os.path.join(REPO_DIR, "outputs", f"libero_eval_{run_id}_profile.json")
# Schema version lives in the filename so rows with different columns never share a header.
manifest_path = os.path.join(REPO_DIR, "outputs", "libero_eval_manifest_v2.csv")
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
    "policy",
    "max_steps",
    "hard_reset",
    "tf_device",
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
    f.attrs["policy"] = cfg.policy
    f.attrs["code_versions"] = json.dumps(versions)

    task_grp = f.create_group(f"task_{cfg.task_id}")

    for trial_idx in range(cfg.num_trials):
        init_state_idx = trial_idx % len(init_states)
        print(f"Running trial {trial_idx + 1}/{cfg.num_trials} (init_state {init_state_idx})...")
        result = run_trial(
            env,
            policy,
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
        with timer.stage(f"io: hdf5 write ({cfg.h5_compression or 'none'})"):
            grp.create_dataset(
                "agentview_images",
                data=result["agent_frames"],
                compression=cfg.h5_compression,
            )
            grp.create_dataset(
                "wrist_images", data=result["wrist_frames"], compression=cfg.h5_compression
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
                "policy": cfg.policy,
                "max_steps": max_steps,
                "hard_reset": cfg.hard_reset,
                "tf_device": cfg.tf_device,
            }
        )
        manifest_file.flush()

    num_successes = sum(r["success"] for r in trial_results)
    f.attrs["num_successes"] = num_successes
    f.attrs["success_rate"] = num_successes / cfg.num_trials

run_wall = time.perf_counter() - run_t0
action_dim = env.env.action_dim  # env.close() deletes env.env
env.close()
print(
    f"Success rate: {num_successes}/{cfg.num_trials} ({num_successes / cfg.num_trials * 100:.1f}%)"
)

timing = timer.report(
    title="Per-stage timing",
    wall=run_wall,
    num_steps=sum(r["num_steps"] for r in trial_results),
)
with open(profile_path, "w") as pf:
    json.dump(
        {
            "run_id": run_id,
            "config": asdict(cfg),
            "environment": environment,
            "code_versions": versions,
            "timing": timing,
        },
        pf,
        indent=2,
    )
print(f"Profile written to {profile_path}")

# %%
# Post-write validation
with h5py.File(output_path, "r") as f:
    assert f.attrs["num_trials"] == cfg.num_trials
    assert f.attrs["num_successes"] == num_successes

    for trial_idx in range(cfg.num_trials):
        grp = cast(h5py.Group, f[f"task_{cfg.task_id}/trial_{trial_idx}"])
        n_steps = cast(int, grp.attrs["num_steps"])

        actions_ds = cast(h5py.Dataset, grp["actions"])
        assert actions_ds.shape == (n_steps, action_dim), (
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
        assert np.all(np.isfinite(rewards)), f"trial {trial_idx}: non-finite reward encountered"

# %%
# Raw outputs live on pod-local disk; a failed copy leaves them there rather than failing the run.
archive = subprocess.run(  # noqa: S603
    ["bash", os.path.join(REPO_DIR, "scripts", "archive_outputs.sh")]  # noqa: S607
)
if archive.returncode != 0:
    print("WARNING: outputs not archived; copy outputs/ off this machine before teardown.")

print("Eval script completed successfully!")
