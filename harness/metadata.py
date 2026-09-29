"""Per-episode metadata and the run environment facts it records."""

import hashlib
import subprocess
from importlib import metadata as importlib_metadata

import numpy as np

from harness.policy import build_prompt
from harness.rollout import NUM_STEPS_WAIT, EpisodeResult, EpisodeSpec
from harness.world import SceneLayout

CONTROL_FREQ_HZ = 20
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "mujoco",
    "robosuite",
    "h5py",
    "imageio",
    "tensorflow",
)


def mjcf_sha256(xml: str) -> str:
    return hashlib.sha256(xml.encode()).hexdigest()


def git_state(path) -> dict:
    """HEAD SHA of the checkout at `path` and whether tracked files differ from it."""

    def git(*args):
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(path), *args],  # noqa: S607
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


def _version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def library_versions(packages=_TRACKED_PACKAGES) -> dict[str, str | None]:
    return {name: _version(name) for name in packages}


def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def environment_info(repo_dir, siblings: dict | None = None) -> dict:
    """Facts about the machine and code that are constant across a run."""
    git = {"harness": git_state(repo_dir)}
    git.update({name: git_state(path) for name, path in (siblings or {}).items()})
    return {"git": git, "versions": library_versions(), "gpu": gpu_name()}


def episode_metadata(
    *,
    spec: EpisodeSpec,
    task_label: str,
    layout: SceneLayout,
    result: EpisodeResult,
    mjcf_xml: str,
    provenance: dict,
    action_stats: dict,
    max_steps: int,
    checkpoint: str,
    unnorm_key: str,
    center_crop: bool,
    environment: dict,
) -> dict:
    return {
        "episode_id": spec.episode_id,
        "suite": spec.suite,
        "task_id": spec.task_id,
        "init_state_idx": spec.init_state_idx,
        "seed": spec.seed,
        "perturbation_config_id": spec.perturbation_config_id,
        "task_instruction": task_label,
        "prompt": build_prompt(task_label),
        **layout.metadata(),
        "mjcf_sha256": mjcf_sha256(mjcf_xml),
        "checkpoint": checkpoint,
        **provenance,
        "unnorm_key": unnorm_key,
        "center_crop": center_crop,
        "image_flip": True,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "num_wait_steps": NUM_STEPS_WAIT,
        "max_steps": max_steps,
        "action_stats": {k: np.asarray(v).tolist() for k, v in action_stats.items()},
        **environment,
        "final_success": result.success,
        "terminated_by": result.terminated_by,
        "num_policy_steps": result.num_policy_steps,
        "episode_length": len(result.executed_actions),
    }
