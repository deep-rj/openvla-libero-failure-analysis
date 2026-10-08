"""Per-episode trace schema and its HDF5 reader/writer.

Constant per-episode keys (episode_id, task_id, seed, perturbation_config_id) live in the
episode metadata; the per-step columns are stacked datasets indexed by step_idx.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from harness.decode import DecodedAction

SCHEMA_VERSION = 1
N_DIMS = 7
SLICE_WIDTH = 256
INT_SENTINEL = -1


@dataclass
class ModelOutput:
    slice_logits: np.ndarray  # [7, 256] fp16, raw logits over token ids 31744..31999
    lse_full: np.ndarray  # [7] fp32, log-sum-exp over the full head
    argmax_full_id: np.ndarray  # [7] int
    chosen_token_id: np.ndarray  # [7] int
    inference_latency_ms: float


@dataclass
class WorldState:
    sim_state: np.ndarray
    object_poses: np.ndarray  # [n_obj, 7]: position + quaternion as exposed by the env
    fixture_joint_qpos: np.ndarray  # [n_joint]
    eef_pos: np.ndarray
    eef_quat_raw: np.ndarray
    gripper_qpos: np.ndarray


@dataclass
class StepRecord:
    """State before action `step_idx`; `done`/`success` result from executing it."""

    step_idx: int
    model: ModelOutput | None  # None on wait steps
    decoded: DecodedAction | None
    executed_action: np.ndarray  # exactly what was passed to env.step
    wall_timestamp: float
    world: WorldState
    done: bool
    success: bool

    @property
    def is_wait_step(self) -> bool:
        return self.model is None


def _model_column(steps, getter, shape, dtype, fill):
    empty = np.full(shape, fill, dtype=dtype)
    return [empty if getter(s) is None else np.asarray(getter(s)) for s in steps]


# name -> (per-step shape or None to infer, dtype, wait-step fill, getter)
_COLUMNS = {
    "step_idx": (None, np.int32, None, lambda s: s.step_idx),
    "is_wait_step": (None, np.bool_, None, lambda s: s.is_wait_step),
    "slice_logits": (
        (N_DIMS, SLICE_WIDTH),
        np.float16,
        np.nan,
        lambda s: s.model and s.model.slice_logits,
    ),
    "lse_full": ((N_DIMS,), np.float32, np.nan, lambda s: s.model and s.model.lse_full),
    "argmax_full_id": (
        (N_DIMS,),
        np.int32,
        INT_SENTINEL,
        lambda s: s.model and s.model.argmax_full_id,
    ),
    "chosen_token_id": (
        (N_DIMS,),
        np.int32,
        INT_SENTINEL,
        lambda s: s.model and s.model.chosen_token_id,
    ),
    "bin_idx": ((N_DIMS,), np.int32, INT_SENTINEL, lambda s: s.decoded and s.decoded.bin_idx),
    "normalized_action": (
        (N_DIMS,),
        np.float32,
        np.nan,
        lambda s: s.decoded and s.decoded.normalized_action,
    ),
    "unnormalized_action": (
        (N_DIMS,),
        np.float32,
        np.nan,
        lambda s: s.decoded and s.decoded.unnormalized_action,
    ),
    "clipped_flag": ((N_DIMS,), np.bool_, False, lambda s: s.decoded and s.decoded.clipped_flag),
    "executed_action": ((N_DIMS,), np.float64, None, lambda s: s.executed_action),
    "inference_latency_ms": (
        (),
        np.float32,
        np.nan,
        lambda s: s.model and s.model.inference_latency_ms,
    ),
    "wall_timestamp": (None, np.float64, None, lambda s: s.wall_timestamp),
    "sim_state": (None, np.float64, None, lambda s: s.world.sim_state),
    "object_poses": (None, np.float32, None, lambda s: s.world.object_poses),
    "fixture_joint_qpos": (None, np.float32, None, lambda s: s.world.fixture_joint_qpos),
    "eef_pos": (None, np.float64, None, lambda s: s.world.eef_pos),
    "eef_quat_raw": (None, np.float64, None, lambda s: s.world.eef_quat_raw),
    "gripper_qpos": (None, np.float64, None, lambda s: s.world.gripper_qpos),
    "done": (None, np.bool_, None, lambda s: s.done),
    "success": (None, np.bool_, None, lambda s: s.success),
}


def _column(name, steps):
    shape, dtype, fill, getter = _COLUMNS[name]
    if fill is not None:
        values = _model_column(steps, getter, shape, dtype, fill)
    else:
        values = [np.asarray(getter(s)) for s in steps]
    expected = shape if shape is not None else values[0].shape
    for idx, value in enumerate(values):
        if value.shape != expected:
            raise ValueError(f"{name}: step {idx} has shape {value.shape}, expected {expected}")
    return np.stack(values).astype(dtype)


def _to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"metadata value of type {type(value).__name__} is not serializable")


def write_episode(path, steps: list[StepRecord], metadata: dict) -> None:
    """Writes one episode file atomically; a failure leaves any existing file untouched."""
    if [s.step_idx for s in steps] != list(range(len(steps))):
        raise ValueError("step_idx must count contiguously from 0")
    columns = {name: _column(name, steps) for name in _COLUMNS}

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["metadata"] = json.dumps(metadata, sort_keys=True, default=_to_json)
            for name, data in columns.items():
                f.create_dataset(name, data=data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Columns that come from the simulator; reading only these keeps labeling blind to the model.
WORLD_COLUMNS = (
    "step_idx",
    "is_wait_step",
    "sim_state",
    "object_poses",
    "fixture_joint_qpos",
    "eef_pos",
    "eef_quat_raw",
    "gripper_qpos",
)


def read_world_columns(path) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as f:
        version = f.attrs["schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {version} unsupported (expected {SCHEMA_VERSION})")
        metadata = json.loads(f.attrs["metadata"])
        arrays = {name: f[name][()] for name in WORLD_COLUMNS}
    metadata["schema_version"] = int(version)
    return arrays, metadata


def read_episode(path) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as f:
        version = f.attrs["schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {version} unsupported (expected {SCHEMA_VERSION})")
        metadata = json.loads(f.attrs["metadata"])
        arrays = {name: f[name][()] for name in f}
    metadata["schema_version"] = int(version)
    return arrays, metadata
