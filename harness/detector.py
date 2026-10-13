"""Block 4 detector: deployable triggers and reference baselines.

Inputs are limited to what a real robot has: the policy's outputs and proprioception. World
state is used only for labels and evaluation, never here. Pure functions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WidthBaseline:
    threshold_m: float = 0.003
    horizon: int = 60
    match_tol: int = 10


DEFAULT_WIDTH = WidthBaseline()


def close_commands(gripper_cmd: np.ndarray) -> list[int]:
    """Steps whose env gripper action switches from open (-1) to close (+1)."""
    closing = gripper_cmd > 0
    return [int(t) for t in np.flatnonzero(closing[1:] & ~closing[:-1]) + 1]


def nearest_command(commands: list[int], anchor: int, tol: int) -> int | None:
    near = [c for c in commands if abs(c - anchor) <= tol]
    return min(near, key=lambda c: abs(c - anchor)) if near else None


def finger_width(gripper_qpos: np.ndarray) -> np.ndarray:
    """Panda finger joints move in opposite signs; their difference is the opening in m."""
    return gripper_qpos[:, 0] - gripper_qpos[:, 1]


def empty_close_step(width: np.ndarray, command: int, params: WidthBaseline) -> int | None:
    """First step in [command, command + horizon] with the fingers closed below the threshold."""
    below = np.flatnonzero(width[command : command + params.horizon + 1] < params.threshold_m)
    return command + int(below[0]) if len(below) else None
