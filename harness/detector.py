"""Block 4 detector: deployable triggers and reference baselines.

Inputs are limited to what a real robot has: the policy's outputs and proprioception. World
state is used only for labels and evaluation, never here. Pure functions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WidthBaseline:
    threshold_m: float = 0.003
    reopen_m: float = 0.005
    match_tol: int = 10


DEFAULT_WIDTH = WidthBaseline()


def close_commands(gripper_cmd: np.ndarray) -> list[int]:
    """Steps whose env gripper action switches from open (-1) to close (+1)."""
    closing = gripper_cmd > 0
    return [int(t) for t in np.flatnonzero(closing[1:] & ~closing[:-1]) + 1]


def nearest_command(commands: list[int], anchor: int, tol: int) -> int | None:
    near = [c for c in commands if abs(c - anchor) <= tol]
    return min(near, key=lambda c: abs(c - anchor)) if near else None


def attempt_end(width: np.ndarray, command: int, reopen_m: float) -> int:
    """First step after `command` whose width exceeds its narrowest since `command` by more than
    `reopen_m`, or len(width). Reads the fingers, not the command: one-step open commands are
    common and barely move them."""
    narrowest = np.minimum.accumulate(width[command:])
    reopened = np.flatnonzero(width[command:] > narrowest + reopen_m)
    return command + int(reopened[0]) if len(reopened) else len(width)


def empty_close_step(width: np.ndarray, command: int, params: WidthBaseline) -> int | None:
    """First step of the attempt begun at `command` with the fingers closed below the threshold."""
    end = attempt_end(width, command, params.reopen_m)
    below = np.flatnonzero(width[command:end] < params.threshold_m)
    return command + int(below[0]) if len(below) else None
