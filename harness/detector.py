"""Failure detector: deployable triggers and reference baselines.

Inputs are limited to what a real robot has: the policy's outputs and proprioception. World
state is used only for labels and evaluation, never here. Pure functions.
"""

from dataclasses import dataclass

import numpy as np

from harness import gate


@dataclass(frozen=True)
class DetectorSpec:
    gripper_window: int = 30
    roll_window: int = 100
    budget: float = 0.2  # share of successful first attempts allowed to alarm
    min_window: int = 10


DEFAULT_SPEC = DetectorSpec()


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


def window_mean(values, is_wait, command: int, length: int, keep=None, min_window: int = 10):
    """Mean of `values` over the policy steps in the `length` steps before `command`, where `keep`
    holds; None if the window has fewer than min(length, min_window) steps or keeps none."""
    window = gate.pre_window(is_wait, command, length, min(length, min_window))
    if window is None:
        return None
    return gate.subset_mean(values, window, np.ones(len(values), bool) if keep is None else keep)


def percentile_rank(reference: np.ndarray, x: float) -> float:
    """Share of `reference` (sorted) at or below `x`."""
    return float(np.searchsorted(reference, x, side="right") / len(reference))


def budget_threshold(success_scores: np.ndarray, budget: float) -> float:
    """A threshold above which at most `budget` of `success_scores` lie."""
    ordered = np.sort(success_scores)[::-1]
    return float(ordered[int(np.floor(budget * len(ordered)))])


def v2_score(gripper: float, roll: float | None, gripper_ref, roll_ref) -> float:
    """Mean percentile rank against successful first attempts; the gripper's alone when roll has
    no stop-filtered steps in its window."""
    ranks = [percentile_rank(gripper_ref, gripper)]
    if roll is not None:
        ranks.append(percentile_rank(roll_ref, roll))
    return float(np.mean(ranks))


def excess_over_chance(alarm: np.ndarray, failed: np.ndarray) -> float:
    """Catch rate on failures minus false-alarm rate on successes: what a random alarm firing as
    often on successes would not achieve."""
    return float(alarm[failed].mean() - alarm[~failed].mean())
