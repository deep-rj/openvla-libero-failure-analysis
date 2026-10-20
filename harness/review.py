"""Held-out review of the failure detector: a grasp-then-stall rule for a sensitivity analysis, and
the steps that drive a window mean. Pure functions."""

from dataclasses import dataclass

import numpy as np

from harness import milestones


@dataclass(frozen=True)
class StallParams:
    empty_m: float = 0.003  # width baseline threshold: below this the fingers closed on nothing
    open_m: float = 0.066  # below the 7 cm open width, so a barely-closed hand is not a grasp
    still_m: float = 0.001  # max width change per step for the fingers to count as settled
    min_settled: int = 30
    max_rise_m: float = 0.02


DEFAULT_STALL = StallParams()


def settled_mask(width: np.ndarray, params: StallParams = DEFAULT_STALL) -> np.ndarray:
    """Steps with the fingers closed on something and not moving."""
    closed = (width > params.empty_m) & (width < milestones.DEFAULT_GRASP.open_width_m)
    return closed & (np.abs(np.gradient(width)) < params.still_m)


def is_stall(outcome: str, settled_steps: int, settled_width_m: float, rise_m: float,
             params: StallParams = DEFAULT_STALL) -> bool:  # fmt: skip
    """A failed attempt whose fingers settle on an object that never rises."""
    return (
        outcome == "failure"
        and settled_steps >= params.min_settled
        and params.empty_m < settled_width_m < params.open_m
        and rise_m < params.max_rise_m
    )


def window_drivers(exp_h: np.ndarray, window: np.ndarray, command: int, floor: float = 0.05):
    """(exp(H) - 1, offset from the command) for window steps above 1 + floor, largest first."""
    excess = exp_h[window] - 1
    keep = excess > floor
    offsets = (window[keep] - command).tolist()
    return sorted(zip(excess[keep].tolist(), offsets, strict=True), reverse=True)
