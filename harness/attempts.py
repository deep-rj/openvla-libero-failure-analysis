"""Grasp attempts and their outcomes, for Block 4 labels and evaluation.

An attempt is a close command at a goal object while no goal object is held; it runs until the
fingers reopen (`detector.attempt_end`). It succeeds if a hold begins before then and does not
end in a slip. Uses world state, so it labels and never feeds the detector. Pure functions.
"""

from dataclasses import dataclass

import numpy as np

from harness import detector, mechanism, milestones


@dataclass(frozen=True)
class AttemptParams:
    near_m: float = mechanism.DEFAULT_PARAMS.near_m
    open_width_m: float = milestones.DEFAULT_GRASP.open_width_m
    reopen_m: float = detector.DEFAULT_WIDTH.reopen_m


DEFAULT_PARAMS = AttemptParams()


@dataclass(frozen=True)
class Attempt:
    command: int
    end: int
    obj: int  # index into the goal-object columns
    outcome: str


Hold = tuple[int, int, bool]  # start, end (exclusive), ends in a slip


def label_attempts(gripper_cmd, width, dist, holds: list[list[Hold]], start: int,
                   params=DEFAULT_PARAMS) -> list[Attempt]:  # fmt: skip
    """Attempts in step order. `dist` is [T, n_goal], each goal object's distance to the pad
    midpoint; `holds` lists each goal object's holds."""
    out = []
    for c in detector.close_commands(gripper_cmd):
        obj = int(np.argmin(dist[c]))
        holding = any(a <= c < e for runs in holds for a, e, _ in runs)
        if c < start or dist[c, obj] >= params.near_m or holding:
            continue
        end = detector.attempt_end(width, c, params.reopen_m)
        begun = sorted(h for runs in holds for h in runs if c <= h[0] < end)
        if width[c:end].min() >= params.open_width_m:
            outcome = "aborted"
        elif not begun:
            outcome = "failure"
        else:
            outcome = "slip" if begun[0][2] else "success"
        out.append(Attempt(c, end, obj, outcome))
    return out
