"""Milestone `(k, phase)` and grasp state as pure functions of extracted world-state fields."""

from dataclasses import dataclass

import numpy as np

PLACE_PHASES = ("none", "grasped", "carried")
FIXTURE_PHASES = ("none", "engaged", "partial")


@dataclass(frozen=True)
class GraspParams:
    lift_m: float = 0.01  # above the settled height
    near_m: float = 0.12  # object origin to end effector
    open_width_m: float = 0.07  # finger width at or above this is open
    window: int = 3  # steps over which the object must move with the end effector
    co_move_m: float = 0.01  # max change in object-to-EEF offset over the window


@dataclass(frozen=True)
class MilestoneParams:
    debounce: int = 1
    engaged_fraction: float = 0.05
    partial_fraction: float = 0.5
    grasp: GraspParams = GraspParams()


DEFAULT_GRASP = GraspParams()
DEFAULT_PARAMS = MilestoneParams()


@dataclass(frozen=True)
class Milestone:
    k: int
    phase: str
    k_step: int | None  # first step at which k subgoals counted at once


def debounced(holds: np.ndarray, n: int, start: int) -> np.ndarray:
    """[T, G]: true at t when the predicate held at every step t-n+1..t, all at or after `start`."""
    holds = np.asarray(holds, dtype=bool)
    out = np.zeros_like(holds)
    for t in range(start + n - 1, len(holds)):
        out[t] = holds[t - n + 1 : t + 1].all(axis=0)
    return out


def counting(stable: np.ndarray, deps: list[tuple[int, ...]]) -> np.ndarray:
    """[T, G]: a subgoal counts when its own predicate is stable and every dependency counts."""
    n_goals = stable.shape[1]
    memo: dict[int, np.ndarray] = {}

    def counts(g, path=()):
        if g in path:
            raise ValueError(f"dependency cycle through subgoal {g}")
        if g not in memo:
            result = stable[:, g].copy()
            for d in deps[g]:
                result &= counts(d, (*path, g))
            memo[g] = result
        return memo[g]

    return np.stack([counts(g) for g in range(n_goals)], axis=1)


def max_simultaneous(count: np.ndarray) -> tuple[int, int | None]:
    """The most subgoals counting at one step, and the first step that reached it."""
    per_step = count.sum(axis=1)
    if len(per_step) == 0 or per_step.max() == 0:
        return 0, None
    k = int(per_step.max())
    return k, int(np.argmax(per_step == k))


def grasped(
    obj_pos: np.ndarray,
    eef_pos: np.ndarray,
    finger_width: np.ndarray,
    settled_z: float,
    params: GraspParams = DEFAULT_GRASP,
) -> np.ndarray:
    """[T]: lifted, near the end effector, fingers not open, and moving with the end effector."""
    offset = obj_pos - eef_pos
    lifted = obj_pos[:, 2] > settled_z + params.lift_m
    near = np.linalg.norm(offset, axis=1) < params.near_m
    closed = finger_width < params.open_width_m
    co_moving = np.zeros(len(obj_pos), dtype=bool)
    w = params.window
    for t in range(w, len(obj_pos)):
        drift = np.linalg.norm(offset[t - w : t + 1] - offset[t], axis=1).max()
        co_moving[t] = drift < params.co_move_m
    return lifted & near & closed & co_moving


def finger_width(gripper_qpos: np.ndarray) -> np.ndarray:
    return gripper_qpos[:, 0] - gripper_qpos[:, 1]


def travel_fraction(q: np.ndarray, threshold: float, start: int) -> np.ndarray:
    """[T]: share of the way from the settled joint value to the predicate threshold, in [0, 1]."""
    q0 = q[start]
    span = threshold - q0
    if not np.isfinite(span) or abs(span) < 1e-9:
        return np.full(len(q), np.nan)
    return np.clip((q - q0) / span, 0.0, 1.0)


def _phase_levels(extracted, info, params: MilestoneParams, start: int) -> np.ndarray:
    """[T, G] phase level per subgoal: 0 none, 1 grasped/engaged, 2 carried/partial."""
    subgoals = info["subgoals"]
    width = finger_width(extracted["gripper_qpos"])
    levels = np.zeros(extracted["goal_holds"].shape, dtype=int)
    for g, sub in enumerate(subgoals):
        if sub["kind"] == "place":
            obj = extracted["object_pos"][:, sub["object_index"]]
            held = grasped(obj, extracted["eef_pos"], width, obj[start, 2], params.grasp)
            levels[:, g] = np.where(held & extracted["over_target"][:, g], 2, held.astype(int))
        else:
            frac = np.nan_to_num(
                travel_fraction(extracted["fixture_q"][:, g], sub["threshold"], start)
            )
            levels[:, g] = np.where(
                frac >= params.partial_fraction, 2, (frac >= params.engaged_fraction).astype(int)
            )
    levels[:start] = 0
    return levels


def compute_milestone(extracted, info, params: MilestoneParams = DEFAULT_PARAMS) -> Milestone:
    subgoals = info["subgoals"]
    n_goals = len(subgoals)
    if info["final_success"]:
        # The state after the succeeding action is never logged, so success sets k directly.
        return Milestone(n_goals, "none", None)
    start = info["num_wait_steps"]
    deps = [tuple(s["depends_on"]) for s in subgoals]
    count = counting(debounced(extracted["goal_holds"], params.debounce, start), deps)
    k, k_step = max_simultaneous(count)

    levels = _phase_levels(extracted, info, params, start)
    at_k = count.sum(axis=1) == k
    at_k[:start] = False
    best_level, best_kind = 0, "place"
    for g, sub in enumerate(subgoals):
        prerequisites_met = np.ones(len(at_k), dtype=bool)
        for d in deps[g]:
            prerequisites_met &= count[:, d]
        open_steps = at_k & ~count[:, g] & prerequisites_met
        if open_steps.any():
            level = int(levels[open_steps, g].max())
            if level > best_level:
                best_level, best_kind = level, sub["kind"]
    names = PLACE_PHASES if best_kind == "place" else FIXTURE_PHASES
    return Milestone(k, names[best_level], k_step)


def closure_fraction(extracted, info, g: int) -> float:
    """Most travel achieved toward a fixture subgoal's threshold after settling; NaN if none."""
    sub = info["subgoals"][g]
    start = info["num_wait_steps"]
    frac = travel_fraction(extracted["fixture_q"][:, g], sub["threshold"], start)
    return float(np.nanmax(frac[start:])) if np.isfinite(frac).any() else float("nan")
