"""Milestone `(k, phase)` and grasp state as pure functions of extracted world-state fields."""

from dataclasses import dataclass

import numpy as np

PLACE_PHASES = ("none", "grasped", "carried")
FIXTURE_PHASES = ("none", "engaged", "partial")


@dataclass(frozen=True)
class GraspParams:
    lift_m: float = 0.01  # above the lowest height in the support window
    support_window: int = 20  # steps; a raised resting surface (plate, caddy) is the support
    near_m: float = 0.12  # object origin to end effector
    open_width_m: float = 0.07  # finger width at or above this is open
    window: int = 3  # steps over which the object must move with the end effector
    co_move_m: float = 0.01  # max change in object-to-EEF offset over the window
    move_m: float = 0.005  # min object travel over the window; still objects keep a fixed offset


@dataclass(frozen=True)
class MilestoneParams:
    debounce: int = 10  # steps placed after release; 0.5 s at 20 Hz
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
    params: GraspParams = DEFAULT_GRASP,
) -> np.ndarray:
    """[T]: starts when a lifted object moves with the end effector with fingers not open;
    continues while the fingers stay closed and the offset stays fixed."""
    offset = obj_pos - eef_pos
    z = obj_pos[:, 2]
    support = np.array(
        [z[max(t - params.support_window, 0) : t + 1].min() for t in range(len(z))]
    )
    lifted = z > support + params.lift_m
    near = np.linalg.norm(offset, axis=1) < params.near_m
    closed = finger_width < params.open_width_m
    held = np.zeros(len(obj_pos), dtype=bool)
    w = params.window
    for t in range(w, len(obj_pos)):
        drift = np.linalg.norm(offset[t - w : t + 1] - offset[t], axis=1).max()
        moved = np.linalg.norm(obj_pos[t] - obj_pos[t - w]) >= params.move_m
        onset = lifted[t] and drift < params.co_move_m and moved
        kept = held[t - 1] and np.linalg.norm(offset[t] - offset[t - 1]) < params.co_move_m
        held[t] = near[t] and closed[t] and (onset or kept)
    return held


def in_contact(
    obj_pos: np.ndarray,
    eef_pos: np.ndarray,
    finger_width: np.ndarray,
    params: GraspParams = DEFAULT_GRASP,
) -> np.ndarray:
    """[T]: fingers not open, object within reach, and its offset to the end effector fixed
    since the previous step; no lift required, so it covers a grip before the object lifts."""
    offset = obj_pos - eef_pos
    steady = np.zeros(len(obj_pos), dtype=bool)
    steady[1:] = np.linalg.norm(np.diff(offset, axis=0), axis=1) < params.co_move_m
    near = np.linalg.norm(offset, axis=1) < params.near_m
    return near & (finger_width < params.open_width_m) & steady


def finger_width(gripper_qpos: np.ndarray) -> np.ndarray:
    return gripper_qpos[:, 0] - gripper_qpos[:, 1]


def travel_fraction(q: np.ndarray, threshold: float, start: int) -> np.ndarray:
    """[T]: share of the way from the settled joint value to the predicate threshold, in [0, 1]."""
    q0 = q[start]
    span = threshold - q0
    if not np.isfinite(span) or abs(span) < 1e-9:
        return np.full(len(q), np.nan)
    return np.clip((q - q0) / span, 0.0, 1.0)


def held_goal_objects(extracted, info, params: MilestoneParams) -> np.ndarray:
    """[T, G]: the subgoal's object is grasped; always false for fixture subgoals."""
    width = finger_width(extracted["gripper_qpos"])
    held = np.zeros(extracted["goal_holds"].shape, dtype=bool)
    for g, sub in enumerate(info["subgoals"]):
        if sub["kind"] == "place":
            obj = extracted["object_pos"][:, sub["object_index"]]
            held[:, g] = grasped(obj, extracted["eef_pos"], width, params.grasp)
    return held


def _phase_levels(extracted, info, params: MilestoneParams, start: int, held) -> np.ndarray:
    """[T, G] phase level per subgoal: 0 none, 1 grasped/engaged, 2 carried/partial."""
    levels = np.zeros(extracted["goal_holds"].shape, dtype=int)
    for g, sub in enumerate(info["subgoals"]):
        if sub["kind"] == "place":
            over = held[:, g] & extracted["over_target"][:, g]
            levels[:, g] = np.where(over, 2, held[:, g].astype(int))
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
    held = held_goal_objects(extracted, info, params)
    # A placement counts only once released: LIBERO's predicate alone accepts a gripped touchdown.
    placed = extracted["goal_holds"] & ~held
    count = counting(debounced(placed, params.debounce, start), deps)
    k, k_step = max_simultaneous(count)

    levels = _phase_levels(extracted, info, params, start, held)
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
