"""Behavior flags as pure functions of extracted world-state fields.

Every flag considers only steps at or after the end of the wait; displacements are measured
from the settled poses at that step.
"""

from dataclasses import dataclass

import numpy as np

from harness import milestones


@dataclass(frozen=True)
class FlagParams:
    displaced_m: float = 0.02
    rest_speed: float = 0.01  # m/s
    dwell: int = 10  # steps at rest, and steps after a release before motion counts
    fall_m: float = 0.03  # drop after release that separates a drop from setting down
    stall_seconds: float = 2.0
    stall_jitter_factor: float = 3.0
    stall_floor: float = 0.005  # m/s; still, whatever the wait-step jitter
    oscillation_seconds: float = 4.0
    oscillation_reversals: int = 6
    oscillation_min_travel: float = 0.02
    chatter_seconds: float = 2.0
    chatter_cycles: int = 3
    off_table_m: float = 0.05
    joint_limit_fraction: float = 0.02
    joint_limit_steps: int = 10


DEFAULT_FLAGS = FlagParams()


def _runs_at_least(mask: np.ndarray, n: int) -> bool:
    run = 0
    for value in mask:
        run = run + 1 if value else 0
        if run >= n:
            return True
    return False


def speed(pos: np.ndarray, hz: float) -> np.ndarray:
    """[T] m/s from consecutive positions; step 0 is 0."""
    out = np.zeros(len(pos))
    out[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * hz
    return out


def displaced(pos: np.ndarray, start: int, d: float) -> np.ndarray:
    """[T]: farther than `d` from the settled pose at `start`; false before it."""
    out = np.linalg.norm(pos - pos[start], axis=1) > d
    out[:start] = False
    return out


def wrong_object(held: np.ndarray, moved: np.ndarray, goal_objects) -> bool:
    """A non-goal object was grasped and moved. `held`, `moved`: [T, n_objects]."""
    others = [o for o in range(held.shape[1]) if o not in set(goal_objects)]
    return any(held[:, o].any() and moved[:, o].any() for o in others)


def collision_disturbance(pos, held, in_goal, start: int, d: float, settle: int) -> bool:
    """The object moved more than `d` while not held, outside the `settle` steps around a
    grasp (fingers closing before onset, settling after release) and not inside its goal."""
    acquiring = np.zeros(len(pos), dtype=bool)
    for onset in np.flatnonzero(held[1:] & ~held[:-1]) + 1:
        acquiring[max(onset - settle, 0) : onset] = True
    ref, ready_at = pos[start], start
    for t in range(start, len(pos)):
        if held[t] or acquiring[t]:
            ref, ready_at = None, (t + 1 + settle) if held[t] else (t + 1)
            continue
        if t < ready_at:
            continue
        if ref is None or in_goal[t]:
            ref = pos[t]
            continue
        if np.linalg.norm(pos[t] - ref) > d:
            return True
    return False


def wrong_target(candidate_holds, held, obj_speed, start: int, rest_speed: float, dwell: int):
    """Released and at rest for `dwell` steps on a receptacle that is not the goal."""
    at_rest = ~held & (obj_speed < rest_speed)
    for c in range(candidate_holds.shape[1]):
        mask = candidate_holds[:, c] & at_rest
        mask[:start] = False
        if _runs_at_least(mask, dwell):
            return True
    return False


def regression(count: np.ndarray, n: int) -> bool:
    """A counted subgoal later stopped counting for at least `n` steps."""
    for g in range(count.shape[1]):
        first = np.argmax(count[:, g]) if count[:, g].any() else None
        if first is not None and _runs_at_least(~count[first:, g], n):
            return True
    return False


def order_violation(stable: np.ndarray, count: np.ndarray, deps) -> bool:
    """A dependent subgoal's predicate held while a prerequisite did not count."""
    for g, prerequisites in enumerate(deps):
        for d in prerequisites:
            if (stable[:, g] & ~count[:, d]).any():
                return True
    return False


def dropped(z, held, over_target, fall_m: float, window: int) -> bool:
    """A release followed by a fall of at least `fall_m` within `window` steps, away from
    the target."""
    for t in range(1, len(z)):
        released = held[t - 1] and not held[t] and not over_target[t - 1]
        if released and z[t - 1] - z[t : t + window].min() >= fall_m:
            return True
    return False


def regrasp_count(held: np.ndarray) -> int:
    onsets = int(held[0]) + int(np.sum(held[1:] & ~held[:-1]))
    return max(onsets - 1, 0)


def stalled(eef_pos, hz: float, start: int, v_min: float, seconds: float) -> bool:
    still = speed(eef_pos, hz) < v_min
    return _runs_at_least(still[start + 1 :], round(seconds * hz))


def _reversal_steps(x: np.ndarray, min_travel: float) -> list[int]:
    """Steps where motion along one axis turns around after at least `min_travel`."""
    turns, direction, extreme = [], 0, x[0]
    for t, value in enumerate(x):
        if direction >= 0 and value < extreme - min_travel:
            if direction > 0:
                turns.append(t)
            direction, extreme = -1, value
        elif direction <= 0 and value > extreme + min_travel:
            if direction < 0:
                turns.append(t)
            direction, extreme = 1, value
        elif direction > 0:
            extreme = max(extreme, value)
        elif direction < 0:
            extreme = min(extreme, value)
    return turns


def oscillation(eef_pos, start: int, window: int, reversals: int, min_travel: float) -> bool:
    """At least `reversals` direction reversals of `min_travel` along one axis within
    `window` steps; a reach down and lift reverses once."""
    for axis in range(eef_pos.shape[1]):
        turns = _reversal_steps(eef_pos[start:, axis], min_travel)
        if any(
            turns[i + reversals - 1] - turns[i] < window
            for i in range(len(turns) - reversals + 1)
        ):
            return True
    return False


def gripper_chatter(width, any_held, start: int, window: int, cycles: int,
                    open_w: float = 0.06, closed_w: float = 0.02) -> bool:  # fmt: skip
    """At least `cycles` empty closes within `window` steps, with nothing held throughout."""
    closes, state = [], "open"
    for t in range(len(width)):
        if state == "open" and width[t] < closed_w:
            state = "closed"
            if t >= start:
                closes.append(t)
        elif state == "closed" and width[t] > open_w:
            state = "open"
    for i in range(len(closes) - cycles + 1):
        first, last = closes[i], closes[i + cycles - 1]
        if last - first < window and not any_held[first : last + 1].any():
            return True
    return False


def object_off_table(z, settled_z: float, drop_m: float) -> bool:
    return bool((z < settled_z - drop_m).any())


def workspace_limit(arm_q, ranges, start: int, fraction: float, n: int) -> bool:
    """An arm joint within `fraction` of its range from either limit for `n` steps."""
    for j, (lo, hi) in enumerate(ranges):
        margin = fraction * (hi - lo)
        near = (arm_q[:, j] < lo + margin) | (arm_q[:, j] > hi - margin)
        if _runs_at_least(near[start:], n):
            return True
    return False


def truncated_mid_action(final_success: bool, held_last: bool, moving_last: bool) -> bool:
    return not final_success and (held_last or moving_last)


def compute_flags(
    extracted, info, params: FlagParams = DEFAULT_FLAGS, ms_params=milestones.DEFAULT_PARAMS
) -> dict:
    start = info["num_wait_steps"]
    hz = info["control_freq_hz"]
    subgoals = info["subgoals"]
    deps = [tuple(s["depends_on"]) for s in subgoals]
    obj_pos = extracted["object_pos"]
    eef = extracted["eef_pos"]
    width = milestones.finger_width(extracted["gripper_qpos"])
    n_objects = obj_pos.shape[1]

    held = np.stack(
        [
            milestones.grasped(obj_pos[:, o], eef, width, ms_params.grasp)
            for o in range(n_objects)
        ],
        axis=1,
    )
    moved = np.stack(
        [displaced(obj_pos[:, o], start, params.displaced_m) for o in range(n_objects)], axis=1
    )
    goal_held = milestones.held_goal_objects(extracted, info, ms_params)
    placed = extracted["goal_holds"] & ~goal_held
    count = milestones.counting(milestones.debounced(placed, ms_params.debounce, start), deps)
    stable = milestones.debounced(extracted["goal_holds"], ms_params.debounce, start)

    place = [(g, s) for g, s in enumerate(subgoals) if s["kind"] == "place"]
    goal_objects = [s["object_index"] for _, s in place]
    in_goal = np.zeros((len(obj_pos), n_objects), dtype=bool)
    for g, s in place:
        in_goal[:, s["object_index"]] |= placed[:, g]
    obj_speed = [speed(obj_pos[:, o], hz) for o in range(n_objects)]

    wait_speed = speed(eef[:start], hz)[1:]
    jitter = float(np.median(wait_speed)) if len(wait_speed) else 0.0
    v_min = max(params.stall_jitter_factor * jitter, params.stall_floor)
    window = round(params.oscillation_seconds * hz)
    chatter_window = round(params.chatter_seconds * hz)
    any_held = held.any(axis=1)

    moving_last = any(obj_speed[o][-1] >= params.rest_speed for o in goal_objects) or any(
        abs(extracted["fixture_q"][-1, g] - extracted["fixture_q"][-2, g]) * hz > 0.01
        for g, s in enumerate(subgoals)
        if s["kind"] == "fixture"
    )
    return {
        "wrong_object": wrong_object(held, moved, goal_objects),
        "wrong_target": any(
            wrong_target(
                extracted["candidate_holds"][:, g, : len(s["candidates"])],
                goal_held[:, g],
                obj_speed[s["object_index"]],
                start,
                params.rest_speed,
                params.dwell,
            )
            for g, s in place
        ),
        "regression": regression(count, ms_params.debounce),
        "order_violation": order_violation(stable, count, deps),
        "dropped": any(
            dropped(
                obj_pos[:, s["object_index"], 2],
                goal_held[:, g],
                extracted["over_target"][:, g],
                params.fall_m,
                params.dwell,
            )
            for g, s in place
        ),
        "collision_disturbance": any(
            collision_disturbance(
                obj_pos[:, o], held[:, o], in_goal[:, o], start, params.displaced_m, params.dwell
            )
            for o in range(n_objects)
        ),
        "regrasp_count": sum(regrasp_count(held[start:, o]) for o in goal_objects),
        "stalled": stalled(eef, hz, start, v_min, params.stall_seconds),
        "oscillation": oscillation(
            eef, start, window, params.oscillation_reversals, params.oscillation_min_travel
        ),
        "gripper_chatter": gripper_chatter(
            width, any_held, start, chatter_window, params.chatter_cycles
        ),
        "object_off_table": any(
            object_off_table(obj_pos[:, o, 2], obj_pos[start, o, 2], params.off_table_m)
            for o in goal_objects
        ),
        "workspace_limit": workspace_limit(
            extracted["arm_q"],
            [j["range"] for j in info["arm_joints"]],
            start,
            params.joint_limit_fraction,
            params.joint_limit_steps,
        ),
        "truncated_mid_action": truncated_mid_action(
            info["final_success"], bool(goal_held[-1].any()), moving_last
        ),
    }
