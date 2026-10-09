"""Behavior flags as pure functions of extracted world-state fields.

Every flag considers only steps at or after the end of the wait; displacements are measured
from the settled poses at that step. Each flag returns the step intervals where it fired, as
end-exclusive `(start, end)` pairs; an empty list means it did not fire.
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
    engaged_seconds: float = 5.0  # an object moving at the cap counts if gripped this recently


DEFAULT_FLAGS = FlagParams()


Intervals = list[tuple[int, int]]


def _runs(mask: np.ndarray, n: int = 1, offset: int = 0) -> Intervals:
    """Runs of at least `n` true steps, shifted by `offset`."""
    out, first = [], None
    for t, value in enumerate([*mask, False]):
        if value and first is None:
            first = t
        elif not value and first is not None:
            if t - first >= n:
                out.append((first + offset, t + offset))
            first = None
    return out


def _merge(intervals) -> Intervals:
    out: Intervals = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _union(interval_lists) -> Intervals:
    return _merge(i for intervals in interval_lists for i in intervals)


def _point(t) -> tuple[int, int]:
    return (int(t), int(t) + 1)


def format_intervals(intervals: Intervals) -> str:
    """`145:175;265`: Python slices, `;`-separated; a one-step interval is its step alone."""
    return ";".join(str(a) if b == a + 1 else f"{a}:{b}" for a, b in intervals)


def parse_intervals(text: str) -> Intervals:
    out = []
    for part in filter(None, text.split(";")):
        a, _, b = part.partition(":")
        out.append((int(a), int(b) if b else int(a) + 1))
    return out


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


def wrong_object(held: np.ndarray, moved: np.ndarray, goal_objects) -> Intervals:
    """Holds of a non-goal object that was grasped and moved. `held`, `moved`: [T, n_objects]."""
    others = [o for o in range(held.shape[1]) if o not in set(goal_objects)]
    return _merge(i for o in others if moved[:, o].any() for i in _runs(held[:, o]))


def collision_disturbance(
    pos, held, in_goal, start: int, d: float, settle: int, contact=None, rest_step=0.0005
) -> Intervals:
    """The object moved more than `d` while not held, outside the `settle` steps around a
    grasp (fingers closing before onset, settling after release), outside an unbroken
    `contact` run that ends in a grasp onset, and not inside its goal. Each firing runs from
    the step the displacement passed `d` until the object moves less than `rest_step` per step
    or is grasped; the reference then resets to where it stopped."""
    acquiring = np.zeros(len(pos), dtype=bool)
    for onset in np.flatnonzero(held[1:] & ~held[:-1]) + 1:
        first = onset - settle
        if contact is not None:
            t = onset - 1
            while t >= 0 and contact[t]:
                t -= 1
            first = min(first, t + 1)
        acquiring[max(first, 0) : onset] = True
    ref, ready_at, out, busy_until = pos[start], start, [], start
    for t in range(start, len(pos)):
        if t < busy_until:
            continue
        if held[t] or acquiring[t]:
            ref, ready_at = None, (t + 1 + settle) if held[t] else (t + 1)
            continue
        if t < ready_at:
            continue
        if ref is None or in_goal[t]:
            ref = pos[t]
            continue
        if np.linalg.norm(pos[t] - ref) > d:
            end = t + 1
            while (
                end < len(pos)
                and not held[end]
                and np.linalg.norm(pos[end] - pos[end - 1]) >= rest_step
            ):
                end += 1
            out.append((t, end))
            ref, busy_until = pos[end - 1], end
    return out


def wrong_target(
    candidate_holds, held, obj_speed, start: int, rest_speed: float, dwell: int
) -> Intervals:
    """Released and at rest for `dwell` steps on a receptacle that is not the goal."""
    at_rest = ~held & (obj_speed < rest_speed)
    found = []
    for c in range(candidate_holds.shape[1]):
        mask = candidate_holds[:, c] & at_rest
        mask[:start] = False
        found += _runs(mask, dwell)
    return _merge(found)


def regression(count: np.ndarray, n: int) -> Intervals:
    """A counted subgoal later stopped counting for at least `n` steps."""
    found = []
    for g in range(count.shape[1]):
        if count[:, g].any():
            first = int(np.argmax(count[:, g]))
            found += _runs(~count[first:, g], n, offset=first)
    return _merge(found)


def order_violation(stable: np.ndarray, count: np.ndarray, deps) -> Intervals:
    """A dependent subgoal's predicate held while a prerequisite did not count."""
    return _merge(
        i
        for g, prerequisites in enumerate(deps)
        for d in prerequisites
        for i in _runs(stable[:, g] & ~count[:, d])
    )


def dropped(z, held, over_target, fall_m: float, window: int) -> Intervals:
    """Releases followed by a fall of at least `fall_m` within `window` steps, away from
    the target; each at the release step."""
    out = []
    for t in range(1, len(z)):
        released = held[t - 1] and not held[t] and not over_target[t - 1]
        if released and z[t - 1] - z[t : t + window].min() >= fall_m:
            out.append(_point(t))
    return out


def regrasps(held: np.ndarray, start: int) -> Intervals:
    """Grasp onsets at or after `start`, after the first; a hold already on at `start` is
    the first."""
    onsets = [start] if held[start] else []
    onsets += list(np.flatnonzero(held[start + 1 :] & ~held[start:-1]) + start + 1)
    return [_point(t) for t in onsets[1:]]


def stalled(eef_pos, hz: float, start: int, v_min: float, seconds: float) -> Intervals:
    still = speed(eef_pos, hz) < v_min
    return _runs(still[start + 1 :], round(seconds * hz), offset=start + 1)


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


def oscillation(
    eef_pos, start: int, window: int, reversals: int, min_travel: float
) -> Intervals:
    """At least `reversals` direction reversals of `min_travel` along one axis within
    `window` steps, from the first to the last reversal; a reach down and lift reverses once."""
    found = []
    for axis in range(eef_pos.shape[1]):
        turns = [t + start for t in _reversal_steps(eef_pos[start:, axis], min_travel)]
        for i in range(len(turns) - reversals + 1):
            first, last = turns[i], turns[i + reversals - 1]
            if last - first < window:
                found.append((first, last + 1))
    return _merge(found)


def gripper_chatter(width, any_held, start: int, window: int, cycles: int,
                    open_w: float = 0.06, closed_w: float = 0.02) -> Intervals:  # fmt: skip
    """At least `cycles` empty closes within `window` steps, with nothing held throughout,
    from the first close to the last."""
    closes, state = [], "open"
    for t in range(len(width)):
        if state == "open" and width[t] < closed_w:
            state = "closed"
            if t >= start:
                closes.append(t)
        elif state == "closed" and width[t] > open_w:
            state = "open"
    found = []
    for i in range(len(closes) - cycles + 1):
        first, last = closes[i], closes[i + cycles - 1]
        if last - first < window and not any_held[first : last + 1].any():
            found.append((first, last + 1))
    return _merge(found)


def object_off_table(z, settled_z: float, drop_m: float) -> Intervals:
    return _runs(z < settled_z - drop_m)


def workspace_limit(arm_q, ranges, start: int, fraction: float, n: int) -> Intervals:
    """An arm joint within `fraction` of its range from either limit for `n` steps."""
    found = []
    for j, (lo, hi) in enumerate(ranges):
        margin = fraction * (hi - lo)
        near = (arm_q[:, j] < lo + margin) | (arm_q[:, j] > hi - margin)
        found += _runs(near[start:], n, offset=start)
    return _merge(found)


def engaged_within(
    obj_pos, eef_pos, width, steps: int, grasp: milestones.GraspParams = milestones.DEFAULT_GRASP
) -> bool:
    """Fingers not open within reach of the object at some point in the last `steps` steps,
    counting the last step and the one `steps` before it."""
    recent = slice(len(obj_pos) - 1 - steps, None)
    near = np.linalg.norm(obj_pos[recent] - eef_pos[recent], axis=1) < grasp.near_m
    return bool((near & (width[recent] < grasp.open_width_m)).any())


def truncated_mid_action(
    final_success: bool, held_last: bool, moving_last: bool, last_step: int
) -> Intervals:
    """The last step, when a failure ends with a goal object held or still moving."""
    return [_point(last_step)] if not final_success and (held_last or moving_last) else []


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

    engaged_steps = round(params.engaged_seconds * hz)
    moving_last = any(
        obj_speed[o][-1] >= params.rest_speed
        and engaged_within(obj_pos[:, o], eef, width, engaged_steps, ms_params.grasp)
        for o in goal_objects
    ) or any(
        abs(extracted["fixture_q"][-1, g] - extracted["fixture_q"][-2, g]) * hz > 0.01
        for g, s in enumerate(subgoals)
        if s["kind"] == "fixture"
    )
    return {
        "wrong_object": wrong_object(held, moved, goal_objects),
        "wrong_target": _union(
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
        "dropped": _union(
            dropped(
                obj_pos[:, s["object_index"], 2],
                goal_held[:, g],
                extracted["over_target"][:, g],
                params.fall_m,
                params.dwell,
            )
            for g, s in place
        ),
        "collision_disturbance": _union(
            collision_disturbance(
                obj_pos[:, o],
                held[:, o],
                in_goal[:, o],
                start,
                params.displaced_m,
                params.dwell,
                contact=milestones.in_contact(obj_pos[:, o], eef, width, ms_params.grasp),
                rest_step=params.rest_speed / hz,
            )
            for o in range(n_objects)
        ),
        "regrasp_count": _union(regrasps(held[:, o], start) for o in goal_objects),
        "stalled": stalled(eef, hz, start, v_min, params.stall_seconds),
        "oscillation": oscillation(
            eef, start, window, params.oscillation_reversals, params.oscillation_min_travel
        ),
        "gripper_chatter": gripper_chatter(
            width, any_held, start, chatter_window, params.chatter_cycles
        ),
        "object_off_table": _union(
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
            info["final_success"], bool(goal_held[-1].any()), moving_last, len(eef) - 1
        ),
    }
