"""Mechanism classifier: assigns a failed episode its mechanism and onset from world state.

Pure functions over extracted world fields (`world_extract`), frozen flags (`flags`) and the
per-task collision boxes (`object_geometry.json`). Rules: `docs/labeling_protocol.md`,
"Mechanism classifier".
"""

import re
from dataclasses import dataclass

import numpy as np

from harness import flags, milestones

COVER_GRID = (15, 9, 9)  # samples across the gap, and over the pad face (y, z)


@dataclass(frozen=True)
class MechanismParams:
    cover_cutoff: float = 0.3
    any_cover_objects: tuple[str, ...] = ("moka_pot",)  # thin handles: any cover is between
    close_rate_m: float = 0.004  # pad gap shrink over 2 steps that starts a close
    close_end_rate_m: float = 0.002
    close_end_cap: int = 20
    near_m: float = 0.12  # pad midpoint to object origin for an attempt on that object
    grasp_follow: int = 30  # steps after a close within which a grasp onset means success
    descent_window: int = 60  # steps before a close searched for its bottom
    descent_m: float = 0.03
    rise_m: float = 0.02
    rise_window: int = 20
    descent_close_gap: int = 5  # steps after an open descent's bottom that a close absorbs it
    slip_leave_m: float = 0.005
    slip_open_m: float = 0.002
    slip_ref: int = 15  # steps before a hold ends that serve as the held reference
    shift_m: float = 0.005
    shift_settle: int = 5  # steps after a grasp onset before the held offset is the reference
    place_near_m: float = 0.10  # release this close to an object target is a placement
    descent_tol_m: float = 0.001
    rest_band_m: float = 0.01  # within this of its last height, a placed object is at rest
    defect_delay: int = 10  # steps after a placement first holds when its state is read


DEFAULT_PARAMS = MechanismParams()


def quat_to_mat(q_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=float) / np.linalg.norm(q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pad_face_cover(pads, grip_q, pad_half, obj_p, obj_q, boxes) -> float:
    """Share of the pad face whose line along the closing axis (grip x), between the pads'
    inner faces, passes through any of the object's boxes (body frame)."""
    grip = quat_to_mat(grip_q)
    half = np.asarray(pad_half, dtype=float)
    gap = abs((pads[1] - pads[0]) @ grip[:, 0]) / 2 - half[0]
    n_x, n_y, n_z = COVER_GRID
    xs = np.linspace(-gap, gap, n_x) if gap > 0 else np.zeros(1)
    ys, zs = np.linspace(-half[1], half[1], n_y), np.linspace(-half[2], half[2], n_z)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1)
    points = np.mean(pads, axis=0) + grid.reshape(-1, 3) @ grip.T
    local = (points - obj_p) @ quat_to_mat(obj_q)
    hit = np.zeros(len(points), dtype=bool)
    for box in boxes:
        in_box = (local - box["pos"]) @ quat_to_mat(box["quat_xyzw"])
        hit |= np.all(np.abs(in_box) <= box["half_size"], axis=1)
    return float(hit.reshape(grid.shape[:3]).any(axis=0).mean())


def pad_gap(pads: np.ndarray) -> np.ndarray:
    """[T] distance between the pad centers; `pads` is [T, 2, 3]."""
    return np.linalg.norm(pads[:, 1] - pads[:, 0], axis=1)


def close_onsets(gap: np.ndarray, start: int, rate: float) -> list[int]:
    closing = np.zeros(len(gap), dtype=bool)
    closing[:-2] = (gap[:-2] - gap[2:]) > rate
    return [t for t in range(start + 1, len(gap)) if closing[t] and not closing[t - 1]]


def close_end(gap: np.ndarray, onset: int, params: MechanismParams = DEFAULT_PARAMS) -> int:
    end = onset + 1
    while (
        end + 2 < len(gap)
        and end < onset + params.close_end_cap
        and gap[end] - gap[end + 2] > params.close_end_rate_m
    ):
        end += 1
    return min(end, len(gap) - 1)


def open_descents(z, closes, start: int, params: MechanismParams = DEFAULT_PARAMS):
    """(descent start, bottom) of descents of the pad midpoint with no close near the bottom,
    followed by a rise or by staying down to the end of the episode."""
    out = []
    for t in range(start + 1, len(z) - 1):
        if not (z[t] < z[t - 1] and z[t] <= z[t + 1]):
            continue
        after = z[t:]
        risen = np.flatnonzero(after >= z[t] + params.rise_m)
        until_rise = after[: risen[0]] if len(risen) else after
        if until_rise.min() < z[t] - params.descent_tol_m:
            continue  # not the bottom: it goes lower before rising again
        s = t
        while s - 1 >= start and z[s - 1] > z[s] - params.descent_tol_m:
            s -= 1
        while s < t and z[s + 1] >= z[s]:  # the start is the last step before z drops
            s += 1
        deep = z[s] - z[t] >= params.descent_m
        rises = z[t : t + params.rise_window].max() - z[t] >= params.rise_m
        parks = z[t:].max() - z[t] < params.rise_m
        quiet = not any(s <= c <= t + params.descent_close_gap for c in closes)
        if deep and (rises or parks) and quiet:
            out.append((s, t))
    return out


def held_runs(held: np.ndarray) -> list[tuple[int, int]]:
    out, first = [], None
    for t, value in enumerate([*held, False]):
        if value and first is None:
            first = t
        elif not value and first is not None:
            out.append((first, t))
            first = None
    return out


def leave_and_open(dist, gap, run_start: int, run_end: int, params=DEFAULT_PARAMS):
    """First steps, around the end of a hold, at which the object has moved away from the pad
    midpoint and at which the fingers have opened; None if not within the window."""
    ref = max(run_start, run_end - params.slip_ref)
    end = min(run_end + params.slip_ref, len(dist))
    leave = next((t for t in range(ref, end) if dist[t] - dist[ref] > params.slip_leave_m), None)
    opened = next((t for t in range(ref, end) if gap[t] - gap[ref] > params.slip_open_m), None)
    return leave, opened


def fell(z, run_end: int, fall_m: float, window: int) -> bool:
    if run_end >= len(z):
        return False
    return bool(z[run_end - 1] - z[run_end : run_end + window].min() >= fall_m)


def first_shift(offset, run_start: int, run_end: int, over, params=DEFAULT_PARAMS):
    """First step of a hold, past the settle and away from the target, at which the object's
    offset to the pad midpoint moved more than `shift_m` from its settled value."""
    ref = run_start + params.shift_settle
    for t in range(ref + 1, run_end):
        if not over[t] and np.linalg.norm(offset[t] - offset[ref]) > params.shift_m:
            return t
    return None


def final_descent_start(z, end: int, tol: float = DEFAULT_PARAMS.descent_tol_m) -> int:
    """Walks back from `end` through steps where z is level or falling, to where it last rose."""
    t = end
    while t - 1 >= 0 and z[t - 1] >= z[t] - tol:
        t -= 1
    return t


def tilt_deg(settled_q, q) -> float:
    """Angle between the object's up axis when settled and the same axis now."""
    up = quat_to_mat(settled_q).T @ [0.0, 0.0, 1.0]
    now = quat_to_mat(q) @ up
    return float(np.degrees(np.arccos(np.clip(now[2], -1.0, 1.0))))


def progress(count: np.ndarray, levels: np.ndarray, deps) -> np.ndarray:
    """[T]: 3 per counted subgoal, plus the highest phase level among subgoals still open whose
    prerequisites count."""
    ready = np.ones(count.shape, dtype=bool)
    for g, prerequisites in enumerate(deps):
        for d in prerequisites:
            ready[:, g] &= count[:, d]
    open_levels = np.where(~count & ready, levels, 0)
    return 3 * count.sum(axis=1) + open_levels.max(axis=1)


@dataclass(frozen=True)
class Event:
    kind: str  # grasp_attempt, slip, held_shift, collision, placement, defective_placement
    step: int  # onset
    obj: int  # object index, -1 for none
    decisive: int | None  # step whose progress recovery is measured against; None: never
    cover: float | None = None
    closed: bool = False  # grasp attempts: the fingers closed (not an open descent)


@dataclass(frozen=True)
class Prediction:
    mechanism: str  # labels.csv vocabulary; "" when abstained
    onset_step: int | None
    abstained: bool
    reason: str


def object_type(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


class _World:
    """Per-episode arrays shared by the detectors."""

    def __init__(self, x, info):
        self.x, self.info = x, info
        self.start = info["num_wait_steps"]
        self.subgoals = info["subgoals"]
        self.names = info["object_names"]
        self.deps = [tuple(s["depends_on"]) for s in self.subgoals]
        self.pads = x["finger_pad_pos"]
        self.mid = self.pads.mean(axis=1)
        self.gap = pad_gap(self.pads)
        self.place = [
            (g, s["object_index"]) for g, s in enumerate(self.subgoals) if s["kind"] == "place"
        ]
        self.goal_objects = [i for _, i in self.place]
        width = milestones.finger_width(x["gripper_qpos"])
        obj = x["object_pos"]
        self.held = np.stack(
            [milestones.grasped(obj[:, o], x["eef_pos"], width) for o in range(obj.shape[1])],
            axis=1,
        )
        self.held[: self.start] = False
        goal_held = np.zeros(x["goal_holds"].shape, dtype=bool)
        for g, i in self.place:
            goal_held[:, g] = self.held[:, i]
        self.goal_held = goal_held
        placed = x["goal_holds"] & ~goal_held
        ms = milestones.DEFAULT_PARAMS
        self.count = milestones.counting(
            milestones.debounced(placed, ms.debounce, self.start), self.deps
        )
        self.placed = placed
        self.levels = milestones._phase_levels(x, info, ms, self.start, goal_held)
        self.progress = progress(self.count, self.levels, self.deps)
        self.width = width

    def subgoal_of(self, i: int) -> int:
        return next(g for g, o in self.place if o == i)

    def nearest_goal(self, t: int):
        d = {i: np.linalg.norm(self.mid[t] - self.x["object_pos"][t, i]) for i in self.goal_objects}
        i = min(d, key=d.get)
        return i, d[i]

    def cover(self, t: int, i: int, boxes) -> float:
        return pad_face_cover(
            self.pads[t],
            self.x["grip_quat"][t],
            self.info["finger_pad_half_size"],
            self.x["object_pos"][t, i],
            self.x["object_quat"][t, i],
            boxes[self.names[i]],
        )


def _grasp_attempts(w: _World, boxes, collisions, params: MechanismParams) -> list[Event]:
    """Failed closes and open descents at a goal object; a descent that coincides with a
    collision on another object was stopped by it, and is left to the collision."""
    out = []
    closes = close_onsets(w.gap, w.start, params.close_rate_m)
    z = w.mid[:, 2]

    def stopped_by_collision(t, i):
        window = range(t - params.descent_close_gap, t + params.grasp_follow + 1)
        return any(c.obj != i and c.step in window for c in collisions)

    def failed_attempt(t, i, dist):
        if dist >= params.near_m or w.held[t, w.goal_objects].any():
            return False
        if stopped_by_collision(t, i):
            return False
        return not w.held[t : t + params.grasp_follow, i].any()

    prev = w.start
    for c in closes:
        i, dist = w.nearest_goal(c)
        if failed_attempt(c, i, dist):
            lo = max(prev, c - params.descent_window)
            bottom = lo + int(np.argmin(z[lo : c + 1]))
            end = close_end(w.gap, c, params)
            cover = max(w.cover(t, i, boxes) for t in range(bottom, end + 1))
            out.append(Event("grasp_attempt", bottom, i, bottom, cover, closed=True))
        prev = c
    for _, bottom in open_descents(z, closes, w.start, params):
        i, dist = w.nearest_goal(bottom)
        if failed_attempt(bottom, i, dist):
            out.append(Event("grasp_attempt", bottom, i, bottom, w.cover(bottom, i, boxes)))
    return out


def _release_kind(w: _World, i: int, run, params: MechanismParams) -> str:
    """ "slip" (object left before the fingers opened, and fell), else "release"."""
    a, e = run
    dist = np.linalg.norm(w.mid - w.x["object_pos"][:, i], axis=1)
    leave, opened = leave_and_open(dist, w.gap, a, e, params)
    falls = fell(
        w.x["object_pos"][:, i, 2], e, flags.DEFAULT_FLAGS.fall_m, flags.DEFAULT_FLAGS.dwell
    )
    if leave is not None and (opened is None or leave < opened) and falls:
        return "slip"
    return "release"


def _slips(w: _World, params: MechanismParams) -> list[Event]:
    out = []
    for i in w.goal_objects:
        dist = np.linalg.norm(w.mid - w.x["object_pos"][:, i], axis=1)
        for a, e in held_runs(w.held[:, i]):
            if e < len(dist) and _release_kind(w, i, (a, e), params) == "slip":
                leave, _ = leave_and_open(dist, w.gap, a, e, params)
                out.append(Event("slip", leave, i, leave))
    return out


def _held_shifts(w: _World, params: MechanismParams) -> list[Event]:
    out = []
    for g, i in w.place:
        offset = w.x["object_pos"][:, i] - w.mid
        for a, e in held_runs(w.held[:, i]):
            t = first_shift(offset, a, e, w.x["over_target"][:, g], params)
            if t is None:
                continue
            ends_in_slip = e < len(offset) and _release_kind(w, i, (a, e), params) == "slip"
            if not (ends_in_slip and e - t <= params.slip_ref):
                out.append(Event("held_shift", t, i, t))
    return out


def _collisions(w: _World, params: MechanismParams) -> list[Event]:
    """Frozen `collision_disturbance`, per object; a push on a goal object while the hand is
    at it and empty belongs to the grasp attempt, not a collision."""
    fp, x = flags.DEFAULT_FLAGS, w.x
    hz = w.info["control_freq_hz"]
    in_goal = np.zeros(w.held.shape, dtype=bool)
    for g, i in w.place:
        in_goal[:, i] |= w.placed[:, g]
    out = []
    for o in range(w.held.shape[1]):
        contact = milestones.in_contact(x["object_pos"][:, o], x["eef_pos"], w.width)
        for t, _ in flags.collision_disturbance(
            x["object_pos"][:, o],
            w.held[:, o],
            in_goal[:, o],
            w.start,
            fp.displaced_m,
            fp.dwell,
            contact=contact,
            rest_step=fp.rest_speed / hz,
        ):
            at_it = np.linalg.norm(w.mid[t] - x["object_pos"][t, o]) < params.near_m
            if o in w.goal_objects and at_it and not w.held[t].any():
                continue
            out.append(Event("collision", t, o, t))
    return out


def _target_xy_distance(w: _World, g: int, t: int) -> float | None:
    target = w.subgoals[g]["target"]
    if target not in w.names:
        return None
    i = w.subgoals[g]["object_index"]
    obj = w.x["object_pos"]
    return float(np.linalg.norm(obj[t, i, :2] - obj[t, w.names.index(target), :2]))


def _placements(w: _World, params: MechanismParams) -> list[Event]:
    out = []
    n = len(w.mid)
    for g, i in w.place:
        z = w.x["object_pos"][:, i, 2]
        over = w.x["over_target"][:, g]
        for a, e in held_runs(w.held[:, i]):
            if w.count[a:, g].any():
                continue
            last = e - 1
            if e == n:
                attempted = over[a:e].any()
                if attempted:  # held to the cap: the attempt ends where it left the target
                    last = a + int(np.flatnonzero(over[a:e])[-1])
            else:
                d = _target_xy_distance(w, g, last)
                near = over[last] or (d is not None and d < params.place_near_m)
                attempted = near and _release_kind(w, i, (a, e), params) == "release"
            if attempted:
                # The descent that brought it to rest, not later nudges at that height; searched
                # from the carry's highest point, since the lift also passes that height.
                peak = a + int(np.argmax(z[a : last + 1]))
                at_rest = peak + int(np.argmax(z[peak : last + 1] <= z[last] + params.rest_band_m))
                out.append(Event("placement", final_descent_start(z, at_rest), i, last))
    return out


def placement_state(x, info, g: int, params: MechanismParams = DEFAULT_PARAMS):
    """Tilt and rest height of subgoal `g`'s object `defect_delay` steps after its predicate
    first holds with the object released; None if it never does."""
    w = _World(x, info)
    return _placement_state(w, g, params)


def _placement_state(w: _World, g: int, params: MechanismParams):
    i = w.subgoals[g]["object_index"]
    placed = w.placed[:, g].copy()
    placed[: w.start] = False
    if not placed.any():
        return None
    p = int(np.argmax(placed))
    t = min(p + params.defect_delay, len(placed) - 1)
    q = w.x["object_quat"]
    return {
        "step": p,
        "tilt": tilt_deg(q[w.start, i], q[t, i]),
        "height": float(w.x["object_pos"][t, i, 2]),
    }


def success_reference(episodes, params: MechanismParams = DEFAULT_PARAMS) -> dict:
    """(task_id, object name) -> tilt and height ranges over successful placements."""
    values: dict = {}
    for x, info in episodes:
        if not info["final_success"]:
            continue
        w = _World(x, info)
        for g, i in w.place:
            state = _placement_state(w, g, params)
            if state is not None:
                key = (info["task_id"], w.names[i])
                values.setdefault(key, []).append((state["tilt"], state["height"]))
    return {
        key: {
            "tilt": (min(v[0] for v in vals), max(v[0] for v in vals)),
            "height": (min(v[1] for v in vals), max(v[1] for v in vals)),
        }
        for key, vals in values.items()
    }


def _defective_placements(w: _World, reference: dict, params: MechanismParams) -> list[Event]:
    out = []
    for g, i in w.place:
        dependents = [h for h, deps in enumerate(w.deps) if g in deps]
        if not dependents or not w.count[:, g].any():
            continue
        if any(w.count[:, h].any() for h in dependents):
            continue
        ref = reference.get((w.info["task_id"], w.names[i]))
        state = _placement_state(w, g, params)
        if ref is None or state is None:
            continue
        lo_h, hi_h = ref["height"]
        if state["tilt"] > ref["tilt"][1] or not lo_h <= state["height"] <= hi_h:
            held_before = np.flatnonzero(w.held[: state["step"], i])
            release = int(held_before[-1]) if len(held_before) else state["step"]
            z = w.x["object_pos"][:, i, 2]
            out.append(Event("defective_placement", final_descent_start(z, release), i, None))
    return out


def _between(event: Event, w: _World, params: MechanismParams) -> bool:
    if object_type(w.names[event.obj]) in params.any_cover_objects:
        return event.cover > 0
    return event.cover >= params.cover_cutoff


def _recovered(event: Event, w: _World, params: MechanismParams, attempts) -> bool:
    """Progress resumed on the event's own subgoal; for events on other objects, on any.
    A miss is also recovered by a later close with the same object between the fingers."""
    if event.decisive is None:
        return False
    if event.kind == "grasp_attempt":
        later = [a for a in attempts if a.obj == event.obj and a.step > event.step and a.closed]
        if not _between(event, w, params) and any(_between(a, w, params) for a in later):
            return True
    prog = w.progress
    if event.obj in w.goal_objects:
        g = w.subgoal_of(event.obj)
        prog = 3 * w.count[:, g] + w.levels[:, g]
    before = prog[max(event.decisive - 1, 0)]
    return bool(len(prog) > event.decisive + 1 and prog[event.decisive + 1 :].max() > before)


def _phase_label(w: _World, t: int) -> str:
    for g, i in w.place:
        if w.held[t, i]:
            return "control: place" if w.x["over_target"][t, g] else "control: transport"
    return "control: grasp"


def _label(event: Event, w: _World, params: MechanismParams) -> Prediction:
    name = w.names[event.obj] if event.obj >= 0 else ""
    if event.kind == "grasp_attempt":
        mechanism = "control: grasp" if _between(event, w, params) else "perception"
        reason = f"grasp attempt on {name}: pad-face cover {event.cover:.2f}"
    elif event.kind == "slip":
        mechanism, reason = "control: grasp", f"{name} left the hand before the fingers opened"
    elif event.kind == "held_shift":
        mechanism, reason = "control: transport", f"{name} shifted in the hand while carried"
    elif event.kind == "collision":
        mechanism, reason = _phase_label(w, event.step), f"collision moved {name}"
    elif event.kind == "placement":
        mechanism, reason = "control: place", f"{name} placement did not count"
    else:
        mechanism = "control: place"
        reason = f"{name} placed outside the successes' range, blocking a dependent subgoal"
    return Prediction(mechanism, event.step, False, reason)


def classify(x, info, flag_intervals, boxes, reference, params=DEFAULT_PARAMS) -> Prediction:
    """Rule order: grounding flags; inaction; else the earliest unrecovered event, labeled by
    its kind."""
    grounding = list(flag_intervals["wrong_object"]) + list(flag_intervals["wrong_target"])
    if grounding:
        onset = min(a for a, _ in grounding)
        return Prediction("grounding", onset, False, "wrong_object or wrong_target fired")
    start, n = info["num_wait_steps"], len(x["finger_pad_pos"])
    # `stalled` is judged from the step after the wait, so a stall from the first policy
    # step starts at start + 1.
    if any(a <= start + 1 and b == n for a, b in flag_intervals.get("stalled", [])):
        return Prediction("inaction", start, False, "stalled from the first policy step to the cap")
    w = _World(x, info)
    collisions = _collisions(w, params)
    attempts = _grasp_attempts(w, boxes, collisions, params)
    events = (
        attempts
        + _slips(w, params)
        + _held_shifts(w, params)
        + collisions
        + _placements(w, params)
        + _defective_placements(w, reference, params)
    )
    for event in sorted(events, key=lambda e: e.step):
        if not _recovered(event, w, params, attempts):
            return _label(event, w, params)
    return Prediction("", None, True, "no unrecovered error found")
