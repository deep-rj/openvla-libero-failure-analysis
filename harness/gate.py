"""Separability gate: pre-grasp windows of model-output metrics, aligned on grasp descents.

Failures and successes are aligned on the same event, the bottom of a grasp descent at a goal
object. Unlike the classifier's onset rule, the bottom excludes a lower point left by an earlier
attempt. Pure functions; pre-registration in `docs/separability_gate.md`.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from harness import mechanism
from harness.pilot_report import merged_bin_logits

GRIPPER_DIM = 6


@dataclass(frozen=True)
class GateParams:
    window: int = 50
    min_window: int = 10
    near_tie_ulps: float = 4.0
    # Pilot onsets sit anywhere from the start to the bottom of a descent (up to the
    # classifier's 60-step descent window), and a slip's onset follows its grasp.
    match_tol: int = 60


DEFAULT_PARAMS = GateParams()


@dataclass(frozen=True)
class Descent:
    bottom: int
    obj: int
    close: int | None  # None for a descent that never closed


def descent_bottom(z, close: int, lo: int, rise: float) -> int:
    """Lowest point of the last descent before `close`: back from the close until z has risen
    `rise` above the lowest point so far, no earlier than `lo`. A lower point left by an earlier
    attempt, before the arm rose again, is not this descent's bottom."""
    best = t = close
    while t - 1 >= lo and z[t - 1] < z[best] + rise:
        t -= 1
        if z[t] < z[best]:
            best = t
    return best


def grasp_descents(x, info, params=mechanism.DEFAULT_PARAMS) -> list[Descent]:
    """Closes and open descents at the nearest goal object within `near_m` while no goal
    object is held, failed or not."""
    w = mechanism._World(x, info)
    closes = mechanism.close_onsets(w.gap, w.start, params.close_rate_m)
    z = w.mid[:, 2]

    def at_goal(t):
        i, dist = w.nearest_goal(t)
        return i if dist < params.near_m and not w.held[t, w.goal_objects].any() else None

    out, prev = [], w.start
    for c in closes:
        i = at_goal(c)
        if i is not None:
            lo = max(prev, c - params.descent_window)
            out.append(Descent(descent_bottom(z, c, lo, params.rise_m), i, c))
        prev = c
    for _, bottom in mechanism.open_descents(z, closes, w.start, params):
        i = at_goal(bottom)
        if i is not None:
            out.append(Descent(bottom, i, None))
    return sorted(out, key=lambda d: d.bottom)


def _hold_start(w, descents: list[Descent], i: int, before: int) -> Descent | None:
    """The closing descent that began object `i`'s last hold starting before `before`."""
    starts = [a for a, _ in mechanism.held_runs(w.held[:, i]) if a < before]
    if not starts:
        return None
    closing = [d for d in descents if d.obj == i and d.close is not None and d.close <= starts[-1]]
    return closing[-1] if closing else None


def success_anchor(x, info, descents: list[Descent], g: int) -> Descent | None:
    """The closing descent that began the last hold of place subgoal `g`'s object."""
    w = mechanism._World(x, info)
    return _hold_start(w, descents, w.subgoals[g]["object_index"], len(w.mid))


def failure_anchor(x, info, descents: list[Descent], onset: int, tol: int) -> Descent | None:
    """A slip (a goal object held at the step before the onset) is anchored on the grasp that
    began the hold; any other onset on the descent nearest it within `tol`."""
    w = mechanism._World(x, info)
    held = [i for i in w.goal_objects if onset > 0 and w.held[onset - 1, i]]
    if held:
        return _hold_start(w, descents, held[0], onset)
    near = [d for d in descents if abs(d.bottom - onset) <= tol]
    return min(near, key=lambda d: abs(d.bottom - onset)) if near else None


def pre_window(is_wait, anchor: int, length: int, min_length: int) -> np.ndarray | None:
    """Non-wait steps in the `length` steps before `anchor`; None if fewer than `min_length`."""
    steps = np.arange(max(anchor - length, 0), anchor)
    steps = steps[~np.asarray(is_wait)[steps]]
    return steps if len(steps) >= min_length else None


def near_tie_rate(margins_ulps, window: np.ndarray, max_ulps: float) -> float:
    return float((np.asarray(margins_ulps)[window] <= max_ulps).mean())


@dataclass(frozen=True)
class Unit:
    episode_id: str
    task_id: int
    obj: str
    group: str  # "failure" or "success"
    value: float


def auroc(failures, successes) -> float:
    """P(a failure's value exceeds a success's), ties counted half."""
    f, s = np.asarray(failures, dtype=float)[:, None], np.asarray(successes, dtype=float)[None]
    return float(((f > s).sum() + 0.5 * (f == s).sum()) / (f.size * s.size))


def _wins_and_pairs(units) -> tuple[float, int]:
    f = [u.value for u in units if u.group == "failure"]
    s = [u.value for u in units if u.group == "success"]
    if not f or not s:
        return 0.0, 0
    return auroc(f, s) * len(f) * len(s), len(f) * len(s)


def pooled_auroc(units) -> float:
    wins, pairs = _wins_and_pairs(units)
    return wins / pairs


def same_task_auroc(units) -> float:
    """AUROC over failure-success pairs within one (task, object) stratum only."""
    strata = defaultdict(list)
    for u in units:
        strata[(u.task_id, u.obj)].append(u)
    totals = [_wins_and_pairs(group) for group in strata.values()]
    return sum(w for w, _ in totals) / sum(p for _, p in totals)


def resample_episodes(units, rng) -> list[Unit]:
    """Episodes drawn with replacement within each (task, outcome); an episode's units travel
    together, since a success's grasps of different objects share a trajectory."""
    by_episode = defaultdict(list)
    for u in units:
        by_episode[u.episode_id].append(u)
    strata = defaultdict(list)
    for ep, us in by_episode.items():
        strata[(us[0].task_id, us[0].group)].append(ep)
    out = []
    for eps in strata.values():
        for k in rng.integers(0, len(eps), len(eps)):
            out.extend(by_episode[eps[k]])
    return out


def bootstrap_ci(units, statistic, level: float, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = [statistic(resample_episodes(units, rng)) for _ in range(n_boot)]
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)


def window_speed(pos, window: np.ndarray, hz: float) -> float:
    """Mean displacement per step from each window step to the next, in units per second."""
    pos = np.asarray(pos, dtype=float)
    return float(np.linalg.norm(pos[window + 1] - pos[window], axis=-1).mean() * hz)


def window_rotation_speed(quats_xyzw, window: np.ndarray, hz: float) -> float:
    """Mean rotation angle per step from each window step to the next, in rad/s."""
    q = np.asarray(quats_xyzw, dtype=float)
    dot = np.abs(np.sum(q[window + 1] * q[window], axis=-1))
    return float((2 * np.arccos(np.clip(dot, 0.0, 1.0))).mean() * hz)


def residualize(y, x) -> np.ndarray:
    """`y` minus its least-squares line on `x`."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return y - np.polyval(np.polyfit(x, y, 1), x)


def bin_probs(slice_logits) -> np.ndarray:
    """[..., 255] probabilities over merged bins, as in `pilot_report.derive_step_metrics`."""
    bins = merged_bin_logits(slice_logits)
    weights = np.exp(bins - bins.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def stop_rank(probs, zero_bin: int) -> np.ndarray:
    """Per step: 0 if the no-motion bin is the top choice, 1 if second, 2 otherwise."""
    order = np.argsort(np.asarray(probs), axis=-1)
    return np.where(order[..., -1] == zero_bin, 0, np.where(order[..., -2] == zero_bin, 1, 2))


def subset_mean(values, window: np.ndarray, keep) -> float | None:
    """Mean of `values` over the window steps where `keep` holds; None if there are none."""
    steps = window[np.asarray(keep)[window]]
    return float(np.asarray(values)[steps].mean()) if len(steps) else None


def far_mass(probs, k: int) -> np.ndarray:
    """Per step: probability on bins at least `k` bins from the top bin."""
    probs = np.asarray(probs)
    top = probs.argmax(axis=-1)[..., None]
    far = np.abs(np.arange(probs.shape[-1]) - top) >= k
    return np.where(far, probs, 0.0).sum(axis=-1)
