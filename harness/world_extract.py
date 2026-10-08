"""Extraction stage: replays logged `sim_state` and records named world-state fields.

Reads only the trace's world columns. Everything downstream (`harness.milestones`) is a pure
function of the extracted file; this is the only stage that needs a simulator.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from harness import schema

EXTRACTION_VERSION = 2
PLACE_PREDICATES = ("in", "on")
FIXTURE_PREDICATES = ("close", "turnon")
# LIBERO's `On` between two objects requires their origins within this xy distance.
ON_OBJECT_XY_RADIUS = 0.03


@dataclass(frozen=True)
class Subgoal:
    predicate: tuple[str, ...]
    kind: str  # "place" or "fixture"
    object_index: int  # row in object_names for place subgoals, -1 otherwise
    target: str
    candidates: tuple[str, ...]  # other receptacles the object could be put in or on
    joint_qpos_addr: int  # -1 for place subgoals
    threshold: float  # joint value at which the predicate turns true; NaN for place
    depends_on: tuple[int, ...]


def _fixture_threshold(fixture, predicate: str) -> float:
    """Joint value at which LIBERO's `is_close`/`turn_on` flips, for the LIBERO-10 fixtures."""
    articulation = fixture.object_properties["articulation"]
    kind = type(fixture).__name__
    if predicate == "turnon" and kind == "FlatStove":
        return float(min(articulation["default_turnon_ranges"]))
    if predicate == "close" and kind in ("WhiteCabinet", "Microwave"):
        return float(min(articulation["default_close_ranges"]))
    raise NotImplementedError(f"no threshold rule for {predicate} on {kind}")


def _parent(inner, name: str) -> str | None:
    site = inner.object_sites_dict.get(name)
    return site.parent_name if site is not None else name


def _receptacles(inner, goal_state) -> list[str]:
    """Regions attached to an object or fixture, plus every goal target; not table regions."""
    names = [n for n, site in inner.object_sites_dict.items() if site.parent_name is not None]
    names += [p[-1] for p in goal_state if p[0] in PLACE_PREDICATES]
    return list(dict.fromkeys(names))


def _dependencies(inner, goal_state) -> list[tuple[int, ...]]:
    """Closing a container depends on each placement into it; closing first must be undone."""
    deps = []
    for pred in goal_state:
        if pred[0] != "close":
            deps.append(())
            continue
        container = pred[1]
        deps.append(
            tuple(
                j
                for j, other in enumerate(goal_state)
                if other[0] == "in" and container in (other[2], _parent(inner, other[2]))
            )
        )
    return deps


def build_subgoals(inner, object_names) -> list[Subgoal]:
    goal_state = [tuple(p) for p in inner.parsed_problem["goal_state"]]
    receptacles = _receptacles(inner, goal_state)
    subgoals = []
    for pred, deps in zip(goal_state, _dependencies(inner, goal_state), strict=True):
        if pred[0] in PLACE_PREDICATES:
            obj, target = pred[1], pred[2]
            subgoals.append(
                Subgoal(
                    predicate=pred,
                    kind="place",
                    object_index=list(object_names).index(obj),
                    target=target,
                    candidates=tuple(r for r in receptacles if r not in (target, obj)),
                    joint_qpos_addr=-1,
                    threshold=float("nan"),
                    depends_on=deps,
                )
            )
        elif pred[0] in FIXTURE_PREDICATES:
            name = pred[1]
            fixture = inner.get_object(_parent(inner, name))
            joints = (
                inner.object_sites_dict[name].joints
                if name in inner.object_sites_dict
                else fixture.joints
            )
            if len(joints) != 1:
                raise NotImplementedError(f"{name}: expected one joint, found {len(joints)}")
            subgoals.append(
                Subgoal(
                    predicate=pred,
                    kind="fixture",
                    object_index=-1,
                    target=name,
                    candidates=(),
                    joint_qpos_addr=int(inner.sim.model.get_joint_qpos_addr(joints[0])),
                    threshold=_fixture_threshold(fixture, pred[0]),
                    depends_on=deps,
                )
            )
        else:
            raise NotImplementedError(f"unsupported goal predicate {pred}")
    return subgoals


def _xy_over_target(inner, target: str, obj_pos: np.ndarray) -> bool:
    """The xy part of LIBERO's In/On test for `target`: the object is over it, at any height."""
    sim = inner.sim
    site = inner.object_sites_dict.get(target)
    if site is None:
        target_pos = sim.data.body_xpos[inner.obj_body_id[target]]
        return bool(np.linalg.norm(obj_pos[:2] - target_pos[:2]) < ON_OBJECT_XY_RADIUS)
    site_pos = sim.data.get_site_xpos(target)
    site_mat = sim.data.get_site_xmat(target)
    half = np.abs(site_mat @ site.size)
    return bool(np.all(np.abs(obj_pos[:2] - site_pos[:2]) < half[:2]))


FIXTURE_POSE_TOL = 1e-6


def place_fixtures(env, logged_poses: dict) -> None:
    """Moves jointless fixtures to the episode's logged poses; reset re-samples them."""
    from harness import world

    inner = env.env
    model = inner.sim.model
    for name, pose in logged_poses.items():
        body = inner.obj_body_id[name]
        if model.body_parentid[body] != 0 or model.body_jntnum[body] != 0:
            raise NotImplementedError(f"{name}: expected a jointless child of the world body")
        model.body_pos[body] = pose[:3]
        model.body_quat[body] = [pose[6], pose[3], pose[4], pose[5]]  # xyzw -> wxyz
    inner.sim.forward()
    placed = world.scene_layout(env).static_fixture_poses
    for name, pose in logged_poses.items():
        if not np.allclose(placed[name], pose, atol=FIXTURE_POSE_TOL):
            raise RuntimeError(f"{name}: fixture pose does not match the episode after placing")


def _eval(inner, pred) -> bool:
    from libero.libero.envs.predicates import eval_predicate_fn

    states = inner.object_states_dict
    return bool(eval_predicate_fn(pred[0], *(states[name] for name in pred[1:])))


def extract_episode(episode_path) -> tuple[dict[str, np.ndarray], dict]:
    """Replays every logged step and evaluates goal and receptacle predicates."""
    from libero.libero import benchmark

    from harness import runner

    arrays, meta = schema.read_world_columns(episode_path)
    suite = benchmark.get_benchmark_dict()[meta["suite"]]()
    env, _ = runner.build_env(suite, meta["task_id"], 64, meta["seed"])
    try:
        env.reset()
        place_fixtures(env, meta["static_fixture_poses"])
        inner = env.env
        object_names = meta["object_names"]
        subgoals = build_subgoals(inner, object_names)
        n_steps, n_goals = len(arrays["step_idx"]), len(subgoals)
        n_cand = max([len(s.candidates) for s in subgoals] + [1])

        goal_holds = np.zeros((n_steps, n_goals), dtype=bool)
        over_target = np.zeros((n_steps, n_goals), dtype=bool)
        candidate_holds = np.zeros((n_steps, n_goals, n_cand), dtype=bool)
        fixture_q = np.full((n_steps, n_goals), np.nan)
        object_pos = np.zeros((n_steps, len(object_names), 3))

        for t, state in enumerate(arrays["sim_state"]):
            inner.sim.set_state_from_flattened(state)
            # Contacts and site/body poses are derived quantities; they are stale until forward().
            inner.sim.forward()
            for i, name in enumerate(object_names):
                object_pos[t, i] = inner.sim.data.body_xpos[inner.obj_body_id[name]]
            for g, sub in enumerate(subgoals):
                goal_holds[t, g] = _eval(inner, sub.predicate)
                if sub.kind == "place":
                    pos = object_pos[t, sub.object_index]
                    over_target[t, g] = _xy_over_target(inner, sub.target, pos)
                    for c, cand in enumerate(sub.candidates):
                        candidate_holds[t, g, c] = _eval(
                            inner, (sub.predicate[0], sub.predicate[1], cand)
                        )
                else:
                    fixture_q[t, g] = inner.sim.data.qpos[sub.joint_qpos_addr]
    finally:
        env.close()

    extracted = {
        "step_idx": arrays["step_idx"],
        "is_wait_step": arrays["is_wait_step"],
        "object_pos": object_pos,
        "eef_pos": arrays["eef_pos"],
        "gripper_qpos": arrays["gripper_qpos"],
        "goal_holds": goal_holds,
        "over_target": over_target,
        "candidate_holds": candidate_holds,
        "fixture_q": fixture_q,
    }
    info = {
        "extraction_version": EXTRACTION_VERSION,
        "episode_id": meta["episode_id"],
        "task_id": meta["task_id"],
        "final_success": meta["final_success"],
        "num_wait_steps": meta["num_wait_steps"],
        "control_freq_hz": meta["control_freq_hz"],
        "object_names": list(object_names),
        "subgoals": [asdict(s) for s in subgoals],
    }
    return extracted, info


def write_extracted(path, extracted: dict, info: dict) -> None:
    """Writes atomically: arrays to `<path>.npz`, the description to `<path>.json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **extracted)
    os.replace(tmp, path.with_suffix(".npz"))
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))


def read_extracted(path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text())
    if info["extraction_version"] != EXTRACTION_VERSION:
        raise ValueError(f"{path}: extraction_version {info['extraction_version']} is stale")
    with np.load(path.with_suffix(".npz")) as data:
        return dict(data), info
