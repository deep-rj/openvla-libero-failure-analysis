"""Stage A: determinism (T1), non-interference (T2), consistency (T3, T5) and the cost report."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harness import checks, runner, schema
from harness.cli import _indices
from harness.rollout import NUM_STEPS_WAIT, EpisodeSpec

# The spec budgets about 2 MB per episode and asks to flag sizes far from it.
_SIZE_BUDGET_BYTES = (1.0e6, 4.0e6)


def compare_actions(a: np.ndarray, b: np.ndarray) -> dict:
    """Bitwise comparison of two executed-action sequences."""
    n = min(len(a), len(b))
    differing = np.flatnonzero((a[:n] != b[:n]).any(axis=-1))
    if len(differing):
        first = int(differing[0])
    elif len(a) != len(b):
        first = n
    else:
        first = None
    max_diff = float(np.abs(a[:n] - b[:n]).max()) if n else 0.0
    return {"identical": first is None, "first_divergent_step": first, "max_abs_diff": max_diff}


def top2_margin(slice_logits: np.ndarray) -> np.ndarray:
    """Per-dimension gap between the two largest logits of a [7, 256] slice."""
    ordered = np.sort(slice_logits.astype(np.float32), axis=-1)
    return ordered[:, -1] - ordered[:, -2]


def cost_report(rows, episode_bytes, baseline_episodes: int, dollars_per_hour: float) -> dict:
    """Projects baseline cost from the measured Stage A episodes."""
    steps = np.array([r["num_steps"] for r in rows], dtype=float)
    per_step = np.array([r["seconds_per_step"] for r in rows], dtype=float)
    episode_seconds = steps * per_step
    hours = baseline_episodes * episode_seconds.mean() / 3600
    mean_bytes = float(np.mean(episode_bytes))
    low, high = _SIZE_BUDGET_BYTES
    return {
        "episodes_measured": len(rows),
        "mean_policy_steps": float(steps.mean()),
        "seconds_per_step": float(episode_seconds.sum() / steps.sum()),
        "mean_episode_seconds": float(episode_seconds.mean()),
        "baseline_episodes": baseline_episodes,
        "dollars_per_hour": dollars_per_hour,
        "projected_gpu_hours": float(hours),
        "projected_dollars": float(hours * dollars_per_hour),
        "mean_episode_bytes": mean_bytes,
        "size_far_from_budget": not (low <= mean_bytes <= high),
    }


def render_markdown(sections: dict, checks: dict, cost: dict) -> str:
    lines = ["# Stage A report", ""]
    for name, values in sections.items():
        lines += [f"## {name.upper()}", *(f"- {k}: {v}" for k, v in values.items()), ""]
    for name, problems in checks.items():
        lines.append(f"## {name.upper()}")
        lines += [f"- {p}" for p in problems] if problems else ["- no violations"]
        lines.append("")
    lines += [
        "## Cost",
        f"- measured seconds per policy step: {cost['seconds_per_step']:.3f}",
        f"- mean policy steps per episode: {cost['mean_policy_steps']:.0f}",
        f"- baseline episodes: {cost['baseline_episodes']}",
        f"- projected GPU-hours: {cost['projected_gpu_hours']:.2f}",
        f"- projected dollars at ${cost['dollars_per_hour']:.2f}/h: "
        f"{cost['projected_dollars']:.2f}",
        f"- mean bytes per episode: {cost['mean_episode_bytes']:.0f}"
        + (" (FAR FROM THE ~2 MB BUDGET)" if cost["size_far_from_budget"] else ""),
        "",
    ]
    return "\n".join(lines)


@dataclass
class StageAConfig:
    out_dir: Path
    tasks: list[int] = field(default_factory=lambda: [0, 7, 3, 9])
    episodes_per_task: int = 2
    determinism_task: int = 0
    max_steps: int | None = None
    truncation_steps: int = 60
    policy: str = "stub"
    parity: bool = False
    parity_steps: int = 30
    baseline_episodes: int = 250
    dollars_per_hour: float = 0.75
    camera_size: int = 256
    allow_dirty: bool = False
    seed: int = 0


def _run(cfg, sub, mode, tasks, init_states, max_steps, factory, video=False):
    run_cfg = runner.RunConfig(
        out_dir=Path(cfg.out_dir) / sub,
        tasks=tasks,
        init_states=init_states,
        mode=mode,
        max_steps=max_steps,
        video=video,
        policy=cfg.policy,
        camera_size=cfg.camera_size,
        allow_dirty=cfg.allow_dirty,
        seed=cfg.seed,
    )
    return runner.run(run_cfg, policy_factory=factory), run_cfg.out_dir


def _write_report(cfg, report, checks_by_name):
    sections = {"t1": report["t1"]}
    for name in ("t2", "truncation"):
        if name in report:
            sections[name] = report[name]
    text = render_markdown(sections, checks_by_name, report.get("cost", _NO_COST))
    out = Path(cfg.out_dir)
    (out / "stage_a_report.md").write_text(text)
    (out / "stage_a_report.json").write_text(json.dumps(report, indent=2, default=float))


_NO_COST = {
    "seconds_per_step": float("nan"),
    "mean_policy_steps": float("nan"),
    "baseline_episodes": 0,
    "projected_gpu_hours": float("nan"),
    "dollars_per_hour": 0.0,
    "projected_dollars": float("nan"),
    "mean_episode_bytes": float("nan"),
    "size_far_from_budget": False,
}


def run_stage_a(cfg: StageAConfig, policy_factory=None) -> dict:
    """Runs T1, T2, T3, T5 and the cost measurement; stops after T1 if it diverges."""
    if cfg.determinism_task not in cfg.tasks:
        raise ValueError("determinism_task must be one of the Stage A tasks")
    det = EpisodeSpec("libero_10", cfg.determinism_task, 0, cfg.seed).episode_id
    task = [cfg.determinism_task]

    _, dir_a = _run(cfg, "t1_a", "reference", task, [0], cfg.max_steps, policy_factory)
    _, dir_b = _run(cfg, "t1_b", "reference", task, [0], cfg.max_steps, policy_factory)
    actions_a = np.load(dir_a / "actions" / f"{det}.npy")
    t1 = compare_actions(actions_a, np.load(dir_b / "actions" / f"{det}.npy"))
    report = {"t1": t1, "stopped_after_t1": not t1["identical"]}

    if not t1["identical"]:
        rows, dir_l = _run(cfg, "t1_logged", "logged", task, [0], cfg.max_steps, policy_factory)
        arrays, _ = schema.read_episode(dir_l / rows[0]["episode_file"])
        step = t1["first_divergent_step"]
        if step < len(arrays["step_idx"]):
            report["t1_margin"] = top2_margin(arrays["slice_logits"][step]).tolist()
        _write_report(cfg, report, {})
        return report

    init_states = list(range(cfg.episodes_per_task))
    rows, dir_main = _run(
        cfg, "logged", "logged", cfg.tasks, init_states, cfg.max_steps, policy_factory, video=True
    )
    t3, t5 = [], []
    for row in rows:
        arrays, meta = schema.read_episode(dir_main / row["episode_file"])
        stats = {k: np.array(v) for k, v in meta["action_stats"].items()}
        sent = list(np.load(dir_main / "actions" / f"{row['episode_id']}.npy"))
        prefix = f"{row['episode_id']}: "
        t3 += [prefix + p for p in checks.check_self_consistency(arrays, stats, sent)]
        t5 += [prefix + p for p in checks.check_wait_alignment(arrays, NUM_STEPS_WAIT)]
        if row["episode_id"] == det:
            report["t2"] = compare_actions(actions_a, arrays["executed_action"])

    rows_t, dir_t = _run(
        cfg, "truncation", "logged", task, [0], cfg.truncation_steps, policy_factory
    )
    arrays, _ = schema.read_episode(dir_t / rows_t[0]["episode_file"])
    t5 += [f"truncation: {p}" for p in checks.check_wait_alignment(arrays, NUM_STEPS_WAIT)]
    report["truncation"] = {
        k: rows_t[0][k] for k in ("terminated_by", "success", "num_steps")
    }

    by_check = {"t3": t3, "t5": t5}
    if cfg.parity:
        _, dir_p = _run(cfg, "parity", "parity", task, [0], cfg.parity_steps, policy_factory)
        mismatched = json.loads((dir_p / "parity" / f"{det}.json").read_text())["mismatched_steps"]
        report["t3_parity"] = mismatched
        by_check["t3_parity"] = [f"policy step {k} differs from vendored" for k in mismatched]

    sizes = [
        sum((dir_main / row[k]).stat().st_size for k in ("episode_file", "video_file") if row[k])
        for row in rows
    ]
    report["cost"] = cost_report(rows, sizes, cfg.baseline_episodes, cfg.dollars_per_hour)
    report.update(t3=t3, t5=t5, stopped_after_t1=False)
    _write_report(cfg, report, by_check)
    return report


def parse_args(argv=None) -> StageAConfig:
    parser = argparse.ArgumentParser(description="Run Stage A: T1, T2, T3, T5 and the cost report.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=_indices, default=[0, 7, 3, 9])
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--determinism-task", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="policy-step budget (default: suite budget)")
    parser.add_argument("--truncation-steps", type=int, default=60)
    parser.add_argument("--policy", choices=["stub", "openvla"], default="stub")
    parser.add_argument("--parity", action="store_true", help="also compare against vendored")
    parser.add_argument("--parity-steps", type=int, default=30)
    parser.add_argument("--baseline-episodes", type=int, default=250)
    parser.add_argument("--dollars-per-hour", type=float, default=0.75)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    return StageAConfig(
        out_dir=args.out,
        tasks=args.tasks,
        episodes_per_task=args.episodes_per_task,
        determinism_task=args.determinism_task,
        max_steps=args.max_steps,
        truncation_steps=args.truncation_steps,
        policy=args.policy,
        parity=args.parity,
        parity_steps=args.parity_steps,
        baseline_episodes=args.baseline_episodes,
        dollars_per_hour=args.dollars_per_hour,
        allow_dirty=args.allow_dirty,
    )


def main(argv=None) -> None:
    cfg = parse_args(argv)
    factory = None
    if cfg.policy == "openvla":
        # One load serves every run; reloading the checkpoint per run would dominate the cost.
        loaded = runner.default_policy_factory(
            runner.RunConfig(out_dir=cfg.out_dir, tasks=[], init_states=[], policy="openvla")
        )
        factory = lambda _cfg: loaded  # noqa: E731
    report = run_stage_a(cfg, factory)
    problems = report.get("t3", []) + report.get("t5", []) + report.get("t3_parity", [])
    passed = report["t1"]["identical"] and report.get("t2", {}).get("identical") and not problems
    print(f"Stage A {'PASS' if passed else 'FAIL'}; report in {cfg.out_dir / 'stage_a_report.md'}")
