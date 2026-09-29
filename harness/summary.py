"""Suite-level results from a run manifest: success rates with n and spread, and throughput."""

import math
from collections import Counter, defaultdict


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(rows) -> dict:
    n = len(rows)
    k = sum(r["success"] for r in rows)
    return {"n": n, "successes": k, "success_rate": k / n, "ci95": wilson_interval(k, n)}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no episodes to summarize")
    wall = [r["num_steps"] * r["seconds_per_step"] for r in rows]
    steps = [r["num_steps"] for r in rows]
    by_task, by_seed = defaultdict(list), defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
        by_seed[r["seed"]].append(r)

    result = {
        **_rate(rows),
        "per_task": {task: _rate(group) for task, group in sorted(by_task.items())},
        "seconds_per_episode": sum(wall) / len(rows),
        "seconds_per_policy_step": sum(wall) / max(sum(steps), 1),
        "mean_policy_steps": sum(steps) / len(rows),
        "terminated_by": dict(Counter(r["terminated_by"] for r in rows)),
    }
    if len(by_seed) > 1:
        result["per_seed"] = {seed: _rate(group) for seed, group in sorted(by_seed.items())}
        rates = [v["success_rate"] for v in result["per_seed"].values()]
        result["seed_rate_range"] = (min(rates), max(rates))
    return result


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(result: dict) -> str:
    low, high = result["ci95"]
    lines = [
        "# Baseline summary",
        "",
        f"- success rate: {result['successes']}/{result['n']} = {_pct(result['success_rate'])} "
        f"(Wilson 95% interval {_pct(low)} to {_pct(high)})",
        f"- seconds per episode: {result['seconds_per_episode']:.1f}",
        f"- seconds per policy step: {result['seconds_per_policy_step']:.3f}",
        f"- mean policy steps per episode: {result['mean_policy_steps']:.0f}",
        f"- terminated by: {result['terminated_by']}",
        "",
        "## Per task",
        "",
        "| task | n | successes | rate | 95% interval |",
        "|---|---|---|---|---|",
    ]
    for task, r in result["per_task"].items():
        lo, hi = r["ci95"]
        lines.append(
            f"| {task} | {r['n']} | {r['successes']} | {_pct(r['success_rate'])} "
            f"| {_pct(lo)} to {_pct(hi)} |"
        )
    if "per_seed" in result:
        lines += ["", "## Per seed", "", "| seed | n | rate |", "|---|---|---|"]
        lines += [
            f"| {seed} | {r['n']} | {_pct(r['success_rate'])} |"
            for seed, r in result["per_seed"].items()
        ]
    return "\n".join(lines) + "\n"


def write_outputs(run_dir) -> dict:
    """Writes summary.md and summary.json next to the run's manifest."""
    import json
    from pathlib import Path

    from harness import manifest

    run_dir = Path(run_dir)
    result = summarize(manifest.read_rows(run_dir / "manifest.csv"))
    (run_dir / "summary.md").write_text(render_markdown(result))
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result
