"""Block 4 reference baseline on the baseline run: an empty close read from finger width alone.

At each closing gate anchor, the policy's close command nearest the anchor is the trigger; the
baseline fires at the first step within the horizon whose finger width is below the threshold.
Reports catches, false alarms and delay after the command over a threshold sweep, the narrowest
successful grasp per object, and the same-task AUROC of minimum width over the horizon.

Reads <anchors>; writes <out>/width_baseline.json and width_baseline.md.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import separability_gate as sg

from harness import detector, gate, metadata, schema

SWEEP_M = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.010, 0.015, 0.020, 0.030)
N_BOOT = 2000


def load(run_dir: Path, anchors_csv: Path, params: detector.WidthBaseline) -> list[dict]:
    with open(anchors_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded"] and r["anchor_closed"] == "1"]
    out, cached = [], None
    for r in rows:
        ep = r["episode_id"]
        if cached is None or cached[0] != ep:
            a, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
            commands = detector.close_commands(a["executed_action"][:, 6])
            cached = (ep, commands, detector.finger_width(a["gripper_qpos"]))
        _, commands, width = cached
        anchor = int(r["anchor_step"])
        command = detector.nearest_command(commands, anchor, params.match_tol)
        row = {
            "episode_id": ep,
            "task_id": int(r["task_id"]),
            "object": r["object"],
            "group": r["group"],
            "anchor_step": anchor,
            "command_step": command,
            "width": None if command is None else width[command : command + params.horizon + 1],
        }
        if command is not None:
            row["min_width_m"] = float(row["width"].min())
            row["delay"] = _delay(row, params)
        out.append(row)
    return out


def _delay(row: dict, params: detector.WidthBaseline) -> int | None:
    """Steps from the command to the baseline firing; the row's width starts at the command."""
    return detector.empty_close_step(row["width"], 0, params)


def sweep(rows: list[dict], params: detector.WidthBaseline) -> list[dict]:
    matched = [r for r in rows if r["command_step"] is not None]
    out = []
    for th in SWEEP_M:
        p = detector.WidthBaseline(th, params.horizon, params.match_tol)
        entry = {"threshold_m": th}
        for group in ("failure", "success"):
            delays = [_delay(r, p) for r in matched if r["group"] == group]
            fired = [d for d in delays if d is not None]
            entry[f"{group}_n"] = len(delays)
            entry[f"{group}_fired"] = len(fired)
            entry[f"{group}_delay_median"] = float(np.median(fired)) if fired else None
        out.append(entry)
    return out


def narrowest_success(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        if r["group"] == "success" and r["command_step"] is not None:
            by[r["object"]].append(r["min_width_m"])
    return {
        o: {"n": len(v), "min_m": min(v), "median_m": float(np.median(v))}
        for o, v in sorted(by.items())
    }


def auroc(rows: list[dict]) -> dict:
    units = [
        gate.Unit(r["episode_id"], r["task_id"], r["object"], r["group"], -r["min_width_m"])
        for r in rows
        if r["command_step"] is not None
    ]
    lo, hi = gate.bootstrap_ci(units, gate.same_task_auroc, sg.EXPLORATORY_LEVEL, N_BOOT, sg.SEED)
    return {"auroc": gate.same_task_auroc(units), "ci": [lo, hi]}


def render(results: dict, params: detector.WidthBaseline, sha: str) -> str:
    r = results["unmatched"]
    a = results["auroc_min_width"]
    lines = [
        "# Width baseline (Block 4 reference)",
        "",
        f"Trigger: the close command nearest each closing gate anchor (within "
        f"{params.match_tol} steps). Fires at the first step in the next {params.horizon} with "
        f"finger width below the threshold; chosen threshold {params.threshold_m * 1000:.0f} mm. "
        f"No close command near the anchor: {r['failure']} failures, {r['success']} successes. "
        f"Harness `{sha[:7]}`.",
        "",
        "| Threshold | failures fired | successes fired | median delay (steps): failures "
        "| successes |",
        "|---|---|---|---|---|",
    ]
    for e in results["sweep"]:
        mark = " (chosen)" if e["threshold_m"] == params.threshold_m else ""
        delay = "-" if e["failure_delay_median"] is None else f"{e['failure_delay_median']:.0f}"
        s_delay = "-" if e["success_delay_median"] is None else f"{e['success_delay_median']:.0f}"
        lines.append(
            f"| < {e['threshold_m'] * 1000:g} mm{mark} | {e['failure_fired']}/{e['failure_n']} "
            f"| {e['success_fired']}/{e['success_n']} | {delay} | {s_delay} |"
        )
    lines += [
        "",
        f"Same-task AUROC of minimum width over the horizon (narrower above): {a['auroc']:.2f}, "
        f"{sg.EXPLORATORY_LEVEL:.2%} CI {a['ci'][0]:.2f} to {a['ci'][1]:.2f}, {N_BOOT} draws, seed "
        f"{sg.SEED}.",
        "",
        "**Narrowest successful grasp per object**",
        "",
        "| Object | n | min (mm) | median (mm) |",
        "|---|---|---|---|",
    ]
    for o, v in results["narrowest_success"].items():
        lines.append(f"| {o} | {v['n']} | {v['min_m'] * 1000:.1f} | {v['median_m'] * 1000:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    params = detector.DEFAULT_WIDTH
    rows = load(args.run_dir, args.anchors, params)
    results = {
        "params": vars(params),
        "unmatched": {
            g: sum(r["command_step"] is None for r in rows if r["group"] == g)
            for g in ("failure", "success")
        },
        "sweep": sweep(rows, params),
        "auroc_min_width": auroc(rows),
        "narrowest_success": narrowest_success(rows),
        "anchors": [{k: v for k, v in r.items() if k != "width"} for r in rows],
    }
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "width_baseline.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "width_baseline.md").write_text(render(results, params, sha))
    print(render(results, params, sha))


if __name__ == "__main__":
    main()
