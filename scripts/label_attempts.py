"""Labels every grasp attempt in a run (`harness.attempts`) and, given the gate anchors, checks the
labels against them: a closing failure anchor should fall on a failed attempt, a success anchor on a
successful one.

Reads <run>/world and <run>/episodes; writes <out>/attempts.csv and attempts.md.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    attempts,
    detector,
    manifest,
    mechanism,
    metadata,
    milestones,
    schema,
    world_extract,
)

COLUMNS = ("episode_id", "task_id", "episode_success", "command_step", "end_step", "object",
           "outcome")  # fmt: skip


def episode_attempts(
    run_dir: Path,
    row: dict,
    attempt_params=attempts.DEFAULT_PARAMS,
    params=mechanism.DEFAULT_PARAMS,
) -> list[dict]:
    ep = row["episode_id"]
    x, info = world_extract.read_extracted(run_dir / "world" / ep)
    arrays, _ = schema.read_episode(run_dir / row["episode_file"])
    w = mechanism._World(x, info)
    goals = w.goal_objects
    dist = np.linalg.norm(w.mid[:, None, :] - x["object_pos"][:, goals], axis=2)
    holds = [
        [
            (a, e, e < len(dist) and mechanism._release_kind(w, i, (a, e), params) == "slip")
            for a, e in mechanism.held_runs(w.held[:, i])
        ]
        for i in goals
    ]
    found = attempts.label_attempts(
        arrays["executed_action"][:, 6],
        milestones.finger_width(x["gripper_qpos"]),
        dist,
        holds,
        start=w.start,
        params=attempt_params,
    )
    return [
        {
            "episode_id": ep,
            "task_id": info["task_id"],
            "episode_success": int(bool(row["success"])),
            "command_step": a.command,
            "end_step": a.end,
            "object": info["object_names"][goals[a.obj]],
            "outcome": a.outcome,
        }
        for a in found
    ]


def check_anchors(rows: list[dict], anchors_csv: Path, tol: int) -> Counter:
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], []).append(r)
    with open(anchors_csv, newline="") as f:
        anchors = [a for a in csv.DictReader(f) if not a["excluded"] and a["anchor_closed"] == "1"]
    out = Counter()
    for a in anchors:
        eps = by_ep.get(a["episode_id"], [])
        command = detector.nearest_command(
            [r["command_step"] for r in eps], int(a["anchor_step"]), tol
        )
        label = next((r["outcome"] for r in eps if r["command_step"] == command), "no attempt")
        out[(a["group"], label)] += 1
    return out


def render(rows: list[dict], check: Counter | None, sha: str) -> str:
    counts = Counter((r["episode_success"], r["outcome"]) for r in rows)
    lines = [
        "# Grasp attempts",
        "",
        "An attempt is a close command within "
        f"{attempts.DEFAULT_PARAMS.near_m * 100:.0f} cm of a goal object while none is held, "
        "running until the fingers reopen. Success: a hold on a goal object begins before then "
        "and does not end in a slip (the classifier's rule). Aborted: the fingers reopen before "
        f"closing past {attempts.DEFAULT_PARAMS.open_width_m * 100:.0f} cm. Harness `{sha[:7]}`.",
        "",
        "| Episode | success | slip | failure | aborted |",
        "|---|---|---|---|---|",
    ]
    for s, name in ((1, "successful"), (0, "failed")):
        n_ep = len({r["episode_id"] for r in rows if r["episode_success"] == s})
        cells = " | ".join(str(counts[(s, o)]) for o in ("success", "slip", "failure", "aborted"))
        lines.append(f"| {name} ({n_ep} with attempts) | {cells} |")
    if check is None:
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "**Check against the gate anchors** (closing anchors; the attempt nearest the anchor, "
        "within 10 steps)",
        "",
        "| Anchor | success | slip | failure | aborted | no attempt |",
        "|---|---|---|---|---|---|",
    ]
    for g in ("failure", "success"):
        cells = " | ".join(
            str(check[(g, o)]) for o in ("success", "slip", "failure", "aborted", "no attempt")
        )
        lines.append(f"| {g} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--anchors", type=Path, help="gate anchors to check the labels against")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        r
        for m in manifest.read_rows(args.run_dir / "manifest.csv")
        for r in episode_attempts(args.run_dir, m)
    ]
    check = args.anchors and check_anchors(rows, args.anchors, detector.DEFAULT_WIDTH.match_tol)
    sha = metadata.git_state(Path(__file__).resolve().parent.parent)["sha"] or ""
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "attempts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "attempts.md").write_text(render(rows, check, sha))
    print(render(rows, check, sha))


if __name__ == "__main__":
    main()
