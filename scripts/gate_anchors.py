"""Writes <out>/anchors.csv: the grasp-descent anchor and pre-window of every gate unit, and
prints the gripper near-tie counts per window pooled over both outcomes (no outcome split).

Units: grasp and perception failures at objects (fixture failures excluded), and each place
subgoal of each success.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import gate, manifest, schema, world_extract
from harness.pilot_report import derive_step_metrics

# Hand-labeled failures at a fixture (stove knob, drawer), which have no grasp descent.
FIXTURE_FAILURES = {
    "libero_10-t02-i007-s0-none",
    "libero_10-t02-i008-s0-none",
    "libero_10-t02-i014-s0-none",
    "libero_10-t03-i008-s0-none",
}
GRASP_MECHANISMS = ("perception", "control: grasp")
COLUMNS = (
    "episode_id", "task_id", "group", "object", "onset_step", "anchor_step", "anchor_closed",
    "window_len", "near_ties", "excluded",
)  # fmt: skip


def units(run_dir: Path, mechanisms_csv: Path, params=gate.DEFAULT_PARAMS):
    with open(mechanisms_csv, newline="") as f:
        failures = {
            r["episode_id"]: int(r["onset_step"])
            for r in csv.DictReader(f)
            if r["mechanism"] in GRASP_MECHANISMS
        }
    for row in manifest.read_rows(run_dir / "manifest.csv"):
        ep = row["episode_id"]
        if not row["success"] and ep not in failures:
            continue
        x, info = world_extract.read_extracted(run_dir / "world" / ep)
        arrays, _ = schema.read_episode(run_dir / row["episode_file"])
        margins = derive_step_metrics(
            arrays["slice_logits"],
            arrays["lse_full"],
            arrays["argmax_full_id"],
            arrays["chosen_token_id"],
        )["bin_margin_ulps"][:, gate.GRIPPER_DIM]
        descents = gate.grasp_descents(x, info)
        base = {"episode_id": ep, "task_id": info["task_id"]}
        if row["success"]:
            targets = [
                (g, None, gate.success_anchor(x, info, descents, g))
                for g, s in enumerate(info["subgoals"])
                if s["kind"] == "place"
            ]
        elif ep in FIXTURE_FAILURES:
            targets = [(None, failures[ep], "fixture")]
        else:
            onset = failures[ep]
            targets = [(None, onset, gate.failure_anchor(x, info, descents, onset, params.match_tol))]
        for g, onset, anchor in targets:
            unit = {
                **base,
                "group": "success" if row["success"] else "failure",
                "onset_step": "" if onset is None else onset,
            }
            if g is not None:
                unit["object"] = info["object_names"][info["subgoals"][g]["object_index"]]
            if anchor == "fixture" or anchor is None:
                yield {**unit, "excluded": anchor or "no anchor"}
                continue
            unit["object"] = info["object_names"][anchor.obj]
            window = gate.pre_window(
                arrays["is_wait_step"], anchor.bottom, params.window, params.min_window
            )
            unit.update(anchor_step=anchor.bottom, anchor_closed=int(anchor.close is not None))
            if window is None:
                yield {**unit, "excluded": "short window"}
                continue
            near = int((margins[window] <= params.near_tie_ulps).sum())
            yield {**unit, "window_len": len(window), "near_ties": near, "excluded": ""}


def pooled_counts(rows) -> str:
    kept = [r for r in rows if not r["excluded"]]
    counts = np.array([r["near_ties"] for r in kept])
    lengths = np.array([r["window_len"] for r in kept])
    q = np.percentile(counts, [25, 50, 75, 90])
    return (
        f"windows: {len(kept)} kept (both outcomes pooled), mean length {lengths.mean():.1f}\n"
        f"near-ties per window: zero in {(counts == 0).mean():.0%}, mean {counts.mean():.2f}, "
        f"p25/p50/p75/p90 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}, max {counts.max()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--mechanisms", type=Path, help="default <run>/labeling/failure_mechanisms.csv"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mechanisms = args.mechanisms or args.run_dir / "labeling" / "failure_mechanisms.csv"
    rows = list(units(args.run_dir, mechanisms))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "anchors.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    excluded = {}
    for r in rows:
        if r["excluded"]:
            key = (r["group"], r["excluded"])
            excluded[key] = excluded.get(key, 0) + 1
    print(f"units: {len(rows)}; excluded: {excluded}")
    print(pooled_counts(rows))


if __name__ == "__main__":
    main()
