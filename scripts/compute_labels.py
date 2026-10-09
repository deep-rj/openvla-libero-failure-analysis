"""Writes the computed label columns (milestone, closure fraction, flags) for a selection.

Reads <run>/world/ (from `extract_world.py`) and writes <run>/labeling/computed.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import flags, milestones, world_extract


def computed_row(extracted, info) -> dict:
    m = milestones.compute_milestone(extracted, info)
    fractions = [
        milestones.closure_fraction(extracted, info, g)
        for g, s in enumerate(info["subgoals"])
        if s["kind"] == "fixture"
    ]
    return {
        "episode_id": info["episode_id"],
        "task_id": info["task_id"],
        "n_subgoals": len(info["subgoals"]),
        "k_completed": m.k,
        "phase": m.phase,
        "k_step": "" if m.k_step is None else m.k_step,
        "closure_fraction": ";".join("" if np.isnan(f) else f"{f:.3f}" for f in fractions),
        **{k: flags.format_intervals(v) for k, v in flags.compute_flags(extracted, info).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()

    selection = args.selection or args.run_dir / "labeling" / "label_selection.json"
    ids = json.loads(selection.read_text())["episode_ids"]
    rows = [computed_row(*world_extract.read_extracted(args.run_dir / "world" / i)) for i in ids]
    out = args.run_dir / "labeling" / "computed.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
