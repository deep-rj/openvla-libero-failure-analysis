"""Computes (k, phase) for every extracted episode and checks the replay against LIBERO.

Writes <run>/world/milestones_n<N>.csv. No logged step may have every subgoal holding: LIBERO
ends the episode on the first such state, and that state is never logged. The last logged step
of a success is exempt (one-substep pose lag in the live check). A violation means the replay
does not reproduce the live scene.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import milestones, world_extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--debounce", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    world_dir = args.run_dir / "world"
    paths = sorted(world_dir.glob("*.json"))
    loaded = [world_extract.read_extracted(p.with_suffix("")) for p in paths]
    for n in args.debounce:
        params = milestones.MilestoneParams(debounce=n)
        rows, problems = [], []
        for extracted, info in loaded:
            n_goals = len(info["subgoals"])
            m = milestones.compute_milestone(extracted, info, params)
            as_failure = milestones.compute_milestone(
                extracted, {**info, "final_success": False}, params
            )
            all_hold = extracted["goal_holds"].all(axis=1)
            if info["final_success"]:
                # The live check reads poses one substep stale; replay can lead it by a step.
                all_hold = all_hold[:-1]
            if all_hold.any():
                problems.append(f"{info['episode_id']}: every subgoal holds at a logged step")
            fractions = [
                milestones.closure_fraction(extracted, info, g)
                for g, s in enumerate(info["subgoals"])
                if s["kind"] == "fixture"
            ]
            rows.append(
                {
                    "episode_id": info["episode_id"],
                    "task_id": info["task_id"],
                    "success": info["final_success"],
                    "n_subgoals": n_goals,
                    "k_completed": m.k,
                    "phase": m.phase,
                    "k_step": "" if m.k_step is None else m.k_step,
                    "k_logged": as_failure.k,
                    "closure_fraction": ";".join(
                        "" if np.isnan(f) else f"{f:.3f}" for f in fractions
                    ),
                }
            )
        out = world_dir / f"milestones_n{n}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"N={n}: {len(rows)} episodes, {len(problems)} inconsistencies -> {out}")
        for p in problems:
            print("  " + p)


if __name__ == "__main__":
    main()
