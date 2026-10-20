"""Window selection for the failure detector on first attempts: same-task AUROC of the window mean
of gripper exp(H) (v1) and of stop-filtered roll exp(H) (v2) before the close command, over a
declared grid. Failed and slipped first attempts against successful ones; the best window per
feature is the one frozen.

Reads <attempts>; writes <out>/window_sweep.json and window_sweep.md.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import separability_gate as sg

from harness import gate, metadata, schema
from harness.pilot_report import derive_step_metrics

GRID = {"gripper": (5, 10, 20, 30, 50, 75, 100), "roll": (5, 10, 20, 30, 50, 75, 100, 150, 200)}
DIM = {"gripper": 6, "roll": 3}
N_BOOT = 2000


def first_attempts(attempts_csv: Path) -> list[dict]:
    with open(attempts_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["outcome"] != "aborted"]
    out, seen = [], set()
    for r in rows:
        key = (r["episode_id"], r["object"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def load(run_dir: Path, rows: list[dict], roll_zero_bin: int) -> dict:
    out = {}
    for ep in {r["episode_id"] for r in rows}:
        a, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
        acting = ~a["is_wait_step"]
        exp_h = np.full(a["slice_logits"].shape[:2], np.nan)
        exp_h[acting] = derive_step_metrics(
            a["slice_logits"][acting],
            a["lse_full"][acting],
            a["argmax_full_id"][acting],
            a["chosen_token_id"][acting],
        )["exp_h"]
        rank = np.full(len(acting), 2)
        rank[acting] = gate.stop_rank(
            gate.bin_probs(a["slice_logits"][acting])[:, 3], roll_zero_bin
        )
        out[ep] = (a["is_wait_step"], exp_h, rank)
    return out


def window_value(series, command: int, feature: str, length: int):
    """Window mean and window length, or None if the window is too short."""
    is_wait, exp_h, rank = series
    min_length = min(length, gate.DEFAULT_PARAMS.min_window)
    window = gate.pre_window(is_wait, command, length, min_length)
    if window is None:
        return None
    keep = rank == 2 if feature == "roll" else np.ones(len(rank), bool)
    v = gate.subset_mean(exp_h[:, DIM[feature]], window, keep)
    return None if v is None else (v, len(window))


def sweep(rows: list[dict], series: dict) -> dict:
    outcome = {r["episode_id"]: r["episode_success"] for r in rows}
    out = {}
    for feature, grid in GRID.items():
        entries = []
        for length in grid:
            units, short = [], 0
            for r in rows:
                got = window_value(series[r["episode_id"]], int(r["command_step"]), feature, length)
                if got is None:
                    continue
                short += got[1] < length
                group = "success" if r["outcome"] == "success" else "failure"
                units.append(
                    gate.Unit(r["episode_id"], int(r["task_id"]), r["object"], group, got[0])
                )
            lo, hi = gate.bootstrap_ci(
                units, gate.same_task_auroc, sg.EXPLORATORY_LEVEL, N_BOOT, sg.SEED, outcome.get
            )
            entries.append(
                {
                    "window": length,
                    "auroc": gate.same_task_auroc(units),
                    "ci": [lo, hi],
                    "n_failure": sum(u.group == "failure" for u in units),
                    "n_success": sum(u.group == "success" for u in units),
                    "n_short": int(short),
                }
            )
        best = max(entries, key=lambda e: e["auroc"])
        out[feature] = {"grid": entries, "chosen": best["window"]}
    return out


def render(results: dict, sha: str) -> str:
    lines = [
        "# Window sweep, first attempts",
        "",
        f"Window mean of exp(H) before the close command: gripper on all steps (v1), roll on steps "
        f"where the no-motion bin is not in the top two (v2). Failed and slipped first attempts "
        f"against successful ones. Same-task AUROC, {sg.EXPLORATORY_LEVEL:.2%} CIs, bootstrap over "
        f"episodes within (task, episode outcome), {N_BOOT} draws, seed {sg.SEED}. Short: windows "
        f"cut by the episode start. Harness `{sha[:7]}`.",
    ]
    for feature, r in results.items():
        lines += [
            "",
            f"**{feature}** (chosen: {r['chosen']} steps)",
            "",
            "| Window | AUROC | failed / successful | short |",
            "|---|---|---|---|",
        ]
        for e in r["grid"]:
            lines.append(
                f"| {e['window']} | {e['auroc']:.2f} ({e['ci'][0]:.2f} to {e['ci'][1]:.2f}) "
                f"| {e['n_failure']} / {e['n_success']} | {e['n_short']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks = json.loads((args.run_dir / "instrument_checks.json").read_text())
    rows = first_attempts(args.attempts)
    series = load(args.run_dir, rows, checks["per_dim"]["3"]["raw_zero_bin"])
    results = sweep(rows, series)
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    (args.out / "window_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "window_sweep.md").write_text(render(results, sha))
    print(render(results, sha))


if __name__ == "__main__":
    main()
