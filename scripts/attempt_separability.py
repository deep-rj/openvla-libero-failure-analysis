"""The gate's exp(H) test at attempt level: window mean of per-dim exp(H) over the policy steps
before each attempt's close command, failed and slipped attempts against successful ones, on all
steps and on steps where the no-motion bin is not in the top two. Subsets: every attempt; the
first attempt per goal object per episode, with all its failures or only those in failed
episodes; retries (attempt 2+ on the object); the attempts nearest the gate anchors. Also the
attempt's index on its object as a score on every attempt.

Reads <attempts>; writes <out>/attempt_separability.json and attempt_separability.md.
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

from harness import detector, gate, metadata, schema
from harness.pilot_report import derive_step_metrics

N_BOOT = 2000
WINDOW = gate.DEFAULT_PARAMS.window
MOTION = len(sg.DIM_NAMES) - 1
VARIANTS = ("all steps", "stop not in top two")
SUBSETS = (
    "every attempt",
    "first per object",
    "first per object, failures in failed episodes only",
    "retries",
    "nearest the gate anchors",
)
INDEX = ("attempt index", "")


def load(run_dir: Path, attempts_csv: Path, zero_bins) -> list[dict]:
    with open(attempts_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["outcome"] != "aborted"]
    out, cached = [], None
    for r in rows:
        ep = r["episode_id"]
        if cached is None or cached[0] != ep:
            a, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
            acting = ~a["is_wait_step"]
            shape = a["slice_logits"].shape[:2]
            exp_h, rank = np.full(shape, np.nan), np.full(shape, 2)
            exp_h[acting] = derive_step_metrics(
                a["slice_logits"][acting],
                a["lse_full"][acting],
                a["argmax_full_id"][acting],
                a["chosen_token_id"][acting],
            )["exp_h"]
            probs = gate.bin_probs(a["slice_logits"][acting])
            for d in range(MOTION):
                rank[acting, d] = gate.stop_rank(probs[:, d], zero_bins[d])
            cached = (ep, a["is_wait_step"], exp_h, rank)
        _, is_wait, exp_h, rank = cached
        command = int(r["command_step"])
        window = gate.pre_window(is_wait, command, WINDOW, gate.DEFAULT_PARAMS.min_window)
        if window is None:
            continue
        values = {}
        for d, name in enumerate(sg.DIM_NAMES):
            values[(name, VARIANTS[0])] = float(exp_h[window, d].mean())
            if d < MOTION:
                values[(name, VARIANTS[1])] = gate.subset_mean(exp_h[:, d], window, rank[:, d] == 2)
        out.append({"row": r, "values": values})
    return out


def subsets(loaded, anchors_csv: Path) -> dict[str, list[dict]]:
    count = {}
    for u in loaded:
        key = (u["row"]["episode_id"], u["row"]["object"])
        count[key] = count.get(key, 0) + 1
        u["values"][INDEX] = float(count[key])
    first = [u for u in loaded if u["values"][INDEX] == 1]
    first_fatal = [
        u for u in first if u["row"]["outcome"] == "success" or u["row"]["episode_success"] == "0"
    ]
    retries = [u for u in loaded if u["values"][INDEX] > 1]
    with open(anchors_csv, newline="") as f:
        anchors = [a for a in csv.DictReader(f) if not a["excluded"] and a["anchor_closed"] == "1"]
    by_ep = {}
    for u in loaded:
        by_ep.setdefault(u["row"]["episode_id"], []).append(u)
    nearest = []
    for a in anchors:
        us = by_ep.get(a["episode_id"], [])
        c = detector.nearest_command(
            [int(u["row"]["command_step"]) for u in us],
            int(a["anchor_step"]),
            detector.DEFAULT_WIDTH.match_tol,
        )
        nearest += [u for u in us if int(u["row"]["command_step"]) == c][:1]
    return dict(zip(SUBSETS, (loaded, first, first_fatal, retries, nearest), strict=True))


def to_units(us, key) -> list[gate.Unit]:
    return [
        gate.Unit(
            u["row"]["episode_id"],
            int(u["row"]["task_id"]),
            u["row"]["object"],
            "success" if u["row"]["outcome"] == "success" else "failure",
            u["values"][key],
        )
        for u in us
        if u["values"].get(key) is not None
    ]


def analyse(groups: dict[str, list[dict]]) -> dict:
    outcome = {u["row"]["episode_id"]: u["row"]["episode_success"] for us in groups.values()
               for u in us}  # fmt: skip
    out = {}

    def test(us, key):
        units = to_units(us, key)
        lo, hi = gate.bootstrap_ci(
            units, gate.same_task_auroc, sg.EXPLORATORY_LEVEL, N_BOOT, sg.SEED, outcome.get
        )
        return {
            "auroc": gate.same_task_auroc(units),
            "ci": [lo, hi],
            "n_failure": sum(u.group == "failure" for u in units),
            "n_success": sum(u.group == "success" for u in units),
        }

    out[f"{SUBSETS[0]} | {INDEX[0]}"] = test(groups[SUBSETS[0]], INDEX)
    for subset, us in groups.items():
        for d, name in enumerate(sg.DIM_NAMES):
            for variant in VARIANTS:
                if variant == VARIANTS[1] and d >= MOTION:
                    continue
                out[f"{subset} | {name} | {variant}"] = test(us, (name, variant))
    return out


def render(results: dict, sha: str) -> str:
    lines = [
        "# Attempt-level separability (exploratory)",
        "",
        f"Window mean of exp(H) over the {WINDOW} policy steps before each attempt's close "
        f"command. Failed and slipped attempts against successful ones, aborted excluded. "
        f"Same-task AUROC, "
        f"failures above successes, {sg.EXPLORATORY_LEVEL:.2%} CIs, bootstrap over episodes "
        f"within (task, episode outcome), {N_BOOT} draws, seed {sg.SEED}. Harness `{sha[:7]}`.",
    ]
    r = results[f"{SUBSETS[0]} | {INDEX[0]}"]
    lines += [
        "",
        f"Attempt index on its object as the score, every attempt: {r['auroc']:.2f} "
        f"({r['ci'][0]:.2f} to {r['ci'][1]:.2f}).",
    ]
    for subset in SUBSETS:
        first = results[f"{subset} | {sg.DIM_NAMES[0]} | {VARIANTS[0]}"]
        lines += [
            "",
            f"**{subset}** ({first['n_failure']} failed, {first['n_success']} successful)",
            "",
            "| Dim | all steps | stop not in top two |",
            "|---|---|---|",
        ]
        for name in sg.DIM_NAMES:
            cells = []
            for variant in VARIANTS:
                r = results.get(f"{subset} | {name} | {variant}")
                if r is None:
                    cells.append("-")
                    continue
                mark = "*" if r["ci"][0] > 0.5 or r["ci"][1] < 0.5 else ""
                cells.append(f"{r['auroc']:.2f}{mark} ({r['ci'][0]:.2f} to {r['ci'][1]:.2f})")
            lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "\\* CI excludes 0.5."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks = json.loads((args.run_dir / "instrument_checks.json").read_text())
    zero_bins = [checks["per_dim"][str(d)]["raw_zero_bin"] for d in range(MOTION)]
    loaded = load(args.run_dir, args.attempts, zero_bins)
    results = analyse(subsets(loaded, args.anchors))
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    (args.out / "attempt_separability.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "attempt_separability.md").write_text(render(results, sha))
    print(render(results, sha))


if __name__ == "__main__":
    main()
