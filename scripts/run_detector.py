"""Failure detector, frozen on the baseline run and evaluated once on held-out init states.

v1 (primary): mean gripper exp(H) over the policy steps before each first attempt's close
command. v2 (secondary): mean percentile rank of v1 and stop-filtered roll exp(H) against
successful development first attempts. Each alarms above a threshold that lets the budget's
share of successful development first attempts alarm.

  freeze   <dev run> --attempts <csv> --out <frozen.json>
  evaluate <run> --attempts <csv> --frozen <frozen.json> --out <dir>
  metrics  --scores <scores.csv> --frozen <frozen.json> --out <dir>

`evaluate` writes per-attempt scores to <dir>/scores.csv; `metrics` recomputes the evaluation
from that file alone, without the raw traces.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import detector_window_sweep as ws
import separability_gate as sg

from harness import detector, detector_eval, metadata, milestones, schema

LEVEL = 0.95
N_BOOT = 2000


def features(run_dir: Path, rows: list[dict], roll_zero_bin: int, spec: detector.DetectorSpec):
    series = ws.load(run_dir, rows, roll_zero_bin)
    out = []
    for r in rows:
        is_wait, exp_h, rank = series[r["episode_id"]]
        c = int(r["command_step"])
        g = detector.window_mean(
            exp_h[:, 6], is_wait, c, spec.gripper_window, None, spec.min_window
        )
        roll = detector.window_mean(
            exp_h[:, 3], is_wait, c, spec.roll_window, rank == 2, spec.min_window
        )
        out.append((g, roll))
    return out


def freeze(args) -> None:
    spec = detector.DEFAULT_SPEC
    checks = json.loads((args.run_dir / "instrument_checks.json").read_text())
    roll_zero_bin = checks["per_dim"]["3"]["raw_zero_bin"]
    rows = ws.first_attempts(args.attempts)
    feats = features(args.run_dir, rows, roll_zero_bin, spec)
    ok = [(r, f) for r, f in zip(rows, feats, strict=True) if f[0] is not None]
    succ = [f for r, f in ok if r["outcome"] == "success"]
    g_ref = np.sort([f[0] for f in succ])
    r_ref = np.sort([f[1] for f in succ if f[1] is not None])
    v2_succ = np.array([detector.v2_score(g, roll, g_ref, r_ref) for g, roll in succ])
    frozen = {
        "spec": asdict(spec),
        "roll_zero_bin": roll_zero_bin,
        "v1_threshold": detector.budget_threshold(g_ref, spec.budget),
        "v2_threshold": detector.budget_threshold(v2_succ, spec.budget),
        "gripper_reference": g_ref.tolist(),
        "roll_reference": r_ref.tolist(),
        "dev": {
            "run": str(args.run_dir),
            "n_failure": sum(r["outcome"] != "success" for r, _ in ok),
            "n_success": len(succ),
        },
        "git": metadata.git_state(sg.REPO),
    }
    args.out.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"frozen: v1 threshold {frozen['v1_threshold']:.5f}, v2 threshold "
          f"{frozen['v2_threshold']:.4f}, from {frozen['dev']}")  # fmt: skip


def _width_delays(run_dir: Path, rows: list[dict]) -> list[int | None]:
    out, widths = [], {}
    for r in rows:
        ep = r["episode_id"]
        if ep not in widths:
            a, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
            widths[ep] = milestones.finger_width(a["gripper_qpos"])
        c = int(r["command_step"])
        fire = detector.empty_close_step(widths[ep], c, detector.DEFAULT_WIDTH)
        out.append(None if fire is None else fire - c)
    return out


def score(run_dir: Path, attempts: Path, frozen: dict) -> list[detector_eval.AttemptScore]:
    spec = detector.DetectorSpec(**frozen["spec"])
    g_ref, r_ref = np.array(frozen["gripper_reference"]), np.array(frozen["roll_reference"])
    rows = ws.first_attempts(attempts)
    feats = features(run_dir, rows, frozen["roll_zero_bin"], spec)
    kept = [(r, f) for r, f in zip(rows, feats, strict=True) if f[0] is not None]
    rows = [r for r, _ in kept]
    delays = _width_delays(run_dir, rows)
    return [
        detector_eval.AttemptScore(
            episode_id=r["episode_id"],
            task_id=int(r["task_id"]),
            object=r["object"],
            episode_success=r["episode_success"],
            failed=r["outcome"] != "success",
            v1=float(g),
            v2=detector.v2_score(g, roll, g_ref, r_ref),
            width_delay=d,
        )
        for (r, (g, roll)), d in zip(kept, delays, strict=True)
    ]


def _write_results(scores, frozen: dict, frozen_path: Path, out: Path) -> None:
    thresholds = {name: frozen[f"{name}_threshold"] for name in ("v1", "v2")}
    metrics = detector_eval.evaluate(scores, thresholds, LEVEL, N_BOOT, sg.SEED)
    results = {
        "n_failure": metrics.pop("n_failure"),
        "n_success": metrics.pop("n_success"),
        "frozen": str(frozen_path),
        **metrics,
    }
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    out.mkdir(parents=True, exist_ok=True)
    (out / "detector_eval.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "detector_eval.md").write_text(render(results, frozen, sha))
    print(render(results, frozen, sha))


def evaluate(args) -> None:
    frozen = json.loads(args.frozen.read_text())
    scores = score(args.run_dir, args.attempts, frozen)
    args.out.mkdir(parents=True, exist_ok=True)
    detector_eval.write_scores(args.out / "scores.csv", scores)
    _write_results(scores, frozen, args.frozen, args.out)


def metrics(args) -> None:
    frozen = json.loads(args.frozen.read_text())
    _write_results(detector_eval.read_scores(args.scores), frozen, args.frozen, args.out)


def render(results: dict, frozen: dict, sha: str) -> str:
    spec = frozen["spec"]
    lines = [
        "# Detector evaluation",
        "",
        f"First attempts: {results['n_failure']} failed or slipped, {results['n_success']} "
        f"successful. Frozen `{results['frozen']}` (harness `{frozen['git']['sha'][:7]}`): "
        f"gripper window {spec['gripper_window']}, roll window {spec['roll_window']}, budget "
        f"{spec['budget']:.0%}. Same-task AUROC and excess over chance (catch rate minus "
        f"false-alarm rate) with {LEVEL:.0%} CIs, bootstrap over episodes within (task, episode "
        f"outcome), {N_BOOT} draws, seed {sg.SEED}. Pass: both CIs exclude chance. Evaluated at "
        f"harness `{sha[:7]}`.",
        "",
        "| | AUROC | caught | false alarms | random alarm | excess over chance | width also "
        "(later, median) | width never | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, label in (("v1", "v1 (primary)"), ("v2", "v2 (secondary)")):
        r = results[name]
        delay = "-" if r["width_delay_median"] is None else f"{r['width_delay_median']:.0f}"
        lines.append(
            f"| {label} | {r['auroc']:.2f} ({r['auroc_ci'][0]:.2f} to {r['auroc_ci'][1]:.2f}) "
            f"| {r['caught']} | {r['false_alarms']} ({r['false_alarm_rate']:.0%}) "
            f"| {r['random_alarm_expected']:.1f} | {r['excess_over_chance']:.2f} "
            f"({r['excess_ci'][0]:.2f} to {r['excess_ci'][1]:.2f}) | {r['width_also_catches']} "
            f"({delay} steps) | {r['width_never_catches']} | {'yes' if r['passes'] else 'no'} |"
        )
    w = results["width"]
    lines += [
        "",
        f"Width baseline on the same first attempts: catches {w['caught']}, false alarms "
        f"{w['false_alarms']}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("run_dir", type=Path)
    f.add_argument("--attempts", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--attempts", type=Path, required=True)
    e.add_argument("--frozen", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("metrics")
    m.add_argument("--scores", type=Path, required=True)
    m.add_argument("--frozen", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "evaluate": evaluate, "metrics": metrics}[args.command](args)


if __name__ == "__main__":
    main()
