"""Block 4 detector, frozen on the baseline run and evaluated once on held-out init states.

v1 (primary): mean gripper exp(H) over the policy steps before each first attempt's close
command. v2 (secondary): mean percentile rank of v1 and stop-filtered roll exp(H) against
successful development first attempts. Each alarms above a threshold that lets the budget's
share of successful development first attempts alarm.

  freeze   <dev run> --attempts <csv> --out <frozen.json>
  evaluate <run> --attempts <csv> --frozen <frozen.json> --out <dir>
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import block4_window_sweep as ws
import separability_gate as sg

from harness import detector, gate, metadata, milestones, schema

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


def _units(rows, values, failed) -> list[gate.Unit]:
    return [
        gate.Unit(r["episode_id"], int(r["task_id"]), r["object"], "failure" if f else "success", v)
        for r, v, f in zip(rows, values, failed, strict=True)
    ]


def _excess(units) -> float:
    f = [u.value for u in units if u.group == "failure"]
    s = [u.value for u in units if u.group == "success"]
    return float(np.mean(f) - np.mean(s))


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


def evaluate(args) -> None:
    frozen = json.loads(args.frozen.read_text())
    spec = detector.DetectorSpec(**frozen["spec"])
    g_ref, r_ref = np.array(frozen["gripper_reference"]), np.array(frozen["roll_reference"])
    rows = ws.first_attempts(args.attempts)
    feats = features(args.run_dir, rows, frozen["roll_zero_bin"], spec)
    kept = [(r, f) for r, f in zip(rows, feats, strict=True) if f[0] is not None]
    rows = [r for r, _ in kept]
    failed = np.array([r["outcome"] != "success" for r in rows])
    outcome = {r["episode_id"]: r["episode_success"] for r in rows}
    scores = {
        "v1": np.array([f[0] for _, f in kept]),
        "v2": np.array([detector.v2_score(g, roll, g_ref, r_ref) for _, (g, roll) in kept]),
    }
    delays = _width_delays(args.run_dir, rows)
    width_fires = np.array([d is not None for d in delays])
    results = {
        "n_failure": int(failed.sum()),
        "n_success": int((~failed).sum()),
        "frozen": str(args.frozen),
        "width": {
            "caught": int(width_fires[failed].sum()),
            "false_alarms": int(width_fires[~failed].sum()),
        },
    }
    for name, score in scores.items():
        alarm = score > frozen[f"{name}_threshold"]
        units = _units(rows, score.tolist(), failed)
        alarm_units = _units(rows, alarm.astype(float).tolist(), failed)
        lo, hi = gate.bootstrap_ci(units, gate.same_task_auroc, LEVEL, N_BOOT, sg.SEED, outcome.get)
        elo, ehi = gate.bootstrap_ci(alarm_units, _excess, LEVEL, N_BOOT, sg.SEED, outcome.get)
        caught = alarm & failed
        also = [delays[i] for i in np.flatnonzero(caught & width_fires)]
        fa_rate = float(alarm[~failed].mean())
        results[name] = {
            "auroc": gate.same_task_auroc(units),
            "auroc_ci": [lo, hi],
            "caught": int(caught.sum()),
            "false_alarms": int((alarm & ~failed).sum()),
            "false_alarm_rate": fa_rate,
            "random_alarm_expected": fa_rate * int(failed.sum()),
            "excess_over_chance": detector.excess_over_chance(alarm, failed),
            "excess_ci": [elo, ehi],
            "width_also_catches": len(also),
            "width_delay_median": float(np.median(also)) if also else None,
            "width_never_catches": int((caught & ~width_fires).sum()),
            "passes": bool(lo > 0.5 and elo > 0),
        }
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detector_eval.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "detector_eval.md").write_text(render(results, frozen, sha))
    print(render(results, frozen, sha))


def render(results: dict, frozen: dict, sha: str) -> str:
    spec = frozen["spec"]
    lines = [
        "# Detector evaluation (Block 4)",
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
    args = parser.parse_args()
    {"freeze": freeze, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
