"""Held-out review of the frozen Block 4 detector (v1).

sample    <run> --attempts <csv> --frozen <json> --out <csv>
          five first attempts per stratum (caught, missed, false alarm), distinct tasks, seed 0
plot      <run> --attempts <csv> --frozen <json> --episodes t00-i035 ... --out <png>
          gripper exp(H) and finger width around each episode's first scored attempt
stall     <run> --attempts <csv> --frozen <json> --out <dir>
          v1 rescored with grasp-then-stall failures excluded (`harness.review.is_stall`)
mechanism <dev run> --attempts <csv> --frozen <json> --mechanisms <csv> --out <dir>
          v1 on development first attempts split by the episode's Block 2 mechanism
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import block4_window_sweep as ws
import separability_gate as sg
from block4_detector import LEVEL, N_BOOT, _excess, _units, _width_delays, features

from harness import detector, gate, mechanism, metadata, milestones, review, world_extract

FLAT = 1.001  # v1 below this: no step in the window departs from a certain gripper token
STRATA = ("caught", "missed", "false_alarm")


def short(episode_id: str) -> str:
    return "-".join(episode_id.split("-")[1:3])


def scored(run_dir: Path, attempts_csv: Path, frozen: dict):
    """First attempts with a v1 score, their scores, and alarms at the frozen threshold."""
    spec = detector.DetectorSpec(**frozen["spec"])
    rows = ws.first_attempts(attempts_csv)
    feats = features(run_dir, rows, frozen["roll_zero_bin"], spec)
    kept = [(r, f[0]) for r, f in zip(rows, feats, strict=True) if f[0] is not None]
    v1 = np.array([v for _, v in kept])
    return [r for r, _ in kept], v1, v1 > frozen["v1_threshold"]


def evaluate(rows: list[dict], v1: np.ndarray, threshold: float) -> dict:
    failed = np.array([r["outcome"] != "success" for r in rows])
    alarm = v1 > threshold
    outcome = {r["episode_id"]: r["episode_success"] for r in rows}
    units = _units(rows, v1.tolist(), failed)
    alarm_units = _units(rows, alarm.astype(float).tolist(), failed)
    lo, hi = gate.bootstrap_ci(units, gate.same_task_auroc, LEVEL, N_BOOT, sg.SEED, outcome.get)
    elo, ehi = gate.bootstrap_ci(alarm_units, _excess, LEVEL, N_BOOT, sg.SEED, outcome.get)
    return {
        "n_failure": int(failed.sum()),
        "n_success": int((~failed).sum()),
        "auroc": gate.same_task_auroc(units),
        "auroc_ci": [lo, hi],
        "caught": int((alarm & failed).sum()),
        "false_alarms": int((alarm & ~failed).sum()),
        "excess_over_chance": _excess(alarm_units),
        "excess_ci": [elo, ehi],
        "passes": bool(lo > 0.5 and elo > 0),
    }


def sample(args) -> None:
    frozen = json.loads(args.frozen.read_text())
    rows, v1, alarm = scored(args.run_dir, args.attempts, frozen)
    failed = np.array([r["outcome"] != "success" for r in rows])
    masks = {"caught": alarm & failed, "missed": ~alarm & failed, "false_alarm": alarm & ~failed}
    rng = random.Random(0)  # noqa: S311
    out = []
    for name in STRATA:
        pool = list(np.flatnonzero(masks[name]))
        rng.shuffle(pool)
        picked, tasks = [], set()
        for i in pool:
            if rows[i]["task_id"] not in tasks and len(picked) < args.per_stratum:
                picked.append(i)
                tasks.add(rows[i]["task_id"])
        for i in sorted(picked, key=lambda i: rows[i]["episode_id"]):
            r = rows[i]
            out.append({"stratum": name, "episode": short(r["episode_id"]), "task": r["task_id"],
                        "object": r["object"], "label": r["outcome"],
                        "episode_success": r["episode_success"], "command_step": r["command_step"],
                        "end_step": r["end_step"], "v1": f"{v1[i]:.4f}"})  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} attempts -> {args.out}")


def plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = json.loads(args.frozen.read_text())
    window, threshold = frozen["spec"]["gripper_window"], frozen["v1_threshold"]
    by_ep = {short(r["episode_id"]): r for r in ws.first_attempts(args.attempts)}
    ink, muted, series, shade = "#1f1f1f", "#8a8a8a", "#2a6fdb", "#e8eefb"
    fig, axes = plt.subplots(len(args.episodes), 2, figsize=(13, 2.4 * len(args.episodes)),
                             sharex=True, squeeze=False)  # fmt: skip
    for row, ep in zip(axes, args.episodes, strict=True):
        r = by_ep[ep]
        c, end = int(r["command_step"]), int(r["end_step"])
        is_wait, exp_h, _ = ws.load(args.run_dir, [r], frozen["roll_zero_bin"])[r["episode_id"]]
        (fire,) = _width_delays(args.run_dir, [r])
        a, _ = sg.schema.read_episode(args.run_dir / "episodes" / f"{r['episode_id']}.h5")
        width = milestones.finger_width(a["gripper_qpos"]) * 1000
        closing = np.flatnonzero(a["executed_action"][:, 6] > 0) - c
        v1 = detector.window_mean(exp_h[:, 6], is_wait, c, window, None,
                                  frozen["spec"]["min_window"])  # fmt: skip
        t = np.arange(len(width)) - c
        for ax in row:
            ax.axvspan(-window, 0, color=shade, lw=0)
            ax.axvline(0, color=ink, lw=1)
            ax.axvline(end - c, color=muted, lw=1, ls=":")
            for s in closing[(closing >= -60) & (closing <= 90)]:
                ax.axvspan(s - 0.5, s + 0.5, ymin=0, ymax=0.06, color=ink, lw=0)
            ax.set_xlim(-60, 90)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].plot(t, exp_h[:, 6], color=series, lw=1.5)
        row[0].axhline(threshold, color=muted, ls="--", lw=1)
        row[0].hlines(v1, -window, 0, color=ink, lw=2.5)
        row[0].set_ylabel("gripper exp(H)")
        row[0].set_title(f"{ep}  v1 {v1:.4f} vs threshold {threshold:.4f}", loc="left", fontsize=10)
        row[1].plot(t, width, color=series, lw=1.5)
        row[1].axhline(detector.DEFAULT_WIDTH.threshold_m * 1000, color=muted, ls="--", lw=1)
        if fire is not None:
            row[1].plot(fire, width[c + fire], "o", ms=8, color=ink)
        row[1].set_ylabel("finger width (mm)")
        when = "never" if fire is None else f"+{fire}"
        row[1].set_title(f"width fires {when}; attempt ends +{end - c}", loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("policy steps from close command")
    legend = (
        f"Shaded: {window}-step v1 window. Solid: close command; bar: window mean (v1). Dashed: v1 "
        "threshold / width threshold. Dotted: attempt end. Ticks: steps commanding close."
    )
    fig.text(0.01, 0.005, legend, fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(args.out)


def stall_features(run_dir: Path, r: dict) -> dict:
    x, info = world_extract.read_extracted(run_dir / "world" / r["episode_id"])
    w = mechanism._World(x, info)
    o = info["object_names"].index(r["object"])
    c, e = int(r["command_step"]), int(r["end_step"])
    width = milestones.finger_width(x["gripper_qpos"])[c:e]
    settled = review.settled_mask(width)
    rise = float((x["object_pos"][c:e, o, 2] - x["object_pos"][c, o, 2]).max())
    dist = np.linalg.norm(w.mid[c:e] - x["object_pos"][c:e, o], axis=1)
    return {
        "settled_steps": int(settled.sum()),
        "settled_width_m": float(np.median(width[settled])) if settled.any() else float("nan"),
        "rise_m": rise,
        "object_dist_m": float(np.median(dist[settled])) if settled.any() else float("nan"),
    }


def stall(args) -> None:
    frozen = json.loads(args.frozen.read_text())
    rows, v1, alarm = scored(args.run_dir, args.attempts, frozen)
    stalls = []
    for i, r in enumerate(rows):
        if r["outcome"] != "failure":
            continue
        f = stall_features(args.run_dir, r)
        if review.is_stall(r["outcome"], f["settled_steps"], f["settled_width_m"], f["rise_m"]):
            stalls.append({"episode": short(r["episode_id"]), "object": r["object"],
                           "alarm": bool(alarm[i]), **f})  # fmt: skip
    excluded = {(s["episode"], s["object"]) for s in stalls}
    keep = [(short(r["episode_id"]), r["object"]) not in excluded for r in rows]
    results = {
        "rule": review.DEFAULT_STALL.__dict__,
        "stalls": stalls,
        "as_frozen": evaluate(rows, v1, frozen["v1_threshold"]),
        "stalls_excluded": evaluate(
            [r for r, k in zip(rows, keep, strict=True) if k], v1[keep], frozen["v1_threshold"]
        ),
        "git": metadata.git_state(sg.REPO),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stall_sensitivity.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# v1 with grasp-then-stall failures excluded",
        "",
        f"Stall: a failed first attempt whose fingers settle (width moving under 1 mm per step) "
        f"between 3 and 66 mm for at least 30 steps on an object that rises under 2 cm "
        f"(`harness.review`). {len(stalls)} found: "
        + ", ".join(f"{s['episode']}{' (alarm)' if s['alarm'] else ''}" for s in stalls)
        + ".",
        "",
        "| | failed | AUROC | caught | false alarms | excess over chance | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, key in (("as frozen", "as_frozen"), ("stalls excluded", "stalls_excluded")):
        m = results[key]
        lines.append(
            f"| {name} | {m['n_failure']} | {m['auroc']:.3f} ({m['auroc_ci'][0]:.3f} to "
            f"{m['auroc_ci'][1]:.3f}) | {m['caught']} | {m['false_alarms']} | "
            f"{m['excess_over_chance']:.3f} ({m['excess_ci'][0]:.3f} to {m['excess_ci'][1]:.3f}) "
            f"| {'yes' if m['passes'] else 'no'} |"
        )
    (args.out / "stall_sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def mechanism_groups(rows, v1, mechanisms: dict, tol: int) -> dict:
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["outcome"] == "success":
            groups["success"].append(i)
            continue
        m = mechanisms.get(r["episode_id"])
        if m is None:
            groups["failed attempt, successful episode"].append(i)
            continue
        onset = m["onset_step"]
        initiator = onset != "" and abs(int(onset) - int(r["command_step"])) <= tol
        groups[m["mechanism"] if initiator else "failed attempt, not the initiator"].append(i)
    return groups


def mechanism_split(args) -> None:
    frozen = json.loads(args.frozen.read_text())
    rows, v1, alarm = scored(args.run_dir, args.attempts, frozen)
    with open(args.mechanisms, newline="") as f:
        mechanisms = {m["episode_id"]: m for m in csv.DictReader(f)}
    results = {"git": metadata.git_state(sg.REPO), "by_tolerance": {}}
    lines = [
        "# v1 by failure mechanism (development, in-sample)",
        "",
        "A failed first attempt takes its episode's Block 2 mechanism when the mechanism's onset "
        "is within the tolerance of the close command. AUROC is same-task against successful "
        f"first attempts, 95% CI. Flat: v1 below {FLAT}.",
    ]
    for tol in args.tol:
        groups = mechanism_groups(rows, v1, mechanisms, tol)
        succ = groups["success"]
        lines += ["", f"Onset within ±{tol} steps:", "",
                  "| group | n | flat | alarm | AUROC |", "|---|---|---|---|---|"]  # fmt: skip
        entry = {}
        for name in sorted(groups, key=lambda k: -len(groups[k])):
            idx = groups[name]
            g = {"n": len(idx), "flat": float(np.mean(v1[idx] < FLAT)),
                 "alarm": float(np.mean(alarm[idx]))}  # fmt: skip
            auroc = ""
            if name != "success" and len(idx) >= 5:
                both = idx + succ
                m = evaluate([dict(rows[i], outcome="failure" if i in idx else "success")
                              for i in both], v1[both], frozen["v1_threshold"])  # fmt: skip
                g.update(auroc=m["auroc"], auroc_ci=m["auroc_ci"])
                auroc = f"{m['auroc']:.2f} ({m['auroc_ci'][0]:.2f} to {m['auroc_ci'][1]:.2f})"
            entry[name] = g
            lines.append(f"| {name} | {g['n']} | {g['flat']:.0%} | {g['alarm']:.0%} | {auroc} |")
        results["by_tolerance"][str(tol)] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mechanism_split.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "mechanism_split.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sample", "plot", "stall", "mechanism"):
        p = sub.add_parser(name)
        p.add_argument("run_dir", type=Path)
        p.add_argument("--attempts", type=Path, required=True)
        p.add_argument("--frozen", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "sample":
            p.add_argument("--per-stratum", type=int, default=5)
        if name == "plot":
            p.add_argument("--episodes", nargs="+", required=True)
        if name == "mechanism":
            p.add_argument("--mechanisms", type=Path, required=True)
            p.add_argument("--tol", type=int, nargs="+", default=[10, 3])
    args = parser.parse_args()
    {"sample": sample, "plot": plot, "stall": stall, "mechanism": mechanism_split}[args.command](
        args
    )


if __name__ == "__main__":
    main()
