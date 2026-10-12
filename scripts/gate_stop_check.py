"""Exploratory check after the gate: does the motion-dim separation sit on steps where "no motion"
competes? Per step and motion dim, the no-motion bin (raw action 0, from instrument_checks.json)
is the top choice, the runner-up, or neither.

Reads <out>/anchors.csv; writes <out>/stop_check.json, stop_check.md and stop_curves.png.
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

MOTION_DIMS = sg.DIM_NAMES[:6]
METRICS = ("stop chosen share", "stop runner-up share", "exp(H), stop in top two",
           "exp(H), stop not in top two")  # fmt: skip


def load(run_dir: Path, anchors_csv: Path, zero_bins, params=gate.DEFAULT_PARAMS):
    with open(anchors_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded"]]
    span = np.arange(-sg.CURVE_BEFORE, sg.CURVE_AFTER + 1)
    out, cached = [], None
    for r in rows:
        ep = r["episode_id"]
        if cached is None or cached[0] != ep:
            a, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
            acting = ~a["is_wait_step"]
            exp_h = np.full(a["slice_logits"].shape[:2], np.nan)
            rank = np.full(a["slice_logits"].shape[:2], -1)
            exp_h[acting] = derive_step_metrics(
                a["slice_logits"][acting],
                a["lse_full"][acting],
                a["argmax_full_id"][acting],
                a["chosen_token_id"][acting],
            )["exp_h"]
            probs = gate.bin_probs(a["slice_logits"][acting])
            for d in range(len(MOTION_DIMS)):
                rank[acting, d] = gate.stop_rank(probs[:, d], zero_bins[d])
            cached = (ep, a["is_wait_step"], exp_h, rank)
        _, is_wait, exp_h, rank = cached
        anchor = int(r["anchor_step"])
        window = gate.pre_window(is_wait, anchor, params.window, params.min_window)
        steps = anchor + span
        valid = (steps >= 0) & (steps < len(is_wait))
        valid[valid] &= ~is_wait[steps[valid]]
        curve_rank = np.full((len(span), len(MOTION_DIMS)), -1)
        curve_rank[valid] = rank[steps[valid], : len(MOTION_DIMS)]
        values = {}
        for d, name in enumerate(MOTION_DIMS):
            rk = rank[:, d]
            values[(name, METRICS[0])] = float((rk[window] == 0).mean())
            values[(name, METRICS[1])] = float((rk[window] == 1).mean())
            values[(name, METRICS[2])] = gate.subset_mean(exp_h[:, d], window, rk <= 1)
            values[(name, METRICS[3])] = gate.subset_mean(exp_h[:, d], window, rk == 2)
        curve_exp_h = np.full((len(span), len(MOTION_DIMS)), np.nan)
        curve_exp_h[valid] = exp_h[steps[valid], : len(MOTION_DIMS)]
        out.append(
            {
                "row": r,
                "values": values,
                "curve_rank": curve_rank,
                "curve_exp_h": curve_exp_h,
                "valid": valid,
                "steps": (is_wait, exp_h, rank),
            }
        )
    return out


def test(loaded, key) -> dict:
    kept = [u for u in loaded if u["values"][key] is not None]
    units = [
        gate.Unit(
            u["row"]["episode_id"],
            int(u["row"]["task_id"]),
            u["row"]["object"],
            u["row"]["group"],
            u["values"][key],
        )
        for u in kept
    ]
    lo, hi = gate.bootstrap_ci(
        units, gate.same_task_auroc, sg.EXPLORATORY_LEVEL, sg.N_BOOT, sg.SEED
    )
    n = {g: sum(u.group == g for u in units) for g in ("failure", "success")}
    return {"auroc": gate.same_task_auroc(units), "ci": [lo, hi], "n": n}


def plot(loaded, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = np.arange(-sg.CURVE_BEFORE, sg.CURVE_AFTER + 1)
    rows = (("stop chosen", 0), ("stop runner-up", 1))
    fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharex=True, sharey="row", facecolor=sg.SURFACE)
    for d, name in enumerate(MOTION_DIMS):
        for row, (label, rank) in enumerate(rows):
            ax = axes[row, d]
            ax.set_facecolor(sg.SURFACE)
            for group in ("success", "failure"):
                us = [u for u in loaded if u["row"]["group"] == group]
                hit = np.stack([u["curve_rank"][:, d] == rank for u in us])
                valid = np.stack([u["valid"] for u in us])
                share = hit.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
                ax.plot(span, share, color=sg.COLORS[group], lw=2,
                        label=f"{sg.PLURAL[group]} (n = {len(us)})")  # fmt: skip
            ax.axvline(0, color=sg.MUTED, lw=1, ls="--")
            ax.grid(color=sg.GRID, lw=0.8)
            ax.tick_params(colors=sg.MUTED, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(sg.GRID)
            if row == 0:
                ax.set_title(name, color=sg.INK, fontsize=11, loc="left")
            if d == 0:
                ax.set_ylabel(f"share of runs: {label}", color=sg.MUTED, fontsize=9)
            if row == 1:
                ax.set_xlabel("steps from the grasp descent's bottom", color=sg.MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=sg.INK)
    fig.suptitle(
        "Where the no-motion bin ranks per step, aligned on the grasp descent",
        color=sg.INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=sg.SURFACE)
    plt.close(fig)


def plot_filtered(loaded, path: Path, min_runs: int = 10) -> None:
    """Per-step mean exp(H) over the runs whose step qualifies; points with fewer than
    `min_runs` qualifying runs are left out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = np.arange(-sg.CURVE_BEFORE, sg.CURVE_AFTER + 1)
    rows = (
        ("all steps", lambda rk: rk >= 0),
        ("stop not in top two", lambda rk: rk == 2),
        ("stop in top two", lambda rk: (rk == 0) | (rk == 1)),
    )
    fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), sharex=True, facecolor=sg.SURFACE)
    for d, name in enumerate(MOTION_DIMS):
        for row, (label, keep) in enumerate(rows):
            ax = axes[row, d]
            ax.set_facecolor(sg.SURFACE)
            for group in ("success", "failure"):
                us = [u for u in loaded if u["row"]["group"] == group]
                h = np.stack([u["curve_exp_h"][:, d] for u in us])
                ok = np.stack([keep(u["curve_rank"][:, d]) for u in us])
                n = ok.sum(axis=0)
                mean = np.where(ok, h, 0.0).sum(axis=0) / np.maximum(n, 1)
                ax.plot(span, np.where(n >= min_runs, mean, np.nan), color=sg.COLORS[group],
                        lw=2, label=f"{sg.PLURAL[group]} (n = {len(us)})")  # fmt: skip
            ax.axvline(0, color=sg.MUTED, lw=1, ls="--")
            ax.grid(color=sg.GRID, lw=0.8)
            ax.tick_params(colors=sg.MUTED, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(sg.GRID)
            if row == 0:
                ax.set_title(name, color=sg.INK, fontsize=11, loc="left")
            if d == 0:
                ax.set_ylabel(f"exp(H): {label}", color=sg.MUTED, fontsize=9)
            if row == 2:
                ax.set_xlabel("steps from the grasp descent's bottom", color=sg.MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=sg.INK)
    fig.suptitle(
        f"Mean exp(H) per step, filtered by where the no-motion bin ranks "
        f"(points with fewer than {min_runs} qualifying runs omitted)",
        color=sg.INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=sg.SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks = json.loads((args.run_dir / "instrument_checks.json").read_text())
    zero_bins = [checks["per_dim"][str(d)]["raw_zero_bin"] for d in range(len(MOTION_DIMS))]
    loaded = load(args.run_dir, args.out / "anchors.csv", zero_bins)
    results = {name: {m: test(loaded, (name, m)) for m in METRICS} for name in MOTION_DIMS}

    def fmt(r):
        return (
            f"{r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] "
            f"({r['n']['failure']}/{r['n']['success']})"
        )

    lines = [
        "# No-motion check (exploratory, after the gate)",
        "",
        f"No-motion bins (x..yaw): {zero_bins}. Same-task AUROC, failures above successes, "
        f"{sg.EXPLORATORY_LEVEL:.2%} CI, bootstrap over episodes, {sg.N_BOOT} draws, seed "
        f"{sg.SEED}; (n failures/successes with qualifying steps). Harness "
        f"`{(metadata.git_state(sg.REPO)['sha'] or '')[:7]}`.",
        "",
        "| Dim | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
        *(
            f"| {name} | " + " | ".join(fmt(results[name][m]) for m in METRICS) + " |"
            for name in MOTION_DIMS
        ),
    ]
    flat = {f"{n}: {m}": r for n, ms in results.items() for m, r in ms.items()}
    (args.out / "stop_check.json").write_text(
        json.dumps({"zero_bins": zero_bins, "results": flat}, indent=2) + "\n"
    )
    (args.out / "stop_check.md").write_text("\n".join(lines) + "\n")
    plot(loaded, args.out / "stop_curves.png")
    plot_filtered(loaded, args.out / "stop_filtered_curves.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
