"""Runs the separability gate from <anchors>/anchors.csv (`scripts/gate_anchors.py`).

Writes gate_results.json, gate_report.md and curves.png to --out. Pre-registration and pass
rule: `docs/separability_gate.md`.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import gate, metadata, schema
from harness.pilot_report import derive_step_metrics

REPO = Path(__file__).resolve().parent.parent
DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
PRIMARY = "gripper exp(H)"
EXPLORATORY = tuple(f"{d} exp(H)" for d in DIM_NAMES[:6])
SECONDARY = "gripper near-tie rate"
PRIMARY_LEVEL = 0.95
EXPLORATORY_LEVEL = 1 - 0.05 / len(EXPLORATORY)
CURVE_BEFORE, CURVE_AFTER = 50, 20
N_BOOT, SEED = 10_000, 0
COLORS = {"failure": "#eb6834", "success": "#2a78d6"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def load(run_dir: Path, anchors_csv: Path, params=gate.DEFAULT_PARAMS):
    """Per kept unit: its window statistics and the aligned per-step exp(H) for the curves."""
    with open(anchors_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded"]]
    span = np.arange(-CURVE_BEFORE, CURVE_AFTER + 1)
    out, cached = [], None
    for r in rows:
        ep = r["episode_id"]
        if cached is None or cached[0] != ep:
            arrays, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
            m = derive_step_metrics(
                arrays["slice_logits"],
                arrays["lse_full"],
                arrays["argmax_full_id"],
                arrays["chosen_token_id"],
            )
            cached = (ep, arrays["is_wait_step"], m["exp_h"], m["bin_margin_ulps"])
        _, is_wait, exp_h, margins = cached
        anchor = int(r["anchor_step"])
        window = gate.pre_window(is_wait, anchor, params.window, params.min_window)
        steps = anchor + span
        valid = (steps >= 0) & (steps < len(is_wait))
        valid[valid] &= ~is_wait[steps[valid]]
        curve = np.full((len(span), len(DIM_NAMES)), np.nan)
        curve[valid] = exp_h[steps[valid]]
        values = {f"{d} exp(H)": float(exp_h[window, k].mean()) for k, d in enumerate(DIM_NAMES)}
        values[SECONDARY] = gate.near_tie_rate(
            margins[:, gate.GRIPPER_DIM], window, params.near_tie_ulps
        )
        out.append({"row": r, "values": values, "curve": curve})
    return out


def test(loaded, metric: str, level: float) -> dict:
    units = [
        gate.Unit(
            u["row"]["episode_id"],
            int(u["row"]["task_id"]),
            u["row"]["object"],
            u["row"]["group"],
            u["values"][metric],
        )
        for u in loaded
    ]
    result = {"level": level}
    for name, stat in (("same_task", gate.same_task_auroc), ("pooled", gate.pooled_auroc)):
        lo, hi = gate.bootstrap_ci(units, stat, level, N_BOOT, SEED)
        result[name] = {"auroc": stat(units), "ci": [lo, hi], "excludes_half": lo > 0.5 or hi < 0.5}
    return result


def plot(loaded, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = np.arange(-CURVE_BEFORE, CURVE_AFTER + 1)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, facecolor=SURFACE)
    for k, ax in enumerate(axes.flat):
        if k == len(DIM_NAMES):
            ax.axis("off")
            continue
        ax.set_facecolor(SURFACE)
        for group in ("success", "failure"):
            curves = np.stack([u["curve"][:, k] for u in loaded if u["row"]["group"] == group])
            n = np.sum(~np.isnan(curves), axis=0)
            mean = np.nanmean(curves, axis=0)
            lo, hi = np.nanpercentile(curves, [25, 75], axis=0)
            label = f"{group}s (n = {len(curves)})"
            ax.fill_between(span, lo, hi, color=COLORS[group], alpha=0.15, linewidth=0)
            ax.plot(span, np.where(n > 0, mean, np.nan), color=COLORS[group], lw=2, label=label)
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.set_title(DIM_NAMES[k], color=INK, fontsize=11, loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if k % 4 == 0:
            ax.set_ylabel("exp(H), effective bins", color=MUTED, fontsize=9)
        if k >= 3:
            ax.set_xlabel("steps from the grasp descent's bottom", color=MUTED, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, labelcolor=INK)
    fig.suptitle(
        "Per-step exp(H) aligned on the grasp descent: mean and interquartile band",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def render(results: dict) -> str:
    def line(metric, r):
        st, po = r["same_task"], r["pooled"]
        return (
            f"| {metric} | {r['level']:.4f} | {st['auroc']:.3f} [{st['ci'][0]:.3f}, "
            f"{st['ci'][1]:.3f}] | {po['auroc']:.3f} [{po['ci'][0]:.3f}, {po['ci'][1]:.3f}] |"
        )

    n = results["n_units"]
    lines = [
        "# Separability gate",
        "",
        f"Units: {n['failure']} failures, {n['success']} success grasps. Same-task AUROC is "
        "decisive; pooled is reported beside it. Bootstrap over episodes within (task, outcome), "
        f"{N_BOOT} draws, seed {SEED}. Harness `{results['git']['sha'][:7]}`.",
        "",
        "| Metric | CI level | Same-task AUROC [CI] | Pooled AUROC [CI] |",
        "|---|---|---|---|",
        line(f"**{PRIMARY}** (primary)", results["primary"]),
        *(line(m, r) for m, r in results["exploratory"].items()),
        line(f"{SECONDARY} (secondary)", results["secondary"]),
        "",
        f"**Gate: {'pass' if results['gate_pass'] else 'fail'}.** "
        f"P1 (failures higher on {PRIMARY}): "
        f"{'supported' if results['p1_supported'] else 'not supported'}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--anchors", type=Path, help="default <out>/anchors.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    anchors = args.anchors or args.out / "anchors.csv"
    loaded = load(args.run_dir, anchors)
    primary = test(loaded, PRIMARY, PRIMARY_LEVEL)
    exploratory = {m: test(loaded, m, EXPLORATORY_LEVEL) for m in EXPLORATORY}
    results = {
        "git": metadata.git_state(REPO),
        "params": vars(gate.DEFAULT_PARAMS) | {"n_boot": N_BOOT, "seed": SEED},
        "n_units": {g: sum(u["row"]["group"] == g for u in loaded) for g in ("failure", "success")},
        "primary": primary,
        "exploratory": exploratory,
        "secondary": test(loaded, SECONDARY, PRIMARY_LEVEL),
        "p1_supported": primary["same_task"]["ci"][0] > 0.5,
        "gate_pass": primary["same_task"]["excludes_half"]
        or any(r["same_task"]["excludes_half"] for r in exploratory.values()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "gate_report.md").write_text(render(results))
    plot(loaded, args.out / "curves.png")
    print(render(results))


if __name__ == "__main__":
    main()
