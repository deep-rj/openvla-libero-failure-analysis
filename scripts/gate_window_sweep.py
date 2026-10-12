"""Exploratory window sweep after the gate: same-task AUROC of mean exp(H) per dim over windows
ending at the grasp descent's bottom (last 10..50 steps) and over disjoint 10-step blocks.
Each on all steps, and on motion dims also on steps where the no-motion bin is not in the top two.

Reads <out>/anchors.csv; writes <out>/window_sweep.json, window_sweep.md, window_sweep.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_stop_check as sc
import separability_gate as sg

from harness import gate, metadata

LENGTHS = (10, 20, 30, 40, 50)
BLOCK = 10
BLOCK_MIN = 5  # non-wait steps a 10-step block needs
N_BOOT = 2000
VARIANTS = {"all steps": "#1f1f1e", "stop not in top two": "#1baf7a"}


def windows(kind: str):
    """(label, steps before the anchor where the window ends, length, minimum length)."""
    if kind == "cumulative":
        return [(f"last {n}", 0, n, min(n, gate.DEFAULT_PARAMS.min_window)) for n in LENGTHS]
    return [(f"{-e - BLOCK} to {-e}", e, BLOCK, BLOCK_MIN) for e in (40, 30, 20, 10, 0)]


def unit_value(u, d: int, end: int, length: int, min_length: int, variant: str):
    is_wait, exp_h, rank = u["steps"]
    window = gate.pre_window(is_wait, int(u["row"]["anchor_step"]) - end, length, min_length)
    if window is None:
        return None
    keep = rank[:, d] == 2 if variant == "stop not in top two" else np.ones(len(rank), bool)
    return gate.subset_mean(exp_h[:, d], window, keep)


def sweep(loaded) -> dict:
    out = {}
    for kind in ("cumulative", "blocks"):
        for label, end, length, min_length in windows(kind):
            for d, name in enumerate(sg.DIM_NAMES):
                for variant in VARIANTS:
                    if variant != "all steps" and d >= len(sc.MOTION_DIMS):
                        continue
                    units = []
                    for u in loaded:
                        v = unit_value(u, d, end, length, min_length, variant)
                        if v is not None:
                            r = u["row"]
                            units.append(
                                gate.Unit(
                                    r["episode_id"], int(r["task_id"]), r["object"], r["group"], v
                                )
                            )
                    lo, hi = gate.bootstrap_ci(
                        units, gate.same_task_auroc, sg.EXPLORATORY_LEVEL, N_BOOT, sg.SEED
                    )
                    out[f"{kind} | {label} | {name} | {variant}"] = {
                        "auroc": gate.same_task_auroc(units),
                        "ci": [lo, hi],
                        "n": {g: sum(x.group == g for x in units) for g in ("failure", "success")},
                    }
    return out


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 7, figsize=(20, 6.5), sharey=True, facecolor=sg.SURFACE)
    markers = {"all steps": ("o", "-"), "stop not in top two": ("s", "--")}
    for row, kind in enumerate(("cumulative", "blocks")):
        labels = [w[0] for w in windows(kind)]
        for d, name in enumerate(sg.DIM_NAMES):
            ax = axes[row, d]
            ax.set_facecolor(sg.SURFACE)
            for k, (variant, color) in enumerate(VARIANTS.items()):
                keys = [f"{kind} | {lab} | {name} | {variant}" for lab in labels]
                if keys[0] not in results:
                    continue
                x = np.arange(len(labels)) + (k - 0.5) * 0.18
                y = np.array([results[key]["auroc"] for key in keys])
                ci = np.array([results[key]["ci"] for key in keys])
                marker, style = markers[variant]
                ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], color=color, lw=2,
                            ls=style, marker=marker, ms=8, capsize=0, elinewidth=1.2,
                            label=variant)  # fmt: skip
            ax.axhline(0.5, color=sg.MUTED, lw=1, ls=":")
            ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.grid(color=sg.GRID, lw=0.8, axis="y")
            ax.tick_params(colors=sg.MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(sg.GRID)
            if row == 0:
                ax.set_title(name, color=sg.INK, fontsize=11, loc="left")
            if d == 0:
                title = "windows ending at the grasp" if kind == "cumulative" else "10-step blocks"
                ax.set_ylabel(f"same-task AUROC\n{title}", color=sg.MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, labelcolor=sg.INK, fontsize=9, loc="lower left")
    fig.suptitle(
        f"Same-task AUROC (failures above successes) by window, with "
        f"{sg.EXPLORATORY_LEVEL:.2%} CIs",
        color=sg.INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=sg.SURFACE)
    plt.close(fig)


def render(results: dict, sha: str) -> str:
    lines = [
        "# Window sweep (exploratory, after the gate)",
        "",
        f"Same-task AUROC [{sg.EXPLORATORY_LEVEL:.2%} CI], bootstrap over episodes, {N_BOOT} "
        f"draws, seed {sg.SEED}. Blocks need {BLOCK_MIN} non-wait steps. Harness `{sha[:7]}`.",
    ]
    for kind in ("cumulative", "blocks"):
        labels = [w[0] for w in windows(kind)]
        for variant in VARIANTS:
            lines += ["", f"**{kind}, {variant}**", "", "| Dim | " + " | ".join(labels) + " |"]
            lines.append("|---|" + "---|" * len(labels))
            for name in sg.DIM_NAMES:
                keys = [f"{kind} | {lab} | {name} | {variant}" for lab in labels]
                if keys[0] not in results:
                    continue
                cells = []
                for key in keys:
                    r = results[key]
                    mark = "*" if r["ci"][0] > 0.5 or r["ci"][1] < 0.5 else ""
                    cells.append(f"{r['auroc']:.2f}{mark}")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "\\* CI excludes 0.5. CIs and n per cell: `window_sweep.json`."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks = json.loads((args.run_dir / "instrument_checks.json").read_text())
    zero_bins = [checks["per_dim"][str(d)]["raw_zero_bin"] for d in range(len(sc.MOTION_DIMS))]
    loaded = sc.load(args.run_dir, args.out / "anchors.csv", zero_bins)
    results = sweep(loaded)
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    (args.out / "window_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "window_sweep.md").write_text(render(results, sha))
    plot(results, args.out / "window_sweep.png")
    print(render(results, sha))


if __name__ == "__main__":
    main()
