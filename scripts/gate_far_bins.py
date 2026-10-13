"""Exploratory Block 4 feature on the baseline: far-bin mass, the probability a step puts on bins
at least 10 away from its top bin (a genuinely different action), per dim, over the pre-grasp
window; on all steps and on steps where the no-motion bin is not in the top two. Also whether it
separates beyond exp(H): its AUROC after removing its linear dependence on window exp(H).

Reads <out>/anchors.csv; writes <out>/far_bins.json, far_bins.md and far_bins_curves.png.
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

FAR_BINS = 10
LENGTHS = (40, 50, 75)
N_BOOT = 2000
MOTION = len(sg.DIM_NAMES) - 1
VARIANTS = ("all steps", "stop not in top two")


def load(run_dir: Path, anchors_csv: Path, zero_bins):
    with open(anchors_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded"]]
    out, cached = [], None
    for r in rows:
        ep = r["episode_id"]
        if cached is None or cached[0] != ep:
            a, _ = schema.read_episode(run_dir / "episodes" / f"{ep}.h5")
            acting = ~a["is_wait_step"]
            shape = a["slice_logits"].shape[:2]
            exp_h, far, rank = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, 2)
            exp_h[acting] = derive_step_metrics(
                a["slice_logits"][acting],
                a["lse_full"][acting],
                a["argmax_full_id"][acting],
                a["chosen_token_id"][acting],
            )["exp_h"]
            probs = gate.bin_probs(a["slice_logits"][acting])
            far[acting] = gate.far_mass(probs, FAR_BINS)
            for d in range(MOTION):
                rank[acting, d] = gate.stop_rank(probs[:, d], zero_bins[d])
            cached = (ep, a["is_wait_step"], exp_h, far, rank)
        out.append({"row": r, "steps": cached[1:]})
    return out


def values(u, d: int, length: int, variant: str, metric: str):
    is_wait, exp_h, far, rank = u["steps"]
    window = gate.pre_window(
        is_wait, int(u["row"]["anchor_step"]), length, gate.DEFAULT_PARAMS.min_window
    )
    keep = rank[:, d] == 2 if variant == VARIANTS[1] else np.ones(len(rank), bool)
    series = far if metric == "far" else exp_h
    return gate.subset_mean(series[:, d], window, keep)


def to_units(loaded, vals) -> list[gate.Unit]:
    return [
        gate.Unit(
            u["row"]["episode_id"],
            int(u["row"]["task_id"]),
            u["row"]["object"],
            u["row"]["group"],
            v,
        )
        for u, v in zip(loaded, vals, strict=True)
        if v is not None
    ]


def ci(units) -> dict:
    lo, hi = gate.bootstrap_ci(units, gate.same_task_auroc, sg.EXPLORATORY_LEVEL, N_BOOT, sg.SEED)
    return {"auroc": gate.same_task_auroc(units), "ci": [lo, hi]}


def analyse(loaded) -> dict:
    out = {}
    for d, name in enumerate(sg.DIM_NAMES):
        for variant in VARIANTS:
            if variant == VARIANTS[1] and d >= MOTION:
                continue
            for n in LENGTHS:
                far = [values(u, d, n, variant, "far") for u in loaded]
                h = [values(u, d, n, variant, "exp_h") for u in loaded]
                both = [(f, e) for f, e in zip(far, h, strict=True) if f is not None]
                f_arr, h_arr = np.array(both).T
                residual = iter(gate.residualize(f_arr, h_arr))
                resid = [None if f is None else float(next(residual)) for f in far]
                out[f"{name} | {variant} | last {n}"] = {
                    "far": ci(to_units(loaded, far)),
                    "far_beyond_exp_h": ci(to_units(loaded, resid)),
                    "pearson_r_far_exp_h": float(np.corrcoef(f_arr, h_arr)[0, 1]),
                    "mean_far": {
                        g: float(
                            np.mean(
                                [
                                    v
                                    for u, v in zip(loaded, far, strict=True)
                                    if v is not None and u["row"]["group"] == g
                                ]
                            )
                        )
                        for g in ("failure", "success")
                    },
                }
    return out


def plot(loaded, path: Path, min_runs: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = np.arange(-sg.CURVE_BEFORE, sg.CURVE_AFTER + 1)
    fig, axes = plt.subplots(2, 7, figsize=(22, 6.5), sharex=True, facecolor=sg.SURFACE)
    for row, variant in enumerate(VARIANTS):
        for d, name in enumerate(sg.DIM_NAMES):
            ax = axes[row, d]
            ax.set_facecolor(sg.SURFACE)
            if row == 1 and d >= MOTION:
                ax.axis("off")
                continue
            for group in ("success", "failure"):
                us = [u for u in loaded if u["row"]["group"] == group]
                curves, oks = [], []
                for u in us:
                    is_wait, _, far, rank = u["steps"]
                    steps = int(u["row"]["anchor_step"]) + span
                    valid = (steps >= 0) & (steps < len(is_wait))
                    c, ok = np.full(len(span), np.nan), np.zeros(len(span), bool)
                    c[valid] = far[steps[valid], d]
                    ok[valid] = ~is_wait[steps[valid]]
                    if row == 1:
                        ok[valid] &= rank[steps[valid], d] == 2
                    curves.append(np.where(ok, c, 0.0))
                    oks.append(ok)
                n = np.sum(oks, axis=0)
                mean = np.sum(curves, axis=0) / np.maximum(n, 1)
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
                ax.set_ylabel(f"far-bin mass: {variant}", color=sg.MUTED, fontsize=9)
            if row == 1 or d >= MOTION:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel("steps from the grasp descent's bottom", color=sg.MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, labelcolor=sg.INK)
    fig.suptitle(
        f"Mean probability on bins at least {FAR_BINS} from the top bin, aligned on the grasp "
        f"descent (points with fewer than {min_runs} qualifying runs omitted)",
        color=sg.INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=sg.SURFACE)
    plt.close(fig)


def render(results: dict, sha: str) -> str:
    def cell(r):
        mark = "*" if r["ci"][0] > 0.5 or r["ci"][1] < 0.5 else ""
        return f"{r['auroc']:.2f}{mark}"

    lines = [
        "# Far-bin mass (exploratory, Block 4)",
        "",
        f"Probability on bins at least {FAR_BINS} from the top bin, window mean. Same-task "
        f"AUROC, failures above successes, {sg.EXPLORATORY_LEVEL:.2%} CIs, bootstrap over "
        f"episodes, {N_BOOT} draws, seed {sg.SEED}. Beyond exp(H): far-bin mass minus its "
        f"least-squares line on the window's exp(H). Harness `{sha[:7]}`.",
    ]
    for variant in VARIANTS:
        lines += [
            "",
            f"**{variant}**",
            "",
            "| Dim | window | mean far mass (fail / succ) | r with exp(H) | AUROC "
            "| beyond exp(H) |",
            "|---|---|---|---|---|---|",
        ]
        for key, r in results.items():
            name, v, window = key.split(" | ")
            if v != variant:
                continue
            m = r["mean_far"]
            lines.append(
                f"| {name} | {window} | {m['failure']:.3f} / {m['success']:.3f} "
                f"| {r['pearson_r_far_exp_h']:.2f} | {cell(r['far'])} "
                f"| {cell(r['far_beyond_exp_h'])} |"
            )
    lines += ["", "\\* CI excludes 0.5. CIs: `far_bins.json`."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checks = json.loads((args.run_dir / "instrument_checks.json").read_text())
    zero_bins = [checks["per_dim"][str(d)]["raw_zero_bin"] for d in range(MOTION)]
    loaded = load(args.run_dir, args.out / "anchors.csv", zero_bins)
    results = analyse(loaded)
    sha = metadata.git_state(sg.REPO)["sha"] or ""
    (args.out / "far_bins.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "far_bins.md").write_text(render(results, sha))
    plot(loaded, args.out / "far_bins_curves.png")
    print(render(results, sha))


if __name__ == "__main__":
    main()
