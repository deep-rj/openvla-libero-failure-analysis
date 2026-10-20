"""Pooled premise-check statistics from logged episodes.

Everything is pooled over non-wait steps and reported per action dimension, with no split
by outcome or phase and no per-episode or time-aligned output, so it can be produced before
the hand labels exist. Entropy is over the 255-bin distribution: the softmax of the 256
slice logits renormalized within the slice, with tokens 31744 and 31745 merged into bin 254.
"""

import json
from pathlib import Path

import numpy as np

from harness import manifest, schema, summary
from harness.decode import ACTION_SLICE_START, N_BINS, VOCAB_SIZE

# Fixed before any data was seen.
EXP_H_CUTOFFS = (1.05, 1.5, 2.0)
MARGIN_ULP_CUTOFFS = (1, 4, 16)
# Above fp32/fp64 summation error; a larger slice lse means lse_full is from another pass.
LSE_TOL = 1e-5
QUANTILES = (5, 25, 50, 75, 95)
DISTANCE_CLASSES = ("adjacent", "2-9", ">=10")
_ANOMALY_KEYS = ("t1", "t2", "t3", "t5", "t3_parity", "truncation")
_METRIC_KEYS = (
    "argmax_text",
    "argmax_padding",
    "chosen_edge",
    "oos_mass",
    "slice_exceeds_full",
    "exp_h",
    "token_margin_ulps",
    "bin_margin_ulps",
    "bin_distance",
)


def bf16_ulp(x) -> np.ndarray:
    """Spacing of bf16 values in the binade of `x` (8 significant bits)."""
    magnitude = np.abs(np.asarray(x, dtype=np.float64))
    _, exponent = np.frexp(magnitude)
    return np.where(magnitude == 0, 2.0**-133, 2.0 ** (exponent - 8.0))


def _top2(values):
    """Largest and second-largest along the last axis, and the index of each."""
    order = np.argsort(values, axis=-1)
    first, second = order[..., -1], order[..., -2]
    pick = lambda idx: np.take_along_axis(values, idx[..., None], axis=-1)[..., 0]  # noqa: E731
    return pick(first), pick(second), first, second


def lse_excess(slice_logits, lse_full) -> np.ndarray:
    """log-sum-exp over the slice minus over the full head, in float64; positive is impossible."""
    x = np.asarray(slice_logits, dtype=np.float64)
    peak = x.max(axis=-1, keepdims=True)
    lse_slice = peak[..., 0] + np.log(np.exp(x - peak).sum(axis=-1))
    return lse_slice - np.asarray(lse_full, dtype=np.float64)


def merged_bin_logits(slice_logits) -> np.ndarray:
    """Per-bin logits [..., 255] in float32; bin i is token 31999 - i."""
    reversed_tokens = np.asarray(slice_logits, dtype=np.float32)[..., ::-1]
    bins = reversed_tokens[..., :N_BINS].copy()
    # The last two slice tokens both decode to bin 254.
    bins[..., N_BINS - 1] = np.logaddexp(
        reversed_tokens[..., N_BINS - 1], reversed_tokens[..., N_BINS]
    )
    return bins


def derive_step_metrics(slice_logits, lse_full, argmax_full_id, chosen_token_id) -> dict:
    """Per-step, per-dimension quantities from logged logits, computed in float32."""
    x = np.asarray(slice_logits, dtype=np.float32)
    excess = lse_excess(x, lse_full)
    share_of_full_head = np.exp(np.minimum(excess, 0.0))

    bins = merged_bin_logits(x)
    weights = np.exp(bins - bins.max(axis=-1, keepdims=True))
    p = weights / weights.sum(axis=-1, keepdims=True)
    entropy = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=-1)

    tok1, tok2, _, _ = _top2(x)
    bin1, bin2, bin_first, bin_second = _top2(bins)
    argmax = np.asarray(argmax_full_id)
    return {
        "argmax_text": argmax < ACTION_SLICE_START,
        "argmax_padding": argmax >= VOCAB_SIZE,
        "chosen_edge": np.asarray(chosen_token_id) == ACTION_SLICE_START,
        "oos_mass": 1.0 - share_of_full_head,
        "slice_exceeds_full": excess > LSE_TOL,
        "exp_h": np.exp(entropy),
        "token_margin_ulps": (tok1 - tok2) / bf16_ulp(tok1),
        "bin_margin_ulps": (bin1 - bin2) / bf16_ulp(bin1),
        "bin_distance": np.abs(bin_first - bin_second),
    }


def pool(metrics_list: list[dict]) -> dict:
    return {k: np.concatenate([m[k] for m in metrics_list]) for k in _METRIC_KEYS}


def _distribution(values, cutoffs=()) -> dict:
    out = {
        "mean": float(values.mean()),
        "quantiles": {str(q): float(np.percentile(values, q)) for q in QUANTILES},
    }
    if len(cutoffs):
        out["fraction_at_most"] = {str(c): float((values <= c).mean()) for c in cutoffs}
    return out


def _distance_class(distance) -> np.ndarray:
    return np.where(distance == 1, 0, np.where(distance <= 9, 1, 2))


def _per_dim(m: dict, d: int) -> dict:
    n = len(m["exp_h"])
    classes = _distance_class(m["bin_distance"][:, d])
    crosstab = {}
    for cutoff in MARGIN_ULP_CUTOFFS:
        under = m["bin_margin_ulps"][:, d] <= cutoff
        crosstab[str(cutoff)] = {
            name: int((under & (classes == k)).sum()) for k, name in enumerate(DISTANCE_CLASSES)
        }
    return {
        "n_steps": n,
        "argmax_outside_slice": {
            "text_fraction": float(m["argmax_text"][:, d].mean()),
            "padding_fraction": float(m["argmax_padding"][:, d].mean()),
            "chosen_edge_fraction": float(m["chosen_edge"][:, d].mean()),
        },
        "oos_mass": {
            **_distribution(m["oos_mass"][:, d]),
            "max": float(m["oos_mass"][:, d].max()),
            "slice_exceeds_full_steps": int(m["slice_exceeds_full"][:, d].sum()),
        },
        "exp_h": _distribution(m["exp_h"][:, d], EXP_H_CUTOFFS),
        "token_margin_ulps": _distribution(m["token_margin_ulps"][:, d], MARGIN_ULP_CUTOFFS),
        "bin_margin_ulps": _distribution(m["bin_margin_ulps"][:, d], MARGIN_ULP_CUTOFFS),
        "near_tie_bin_distance": crosstab,
    }


def summarize_pooled(metrics: dict) -> dict:
    return {"per_dim": {d: _per_dim(metrics, d) for d in range(metrics["exp_h"].shape[1])}}


def build_report(run_dir, validation_report: dict | None = None, episode_ids=None) -> dict:
    """Pools every logged episode of a run (or only `episode_ids`) into one report."""
    run_dir = Path(run_dir)
    rows = manifest.read_rows(run_dir / "manifest.csv")
    if episode_ids is not None:
        rows = [r for r in rows if r["episode_id"] in episode_ids]

    outcomes: dict[str, dict] = {}
    for row in rows:
        task = outcomes.setdefault(str(row["task_id"]), {"successes": 0, "failures": 0})
        task["successes" if row["success"] else "failures"] += 1

    metrics, n_steps = [], 0
    for row in rows:
        if not row["episode_file"]:
            continue
        arrays, _ = schema.read_episode(run_dir / row["episode_file"])
        acting = ~arrays["is_wait_step"]
        metrics.append(
            derive_step_metrics(
                arrays["slice_logits"][acting],
                arrays["lse_full"][acting],
                arrays["argmax_full_id"][acting],
                arrays["chosen_token_id"][acting],
            )
        )
        n_steps += int(acting.sum())

    successes = sum(r["success"] for r in rows)
    validation = validation_report or {}
    return {
        "n_episodes": len(metrics),
        "n_steps_pooled": n_steps,
        "outcomes_per_task": outcomes,
        "suite_success": {
            "n": len(rows),
            "successes": successes,
            "ci95": summary.wilson_interval(successes, len(rows)),
        },
        "cutoffs": {
            "exp_h": list(EXP_H_CUTOFFS),
            "margin_ulps": list(MARGIN_ULP_CUTOFFS),
            "quantiles": list(QUANTILES),
        },
        "pooled": summarize_pooled(pool(metrics)),
        "anomalies": {k: validation[k] for k in _ANOMALY_KEYS if k in validation},
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Pilot report",
        "",
        f"Pooled over {report['n_steps_pooled']} non-wait steps from {report['n_episodes']} "
        "episodes; per action dimension; no split by outcome or phase.",
        "",
        "## Outcomes per task",
        "",
        "| task | successes | failures |",
        "|---|---|---|",
    ]
    lines += [
        f"| {task} | {c['successes']} | {c['failures']} |"
        for task, c in sorted(report["outcomes_per_task"].items(), key=lambda kv: int(kv[0]))
    ]
    lines += [
        "",
        "## Out-of-slice premise check",
        "",
        "| dim | argmax text | argmax padding | chose 31744 | mass median | mass p95 | mass max "
        "| slice lse > full |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d, s in report["pooled"]["per_dim"].items():
        o, m = s["argmax_outside_slice"], s["oos_mass"]
        lines.append(
            f"| {d} | {o['text_fraction']:.4f} | {o['padding_fraction']:.4f} "
            f"| {o['chosen_edge_fraction']:.4f} | {m['quantiles']['50']:.4g} "
            f"| {m['quantiles']['95']:.4g} | {m['max']:.4g} | {m['slice_exceeds_full_steps']} |"
        )
    lines += ["", "## Effective bins exp(H)", ""]
    header = "| dim | q5 | q25 | q50 | q75 | q95 | " + " | ".join(
        f"≤{c}" for c in report["cutoffs"]["exp_h"]
    )
    lines += [header + " |", "|" + "---|" * (6 + len(report["cutoffs"]["exp_h"]))]
    for d, s in report["pooled"]["per_dim"].items():
        e = s["exp_h"]
        qs = " | ".join(f"{e['quantiles'][str(q)]:.3g}" for q in QUANTILES)
        fr = " | ".join(f"{e['fraction_at_most'][str(c)]:.3f}" for c in EXP_H_CUTOFFS)
        lines.append(f"| {d} | {qs} | {fr} |")
    for name, title in (("token_margin_ulps", "token"), ("bin_margin_ulps", "bin")):
        lines += ["", f"## Top-2 {title} margin, fraction at most N ulps", ""]
        lines += ["| dim | " + " | ".join(f"≤{c}" for c in MARGIN_ULP_CUTOFFS) + " |"]
        lines += ["|" + "---|" * (1 + len(MARGIN_ULP_CUTOFFS))]
        for d, s in report["pooled"]["per_dim"].items():
            fr = " | ".join(
                f"{s[name]['fraction_at_most'][str(c)]:.4f}" for c in MARGIN_ULP_CUTOFFS
            )
            lines.append(f"| {d} | {fr} |")
    lines += ["", "## Near ties by distance between the top two bins", ""]
    lines += ["| dim | cutoff (ulps) | adjacent | 2-9 | >=10 |", "|---|---|---|---|---|"]
    for d, s in report["pooled"]["per_dim"].items():
        for cutoff, counts in s["near_tie_bin_distance"].items():
            row = " | ".join(str(counts[name]) for name in DISTANCE_CLASSES)
            lines.append(f"| {d} | {cutoff} | {row} |")
    lines += ["", "## Validation-run anomalies", ""]
    anomalies = report["anomalies"]
    if not anomalies:
        lines.append("- none supplied")
    for name, value in anomalies.items():
        lines.append(f"- {name}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def write_outputs(run_dir, validation_report_path=None, episode_ids=None) -> dict:
    run_dir = Path(run_dir)
    validation = json.loads(Path(validation_report_path).read_text()) if validation_report_path else None
    report = build_report(run_dir, validation, episode_ids)
    (run_dir / "pilot_report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "pilot_report.md").write_text(render_markdown(report))
    return report
