"""Instrument-validity checks over a logged run: invariants that hold by construction.

Each check counts violations of something that cannot happen if the logits were captured,
stored and summarized correctly. Everything is pooled per action dimension; the only
per-episode output is the location of a violation, never a model-output value.
"""

import json
from pathlib import Path

import numpy as np

from harness import checks, decode, manifest, schema
from harness.pilot_report import derive_step_metrics, merged_bin_logits
from harness.rollout import NUM_STEPS_WAIT

LSE_TOL = 1e-5
EXP_H_RTOL = 1e-5
_N_EXAMPLES = 20
_EDGE_TOKENS = (0, 1)  # slice indices of tokens 31744 and 31745


def non_bf16(slice_logits) -> np.ndarray:
    """True where a stored logit has more significant bits than bf16 holds.

    fp16 subnormals are excluded: storing a tiny bf16 value as fp16 rounds it.
    """
    x = np.asarray(slice_logits, dtype=np.float16)
    bits = x.astype(np.float32).view(np.uint32)
    return ((bits & 0xFFFF) != 0) & (np.abs(x) >= np.finfo(np.float16).tiny)


def lse_excess(slice_logits, lse_full) -> np.ndarray:
    """log-sum-exp over the slice minus over the full head; positive is impossible."""
    x = np.asarray(slice_logits, dtype=np.float64)
    peak = x.max(axis=-1, keepdims=True)
    lse_slice = peak[..., 0] + np.log(np.exp(x - peak).sum(axis=-1))
    return lse_slice - np.asarray(lse_full, dtype=np.float64)


def bin_margin_nats(slice_logits) -> np.ndarray:
    top2 = np.sort(merged_bin_logits(slice_logits), axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


def entropy_bound_violations(exp_h, margin_nats, rtol=EXP_H_RTOL) -> np.ndarray:
    """A top-2 margin of d nats caps p_max at 1/(1 + e^-d), and exp(H) >= 1/p_max."""
    bound = 1.0 + np.exp(-np.asarray(margin_nats, dtype=np.float64))
    return np.asarray(exp_h, dtype=np.float64) < bound * (1.0 - rtol)


def merge_invariant(slice_logits, token_margin_ulps, bin_margin_ulps):
    """(covered, violated): the merged edge bin reaches at most max(edge logits) + ln 2."""
    x = np.asarray(slice_logits, dtype=np.float32)
    second = np.sort(x, axis=-1)[..., -2]
    edges = x[..., list(_EDGE_TOKENS)].max(axis=-1)
    covered = edges < second - np.log(2.0)
    return covered, covered & (token_margin_ulps != bin_margin_ulps)


def raw_action_bin(raw, q01, q99, mask) -> np.ndarray:
    """The bin OpenVLA's tokenizer assigns to a raw action (BOUNDS_Q99 normalization)."""
    raw, q01, q99 = (np.asarray(a, dtype=np.float64) for a in (raw, q01, q99))
    scaled = np.clip(2 * (raw - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)
    normalized = np.clip(np.where(mask, scaled, raw), -1, 1)
    digitized = np.digitize(normalized, np.linspace(-1, 1, decode.N_ACTION_BINS))
    return decode.token_to_bin(decode.VOCAB_SIZE - digitized)[0]


def occupancy(counts) -> dict:
    counts = np.asarray(counts)
    total = counts.sum()
    occupied = np.flatnonzero(counts)
    cumulative = np.cumsum(np.sort(counts)[::-1])
    span = occupied[-1] - occupied[0] + 1 if len(occupied) else 0
    top = np.argsort(counts)[::-1][:5]
    return {
        "occupied_bins": len(occupied),
        "bins_for_90_percent": int(np.searchsorted(cumulative, 0.9 * total) + 1),
        "empty_fraction_within_range": float(1 - len(occupied) / span) if span else 0.0,
        "top_bins": [
            {"bin": int(b), "fraction": float(counts[b] / total)} for b in top if counts[b]
        ],
    }


def _violations(messages: list[str]) -> dict:
    return {"count": len(messages), "examples": messages[:_N_EXAMPLES]}


def check_run(run_dir, lse_tol=LSE_TOL, exp_h_rtol=EXP_H_RTOL) -> dict:
    run_dir = Path(run_dir)
    n_dims = schema.N_DIMS
    counts = {
        k: np.zeros(n_dims, dtype=np.int64)
        for k in (
            "lse_excess_over_tol",
            "lse_clamped",
            "non_bf16_logits",
            "exp_h_out_of_range",
            "entropy_bound_violations",
            "merge_covered",
            "merge_violations",
        )
    }
    lse_excess_max = np.full(n_dims, -np.inf)
    exp_h_min, exp_h_max = np.full(n_dims, np.inf), np.full(n_dims, -np.inf)
    chosen_bins = np.zeros((n_dims, decode.N_BINS), dtype=np.int64)
    t3, t5, location = [], [], {}
    n_episodes = n_steps = 0
    stats = None

    for row in manifest.read_rows(run_dir / "manifest.csv"):
        if not row["episode_file"]:
            continue
        arrays, meta = schema.read_episode(run_dir / row["episode_file"])
        stats = {k: np.array(v) for k, v in meta["action_stats"].items()}
        sent = list(np.load(run_dir / "actions" / f"{row['episode_id']}.npy"))
        prefix = f"{row['episode_id']}: "
        t3 += [prefix + p for p in checks.check_self_consistency(arrays, stats, sent)]
        t5 += [prefix + p for p in checks.check_wait_alignment(arrays, NUM_STEPS_WAIT)]

        acting = ~arrays["is_wait_step"]
        x = arrays["slice_logits"][acting]
        m = derive_step_metrics(
            x,
            arrays["lse_full"][acting],
            arrays["argmax_full_id"][acting],
            arrays["chosen_token_id"][acting],
        )
        excess = lse_excess(x, arrays["lse_full"][acting])
        covered, violated = merge_invariant(x, m["token_margin_ulps"], m["bin_margin_ulps"])
        flags = {
            "lse_excess_over_tol": excess > lse_tol,
            "lse_clamped": excess > 0,
            "non_bf16_logits": non_bf16(x).sum(axis=-1),
            "exp_h_out_of_range": (m["exp_h"] < 1 - exp_h_rtol)
            | (m["exp_h"] > decode.N_BINS * (1 + exp_h_rtol)),
            "entropy_bound_violations": entropy_bound_violations(
                m["exp_h"], bin_margin_nats(x), exp_h_rtol
            ),
            "merge_covered": covered,
            "merge_violations": violated,
        }
        for name, flag in flags.items():
            counts[name] += flag.sum(axis=0)
        for name in ("lse_excess_over_tol", "entropy_bound_violations", "merge_violations"):
            steps = np.flatnonzero(flags[name].any(axis=-1))
            if len(steps) and name not in location:
                location[name] = f"{row['episode_id']} policy step {int(steps[0])}"

        lse_excess_max = np.maximum(lse_excess_max, excess.max(axis=0))
        exp_h_min = np.minimum(exp_h_min, m["exp_h"].min(axis=0))
        exp_h_max = np.maximum(exp_h_max, m["exp_h"].max(axis=0))
        for d in range(n_dims):
            chosen_bins[d] += np.bincount(arrays["bin_idx"][acting][:, d], minlength=decode.N_BINS)
        n_episodes += 1
        n_steps += int(acting.sum())

    zero_bins = (
        raw_action_bin(np.zeros(n_dims), stats["q01"], stats["q99"], stats["mask"])
        if stats is not None
        else np.full(n_dims, -1)
    )
    per_dim = {}
    for d in range(n_dims):
        total = chosen_bins[d].sum()
        per_dim[d] = {
            **{name: int(c[d]) for name, c in counts.items()},
            "lse_excess_max": float(lse_excess_max[d]),
            "exp_h_min": float(exp_h_min[d]),
            "exp_h_max": float(exp_h_max[d]),
            "chosen_bins": occupancy(chosen_bins[d]),
            "raw_zero_bin": int(zero_bins[d]),
            "raw_zero_bin_fraction": float(chosen_bins[d][zero_bins[d]] / total) if total else 0.0,
        }
    return {
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "tolerances": {"lse": lse_tol, "exp_h_rtol": exp_h_rtol},
        "per_dim": per_dim,
        "first_violation": location,
        "t3": _violations(t3),
        "t5": _violations(t5),
        "chosen_bin_counts": chosen_bins.tolist(),
    }


def render_markdown(report: dict) -> str:
    tol = report["tolerances"]
    lines = [
        "# Instrument checks",
        "",
        f"{report['n_steps']} non-wait steps from {report['n_episodes']} episodes. "
        f"Tolerances: lse {tol['lse']:g}, exp(H) relative {tol['exp_h_rtol']:g}.",
        "",
        "## Invariants (counts of violating steps; non-bf16 counts logits)",
        "",
        "| dim | lse > tol | lse > 0 | max lse excess | non-bf16 | exp(H) out of [1, 255] "
        "| exp(H) min | exp(H) max | entropy bound | merge covered | merge violations |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, s in report["per_dim"].items():
        lines.append(
            f"| {d} | {s['lse_excess_over_tol']} | {s['lse_clamped']} | {s['lse_excess_max']:.3g} "
            f"| {s['non_bf16_logits']} | {s['exp_h_out_of_range']} | {s['exp_h_min']:.6g} "
            f"| {s['exp_h_max']:.4g} | {s['entropy_bound_violations']} | {s['merge_covered']} "
            f"| {s['merge_violations']} |"
        )
    for name in ("t3", "t5"):
        v = report[name]
        lines += ["", f"## {name.upper()} over every episode: {v['count']} violations", ""]
        lines += [f"- {e}" for e in v["examples"]]
    if report["first_violation"]:
        lines += ["", "## First violation of each invariant", ""]
        lines += [f"- {k}: {v}" for k, v in report["first_violation"].items()]
    lines += [
        "",
        "## Chosen-bin occupancy",
        "",
        "| dim | occupied bins | bins for 90% | empty within range | bin of raw 0 "
        "| share on raw-0 bin | top bins (bin: share) |",
        "|---|---|---|---|---|---|---|",
    ]
    for d, s in report["per_dim"].items():
        o = s["chosen_bins"]
        top = ", ".join(f"{b['bin']}: {b['fraction']:.3f}" for b in o["top_bins"])
        lines.append(
            f"| {d} | {o['occupied_bins']} | {o['bins_for_90_percent']} "
            f"| {o['empty_fraction_within_range']:.3f} | {s['raw_zero_bin']} "
            f"| {s['raw_zero_bin_fraction']:.3f} | {top} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(run_dir) -> dict:
    run_dir = Path(run_dir)
    report = check_run(run_dir)
    (run_dir / "instrument_checks.json").write_text(json.dumps(report, indent=2))
    (run_dir / "instrument_checks.md").write_text(render_markdown(report))
    return report
