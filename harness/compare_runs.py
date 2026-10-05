"""Bitwise comparison of two runs, matching episodes by (task, init state) and ignoring seed."""

from pathlib import Path

import numpy as np

from harness import manifest, schema

_STREAMS = {
    "action": "executed_action",
    "token": "chosen_token_id",
    "logit": "slice_logits",
    "state": "sim_state",
}


def _mismatch_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row inequality over the common prefix; missing-value sentinels (NaN) compare equal."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    same = a == b
    if a.dtype.kind == "f":
        same |= np.isnan(a) & np.isnan(b)
    return ~same.reshape(n, -1).all(axis=1)


def _first(mask: np.ndarray) -> int | None:
    return int(np.flatnonzero(mask)[0]) if mask.any() else None


def _compare_episode(path_a: Path, path_b: Path) -> dict:
    arrays_a, _ = schema.read_episode(path_a)
    arrays_b, _ = schema.read_episode(path_b)
    length_a, length_b = len(arrays_a["step_idx"]), len(arrays_b["step_idx"])

    first = {
        name: _first(_mismatch_rows(arrays_a[key], arrays_b[key])) for name, key in _STREAMS.items()
    }
    first_action = min(
        (s for s in (first["action"], first["token"]) if s is not None), default=None
    )
    return {
        "length_a": length_a,
        "length_b": length_b,
        "first_action_divergence": first_action,
        "first_logit_divergence": first["logit"],
        "first_state_divergence": first["state"],
        "actions_identical": first_action is None,
        "logits_identical": first["logit"] is None,
        "states_identical": first["state"] is None,
        "identical": length_a == length_b
        and all(step is None for step in (first_action, first["logit"], first["state"])),
    }


def _by_key(rows: list[dict]) -> dict[tuple[int, int], dict]:
    return {(r["task_id"], r["init_state_idx"]): r for r in rows}


def compare_runs(run_a, run_b) -> dict:
    run_a, run_b = Path(run_a), Path(run_b)
    rows_a = _by_key(manifest.read_rows(run_a / "manifest.csv"))
    rows_b = _by_key(manifest.read_rows(run_b / "manifest.csv"))
    matched = sorted(rows_a.keys() & rows_b.keys())

    episodes = []
    for key in matched:
        ra, rb = rows_a[key], rows_b[key]
        detail = _compare_episode(run_a / ra["episode_file"], run_b / rb["episode_file"])
        episodes.append(
            {
                "task_id": key[0],
                "init_state_idx": key[1],
                "id_a": ra["episode_id"],
                "id_b": rb["episode_id"],
                "success_a": ra["success"],
                "success_b": rb["success"],
                "outcome_flip": ra["success"] != rb["success"],
                **detail,
            }
        )
    return {
        "n_matched": len(episodes),
        "n_identical": sum(e["identical"] for e in episodes),
        "n_action_divergent": sum(not e["actions_identical"] for e in episodes),
        "n_outcome_flips": sum(e["outcome_flip"] for e in episodes),
        "only_in_a": sorted(rows_a.keys() - rows_b.keys()),
        "only_in_b": sorted(rows_b.keys() - rows_a.keys()),
        "episodes": episodes,
    }


def _cell(step) -> str:
    return "-" if step is None else str(step)


def render_markdown(report: dict) -> str:
    lines = [
        "# Run comparison",
        "",
        f"{report['n_matched']} matched episodes: {report['n_identical']} identical, "
        f"{report['n_action_divergent']} with diverging actions, "
        f"{report['n_outcome_flips']} outcome flips. "
        f"Unmatched: {len(report['only_in_a'])} only in A, {len(report['only_in_b'])} only in B.",
        "",
        "| task | init | A | B | identical | first action | first logit | first state | "
        "steps A/B | success A/B |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {e['task_id']} | {e['init_state_idx']} | {e['id_a']} | {e['id_b']} "
        f"| {'identical' if e['identical'] else 'differs'} "
        f"| {_cell(e['first_action_divergence'])} | {_cell(e['first_logit_divergence'])} "
        f"| {_cell(e['first_state_divergence'])} | {e['length_a']}/{e['length_b']} "
        f"| {e['success_a']}/{e['success_b']} |"
        for e in report["episodes"]
    )
    return "\n".join(lines) + "\n"
