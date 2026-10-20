"""Block 4 evaluation metrics from per-attempt detector scores, so the held-out result is
recomputable from a committed scores file without the raw traces."""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from harness import detector, gate


@dataclass(frozen=True)
class AttemptScore:
    episode_id: str
    task_id: int
    object: str
    episode_success: str  # bootstrap stratum only
    failed: bool
    v1: float
    v2: float
    width_delay: int | None  # steps from close command to the width baseline's firing


def write_scores(path: Path, scores: list[AttemptScore]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[x.name for x in fields(AttemptScore)])
        w.writeheader()
        for s in scores:
            row = asdict(s)
            row["width_delay"] = "" if s.width_delay is None else s.width_delay
            row["v1"], row["v2"] = repr(s.v1), repr(s.v2)
            w.writerow(row)


def read_scores(path: Path) -> list[AttemptScore]:
    with open(path, newline="") as f:
        return [
            AttemptScore(
                episode_id=r["episode_id"],
                task_id=int(r["task_id"]),
                object=r["object"],
                episode_success=r["episode_success"],
                failed=r["failed"] == "True",
                v1=float(r["v1"]),
                v2=float(r["v2"]),
                width_delay=None if r["width_delay"] == "" else int(r["width_delay"]),
            )
            for r in csv.DictReader(f)
        ]


def _units(scores, values, failed) -> list[gate.Unit]:
    return [
        gate.Unit(s.episode_id, s.task_id, s.object, "failure" if f else "success", v)
        for s, v, f in zip(scores, values, failed, strict=True)
    ]


def _excess(units) -> float:
    f = [u.value for u in units if u.group == "failure"]
    s = [u.value for u in units if u.group == "success"]
    return float(np.mean(f) - np.mean(s))


def evaluate(
    scores: list[AttemptScore], thresholds: dict[str, float], level: float, n_boot: int, seed: int
) -> dict:
    """Same-task AUROC and excess over chance with episode-bootstrap CIs per score; passes when
    both CIs exclude chance."""
    failed = np.array([s.failed for s in scores])
    outcome = {s.episode_id: s.episode_success for s in scores}
    delays = [s.width_delay for s in scores]
    width_fires = np.array([d is not None for d in delays])
    results = {
        "n_failure": int(failed.sum()),
        "n_success": int((~failed).sum()),
        "width": {
            "caught": int(width_fires[failed].sum()),
            "false_alarms": int(width_fires[~failed].sum()),
        },
    }
    for name, threshold in thresholds.items():
        score = np.array([getattr(s, name) for s in scores])
        alarm = score > threshold
        units = _units(scores, score.tolist(), failed)
        alarm_units = _units(scores, alarm.astype(float).tolist(), failed)
        lo, hi = gate.bootstrap_ci(units, gate.same_task_auroc, level, n_boot, seed, outcome.get)
        elo, ehi = gate.bootstrap_ci(alarm_units, _excess, level, n_boot, seed, outcome.get)
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
    return results
