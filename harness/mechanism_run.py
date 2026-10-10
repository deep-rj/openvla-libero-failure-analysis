"""Runs the mechanism classifier over a run directory and compares it with hand labels."""

import csv
import json
from collections import Counter
from pathlib import Path

from harness import flags, mechanism, world_extract

PREDICTION_COLUMNS = ("episode_id", "mechanism", "onset_step", "abstained", "reason")


def load_heldout(path: Path, include_heldout: bool) -> set[str]:
    """Held-out episode IDs; a missing list is an error unless held-out episodes are included."""
    if not path.exists():
        if include_heldout:
            return set()
        raise FileNotFoundError(f"{path}: held-out list needed to skip held-out episodes")
    return set(json.loads(path.read_text())["episode_ids"])


def select_episodes(failures, heldout: set[str], include_heldout: bool) -> list[str]:
    return [ep for ep in failures if include_heldout or ep not in heldout]


def classify_run(run_dir: Path, episode_ids, params=mechanism.DEFAULT_PARAMS) -> dict:
    """episode_id -> Prediction. The placement reference uses every successful episode."""
    world = run_dir / "world"
    geometry = json.loads((run_dir / "object_geometry.json").read_text())
    successes = []
    for path in sorted(world.glob("*.json")):
        if path.stem.startswith("milestones"):
            continue
        if json.loads(path.read_text())["final_success"]:
            successes.append(world_extract.read_extracted(path.with_suffix("")))
    reference = mechanism.success_reference(successes, params)
    out = {}
    for ep in episode_ids:
        x, info = world_extract.read_extracted(world / ep)
        boxes = geometry[str(info["task_id"])]
        out[ep] = mechanism.classify(
            x, info, flags.compute_flags(x, info), boxes, reference, params
        )
    return out


def write_predictions(path: Path, predictions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for ep, p in predictions.items():
            writer.writerow(
                {
                    "episode_id": ep,
                    "mechanism": p.mechanism,
                    "onset_step": "" if p.onset_step is None else p.onset_step,
                    "abstained": int(p.abstained),
                    "reason": p.reason,
                }
            )


def read_predictions(path: Path) -> dict:
    with open(path, newline="") as f:
        return {
            row["episode_id"]: mechanism.Prediction(
                row["mechanism"],
                int(row["onset_step"]) if row["onset_step"] else None,
                row["abstained"] == "1",
                row["reason"],
            )
            for row in csv.DictReader(f)
        }


def dev_report(predictions: dict, labels: dict) -> dict:
    """Agreement on decided episodes, abstentions, confusion (label, predicted) and signed
    onset offsets (predicted minus hand) for episodes with both."""
    eps = [ep for ep in predictions if ep in labels]
    decided = [ep for ep in eps if not predictions[ep].abstained]
    confusion = Counter((labels[ep]["mechanism"], predictions[ep].mechanism) for ep in decided)
    return {
        "n": len(eps),
        "decided": len(decided),
        "agree": sum(labels[ep]["mechanism"] == predictions[ep].mechanism for ep in decided),
        "abstained": [ep for ep in eps if predictions[ep].abstained],
        "confusion": dict(confusion),
        "onset_offsets": {
            ep: predictions[ep].onset_step - int(labels[ep]["first_error_step"])
            for ep in decided
            if predictions[ep].onset_step is not None
        },
    }


def render_dev_report(report: dict) -> str:
    offsets = sorted(report["onset_offsets"].values())
    within = sum(abs(o) <= 10 for o in offsets)
    lines = [
        "# Mechanism classifier: pilot development report",
        "",
        "The pilot 20 are the development set the rules were written against, so these numbers",
        "are optimistic; the held-out labels are the test.",
        "",
        f"- Agreement: {report['agree']}/{report['decided']} decided.",
        f"- Abstentions: {len(report['abstained'])}/{report['n']} abstained"
        + (f" ({', '.join(report['abstained'])})." if report["abstained"] else "."),
        f"- Onset offsets (predicted minus hand, steps): within ±10 for {within}/{len(offsets)};"
        f" sorted: {offsets}.",
        "",
        "| hand label | predicted | n |",
        "|---|---|---|",
    ]
    for (label, predicted), n in sorted(report["confusion"].items()):
        lines.append(f"| {label} | {predicted} | {n} |")
    if report["onset_offsets"]:
        lines += ["", "| episode | onset offset |", "|---|---|"]
        lines += [f"| {ep} | {o:+d} |" for ep, o in sorted(report["onset_offsets"].items())]
    return "\n".join(lines) + "\n"
