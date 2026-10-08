"""Picks episodes for hand labeling and writes the empty label template.

Selection reads only outcome and task, never model outputs, so it cannot leak into the
blind onset labeling.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from harness import manifest

# Hand-labeled only; milestone and flags are computed (`scripts/compute_labels.py`).
LABEL_COLUMNS = [
    "episode_id",
    "mechanism",
    "mechanism_notes",
    "first_error_step",
    "no_return_step",
    "no_onset",
    "computed_check",
    "notes",
]


def select_failed_episodes(rows: list[dict], n: int, seed: int) -> list[str]:
    """Draws up to `n` failed episodes, spread as evenly across tasks as failures allow."""
    rng = random.Random(seed)  # noqa: S311
    queues = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["episode_id"]):
        if not row["success"]:
            queues[row["task_id"]].append(row["episode_id"])
    for task in sorted(queues):
        rng.shuffle(queues[task])

    picked = []
    while len(picked) < n and any(queues.values()):
        for task in sorted(queues):
            if queues[task] and len(picked) < n:
                picked.append(queues[task].pop())
    return sorted(picked)


def write_label_template(path, episode_ids: list[str]) -> None:
    """Writes an empty template; refuses to overwrite an existing (possibly filled) file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite labels")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows({"episode_id": episode_id} for episode_id in episode_ids)


def write_selection(run_dir, n: int, seed: int) -> list[str]:
    """Selects episodes from a run's manifest and writes the selection and empty template."""
    run_dir = Path(run_dir)
    rows = manifest.read_rows(run_dir / "manifest.csv")
    ids = select_failed_episodes(rows, n, seed)
    by_id = {r["episode_id"]: r for r in rows}

    out = run_dir / "labeling"
    out.mkdir(exist_ok=True)
    task_counts = Counter(by_id[i]["task_id"] for i in ids)
    record = {
        "seed": seed,
        "n_requested": n,
        "episode_ids": ids,
        "videos": [by_id[i]["video_file"] for i in ids],
        "per_task_counts": {str(t): c for t, c in sorted(task_counts.items())},
    }
    (out / "label_selection.json").write_text(json.dumps(record, indent=2))
    write_label_template(out / "labels_template.csv", ids)
    return ids
