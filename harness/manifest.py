"""Run-level manifest: one CSV row per finished episode."""

import csv
from pathlib import Path

FIELDS = [
    "episode_id",
    "task_suite",
    "task_id",
    "task_instruction",
    "init_state_idx",
    "seed",
    "success",
    "num_steps",
    "terminated_by",
    "episode_file",
    "video_file",
    "git_sha",
    "seconds_per_step",
]


def append_row(path, row: dict) -> None:
    missing, unknown = set(FIELDS) - set(row), set(row) - set(FIELDS)
    if missing:
        raise ValueError(f"manifest row is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"manifest row has unknown fields: {sorted(unknown)}")

    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if header != FIELDS:
            raise ValueError(f"manifest header {header} does not match {FIELDS}")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_episode_ids(path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row["episode_id"] for row in csv.DictReader(f)}
