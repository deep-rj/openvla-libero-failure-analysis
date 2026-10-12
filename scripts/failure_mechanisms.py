"""Writes one row per failed episode: computed milestone and mechanism, hand label first."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import mechanism_run, world_extract
from scripts.compute_labels import computed_row

LABELS = Path("docs/results/baseline/labeling")
HAND_SETS = {
    "pilot": "labels.csv",
    "heldout": "heldout_labels.csv",
    "abstention": "abstention_labels.csv",
}


def read_rows(path: Path) -> dict:
    with open(path, newline="") as f:
        return {row["episode_id"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_rows(args.run_dir / "manifest.csv")
    failures = [ep for ep, row in manifest.items() if row["success"] == "False"]
    computed = {}
    for ep in failures:
        x, info = world_extract.read_extracted(args.run_dir / "world" / ep)
        computed[ep] = computed_row(x, info)
    hand = {name: read_rows(args.labels / file) for name, file in HAND_SETS.items()}
    predicted = mechanism_run.read_predictions(
        args.run_dir / "labeling" / "mechanism_predicted.csv"
    )
    rows = mechanism_run.failure_mechanisms(failures, computed, hand, predicted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mechanism_run.FAILURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
