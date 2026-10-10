"""Compares mechanism_predicted.csv with hand labels: agreement, abstentions, onset offsets."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import mechanism_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="mechanism_predicted.csv")
    parser.add_argument("labels", type=Path, help="hand labels CSV (labels.csv)")
    parser.add_argument("--out", type=Path, required=True, help="markdown report path")
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = {row["episode_id"]: row for row in csv.DictReader(f)}
    predictions = mechanism_run.read_predictions(args.predictions)
    missing = sorted(set(labels) - set(predictions))
    if missing:
        sys.exit(f"no prediction for {len(missing)} labeled episodes, e.g. {missing[0]}")
    report = mechanism_run.dev_report(predictions, labels)
    args.out.write_text(mechanism_run.render_dev_report(report))
    print(
        f"agreement {report['agree']}/{report['decided']}, abstained "
        f"{len(report['abstained'])}/{report['n']} -> {args.out}"
    )


if __name__ == "__main__":
    main()
