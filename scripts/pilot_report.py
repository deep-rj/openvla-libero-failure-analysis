"""Pooled premise-check statistics for a run; no per-episode or time-aligned output."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import pilot_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage-a", type=Path, help="stage_a_report.json to list anomalies from")
    parser.add_argument(
        "--only-selected", action="store_true", help="pool only the episodes in labeling/"
    )
    args = parser.parse_args()

    ids = None
    if args.only_selected:
        record = json.loads((args.run_dir / "labeling" / "label_selection.json").read_text())
        ids = set(record["episode_ids"])
    report = pilot_report.write_outputs(args.run_dir, args.stage_a, ids)
    print(pilot_report.render_markdown(report))


if __name__ == "__main__":
    main()
