"""Writes <run>/labeling/mechanism_predicted.csv: mechanism and onset per failed episode.

Held-out episodes (<run>/labeling/heldout_label_selection.json) are skipped unless
--include-heldout is passed; without that list the script refuses to run.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import mechanism_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--selection", type=Path, help="selection JSON; default every failure")
    parser.add_argument(
        "--heldout", type=Path, help="default <run>/labeling/heldout_label_selection.json"
    )
    parser.add_argument("--include-heldout", action="store_true")
    args = parser.parse_args()

    heldout_path = args.heldout or args.run_dir / "labeling" / "heldout_label_selection.json"
    heldout = mechanism_run.load_heldout(heldout_path, args.include_heldout)
    if args.selection:
        ids = json.loads(args.selection.read_text())["episode_ids"]
    else:
        ids = [
            p.stem
            for p in sorted((args.run_dir / "world").glob("*.json"))
            if not p.stem.startswith("milestones")
            and not json.loads(p.read_text())["final_success"]
        ]
    ids = mechanism_run.select_episodes(ids, heldout, args.include_heldout)
    predictions = mechanism_run.classify_run(args.run_dir, ids)
    out = args.run_dir / "labeling" / "mechanism_predicted.csv"
    mechanism_run.write_predictions(out, predictions)
    counts = Counter(p.mechanism or "abstained" for p in predictions.values())
    print(f"wrote {len(predictions)} rows -> {out}")
    for mechanism, n in sorted(counts.items()):
        print(f"  {mechanism}: {n}")


if __name__ == "__main__":
    main()
