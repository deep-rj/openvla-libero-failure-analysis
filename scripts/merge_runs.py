"""Merges run directories from sharded pods into one run directory."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import merge_runs, summary
from harness.cli import parse_indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+", help="run directories to merge")
    parser.add_argument("--out", type=Path, required=True, help="new, empty output directory")
    parser.add_argument("--tasks", help='expected task ids, e.g. "0:10"')
    parser.add_argument("--init-states", help='expected init-state indices, e.g. "0:25"')
    parser.add_argument("--allow-mixed-commits", action="store_true")
    args = parser.parse_args()

    expected = {}
    if args.tasks and args.init_states:
        expected = {
            "expected_tasks": parse_indices(args.tasks),
            "expected_init_states": parse_indices(args.init_states),
        }
    try:
        report = merge_runs.merge_runs(
            args.shards, args.out, allow_mixed_commits=args.allow_mixed_commits, **expected
        )
    except merge_runs.MergeError as err:
        print(f"merge refused: {err}")
        return 1

    print(f"merged {report['n_episodes']} episodes into {report['out_dir']}")
    if expected:
        missing = report["missing"]
        print(f"missing {len(missing)} of the expected episodes")
        by_init: dict[int, list[int]] = {}
        for task, init in missing:
            by_init.setdefault(init, []).append(task)
        for init, tasks in sorted(by_init.items()):
            print(f"  init state {init}: tasks {tasks}")
    print()
    print(summary.render_markdown(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
