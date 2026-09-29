"""T4: restores a logged mid-episode state into a fresh env and compares the named state."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.restore_check import check_restore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="episode .h5 file")
    parser.add_argument("--step", type=int, help="step to restore (default: mid-episode)")
    args = parser.parse_args()

    problems = check_restore(args.episode, step=args.step)
    print("\n".join(problems) if problems else "restore check passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
