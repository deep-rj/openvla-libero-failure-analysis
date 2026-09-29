"""Draws failed episodes for hand labeling and writes the selection and an empty label CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import labeling  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ids = labeling.write_selection(args.run_dir, args.n, args.seed)
    print(f"selected {len(ids)} failed episodes; see {args.run_dir / 'labeling'}")


if __name__ == "__main__":
    main()
