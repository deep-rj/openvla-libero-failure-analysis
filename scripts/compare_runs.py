"""Compares two runs episode by episode, matched by task and init state."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import compare_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--out", type=Path, help="also write the markdown report here")
    args = parser.parse_args()

    text = compare_runs.render_markdown(compare_runs.compare_runs(args.run_a, args.run_b))
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
