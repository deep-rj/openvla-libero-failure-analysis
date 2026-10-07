"""Instrument-validity checks over every episode of a run; pooled, no model-output values."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import instrument_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = instrument_checks.write_outputs(args.run_dir)
    print(instrument_checks.render_markdown(report))


if __name__ == "__main__":
    main()
