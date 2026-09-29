"""Runs LIBERO episodes with the trace logger; see `python scripts/run_eval.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cli import main

if __name__ == "__main__":
    main()
