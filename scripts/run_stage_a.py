"""Runs Stage A on the pod; see `python scripts/run_stage_a.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.stage_a import main  # noqa: E402

if __name__ == "__main__":
    main()
