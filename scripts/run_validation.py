"""Runs the validation checks on the pod; see `python scripts/run_validation.py --help`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.validation import main  # noqa: E402

if __name__ == "__main__":
    main()
