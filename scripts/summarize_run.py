"""Summarizes a finished run: suite success rate with n and interval, per-task rates, throughput."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import summary  # noqa: E402

if __name__ == "__main__":
    print(summary.render_markdown(summary.write_outputs(Path(sys.argv[1]))))
