"""Command-line entry point for `scripts/run_eval.py`."""

import argparse
import time
from pathlib import Path

from harness.runner import REPO_DIR, RunConfig, run


def parse_indices(text: str) -> list[int]:
    """Parses "0:25", "0,3,5" or "2:4,9" into an ordered list of non-negative ints."""
    out = []
    for part in text.split(","):
        bounds = part.split(":")
        if len(bounds) > 2 or not all(b.isdigit() for b in bounds):
            raise ValueError(f"bad index list {text!r}")
        if len(bounds) == 1:
            out.append(int(bounds[0]))
            continue
        start, stop = int(bounds[0]), int(bounds[1])
        if stop <= start:
            raise ValueError(f"empty range {part!r} in {text!r}")
        out.extend(range(start, stop))
    return out


def _indices(text: str) -> list[int]:
    try:
        return parse_indices(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err


def parse_args(argv=None) -> RunConfig:
    parser = argparse.ArgumentParser(description="Run LIBERO episodes and write traces.")
    parser.add_argument("--tasks", type=_indices, required=True, help='task ids, e.g. "0:10"')
    parser.add_argument(
        "--init-states", type=_indices, required=True, help='init-state indices, e.g. "0:25"'
    )
    parser.add_argument("--out", type=Path, help="output directory (default: outputs/runs/<time>)")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--policy", choices=["stub", "openvla"], default="stub")
    parser.add_argument("--mode", choices=["logged", "reference", "parity"], default="logged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="policy-step budget (default: suite budget)")
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    out = args.out or REPO_DIR / "outputs" / "runs" / time.strftime("%Y%m%d_%H%M%S")
    return RunConfig(
        out_dir=out,
        tasks=args.tasks,
        init_states=args.init_states,
        suite=args.suite,
        seed=args.seed,
        policy=args.policy,
        mode=args.mode,
        max_steps=args.max_steps,
        video=args.video,
        allow_dirty=args.allow_dirty,
    )


def main(argv=None) -> None:
    cfg = parse_args(argv)
    rows = run(cfg)
    successes = sum(r["success"] for r in rows)
    print(f"{len(rows)} episodes run, {successes} successes; outputs in {cfg.out_dir}")
