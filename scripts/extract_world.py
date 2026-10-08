"""Replays a run's traces and writes extracted world-state fields to <run>/world/."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import world_extract


def _extract(args) -> str:
    trace, out = args
    extracted, info = world_extract.extract_episode(trace)
    world_extract.write_extracted(out, extracted, info)
    return trace.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--selection", type=Path, help="label_selection.json; default is every episode"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="re-extract existing files")
    args = parser.parse_args()

    if args.selection:
        ids = json.loads(args.selection.read_text())["episode_ids"]
    else:
        ids = sorted(p.stem for p in (args.run_dir / "episodes").glob("*.h5"))
    out_dir = args.run_dir / "world"
    todo = [
        (args.run_dir / "episodes" / f"{i}.h5", out_dir / i)
        for i in ids
        if args.force or not (out_dir / f"{i}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for done, episode_id in enumerate(pool.map(_extract, todo), 1):
            print(f"[{done}/{len(todo)}] {episode_id}", flush=True)


if __name__ == "__main__":
    main()
