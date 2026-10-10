"""Writes each task's object collision boxes (body frame) to <run>/object_geometry.json."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import runner, world, world_extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    from libero.libero import benchmark

    config = json.loads((args.run_dir / "run_config.json").read_text())["config"]
    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    geometry = {}
    for task_id in config["tasks"]:
        env, _ = runner.build_env(suite, task_id, 64, config["seed"])
        try:
            env.reset()
            names = world.scene_layout(env).object_names
            geometry[str(task_id)] = world_extract.object_collision_boxes(env.env, names)
        finally:
            env.close()
        print(f"task {task_id}: {len(geometry[str(task_id)])} objects", flush=True)
    out = args.run_dir / "object_geometry.json"
    out.write_text(json.dumps(geometry, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
