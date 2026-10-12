"""Exploratory check after the gate: is per-dimension exp(H) in the pre-grasp window explained by
how fast the arm moves? Arm speed comes from world state, not from model outputs.

Reads <out>/anchors.csv; writes <out>/motion_check.json and motion_check.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import separability_gate as sg

from harness import gate, metadata, world_extract

SPEED_OF_DIM = {"x": "translation", "y": "translation", "z": "translation"}
SPEED_OF_DIM |= {"roll": "rotation", "pitch": "rotation", "yaw": "rotation"}


def speeds(run_dir: Path, loaded, params=gate.DEFAULT_PARAMS) -> None:
    world = {}
    for u in loaded:
        r = u["row"]
        ep = r["episode_id"]
        if ep not in world:
            x, info = world_extract.read_extracted(run_dir / "world" / ep)
            world[ep] = (x, info)
        x, info = world[ep]
        window = gate.pre_window(
            x["is_wait_step"], int(r["anchor_step"]), params.window, params.min_window
        )
        hz = info["control_freq_hz"]
        u["values"]["translation"] = gate.window_speed(x["eef_pos"], window, hz)
        u["values"]["rotation"] = gate.window_rotation_speed(x["grip_quat"], window, hz)


def units(loaded, values) -> list[gate.Unit]:
    return [
        gate.Unit(
            u["row"]["episode_id"],
            int(u["row"]["task_id"]),
            u["row"]["object"],
            u["row"]["group"],
            v,
        )
        for u, v in zip(loaded, values, strict=True)
    ]


def auroc_with_ci(us, level: float) -> dict:
    lo, hi = gate.bootstrap_ci(us, gate.same_task_auroc, level, sg.N_BOOT, sg.SEED)
    return {"auroc": gate.same_task_auroc(us), "ci": [lo, hi], "level": level}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaded = sg.load(args.run_dir, args.out / "anchors.csv")
    speeds(args.run_dir, loaded)
    out = {"git": metadata.git_state(sg.REPO), "speed": {}, "dims": {}}
    for kind in ("translation", "rotation"):
        v = [u["values"][kind] for u in loaded]
        out["speed"][kind] = auroc_with_ci(units(loaded, v), sg.PRIMARY_LEVEL)
    for dim, kind in SPEED_OF_DIM.items():
        h = np.array([u["values"][f"{dim} exp(H)"] for u in loaded])
        s = np.array([u["values"][kind] for u in loaded])
        out["dims"][dim] = {
            "speed": kind,
            "pearson_r": float(np.corrcoef(h, s)[0, 1]),
            "raw": auroc_with_ci(units(loaded, h), sg.EXPLORATORY_LEVEL),
            "residual": auroc_with_ci(units(loaded, gate.residualize(h, s)), sg.EXPLORATORY_LEVEL),
        }

    def fmt(r):
        return f"{r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"

    lines = [
        "# Motion check (exploratory, after the gate)",
        "",
        "Same-task AUROC, failures above successes; bootstrap over episodes within "
        f"(task, outcome), {sg.N_BOOT} draws, seed {sg.SEED}. Speed is the end effector's from "
        "world state over the "
        f"same pre-grasp window. Harness `{(out['git']['sha'] or '')[:7]}`.",
        "",
        "| Arm speed in the window | AUROC [95% CI] |",
        "|---|---|",
        *(f"| {k} | {fmt(r)} |" for k, r in out["speed"].items()),
        "",
        "| Dim | Speed | r(exp(H), speed) | AUROC raw [99.17% CI] "
        "| AUROC speed-removed [99.17% CI] |",
        "|---|---|---|---|---|",
        *(
            f"| {d} | {r['speed']} | {r['pearson_r']:.2f} "
            f"| {fmt(r['raw'])} | {fmt(r['residual'])} |"
            for d, r in out["dims"].items()
        ),
        "",
        "Speed-removed: exp(H) minus its least-squares line on speed, fitted once over all units.",
    ]
    (args.out / "motion_check.json").write_text(json.dumps(out, indent=2) + "\n")
    (args.out / "motion_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
