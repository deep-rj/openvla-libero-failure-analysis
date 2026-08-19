"""CLI to inspect LIBERO rollout .h5 files: list trials or render camera frames as video.

Usage:
    python scripts/inspect_h5.py outputs/libero_eval_20260818_233042.h5 --list
    python scripts/inspect_h5.py outputs/libero_eval_20260818_233042.h5 --task 0 --trial 1
"""

import argparse
import os

import cv2
import h5py
import numpy as np

REPO_DIR = os.environ.get(
    "REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DEFAULT_FPS = 20  # matches LIBERO's control_freq, so playback matches real time


def discover_groups(f: h5py.File) -> list[str]:
    """Finds every group that holds agentview_images, regardless of nesting depth."""
    groups = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and "agentview_images" in obj:
            groups.append(name)

    f.visititems(visitor)
    return sorted(groups)


def resolve_group(f: h5py.File, args: argparse.Namespace) -> str:
    groups = discover_groups(f)
    if not groups:
        raise ValueError(f"No trial/episode groups found in {args.h5_path}")

    if args.episode is not None:
        candidate = args.episode
    elif args.task is not None or args.trial is not None:
        candidate = f"task_{args.task if args.task is not None else 0}/trial_{args.trial if args.trial is not None else 0}"
    else:
        return groups[0]

    if candidate not in groups:
        raise ValueError(f"Group {candidate!r} not found. Available groups: {groups}")
    return candidate


def print_structure(f: h5py.File, h5_path: str) -> None:
    print(h5_path)
    if f.attrs:
        attrs = "  ".join(f"{k}={v!r}" for k, v in f.attrs.items())
        print(attrs)
    print()

    for group_path in discover_groups(f):
        grp = f[group_path]
        success = grp.attrs.get("success", "?")
        num_steps = grp.attrs.get("num_steps", grp["agentview_images"].shape[0])
        frame_shape = grp["agentview_images"].shape[1:]
        print(
            f"{group_path:30s} success={success!s:6s} num_steps={num_steps!s:6s} "
            f"frame_shape={frame_shape}"
        )


def build_frames(grp: h5py.Group, camera: str) -> np.ndarray:
    if camera == "agent":
        return grp["agentview_images"][:]
    if camera == "wrist":
        return grp["wrist_images"][:]
    # "both": agentview_images.shape[1:] == wrist_images.shape[1:] is asserted at
    # write time by libero_eval.py, so a plain hstack is always safe here.
    agent = grp["agentview_images"][:]
    wrist = grp["wrist_images"][:]
    return np.concatenate([agent, wrist], axis=2)


def write_video(frames: np.ndarray, out_path: str, fps: int) -> None:
    if len(frames) == 0:
        raise ValueError("Cannot write a video from zero frames")

    height, width = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # cv2.VideoWriter wants (width, height), the reverse of frames.shape's (N, H, W, C)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames:
        # h5 stores RGB observations; cv2.VideoWriter expects BGR
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_frame_grid(frames: np.ndarray, out_path: str, n: int = 6) -> None:
    if len(frames) == 0:
        raise ValueError("Cannot write a frame grid from zero frames")

    n = min(n, len(frames))
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    grid = np.hstack([frames[i] for i in indices])
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def default_output_path(h5_path: str, group_path: str, ext: str) -> str:
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    suffix = group_path.replace("/", "_")
    return os.path.join(REPO_DIR, "outputs", "videos", f"{stem}_{suffix}.{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="path to a LIBERO rollout .h5 file")
    parser.add_argument(
        "--list", action="store_true", help="print file structure and exit"
    )
    parser.add_argument("--task", type=int, default=None, help="task id")
    parser.add_argument("--trial", type=int, default=None, help="trial index")
    parser.add_argument(
        "--episode", default=None, help="episode group name (smoke-test-style files)"
    )
    parser.add_argument(
        "--camera",
        choices=["agent", "wrist", "both"],
        default="both",
        help="which camera view(s) to render",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--frames", action="store_true", help="also write a static frame-grid PNG"
    )
    parser.add_argument("--out", default=None, help="override the output video path")
    args = parser.parse_args()

    with h5py.File(args.h5_path, "r") as f:
        if args.list:
            print_structure(f, args.h5_path)
            return

        group_path = resolve_group(f, args)
        frames = build_frames(f[group_path], args.camera)

        out_path = args.out or default_output_path(args.h5_path, group_path, "mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_video(frames, out_path, args.fps)
        print(f"Wrote video: {out_path}")

        if args.frames:
            grid_path = default_output_path(args.h5_path, group_path, "png")
            write_frame_grid(frames, grid_path)
            print(f"Wrote frame grid: {grid_path}")


if __name__ == "__main__":
    main()
