"""Labeling videos: one frame per control step with a burned-in step index.

The overlay carries the step index only, so labeling stays blind to model outputs.
"""

import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

FPS = 20
OVERLAY_HEIGHT = 24
_OVERLAY_WIDTH = 120
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay_step(frame: np.ndarray, step_idx: int, is_wait: bool) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (_OVERLAY_WIDTH - 1, OVERLAY_HEIGHT - 1), (0, 0, 0), thickness=-1)
    label = f"step {step_idx}" + (" WAIT" if is_wait else "")
    cv2.putText(out, label, (4, 16), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compose_frame(agent: np.ndarray, wrist: np.ndarray, step_idx: int, is_wait: bool) -> np.ndarray:
    """Agentview left, wrist right; env frames arrive rotated 180 degrees and are made upright."""
    agent_up = np.ascontiguousarray(agent[::-1, ::-1])
    wrist_up = np.ascontiguousarray(wrist[::-1, ::-1])
    return np.concatenate([overlay_step(agent_up, step_idx, is_wait), wrist_up], axis=1)


def write_video(path, agent_frames, wrist_frames, is_wait_steps) -> None:
    """Writes frame k as step k, atomically."""
    if not len(agent_frames) == len(wrist_frames) == len(is_wait_steps):
        raise ValueError(
            f"frames must align: {len(agent_frames)} agent, {len(wrist_frames)} wrist, "
            f"{len(is_wait_steps)} wait flags"
        )
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.mp4")
    try:
        with imageio.get_writer(
            tmp,
            fps=FPS,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "18"],
        ) as writer:
            for k, (agent, wrist, wait) in enumerate(
                zip(agent_frames, wrist_frames, is_wait_steps, strict=True)
            ):
                writer.append_data(compose_frame(agent, wrist, k, bool(wait)))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
