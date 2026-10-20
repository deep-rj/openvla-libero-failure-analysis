"""Renders the README demo: a caught failure and a false alarm, video beside the gripper's
per-step exp(H), both aligned on the scored close command.

  make_demo_clip.py <run_dir> --attempts <csv> --frozen <frozen.json> --out <gif> [--pair <name>]
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import block4_window_sweep as ws

from harness import detector

PAIRS = {
    "same-task": {
        "episodes": (
            ("libero_10-t07-i028-s0-none", "Grasp fails: the box slips out mid-lift"),
            ("libero_10-t07-i037-s0-none", "Grasp succeeds: same task, same box"),
        ),
        "after": 50,
        "caption": "Both trip the alarm, with nearly the same score.",
    },
    "basket": {
        "episodes": (
            ("libero_10-t00-i035-s0-none", "Grasp fails: closes on nothing, carries an empty hand"),
            ("libero_10-t07-i037-s0-none", "Grasp succeeds"),
        ),
        "after": 55,
        "caption": "Both trip the alarm. The successful grasp scores higher.",
    },
}
BEFORE = 40  # policy steps shown before the close command
FPS = 10  # half the 20 Hz control rate
HOLD_S = 3.0
PALETTE_SIZE = 128

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
SERIES, WINDOW, CRITICAL = "#2a78d6", "#e6eefa", "#d03b3b"


def load(run_dir: Path, attempts: Path, frozen: dict, episodes) -> list[dict]:
    rows = {r["episode_id"]: r for r in ws.first_attempts(attempts)}
    spec = detector.DetectorSpec(**frozen["spec"])
    out = []
    for ep, title in episodes:
        r = rows[ep]
        c = int(r["command_step"])
        is_wait, exp_h, _ = ws.load(run_dir, [r], frozen["roll_zero_bin"])[ep]
        gripper = exp_h[:, 6]
        score = detector.window_mean(
            gripper, is_wait, c, spec.gripper_window, None, spec.min_window
        )
        video = [f for f in imageio.get_reader(run_dir / "videos" / f"{ep}.mp4")]  # noqa: C416
        out.append({"title": title, "c": c, "g": gripper, "score": score, "video": video})
    return out


def render(episodes: list[dict], frozen: dict, after: int, caption_text: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window, threshold = frozen["spec"]["gripper_window"], frozen["v1_threshold"]
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(
        2, 2, figsize=(9.6, 5.4), dpi=72, gridspec_kw={"width_ratios": [1.15, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    offsets = np.arange(-BEFORE, after + 1)
    art = []
    for (ax_v, ax_t), e in zip(axes, episodes, strict=True):
        c, n = e["c"], len(e["video"])
        last = min(after, n - 1 - c)
        ax_v.axis("off")
        img = ax_v.imshow(e["video"][c - BEFORE])
        ax_v.set_title(e["title"], loc="left", fontsize=10, color=INK)
        ax_t.set_facecolor(SURFACE)
        ax_t.axvspan(-window, 0, color=WINDOW, lw=0)
        ax_t.text(-window / 2, 2.08, f"scored window ({window} steps)", ha="center",
                  fontsize=8, color=MUTED)  # fmt: skip
        ax_t.axvline(0, color=INK_2, lw=1)
        ax_t.text(1.5, 0.94, "close", fontsize=9, color=INK_2)
        ax_t.set_xlim(-BEFORE, after)
        ax_t.set_ylim(0.9, 2.15)
        ax_t.set_yticks([1, 2], ["1 certain", "2 coin flip"])
        ax_t.set_ylabel("gripper uncertainty, exp(H)")
        ax_t.spines[["top", "right"]].set_visible(False)
        ax_t.spines[["left", "bottom"]].set_color(MUTED)
        ax_t.tick_params(colors=MUTED)
        (line,) = ax_t.plot([], [], color=SERIES, lw=2)
        (dot,) = ax_t.plot([], [], "o", ms=5, color=SERIES)
        badge = ax_t.text(
            22, 1.62, f"ALARM\nscore {e['score']:.3f} > {threshold:.3f}", ha="center",
            fontsize=9, color=SURFACE, fontweight="bold", visible=False,
            bbox={"boxstyle": "round,pad=0.4", "fc": CRITICAL, "ec": CRITICAL},
        )  # fmt: skip
        art.append((e, img, line, dot, badge, last))
    axes[1][1].set_xlabel("policy steps from the close command (20 Hz)")
    caption = fig.text(0.5, 0.015, caption_text, ha="center", fontsize=11, fontweight="bold",
                       color=INK, visible=False)  # fmt: skip
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    frames = []
    for d in offsets:
        for e, img, line, dot, badge, last in art:
            k = min(d, last)
            img.set_data(e["video"][e["c"] + k])
            xs = offsets[offsets <= k]
            line.set_data(xs, e["g"][e["c"] + xs])
            dot.set_data([k], [e["g"][e["c"] + k]])
            badge.set_visible(d >= 0)
        caption.set_visible(d == offsets[-1])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    write_gif(out, frames)


def write_gif(out: Path, frames: list[np.ndarray]) -> None:
    """One shared palette, so static chart regions stay identical between frames and compress."""
    from PIL import Image

    palette = Image.fromarray(np.concatenate(frames[::12], axis=0)).quantize(PALETTE_SIZE)
    images = [
        Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames
    ]
    durations = [1000 // FPS] * (len(images) - 1) + [int(HOLD_S * 1000)]
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB, {len(images)} frames")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", choices=PAIRS, default="same-task")
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text())
    pair = PAIRS[args.pair]
    episodes = load(args.run_dir, args.attempts, frozen, pair["episodes"])
    render(episodes, frozen, pair["after"], pair["caption"], args.out)


if __name__ == "__main__":
    main()
