import imageio.v2 as imageio
import numpy as np
import pytest

from harness import video

H = W = 64


def _solid(value, size=(H, W)):
    return np.full((*size, 3), value, dtype=np.uint8)


def test_compose_puts_agentview_left_wrist_right_and_flips_both_upright():
    agent, wrist = _solid(0), _solid(0)
    agent[0, 0] = 255
    wrist[0, 0] = 255
    frame = video.compose_frame(agent, wrist, step_idx=0, is_wait=False)

    assert frame.shape == (H, 2 * W, 3)
    assert frame.dtype == np.uint8
    assert frame[H - 1, W - 1].tolist() == [255, 255, 255]
    assert frame[H - 1, 2 * W - 1].tolist() == [255, 255, 255]
    assert frame[H - 1, 0].tolist() == [0, 0, 0]


def test_overlay_changes_only_the_top_left_band_and_depends_on_the_step():
    base = _solid(0, size=(256, 256))
    a = video.overlay_step(base, 7, is_wait=False)
    b = video.overlay_step(base, 8, is_wait=False)

    assert base.max() == 0
    changed = np.argwhere((a != base).any(axis=-1))
    assert changed[:, 0].max() < video.OVERLAY_HEIGHT
    assert changed[:, 1].max() < 256 // 2
    assert not np.array_equal(a, b)


def test_wait_steps_are_marked_differently_from_action_steps():
    base = _solid(0, size=(256, 256))
    assert not np.array_equal(
        video.overlay_step(base, 3, is_wait=True), video.overlay_step(base, 3, is_wait=False)
    )


def test_overlay_is_deterministic():
    base = _solid(90, size=(256, 256))
    assert np.array_equal(
        video.overlay_step(base, 123, False), video.overlay_step(base, 123, False)
    )


def _write(path, n, size=(H, W)):
    agent = [_solid(10 + 8 * k, size) for k in range(n)]
    wrist = [_solid(200, size) for _ in range(n)]
    is_wait = [k < 3 for k in range(n)]
    video.write_video(path, agent, wrist, is_wait)


def test_video_has_one_frame_per_step_in_order_at_20_fps(tmp_path):
    path = tmp_path / "ep.mp4"
    _write(path, 25, size=(128, 128))

    reader = imageio.get_reader(path)
    frames = [f for f in reader]
    assert reader.get_meta_data()["fps"] == 20
    assert len(frames) == 25
    assert frames[0].shape == (128, 256, 3)
    # Bottom-right of the agent half is clear of the overlay; brightness encodes the step.
    means = [f[64:, 64:128].mean() for f in frames]
    expected = [10 + 8 * k for k in range(25)]
    assert np.abs(np.array(means) - expected).max() < 6


def test_mismatched_stream_lengths_are_rejected_without_leaving_files(tmp_path):
    path = tmp_path / "ep.mp4"
    with pytest.raises(ValueError, match="frames"):
        video.write_video(path, [_solid(0)] * 3, [_solid(0)] * 2, [False] * 3)
    assert list(tmp_path.iterdir()) == []


def test_finished_video_leaves_no_temp_file(tmp_path):
    _write(tmp_path / "ep.mp4", 4)
    assert [p.name for p in tmp_path.iterdir()] == ["ep.mp4"]
