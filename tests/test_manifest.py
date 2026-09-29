import csv

import pytest

from harness import manifest

ROW = {
    "episode_id": "libero_10-t00-i000-s0-none",
    "task_suite": "libero_10",
    "task_id": 0,
    "task_instruction": "put both the alphabet soup and the tomato sauce in the basket",
    "init_state_idx": 0,
    "seed": 0,
    "success": True,
    "num_steps": 279,
    "terminated_by": "success",
    "episode_file": "libero_10-t00-i000-s0-none.h5",
    "video_file": "libero_10-t00-i000-s0-none.mp4",
    "git_sha": "abc123",
    "seconds_per_step": 0.27,
}


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_first_row_writes_the_header_and_later_rows_append(tmp_path):
    path = tmp_path / "manifest.csv"
    manifest.append_row(path, ROW)
    manifest.append_row(path, {**ROW, "episode_id": "other", "success": False})

    rows = _rows(path)
    assert [r["episode_id"] for r in rows] == [ROW["episode_id"], "other"]
    assert rows[0]["success"] == "True"
    assert list(rows[0]) == manifest.FIELDS


def test_missing_or_unknown_fields_are_rejected(tmp_path):
    path = tmp_path / "manifest.csv"
    with pytest.raises(ValueError, match="missing"):
        manifest.append_row(path, {"episode_id": "x"})
    with pytest.raises(ValueError, match="unknown"):
        manifest.append_row(path, {**ROW, "extra": 1})
    assert not path.exists()


def test_a_manifest_with_a_different_header_is_not_appended_to(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("episode_id,something_else\nx,1\n")
    with pytest.raises(ValueError, match="header"):
        manifest.append_row(path, ROW)


def test_completed_episode_ids_are_read_back_for_resuming(tmp_path):
    path = tmp_path / "manifest.csv"
    assert manifest.completed_episode_ids(path) == set()
    manifest.append_row(path, ROW)
    assert manifest.completed_episode_ids(path) == {ROW["episode_id"]}
