import csv
import json
from collections import Counter

import pytest

from harness import labeling, manifest


def _rows(failures_by_task, successes_per_task=2):
    rows = []
    for task, n_fail in failures_by_task.items():
        rows += [
            {"episode_id": f"t{task}-f{i:02d}", "task_id": task, "success": False}
            for i in range(n_fail)
        ]
        rows += [
            {"episode_id": f"t{task}-s{i:02d}", "task_id": task, "success": True}
            for i in range(successes_per_task)
        ]
    return rows


def _task_counts(ids):
    return Counter(int(i.split("-")[0][1:]) for i in ids)


def test_only_failed_episodes_are_selected():
    ids = labeling.select_failed_episodes(_rows({0: 6, 1: 6}), n=8, seed=0)
    assert len(ids) == 8
    assert all("-f" in i for i in ids)


def test_fewer_failures_than_requested_selects_all_of_them():
    ids = labeling.select_failed_episodes(_rows({0: 2, 1: 1}), n=20, seed=0)
    assert sorted(ids) == ["t0-f00", "t0-f01", "t1-f00"]


def test_selection_is_spread_across_tasks_as_evenly_as_failures_allow():
    ids = labeling.select_failed_episodes(_rows({0: 5, 1: 5}), n=4, seed=0)
    assert _task_counts(ids) == {0: 2, 1: 2}

    ids = labeling.select_failed_episodes(_rows({0: 10, 3: 10, 7: 2}), n=9, seed=0)
    counts = _task_counts(ids)
    assert counts[7] == 2
    assert abs(counts[0] - counts[3]) <= 1
    assert sum(counts.values()) == 9


def test_the_same_seed_gives_the_same_draw_and_another_seed_gives_another():
    rows = _rows({0: 30, 1: 30})
    first = labeling.select_failed_episodes(rows, n=10, seed=0)
    assert labeling.select_failed_episodes(rows, n=10, seed=0) == first
    assert labeling.select_failed_episodes(rows, n=10, seed=1) != first


def test_row_order_does_not_change_the_draw():
    rows = _rows({0: 30, 1: 30})
    assert labeling.select_failed_episodes(rows, 10, 0) == labeling.select_failed_episodes(
        rows[::-1], 10, 0
    )


def test_selection_ignores_everything_but_outcome_and_task():
    rows = _rows({0: 30, 1: 30})
    decorated = [
        {**r, "seconds_per_step": 0.1 * i, "num_steps": 100 + i, "entropy": i}
        for i, r in enumerate(rows)
    ]
    assert labeling.select_failed_episodes(decorated, 10, 0) == labeling.select_failed_episodes(
        rows, 10, 0
    )


def test_the_template_has_only_the_hand_labeled_columns_and_episode_ids_filled(tmp_path):
    path = tmp_path / "labels.csv"
    labeling.write_label_template(path, ["a", "b"])

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == labeling.LABEL_COLUMNS
    assert labeling.LABEL_COLUMNS == [
        "episode_id",
        "mechanism",
        "mechanism_notes",
        "first_error_step",
        "no_return_step",
        "no_onset",
        "computed_check",
        "notes",
    ]
    assert [r["episode_id"] for r in rows] == ["a", "b"]
    assert all(v == "" for r in rows for k, v in r.items() if k != "episode_id")


def test_an_existing_label_file_is_never_overwritten(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("hand written labels")
    with pytest.raises(FileExistsError):
        labeling.write_label_template(path, ["a"])
    assert path.read_text() == "hand written labels"


def _manifest_row(episode_id, task, success):
    return {
        "episode_id": episode_id, "task_suite": "libero_10", "task_id": task,
        "task_instruction": "t", "init_state_idx": 0, "seed": 0, "success": success,
        "num_steps": 100, "terminated_by": "success" if success else "max_steps",
        "episode_file": f"episodes/{episode_id}.h5", "video_file": f"videos/{episode_id}.mp4",
        "git_sha": "x", "seconds_per_step": 0.25,
    }  # fmt: skip


def test_write_selection_records_the_seed_ids_and_videos(tmp_path):
    for i in range(6):
        manifest.append_row(tmp_path / "manifest.csv", _manifest_row(f"e{i}", i % 2, i % 3 == 0))

    selected = labeling.write_selection(tmp_path, n=3, seed=5)

    record = json.loads((tmp_path / "labeling" / "label_selection.json").read_text())
    assert record["seed"] == 5
    assert record["episode_ids"] == selected
    assert record["videos"] == [f"videos/{i}.mp4" for i in selected]
    assert (tmp_path / "labeling" / "labels_template.csv").exists()
    assert set(record) == {"seed", "n_requested", "episode_ids", "videos", "per_task_counts"}
