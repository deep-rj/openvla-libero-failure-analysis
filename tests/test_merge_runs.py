import json

import numpy as np
import pytest
from factories import episode_id, make_episode, write_run

from harness import manifest, merge_runs, pilot_report


def _ep(task, init, **kw):
    steps = make_episode(np.random.default_rng(100 * task + init))
    return {"task": task, "init": init, "steps": steps, **kw}


def _tree(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def _shards(tmp_path, **cfg_b):
    a = write_run(
        tmp_path / "a",
        [_ep(0, 0, success=True), _ep(1, 0)],
        config={"tasks": [0, 1], "init_states": [0], "out_dir": "outputs/a"},
        environment={"gpu": "4090", "pod": "a"},
    )
    b = write_run(
        tmp_path / "b",
        [_ep(0, 1), _ep(1, 1, success=True)],
        config={"tasks": [0, 1], "init_states": [1], "out_dir": "outputs/b", **cfg_b},
        environment={"gpu": "4090", "pod": "b"},
    )
    return a, b


def test_disjoint_shards_merge_into_one_run_with_a_sorted_manifest(tmp_path):
    a, b = _shards(tmp_path)
    report = merge_runs.merge_runs([b, a], tmp_path / "merged")

    out = tmp_path / "merged"
    rows = manifest.read_rows(out / "manifest.csv")
    assert [(r["task_id"], r["init_state_idx"]) for r in rows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert report["n_episodes"] == 4
    for r in rows:
        assert (out / r["episode_file"]).exists() and (out / r["video_file"]).exists()
        assert (out / f"actions/{r['episode_id']}.npy").exists()


def test_the_merged_run_records_where_each_episode_came_from(tmp_path):
    a, b = _shards(tmp_path)
    merge_runs.merge_runs([a, b], tmp_path / "merged")
    record = json.loads((tmp_path / "merged/merged_from.json").read_text())
    shards = {s["environment"]["pod"]: s for s in record["shards"]}
    assert shards["a"]["episode_ids"] == [episode_id(0, 0), episode_id(1, 0)]
    assert shards["b"]["episode_ids"] == [episode_id(0, 1), episode_id(1, 1)]
    config = json.loads((tmp_path / "merged/run_config.json").read_text())["config"]
    assert config["tasks"] == [0, 1] and config["init_states"] == [0, 1]


def test_sources_are_left_untouched(tmp_path):
    a, b = _shards(tmp_path)
    before = (_tree(a), _tree(b))
    merge_runs.merge_runs([a, b], tmp_path / "merged")
    assert (_tree(a), _tree(b)) == before


def test_duplicate_episodes_are_refused(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0)])
    b = write_run(tmp_path / "b", [_ep(0, 0)])
    with pytest.raises(merge_runs.MergeError, match="duplicate"):
        merge_runs.merge_runs([a, b], tmp_path / "merged")


@pytest.mark.parametrize("field,value", [("seed", 1), ("policy", "stub"), ("camera_size", 64)])
def test_shards_with_different_run_settings_are_refused(tmp_path, field, value):
    a, b = _shards(tmp_path, **{field: value})
    with pytest.raises(merge_runs.MergeError, match=field):
        merge_runs.merge_runs([a, b], tmp_path / "merged")


def test_shards_from_different_commits_are_refused_unless_allowed(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0)], git_sha="aaa")
    b = write_run(tmp_path / "b", [_ep(0, 1)], git_sha="bbb")
    with pytest.raises(merge_runs.MergeError, match="commit"):
        merge_runs.merge_runs([a, b], tmp_path / "merged")
    merge_runs.merge_runs([a, b], tmp_path / "merged", allow_mixed_commits=True)


def test_a_non_empty_output_directory_is_refused(tmp_path):
    a, b = _shards(tmp_path)
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged/keep.txt").write_text("x")
    with pytest.raises(merge_runs.MergeError, match="not empty"):
        merge_runs.merge_runs([a, b], tmp_path / "merged")


def test_a_manifest_row_without_its_episode_file_is_refused(tmp_path):
    a, b = _shards(tmp_path)
    (a / f"episodes/{episode_id(0, 0)}.h5").unlink()
    with pytest.raises(merge_runs.MergeError, match="missing"):
        merge_runs.merge_runs([a, b], tmp_path / "merged")
    assert not (tmp_path / "merged/manifest.csv").exists()


def test_missing_episodes_are_listed_against_the_expected_grid(tmp_path):
    a, b = _shards(tmp_path)
    report = merge_runs.merge_runs(
        [a, b], tmp_path / "merged", expected_tasks=[0, 1, 2], expected_init_states=[0, 1]
    )
    assert report["missing"] == [(2, 0), (2, 1)]


def test_downstream_tools_read_the_merged_run(tmp_path):
    a, b = _shards(tmp_path)
    merge_runs.merge_runs([a, b], tmp_path / "merged")
    report = pilot_report.build_report(tmp_path / "merged")
    assert report["n_episodes"] == 4 and report["suite_success"]["successes"] == 2
