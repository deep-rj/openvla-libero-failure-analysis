import h5py
import numpy as np
import pytest
from factories import make_episode, make_step

from harness import schema

META = {
    "episode_id": "libero_10-t00-i000-s0-none",
    "task_id": 0,
    "seed": 0,
    "object_names": ["alphabet_soup_1", "basket_1"],
    "final_success": False,
}


def _write(tmp_path, steps, meta=META, name="ep.h5"):
    path = tmp_path / name
    schema.write_episode(path, steps, meta)
    return path


def test_episode_round_trips_arrays_and_metadata(tmp_path):
    steps = make_episode(np.random.default_rng(0))
    arrays, meta = schema.read_episode(_write(tmp_path, steps))

    assert arrays["step_idx"].tolist() == list(range(30))
    assert arrays["slice_logits"].shape == (30, 7, 256)
    assert arrays["slice_logits"].dtype == np.float16
    assert np.array_equal(arrays["slice_logits"][15], steps[15].model.slice_logits)
    assert np.array_equal(arrays["sim_state"][3], steps[3].world.sim_state)
    assert arrays["object_poses"].shape == (30, 3, 7)
    assert arrays["fixture_joint_qpos"].shape == (30, 1)
    assert arrays["executed_action"].dtype == np.float64
    assert np.array_equal(arrays["executed_action"][20], steps[20].executed_action)
    assert meta["episode_id"] == META["episode_id"]
    assert meta["object_names"] == META["object_names"]
    assert meta["final_success"] is False
    assert meta["schema_version"] == schema.SCHEMA_VERSION


def test_wait_steps_carry_sentinels_in_every_model_field(tmp_path):
    arrays, _ = schema.read_episode(_write(tmp_path, make_episode(np.random.default_rng(0))))
    wait, act = arrays["is_wait_step"], ~arrays["is_wait_step"]

    assert wait.sum() == 10
    assert not wait[10:].any()
    assert np.isnan(arrays["slice_logits"][wait].astype(np.float32)).all()
    for name in ("lse_full", "normalized_action", "unnormalized_action"):
        assert np.isnan(arrays[name][wait]).all(), name
    assert np.isnan(arrays["inference_latency_ms"][wait]).all()
    for name in ("argmax_full_id", "chosen_token_id", "bin_idx"):
        assert (arrays[name][wait] == -1).all(), name
        assert (arrays[name][act] >= 0).all(), name
    assert not arrays["clipped_flag"][wait].any()


def test_wait_step_logs_the_action_actually_sent(tmp_path):
    arrays, _ = schema.read_episode(_write(tmp_path, make_episode(np.random.default_rng(0))))
    assert arrays["executed_action"][0].tolist() == [0, 0, 0, 0, 0, 0, -1.0]


def test_empty_fixture_and_object_axes_are_supported(tmp_path):
    steps = make_episode(np.random.default_rng(0), n_obj=0, n_joint=0)
    arrays, _ = schema.read_episode(_write(tmp_path, steps))
    assert arrays["object_poses"].shape == (30, 0, 7)
    assert arrays["fixture_joint_qpos"].shape == (30, 0)


def test_failed_write_leaves_no_file_and_no_temp(tmp_path):
    rng = np.random.default_rng(0)
    steps = make_episode(rng)
    steps[12].model.slice_logits = np.zeros((7, 255), dtype=np.float16)
    with pytest.raises(ValueError, match="slice_logits"):
        _write(tmp_path, steps)
    assert list(tmp_path.iterdir()) == []


def test_failed_rewrite_keeps_the_previous_complete_episode(tmp_path):
    rng = np.random.default_rng(0)
    path = _write(tmp_path, make_episode(rng))
    before = path.read_bytes()
    bad = make_episode(rng)
    bad[12].world.sim_state = np.zeros(5)
    with pytest.raises(ValueError, match="sim_state"):
        _write(tmp_path, bad)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["ep.h5"]


def test_unknown_schema_version_is_rejected_on_read(tmp_path):
    path = _write(tmp_path, make_episode(np.random.default_rng(0)))
    with h5py.File(path, "r+") as f:
        f.attrs["schema_version"] = schema.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        schema.read_episode(path)


def test_step_indices_must_be_contiguous_from_zero(tmp_path):
    rng = np.random.default_rng(0)
    steps = [make_step(rng, i, wait=True) for i in (0, 1, 3)]
    with pytest.raises(ValueError, match="step_idx"):
        _write(tmp_path, steps)


def test_episode_size_matches_the_budget(tmp_path):
    steps = make_episode(np.random.default_rng(0), n_obj=8, n_joint=3, n_act=410, sim_state_len=123)
    size = _write(tmp_path, steps).stat().st_size
    assert 1.5e6 < size < 2.5e6, size
