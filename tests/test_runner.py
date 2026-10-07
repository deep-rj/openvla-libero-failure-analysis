import json

import numpy as np
import pytest

pytest.importorskip("libero.libero")

from harness import checks, manifest, metadata, policy, rollout, runner, schema

pytestmark = pytest.mark.env

TASK = 9


def _cfg(tmp_path, **overrides):
    kwargs = {
        "out_dir": tmp_path,
        "tasks": [TASK],
        "init_states": [0, 1],
        "max_steps": 4,
        "camera_size": 64,
        "policy": "stub",
    }
    return runner.RunConfig(**{**kwargs, **overrides})


def _stub_factory(cfg):
    return policy.StubPolicy(seed=cfg.seed, out_of_slice_rate=0.2), {}


def test_a_run_writes_episode_files_videos_manifest_and_config(tmp_path):
    rows = runner.run(_cfg(tmp_path), policy_factory=_stub_factory)

    ids = [r["episode_id"] for r in rows]
    assert ids == ["libero_10-t09-i000-s0-none", "libero_10-t09-i001-s0-none"]
    for episode_id in ids:
        assert (tmp_path / "episodes" / f"{episode_id}.h5").exists()
        assert (tmp_path / "videos" / f"{episode_id}.mp4").exists()
    assert manifest.completed_episode_ids(tmp_path / "manifest.csv") == set(ids)
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["config"]["tasks"] == [TASK]
    assert "git" in config["environment"]


def test_written_episodes_carry_metadata_and_pass_the_consistency_checks(tmp_path):
    rows = runner.run(_cfg(tmp_path), policy_factory=_stub_factory)
    arrays, meta = schema.read_episode(tmp_path / "episodes" / f"{rows[0]['episode_id']}.h5")

    assert meta["task_id"] == TASK
    assert meta["init_state_idx"] == 0
    assert meta["object_names"] and meta["fixture_joints"]
    assert meta["terminated_by"] == "max_steps"
    assert checks.check_wait_alignment(arrays, rollout.NUM_STEPS_WAIT) == []
    stats = {k: np.array(v) for k, v in meta["action_stats"].items()}
    sent = arrays["executed_action"]
    assert checks.check_self_consistency(arrays, stats, list(sent)) == []


def test_completed_episodes_are_skipped_on_a_rerun(tmp_path):
    runner.run(_cfg(tmp_path), policy_factory=_stub_factory)
    again = runner.run(_cfg(tmp_path), policy_factory=_stub_factory)
    assert again == []
    assert len(manifest.completed_episode_ids(tmp_path / "manifest.csv")) == 2


def test_a_run_summarizes_its_whole_manifest_including_resumed_episodes(tmp_path):
    runner.run(_cfg(tmp_path, init_states=[0]), policy_factory=_stub_factory)
    runner.run(_cfg(tmp_path), policy_factory=_stub_factory)
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["n"] == 2
    assert (tmp_path / "summary.md").exists()


def test_reference_mode_saves_actions_and_no_trace(tmp_path):
    rows = runner.run(
        _cfg(tmp_path, mode="reference", init_states=[0], video=False),
        policy_factory=_stub_factory,
    )
    episode_id = rows[0]["episode_id"]
    actions = np.load(tmp_path / "actions" / f"{episode_id}.npy")
    assert actions.shape == (rollout.NUM_STEPS_WAIT + 4, 7)
    assert not (tmp_path / "episodes").exists()


def test_logged_and_reference_runs_send_identical_actions(tmp_path):
    logged_dir, reference_dir = tmp_path / "logged", tmp_path / "reference"
    cfg = {"init_states": [0], "video": False}
    runner.run(_cfg(logged_dir, **cfg), policy_factory=_stub_factory)
    runner.run(_cfg(reference_dir, mode="reference", **cfg), policy_factory=_stub_factory)

    episode_id = "libero_10-t09-i000-s0-none"
    arrays, _ = schema.read_episode(logged_dir / "episodes" / f"{episode_id}.h5")
    reference = np.load(reference_dir / "actions" / f"{episode_id}.npy")
    assert np.array_equal(arrays["executed_action"], reference)


def test_a_real_policy_refuses_to_run_from_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "git_state", lambda path: {"sha": "a" * 40, "dirty": True})
    with pytest.raises(RuntimeError, match="dirty"):
        runner.run(_cfg(tmp_path, policy="openvla"), policy_factory=_stub_factory)
    assert not (tmp_path / "manifest.csv").exists()


def test_allow_dirty_overrides_the_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "git_state", lambda path: {"sha": "a" * 40, "dirty": True})
    rows = runner.run(
        _cfg(tmp_path, policy="openvla", allow_dirty=True, init_states=[0], video=False),
        policy_factory=_stub_factory,
    )
    assert len(rows) == 1


def test_every_mode_saves_the_actions_the_env_received(tmp_path):
    rows = runner.run(_cfg(tmp_path, init_states=[0], video=False), policy_factory=_stub_factory)
    episode_id = rows[0]["episode_id"]
    arrays, _ = schema.read_episode(tmp_path / "episodes" / f"{episode_id}.h5")
    received = np.load(tmp_path / "actions" / f"{episode_id}.npy")
    assert np.array_equal(received, arrays["executed_action"])


def test_parity_mode_writes_the_mismatched_steps_per_episode(tmp_path):
    from factories import DriftedPolicy, PureFakePolicy

    for name, factory_policy, expected in (
        ("same", PureFakePolicy(), []),
        ("drift", DriftedPolicy(), [10, 11, 12, 13]),
    ):
        out = tmp_path / name
        runner.run(
            _cfg(out, mode="parity", init_states=[0], video=False),
            policy_factory=lambda cfg, p=factory_policy: (p, {}),
        )
        result = json.loads((out / "parity" / "libero_10-t09-i000-s0-none.json").read_text())
        assert result["mismatched_steps"] == expected
        assert (out / "episodes" / "libero_10-t09-i000-s0-none.h5").exists()
