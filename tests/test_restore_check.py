import h5py
import pytest

pytest.importorskip("libero.libero")

from harness import policy, restore_check, runner

pytestmark = pytest.mark.env


@pytest.fixture(scope="module")
def episode_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = runner.RunConfig(
        out_dir=out,
        tasks=[9],
        init_states=[0],
        max_steps=6,
        video=False,
        camera_size=64,
    )
    rows = runner.run(cfg, policy_factory=lambda c: (policy.StubPolicy(seed=1), {}))
    return out / rows[0]["episode_file"]


def _copy_and_edit(episode_path, tmp_path, edit):
    copy = tmp_path / "copy.h5"
    copy.write_bytes(episode_path.read_bytes())
    with h5py.File(copy, "r+") as f:
        edit(f)
    return copy


def test_a_freshly_written_episode_restores_cleanly(episode_path):
    assert restore_check.check_restore(episode_path) == []


def test_an_early_wait_step_also_restores_cleanly(episode_path):
    assert restore_check.check_restore(episode_path, step=3) == []


def test_a_changed_fixture_joint_position_is_reported(episode_path, tmp_path):
    def edit(f):
        f["fixture_joint_qpos"][12, 0] += 0.5

    problems = restore_check.check_restore(_copy_and_edit(episode_path, tmp_path, edit), step=12)
    assert any("fixture_joint_qpos" in p for p in problems)


def test_object_poses_that_disagree_with_sim_state_are_reported(episode_path, tmp_path):
    def edit(f):
        f["object_poses"][12, 0, 0] += 0.1

    problems = restore_check.check_restore(_copy_and_edit(episode_path, tmp_path, edit), step=12)
    assert any("object_poses" in p for p in problems)


def test_a_corrupted_sim_state_is_reported(episode_path, tmp_path):
    def edit(f):
        f["sim_state"][12, 5] += 0.2

    problems = restore_check.check_restore(_copy_and_edit(episode_path, tmp_path, edit), step=12)
    assert problems


def test_an_out_of_range_step_is_rejected(episode_path):
    with pytest.raises(ValueError, match="step"):
        restore_check.check_restore(episode_path, step=999)
