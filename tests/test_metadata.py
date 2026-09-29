import json
import re

import numpy as np
from factories import make_step

from harness import metadata, policy, rollout, schema
from harness.world import FixtureJoint, SceneLayout

SPEC = rollout.EpisodeSpec("libero_10", task_id=9, init_state_idx=4, seed=2)
LAYOUT = SceneLayout(
    object_names=("porcelain_mug_1", "white_yellow_mug_1"),
    fixture_joints=(FixtureJoint("microwave_1_microjoint", "hinge", (-2.09, 0.0), 23),),
    static_fixture_poses={"microwave_1": (0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0)},
)
ENV_INFO = {"git": {"sha": "a" * 40, "dirty": False}, "versions": {"torch": "2.2.0"}, "gpu": None}


def _result(success=False, terminated_by="max_steps"):
    return rollout.EpisodeResult(
        executed_actions=np.zeros((25, 7)),
        success=success,
        terminated_by=terminated_by,
        num_policy_steps=15,
        wall_s=1.0,
        steps=None,
        agent_frames=None,
        wrist_frames=None,
        is_wait_steps=[True] * 10 + [False] * 15,
    )


def _build(**overrides):
    kwargs = {
        "spec": SPEC,
        "task_label": "put the yellow and white mug in the microwave and close it",
        "layout": LAYOUT,
        "result": _result(),
        "mjcf_xml": "<mujoco/>",
        "provenance": {"hub_code_commit": "c" * 40, "checkpoint_commit": "d" * 40},
        "action_stats": policy.LIBERO_10_ACTION_STATS,
        "max_steps": 520,
        "checkpoint": "openvla/openvla-7b-finetuned-libero-10",
        "unnorm_key": "libero_10",
        "center_crop": True,
        "environment": ENV_INFO,
    }
    return metadata.episode_metadata(**{**kwargs, **overrides})


def test_prompt_format_is_the_openvla_template():
    assert policy.build_prompt("Pick Up The Book") == (
        "In: What action should the robot take to pick up the book?\nOut:"
    )


def test_metadata_records_everything_the_spec_asks_for():
    meta = _build()
    for key in (
        "episode_id", "task_id", "seed", "perturbation_config_id", "init_state_idx",
        "object_names", "fixture_joints", "static_fixture_poses", "mjcf_sha256",
        "checkpoint", "checkpoint_commit", "hub_code_commit", "unnorm_key", "center_crop",
        "image_flip", "prompt", "task_instruction", "git", "versions", "gpu",
        "final_success", "episode_length", "terminated_by", "control_freq_hz",
        "num_wait_steps", "max_steps", "action_stats",
    ):  # fmt: skip
        assert key in meta, key


def test_metadata_values_come_from_the_spec_result_and_layout():
    meta = _build(result=_result(success=True, terminated_by="success"))
    assert meta["episode_id"] == "libero_10-t09-i004-s2-none"
    assert meta["object_names"] == ["porcelain_mug_1", "white_yellow_mug_1"]
    assert meta["fixture_joints"][0]["name"] == "microwave_1_microjoint"
    assert meta["final_success"] is True
    assert meta["episode_length"] == 25
    assert meta["prompt"].startswith("In: What action should the robot take to put the yellow")
    assert meta["image_flip"] is True
    assert meta["control_freq_hz"] == 20
    assert meta["num_wait_steps"] == rollout.NUM_STEPS_WAIT
    assert meta["action_stats"]["mask"] == [True] * 6 + [False]


def test_metadata_survives_an_episode_file_round_trip(tmp_path):
    steps = [make_step(np.random.default_rng(0), k, wait=True) for k in range(2)]
    schema.write_episode(tmp_path / "ep.h5", steps, _build())
    _, meta = schema.read_episode(tmp_path / "ep.h5")
    assert meta["static_fixture_poses"]["microwave_1"][2] == 2.0
    json.dumps(meta)


def test_mjcf_hash_changes_with_the_model():
    assert metadata.mjcf_sha256("<a/>") == metadata.mjcf_sha256("<a/>")
    assert metadata.mjcf_sha256("<a/>") != metadata.mjcf_sha256("<b/>")


def test_git_state_reports_a_sha_and_dirtiness_for_a_repo():
    state = metadata.git_state(".")
    assert re.fullmatch(r"[0-9a-f]{40}", state["sha"])
    assert isinstance(state["dirty"], bool)


def test_git_state_outside_a_repo_is_unknown(tmp_path):
    assert metadata.git_state(tmp_path) == {"sha": None, "dirty": None}


def test_library_versions_report_missing_packages_as_none():
    versions = metadata.library_versions(["numpy", "definitely-not-installed"])
    assert versions == {"numpy": np.__version__, "definitely-not-installed": None}
