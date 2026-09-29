from pathlib import Path

import pytest

from harness import cli


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0:3", [0, 1, 2]), ("0,3,5", [0, 3, 5]), ("7", [7]), ("2:4,9", [2, 3, 9])],
)
def test_index_lists_accept_ranges_and_commas(text, expected):
    assert cli.parse_indices(text) == expected


@pytest.mark.parametrize("text", ["", "a", "3:1", "-1", "1:2:3"])
def test_bad_index_lists_are_rejected(text):
    with pytest.raises(ValueError):
        cli.parse_indices(text)


def test_defaults_describe_a_logged_stub_run():
    cfg = cli.parse_args(["--tasks", "0", "--init-states", "0:2", "--out", "runs/x"])
    assert cfg.policy == "stub"
    assert cfg.mode == "logged"
    assert cfg.tasks == [0]
    assert cfg.init_states == [0, 1]
    assert cfg.out_dir == Path("runs/x")
    assert cfg.video is True
    assert cfg.allow_dirty is False
    assert cfg.max_steps is None


def test_flags_map_onto_the_run_config():
    cfg = cli.parse_args(
        [
            "--tasks", "0:10", "--init-states", "0:25", "--out", "o", "--policy", "openvla",
            "--mode", "reference", "--max-steps", "60", "--no-video", "--allow-dirty",
            "--seed", "3",
        ]
    )  # fmt: skip
    assert cfg.tasks == list(range(10))
    assert len(cfg.init_states) == 25
    assert (cfg.policy, cfg.mode, cfg.max_steps, cfg.seed) == ("openvla", "reference", 60, 3)
    assert cfg.video is False
    assert cfg.allow_dirty is True


def test_unknown_policy_or_mode_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["--tasks", "0", "--init-states", "0", "--out", "o", "--policy", "gpt"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--tasks", "0", "--init-states", "0", "--out", "o", "--mode", "x"])
