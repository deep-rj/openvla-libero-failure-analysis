import itertools
import json

import pytest

pytest.importorskip("libero.libero")

from harness import policy, validation

pytestmark = pytest.mark.env

TASK = 9


def _cfg(tmp_path, **overrides):
    kwargs = {
        "out_dir": tmp_path,
        "tasks": [TASK],
        "episodes_per_task": 1,
        "determinism_task": TASK,
        "max_steps": 3,
        "truncation_steps": 2,
        "camera_size": 64,
    }
    return validation.ValidationConfig(**{**kwargs, **overrides})


def _same_seed(cfg):
    return policy.StubPolicy(seed=0, out_of_slice_rate=0.2), {}


def test_a_deterministic_policy_passes_every_validation_check(tmp_path):
    report = validation.run_validation(_cfg(tmp_path), policy_factory=_same_seed)

    assert report["t1"]["identical"] is True
    assert report["t2"]["identical"] is True
    assert report["t3"] == []
    assert report["t5"] == []
    assert report["truncation"]["terminated_by"] == "max_steps"
    assert report["truncation"]["success"] is False
    assert report["stopped_after_t1"] is False
    assert report["cost"]["projected_dollars"] > 0
    assert report["cost"]["mean_episode_bytes"] > 0
    assert (tmp_path / "validation_report.md").read_text().startswith("# Validation report")
    assert json.loads((tmp_path / "validation_report.json").read_text())["t1"]["identical"] is True


def test_a_nondeterministic_policy_stops_after_t1_with_the_margin_at_the_divergent_step(tmp_path):
    seeds = itertools.count()

    def different_each_run(cfg):
        return policy.StubPolicy(seed=next(seeds)), {}

    report = validation.run_validation(_cfg(tmp_path), policy_factory=different_each_run)

    assert report["t1"]["identical"] is False
    assert report["stopped_after_t1"] is True
    assert len(report["t1_margin"]) == 7
    assert "t2" not in report
    assert not (tmp_path / "logged").exists()
    assert (tmp_path / "validation_report.md").exists()
