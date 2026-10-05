import dataclasses

import numpy as np
from factories import make_episode, write_run

from harness import compare_runs


def _steps(seed):
    return make_episode(np.random.default_rng(seed))


def _ep(task, init, seed=0, *, success=False, steps=None, ep_seed=None):
    if steps is None:
        steps = _steps(100 * task + init if ep_seed is None else ep_seed)
    return {"task": task, "init": init, "seed": seed, "success": success, "steps": steps}


def _only(report):
    (episode,) = report["episodes"]
    return episode


def test_identical_runs_match_every_episode_and_report_no_divergence(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0), _ep(3, 1)])
    b = write_run(tmp_path / "b", [_ep(0, 0), _ep(3, 1)])
    report = compare_runs.compare_runs(a, b)
    assert report["n_matched"] == 2 and report["n_identical"] == 2
    assert report["n_action_divergent"] == 0 and report["n_outcome_flips"] == 0
    assert report["only_in_a"] == [] and report["only_in_b"] == []


def test_the_first_action_divergence_is_reported(tmp_path):
    steps = _steps(1)
    changed = list(steps)
    changed[15] = dataclasses.replace(steps[15], executed_action=steps[15].executed_action + 1e-9)
    a = write_run(tmp_path / "a", [_ep(0, 0, steps=steps)])
    b = write_run(tmp_path / "b", [_ep(0, 0, steps=changed)])
    ep = _only(compare_runs.compare_runs(a, b))
    assert not ep["identical"] and not ep["actions_identical"]
    assert ep["first_action_divergence"] == 15


def test_logit_only_differences_leave_the_actions_identical(tmp_path):
    steps = _steps(2)
    changed = list(steps)
    logits = steps[12].model.slice_logits.copy()
    logits[0, 100] = np.float16(logits[0, 100] + np.float16(0.0625))
    changed[12] = dataclasses.replace(
        steps[12], model=dataclasses.replace(steps[12].model, slice_logits=logits)
    )
    a = write_run(tmp_path / "a", [_ep(0, 0, steps=steps)])
    b = write_run(tmp_path / "b", [_ep(0, 0, steps=changed)])
    ep = _only(compare_runs.compare_runs(a, b))
    assert ep["actions_identical"] and not ep["logits_identical"]
    assert ep["first_logit_divergence"] == 12 and ep["first_action_divergence"] is None
    assert not ep["identical"]


def test_wait_steps_with_missing_logits_do_not_count_as_differences(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0)])
    b = write_run(tmp_path / "b", [_ep(0, 0)])
    assert _only(compare_runs.compare_runs(a, b))["logits_identical"]


def test_state_divergence_is_found_independently_of_actions(tmp_path):
    steps = _steps(3)
    changed = list(steps)
    world = steps[7].world
    changed[7] = dataclasses.replace(
        steps[7], world=dataclasses.replace(world, sim_state=world.sim_state + 1e-15)
    )
    a = write_run(tmp_path / "a", [_ep(0, 0, steps=steps)])
    b = write_run(tmp_path / "b", [_ep(0, 0, steps=changed)])
    ep = _only(compare_runs.compare_runs(a, b))
    assert ep["actions_identical"] and ep["first_state_divergence"] == 7


def test_episodes_are_matched_by_task_and_init_state_not_by_seed(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0, seed=0, ep_seed=5)])
    b = write_run(tmp_path / "b", [_ep(0, 0, seed=1, ep_seed=5)])
    ep = _only(compare_runs.compare_runs(a, b))
    assert ep["id_a"].endswith("-s0-none") and ep["id_b"].endswith("-s1-none")
    assert ep["identical"]


def test_outcome_flips_are_counted(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0, success=True), _ep(1, 0, success=False)])
    b = write_run(tmp_path / "b", [_ep(0, 0, success=False), _ep(1, 0, success=False)])
    report = compare_runs.compare_runs(a, b)
    assert report["n_outcome_flips"] == 1


def test_unmatched_episodes_are_listed_on_each_side(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0), _ep(2, 0)])
    b = write_run(tmp_path / "b", [_ep(0, 0), _ep(4, 1)])
    report = compare_runs.compare_runs(a, b)
    assert report["n_matched"] == 1
    assert report["only_in_a"] == [(2, 0)] and report["only_in_b"] == [(4, 1)]


def test_episodes_of_different_length_are_not_identical(tmp_path):
    rng = np.random.default_rng(9)
    long_steps = make_episode(rng, n_act=25)
    a = write_run(tmp_path / "a", [_ep(0, 0, steps=long_steps)])
    b = write_run(tmp_path / "b", [_ep(0, 0, steps=long_steps[:-5])])
    ep = _only(compare_runs.compare_runs(a, b))
    assert not ep["identical"] and (ep["length_a"], ep["length_b"]) == (35, 30)
    assert ep["actions_identical"]


def test_markdown_lists_each_matched_episode(tmp_path):
    a = write_run(tmp_path / "a", [_ep(0, 0)])
    b = write_run(tmp_path / "b", [_ep(0, 0)])
    text = compare_runs.render_markdown(compare_runs.compare_runs(a, b))
    assert text.startswith("# Run comparison") and "identical" in text
