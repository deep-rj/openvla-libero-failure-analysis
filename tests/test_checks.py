import numpy as np
import pytest
from factories import MASK, Q01, Q99, make_episode

from harness import checks, schema

STATS = {"q01": Q01, "q99": Q99, "mask": MASK}


@pytest.fixture
def episode(tmp_path):
    steps = make_episode(np.random.default_rng(0))
    schema.write_episode(tmp_path / "ep.h5", steps, {})
    arrays, _ = schema.read_episode(tmp_path / "ep.h5")
    return arrays, [s.executed_action.copy() for s in steps]


def test_a_consistent_episode_has_no_violations(episode):
    arrays, sent = episode
    assert checks.check_self_consistency(arrays, STATS, sent) == []
    assert checks.check_wait_alignment(arrays, expected_wait_steps=10) == []


def test_chosen_token_differing_from_the_argmax_is_reported(episode):
    arrays, sent = episode
    arrays["chosen_token_id"][15, 2] += 1
    problems = checks.check_self_consistency(arrays, STATS, sent)
    assert any("step 15" in p and "argmax" in p for p in problems)


def test_a_wrong_clipped_flag_is_reported(episode):
    arrays, sent = episode
    arrays["clipped_flag"][12, 0] = True
    problems = checks.check_self_consistency(arrays, STATS, sent)
    assert any("step 12" in p and "clipped" in p for p in problems)


def test_out_of_slice_tokens_must_be_flagged_as_clipped(episode):
    arrays, sent = episode
    arrays["chosen_token_id"][13, 1] = 29871
    arrays["argmax_full_id"][13, 1] = 29871
    problems = checks.check_self_consistency(arrays, STATS, sent)
    assert any("step 13" in p and "clipped" in p for p in problems)


@pytest.mark.parametrize(
    "stage", ["bin_idx", "normalized_action", "unnormalized_action", "executed_action"]
)
def test_decoded_stages_that_do_not_follow_from_the_token_are_reported(episode, stage):
    arrays, sent = episode
    arrays[stage][14, 3] += 1
    problems = checks.check_self_consistency(arrays, STATS, sent)
    assert any("step 14" in p and stage in p for p in problems)


def test_an_executed_action_that_differs_from_what_was_sent_is_reported(episode):
    arrays, sent = episode
    sent[16][0] += 1e-9
    problems = checks.check_self_consistency(arrays, STATS, sent)
    assert any("step 16" in p and "sent" in p for p in problems)


def test_wait_step_actions_are_also_compared_with_what_was_sent(episode):
    arrays, sent = episode
    sent[4][6] = 1.0
    assert any("step 4" in p for p in checks.check_self_consistency(arrays, STATS, sent))


def test_the_sent_action_count_must_match_the_step_count(episode):
    arrays, sent = episode
    assert checks.check_self_consistency(arrays, STATS, sent[:-1])


def test_wait_alignment_requires_the_first_action_step_at_the_wait_count(tmp_path):
    schema.write_episode(tmp_path / "e.h5", make_episode(np.random.default_rng(0), n_wait=9), {})
    arrays, _ = schema.read_episode(tmp_path / "e.h5")
    assert any("first" in p for p in checks.check_wait_alignment(arrays, expected_wait_steps=10))


def test_a_model_output_on_a_wait_step_is_reported(episode):
    arrays, _ = episode
    arrays["chosen_token_id"][2, 0] = 31800
    problems = checks.check_wait_alignment(arrays, expected_wait_steps=10)
    assert any("step 2" in p for p in problems)


def test_logits_on_a_wait_step_must_be_sentinel(episode):
    arrays, _ = episode
    arrays["slice_logits"][3, 0, 0] = 1.0
    assert any("step 3" in p for p in checks.check_wait_alignment(arrays, 10))


def test_a_missing_model_output_after_the_wait_steps_is_reported(episode):
    arrays, _ = episode
    arrays["chosen_token_id"][20, :] = -1
    problems = checks.check_wait_alignment(arrays, expected_wait_steps=10)
    assert any("step 20" in p for p in problems)
