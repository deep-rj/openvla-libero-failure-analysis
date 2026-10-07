import math

import numpy as np
import pytest

from harness import decode, manifest, policy, schema
from harness import instrument_checks as ic
from harness import pilot_report as pr

N_TOKENS = 256


def _slice(dims: dict[int, dict[int, float]], base=-50.0):
    """One step: per dimension, {slice index j: logit}; every other logit is `base`."""
    x = np.full((1, 7, N_TOKENS), base, dtype=np.float16)
    for dim, tokens in dims.items():
        for j, value in tokens.items():
            x[0, dim, j] = value
    return x


def _lse(x):
    x = np.asarray(x, np.float64)
    peak = x.max(-1, keepdims=True)
    return peak[..., 0] + np.log(np.exp(x - peak).sum(-1))


def _metrics(x):
    ids = decode.ACTION_SLICE_START + x.astype(np.float32).argmax(-1)
    return pr.derive_step_metrics(x, _lse(x).astype(np.float32), ids, ids)


def test_logits_with_more_than_eight_significant_bits_are_not_bf16():
    x = np.array([1.0, 1.0 + 2**-7, 1.0 + 2**-9, -24.5, 0.0], dtype=np.float16)
    assert ic.non_bf16(x).tolist() == [False, False, True, False, False]


def test_fp16_subnormals_are_not_counted_since_storage_rounds_them():
    x = np.array([2**-20 + 2**-24], dtype=np.float16)
    assert not ic.non_bf16(x)[0]


def test_lse_excess_is_positive_only_when_the_slice_outweighs_the_full_head():
    x = _slice({0: {10: 3.0}}, base=0.0)
    lse_slice = _lse(x)
    assert ic.lse_excess(x, lse_slice + math.log(2)) == pytest.approx(np.full((1, 7), -math.log(2)))
    assert ic.lse_excess(x, lse_slice - 0.01) == pytest.approx(np.full((1, 7), 0.01), abs=1e-6)


def test_an_exact_tie_cannot_have_fewer_than_two_effective_bins():
    exp_h = np.array([1.0, 2.0, 1.36, 1.38])
    margin_nats = np.array([0.0, 0.0, 1.0, 1.0])
    assert ic.entropy_bound_violations(exp_h, margin_nats).tolist() == [True, False, True, False]


def test_entropy_and_margins_from_the_report_satisfy_the_bound():
    rng = np.random.default_rng(0)
    x = (rng.normal(scale=3, size=(500, 7, N_TOKENS))).astype(np.float16)
    m = _metrics(x)
    assert not ic.entropy_bound_violations(m["exp_h"], ic.bin_margin_nats(x)).any()


def test_margins_must_agree_when_neither_edge_token_can_reach_the_top_two():
    x = _slice({0: {100: 10.0, 120: 9.5}})
    m = _metrics(x)
    covered, violated = ic.merge_invariant(x, m["token_margin_ulps"], m["bin_margin_ulps"])
    assert covered[0, 0] and not violated[0, 0]

    _, violated = ic.merge_invariant(x, m["token_margin_ulps"], m["bin_margin_ulps"] + 1.0)
    assert violated[0, 0]


def test_the_merge_invariant_does_not_cover_steps_with_an_edge_token_near_the_top():
    x = _slice({0: {100: 10.0, 120: 9.5, 0: 9.0}})
    m = _metrics(x)
    covered, _ = ic.merge_invariant(x, m["token_margin_ulps"], m["bin_margin_ulps"])
    assert not covered[0, 0]


def test_raw_action_zero_maps_to_the_bin_the_tokenizer_would_train_on():
    bins = ic.raw_action_bin(
        np.zeros(2), np.array([-1.0, 0.0]), np.array([1.0, 1.0]), [True, False]
    )
    assert bins.tolist() == [127, 127]
    edge = ic.raw_action_bin(np.array([1.0]), np.array([-1.0]), np.array([1.0]), [True])
    assert edge.tolist() == [254]


def test_occupancy_reports_spread_and_gaps_of_chosen_bins():
    counts = np.zeros(decode.N_BINS, dtype=int)
    counts[[10, 12, 14]] = [80, 15, 5]
    occ = ic.occupancy(counts)
    assert occ["occupied_bins"] == 3
    assert occ["bins_for_90_percent"] == 2
    assert occ["empty_fraction_within_range"] == pytest.approx(0.4)
    assert occ["top_bins"][0] == {"bin": 10, "fraction": pytest.approx(0.8)}


def _clean_episode(rng, n_wait=3, n_act=6):
    from factories import MASK, Q01, Q99, make_step

    steps = []
    for i in range(n_wait + n_act):
        step = make_step(rng, i, wait=i < n_wait)
        if step.model is not None:
            full = rng.normal(scale=4, size=(7, policy.HEAD_WIDTH)).astype(np.float32)
            full[:, decode.ACTION_SLICE_START + 40] += 30.0
            full = (full.view(np.uint32) & 0xFFFF0000).view(np.float32)
            model = policy.summarize_logits(full, full.argmax(-1), 0.0)
            decoded = decode.decode_action(model.chosen_token_id, Q01, Q99, MASK)
            step = type(step)(**{**step.__dict__, "model": model, "decoded": decoded,
                                 "executed_action": decoded.executed_action})  # fmt: skip
        steps.append(step)
    return steps


def _write_run(tmp_path, episodes):
    from factories import MASK, Q01, Q99

    meta = {"action_stats": {"q01": Q01.tolist(), "q99": Q99.tolist(), "mask": MASK.tolist()}}
    (tmp_path / "episodes").mkdir()
    (tmp_path / "actions").mkdir()
    for i, steps in enumerate(episodes):
        episode_id = f"e{i}"
        schema.write_episode(tmp_path / f"episodes/{episode_id}.h5", steps, meta)
        np.save(
            tmp_path / f"actions/{episode_id}.npy", np.stack([s.executed_action for s in steps])
        )
        manifest.append_row(
            tmp_path / "manifest.csv",
            {
                "episode_id": episode_id, "task_suite": "libero_10", "task_id": 0,
                "task_instruction": "t", "init_state_idx": i, "seed": 0, "success": True,
                "num_steps": len(steps), "terminated_by": "success",
                "episode_file": f"episodes/{episode_id}.h5", "video_file": "",
                "git_sha": "x", "seconds_per_step": 0.25,
            },
        )  # fmt: skip


def test_a_consistent_run_has_no_violations(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "NUM_STEPS_WAIT", 3)
    rng = np.random.default_rng(0)
    _write_run(tmp_path, [_clean_episode(rng), _clean_episode(rng)])

    report = ic.check_run(tmp_path)

    assert report["n_episodes"] == 2
    assert report["n_steps"] == 12
    assert report["t3"] == {"count": 0, "examples": []}
    assert report["t5"] == {"count": 0, "examples": []}
    for d in range(7):
        s = report["per_dim"][d]
        assert s["lse_excess_over_tol"] == 0
        assert s["non_bf16_logits"] == 0
        assert s["exp_h_out_of_range"] == 0
        assert s["entropy_bound_violations"] == 0
        assert s["merge_violations"] == 0
        assert s["merge_covered"] == 12
        assert s["chosen_bins"]["occupied_bins"] >= 1
    assert ic.render_markdown(report).startswith("# Instrument checks")


def test_a_full_head_lse_below_the_slice_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(ic, "NUM_STEPS_WAIT", 3)
    steps = _clean_episode(np.random.default_rng(0))
    k = 4
    model = steps[k].model
    steps[k] = type(steps[k])(**{**steps[k].__dict__, "model": type(model)(
        **{**model.__dict__, "lse_full": model.lse_full - np.float32(0.01)})})  # fmt: skip
    _write_run(tmp_path, [steps])

    report = ic.check_run(tmp_path)

    assert [report["per_dim"][d]["lse_excess_over_tol"] for d in range(7)] == [1] * 7
