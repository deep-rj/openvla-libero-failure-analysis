import numpy as np
import pytest

from harness import decode, policy

V = 32064


def _logits(rng, out_of_slice_scale=1.0):
    logits = rng.normal(scale=out_of_slice_scale, size=(7, V)).astype(np.float32)
    logits[:, 31744:32000] = rng.normal(scale=4, size=(7, 256))
    return logits


def test_summary_slices_the_action_tokens_and_keeps_them_in_fp16():
    logits = _logits(np.random.default_rng(0))
    out = policy.summarize_logits(logits, chosen=logits.argmax(-1), latency_ms=1.5)

    assert out.slice_logits.shape == (7, 256)
    assert out.slice_logits.dtype == np.float16
    assert np.array_equal(out.slice_logits, logits[:, 31744:32000].astype(np.float16))
    assert out.inference_latency_ms == 1.5


def test_summary_lse_covers_the_full_head_not_just_the_slice():
    logits = _logits(np.random.default_rng(1))
    logits[3, 100] = 50.0
    out = policy.summarize_logits(logits, chosen=logits.argmax(-1), latency_ms=0.0)

    reference = np.log(np.exp(logits.astype(np.float64) - logits.max(-1, keepdims=True)).sum(-1))
    reference += logits.max(-1)
    assert out.lse_full.dtype == np.float32
    assert out.lse_full == pytest.approx(reference, rel=1e-6)
    slice_lse = np.log(np.exp(out.slice_logits.astype(np.float64)).sum(-1))
    assert slice_lse[3] < out.lse_full[3] - 10


def test_summary_argmax_is_over_the_full_head_and_may_leave_the_slice():
    logits = _logits(np.random.default_rng(2))
    logits[5, 29871] = 100.0
    logits[6, 32010] = 100.0
    out = policy.summarize_logits(logits, chosen=logits.argmax(-1), latency_ms=0.0)

    assert out.argmax_full_id.dtype == np.int32
    assert out.argmax_full_id[5] == 29871
    assert out.argmax_full_id[6] == 32010
    assert np.array_equal(out.argmax_full_id, out.chosen_token_id)


def test_summary_reports_the_generated_token_separately_from_the_argmax():
    logits = _logits(np.random.default_rng(3))
    chosen = logits.argmax(-1).copy()
    chosen[2] = 31900
    out = policy.summarize_logits(logits, chosen=chosen, latency_ms=0.0)
    assert out.chosen_token_id[2] == 31900
    assert out.argmax_full_id[2] != 31900


def test_bf16_valued_logits_survive_the_fp16_slice_exactly():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    logits = torch.tensor(_logits(rng, out_of_slice_scale=10.0)).bfloat16().float().numpy()
    out = policy.summarize_logits(logits, chosen=logits.argmax(-1), latency_ms=0.0)
    assert np.array_equal(out.slice_logits.astype(np.float32), logits[:, 31744:32000])


def test_stub_is_reproducible_from_its_seed():
    a = policy.StubPolicy(seed=7).predict(None, "pick up the book")
    b = policy.StubPolicy(seed=7).predict(None, "pick up the book")
    c = policy.StubPolicy(seed=8).predict(None, "pick up the book")
    assert np.array_equal(a.slice_logits, b.slice_logits, equal_nan=True)
    assert not np.array_equal(a.slice_logits, c.slice_logits)


def test_stub_generation_is_greedy_over_the_full_head():
    stub = policy.StubPolicy(seed=0, out_of_slice_rate=0.5)
    for _ in range(30):
        out = stub.predict(None, "task")
        assert np.array_equal(out.chosen_token_id, out.argmax_full_id)


def test_stub_can_emit_out_of_slice_tokens_that_decode_as_clipped():
    stub = policy.StubPolicy(seed=0, out_of_slice_rate=0.5)
    clipped = 0
    for _ in range(30):
        out = stub.predict(None, "task")
        stats = stub.action_stats
        stages = decode.decode_action(
            out.chosen_token_id, stats["q01"], stats["q99"], stats["mask"]
        )
        clipped += int(stages.clipped_flag.sum())
    assert clipped > 0


def test_stub_without_injection_stays_inside_the_action_slice():
    stub = policy.StubPolicy(seed=0)
    ids = np.concatenate([stub.predict(None, "task").chosen_token_id for _ in range(30)])
    assert ((ids >= 31745) & (ids <= 31999)).all()


def test_stub_actions_stats_match_the_libero_layout():
    stats = policy.StubPolicy(seed=0).action_stats
    assert stats["mask"].tolist() == [True] * 6 + [False]
    assert all(np.asarray(stats[k]).shape == (7,) for k in ("q01", "q99"))


def test_stub_reference_action_is_the_decoded_greedy_action_of_the_same_seed():
    stats = policy.LIBERO_10_ACTION_STATS
    logged = policy.StubPolicy(seed=5, out_of_slice_rate=0.3).predict(None, "task")
    expected = decode.decode_action(
        logged.chosen_token_id, stats["q01"], stats["q99"], stats["mask"]
    ).unnormalized_action
    reference = policy.StubPolicy(seed=5, out_of_slice_rate=0.3).reference_action(None, "task")
    assert np.array_equal(reference, expected)
