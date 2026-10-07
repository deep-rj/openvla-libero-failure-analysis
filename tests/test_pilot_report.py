import json
import math

import numpy as np
import pytest

from harness import pilot_report as pr

N_TOKENS = 256


def _slice(dims: dict[int, dict[int, float]], base=-50.0):
    """One step: per dimension, {slice index j: logit}; every other logit is `base`."""
    x = np.full((1, 7, N_TOKENS), base, dtype=np.float16)
    for dim, tokens in dims.items():
        for j, value in tokens.items():
            x[0, dim, j] = value
    return x


def _metrics(x, lse_full=None, argmax=None, chosen=None):
    x32 = x.astype(np.float32)
    peak = x32.max(-1)
    lse_slice = peak + np.log(np.exp(x32 - peak[..., None]).sum(-1))
    lse_full = lse_slice if lse_full is None else lse_full
    ids = 31744 + x32.argmax(-1) if argmax is None else argmax
    return pr.derive_step_metrics(x, lse_full, ids, ids if chosen is None else chosen)


def test_a_flat_slice_has_the_entropy_of_255_bins_with_the_edge_pair_merged():
    m = _metrics(np.zeros((1, 7, N_TOKENS), dtype=np.float16))
    p = np.array([1 / 256] * 254 + [2 / 256])
    expected = math.exp(-(p * np.log(p)).sum())
    assert m["exp_h"] == pytest.approx(np.full((1, 7), expected), rel=1e-4)


def test_one_dominant_logit_has_about_one_effective_bin():
    m = _metrics(_slice({0: {100: 40.0}}, base=0.0))
    assert m["exp_h"][0, 0] == pytest.approx(1.0, abs=1e-4)


def test_the_two_edge_tokens_count_as_one_action():
    m = _metrics(_slice({0: {0: 10.0, 1: 10.0}}, base=0.0))
    assert m["exp_h"][0, 0] < 1.2


def test_out_of_slice_mass_is_one_minus_the_slice_share_of_the_full_head():
    x = _slice({0: {10: 3.0}}, base=0.0)
    lse_slice = np.log(np.exp(x.astype(np.float32)).sum(-1))
    m = _metrics(x, lse_full=lse_slice + math.log(2))
    assert m["oos_mass"] == pytest.approx(np.full((1, 7), 0.5), abs=1e-5)
    assert _metrics(x)["oos_mass"] == pytest.approx(np.zeros((1, 7)), abs=1e-6)


def test_argmax_outside_the_slice_is_split_into_text_and_padding_tokens():
    x = _slice({})
    argmax = np.array([[29871, 32010, 31800, 31744, 0, 32063, 31999]])
    m = _metrics(x, argmax=argmax)
    assert m["argmax_text"][0].tolist() == [True, False, False, False, True, False, False]
    assert m["argmax_padding"][0].tolist() == [False, True, False, False, False, True, False]
    assert m["chosen_edge"][0].tolist() == [False, False, False, True, False, False, False]


def test_token_margins_are_measured_in_bf16_ulps_of_the_top_logit():
    x = _slice(
        {
            0: {10: 12.0, 20: 11.9375},  # binade [8, 16): ulp 0.0625
            1: {10: 20.0, 20: 19.75},  # binade [16, 32): ulp 0.125
            2: {10: 5.0, 20: 4.0},  # binade [4, 8): ulp 0.03125
        }
    )
    m = _metrics(x)
    assert m["token_margin_ulps"][0, :3] == pytest.approx([1.0, 2.0, 32.0])


def test_bin_distance_between_the_top_two_bins():
    x = _slice(
        {
            0: {100: 10.0, 101: 9.0},
            1: {100: 10.0, 105: 9.0},
            2: {100: 10.0, 150: 9.0},
        }
    )
    assert _metrics(x)["bin_distance"][0, :3].tolist() == [1, 5, 50]


def test_bin_margin_uses_the_merged_edge_bin_while_token_margin_does_not():
    x = _slice({0: {0: 10.0, 1: 10.0, 50: 9.0}})
    m = _metrics(x)
    assert m["token_margin_ulps"][0, 0] == 0.0
    merged = np.logaddexp(10.0, 10.0)
    assert m["bin_margin_ulps"][0, 0] == pytest.approx((merged - 9.0) / 0.0625, rel=1e-3)
    assert m["bin_distance"][0, 0] == 254 - (255 - 50)


def test_summary_reports_quantiles_cutoff_fractions_and_the_bin_distance_crosstab():
    n = 200
    rng = np.random.default_rng(0)
    metrics = {
        "argmax_text": np.zeros((n, 7), bool),
        "argmax_padding": np.zeros((n, 7), bool),
        "chosen_edge": np.zeros((n, 7), bool),
        "oos_mass": rng.uniform(0, 0.1, (n, 7)),
        "slice_exceeds_full": np.zeros((n, 7), bool),
        "exp_h": np.tile(np.linspace(1.0, 3.0, n)[:, None], (1, 7)),
        "token_margin_ulps": np.tile(np.linspace(0, 40, n)[:, None], (1, 7)),
        "bin_margin_ulps": np.tile(np.linspace(0, 40, n)[:, None], (1, 7)),
        "bin_distance": np.tile(np.arange(n)[:, None] % 30 + 1, (1, 7)),
    }
    out = pr.summarize_pooled(metrics)

    dim0 = out["per_dim"][0]
    assert dim0["n_steps"] == n
    assert set(dim0["exp_h"]["quantiles"]) == {"5", "25", "50", "75", "95"}
    assert dim0["exp_h"]["quantiles"]["50"] == pytest.approx(2.0, abs=0.02)
    assert dim0["exp_h"]["fraction_at_most"]["1.05"] == pytest.approx(5 / 200)
    assert dim0["exp_h"]["fraction_at_most"]["2.0"] == pytest.approx(0.5, abs=0.02)
    assert dim0["token_margin_ulps"]["fraction_at_most"]["16"] == pytest.approx(0.4, abs=0.02)
    crosstab = dim0["near_tie_bin_distance"]["16"]
    assert set(crosstab) == {"adjacent", "2-9", ">=10"}
    assert sum(crosstab.values()) == round(dim0["bin_margin_ulps"]["fraction_at_most"]["16"] * n)


def test_pooled_metrics_concatenate_across_episodes():
    a = _metrics(_slice({0: {10: 5.0}}, base=0.0))
    b = _metrics(np.concatenate([_slice({0: {10: 5.0}}, base=0.0)] * 3))
    pooled = pr.pool([a, b])
    assert pooled["exp_h"].shape == (4, 7)


def test_the_report_is_pooled_only_and_names_no_episode(tmp_path):
    from factories import make_episode

    from harness import manifest, schema

    rng = np.random.default_rng(0)
    for i, task in enumerate([0, 0, 3]):
        episode_id = f"libero_10-t{task:02d}-i{i:03d}-s0-none"
        (tmp_path / "episodes").mkdir(exist_ok=True)
        schema.write_episode(tmp_path / f"episodes/{episode_id}.h5", make_episode(rng), {})
        manifest.append_row(
            tmp_path / "manifest.csv",
            {
                "episode_id": episode_id, "task_suite": "libero_10", "task_id": task,
                "task_instruction": "t", "init_state_idx": i, "seed": 0, "success": i != 1,
                "num_steps": 20, "terminated_by": "max_steps",
                "episode_file": f"episodes/{episode_id}.h5", "video_file": "",
                "git_sha": "x", "seconds_per_step": 0.25,
            },
        )  # fmt: skip

    report = pr.build_report(tmp_path)

    assert report["outcomes_per_task"] == {
        "0": {"successes": 1, "failures": 1},
        "3": {"successes": 1, "failures": 0},
    }
    assert report["n_episodes"] == 3
    assert report["n_steps_pooled"] == 3 * 20
    text = json.dumps(report)
    assert "libero_10-t" not in text
    assert "per_dim" in report["pooled"]
    assert pr.render_markdown(report).startswith("# Pilot report")


def test_stage_a_anomalies_are_passed_through_when_given(tmp_path):
    from factories import make_episode

    from harness import manifest, schema

    (tmp_path / "episodes").mkdir()
    schema.write_episode(tmp_path / "episodes/a.h5", make_episode(np.random.default_rng(0)), {})
    manifest.append_row(
        tmp_path / "manifest.csv",
        {
            "episode_id": "a", "task_suite": "libero_10", "task_id": 0, "task_instruction": "t",
            "init_state_idx": 0, "seed": 0, "success": True, "num_steps": 20,
            "terminated_by": "success", "episode_file": "episodes/a.h5", "video_file": "",
            "git_sha": "x", "seconds_per_step": 0.25,
        },
    )  # fmt: skip
    stage_a = {"t1": {"identical": True}, "t3": ["e1: step 3 clipped_flag"], "t5": []}

    report = pr.build_report(tmp_path, stage_a_report=stage_a)
    assert report["anomalies"]["t3"] == ["e1: step 3 clipped_flag"]
    assert report["anomalies"]["t1"] == {"identical": True}


def test_only_the_requested_episodes_are_pooled(tmp_path):
    from factories import make_episode

    from harness import manifest, schema

    (tmp_path / "episodes").mkdir()
    rng = np.random.default_rng(0)
    for episode_id in ("a", "b"):
        schema.write_episode(tmp_path / f"episodes/{episode_id}.h5", make_episode(rng), {})
        manifest.append_row(
            tmp_path / "manifest.csv",
            {
                "episode_id": episode_id, "task_suite": "libero_10", "task_id": 0,
                "task_instruction": "t", "init_state_idx": 0, "seed": 0, "success": False,
                "num_steps": 20, "terminated_by": "max_steps",
                "episode_file": f"episodes/{episode_id}.h5", "video_file": "",
                "git_sha": "x", "seconds_per_step": 0.25,
            },
        )  # fmt: skip
    report = pr.build_report(tmp_path, episode_ids={"a"})
    assert report["n_episodes"] == 1
    assert report["n_steps_pooled"] == 20


def test_a_slice_lse_above_the_full_head_is_flagged_rather_than_clamped_away():
    x = _slice({0: {10: 3.0}}, base=0.0)
    lse_slice = np.log(np.exp(x.astype(np.float64)).sum(-1))
    m = _metrics(x, lse_full=(lse_slice - 0.01).astype(np.float32))
    assert m["oos_mass"] == pytest.approx(np.zeros((1, 7)), abs=1e-6)
    assert m["slice_exceeds_full"].all()
    assert not _metrics(x)["slice_exceeds_full"].any()


def test_the_report_counts_steps_where_the_slice_exceeds_the_full_head():
    m = _metrics(np.concatenate([_slice({0: {10: 3.0}}, base=0.0)] * 2))
    m["slice_exceeds_full"][1, 0] = True
    s = pr.summarize_pooled(m)["per_dim"]
    assert s[0]["oos_mass"]["slice_exceeds_full_steps"] == 1
    assert s[1]["oos_mass"]["slice_exceeds_full_steps"] == 0
