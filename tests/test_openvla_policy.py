from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("experiments.robot.openvla_utils")

from experiments.robot import openvla_utils  # noqa: E402

from harness import openvla_policy as op  # noqa: E402
from harness.policy import LIBERO_10_ACTION_STATS  # noqa: E402

V = 32064
UNNORM = "libero_10"


class FakeBatch(dict):
    def to(self, *args, **kwargs):
        return FakeBatch(self)


class FakeProcessor:
    def __init__(self, last_prompt_token=2):
        self.calls = []
        self._last = last_prompt_token

    def __call__(self, prompt, image):
        self.calls.append((prompt, np.array(image)))
        return FakeBatch(
            input_ids=torch.tensor([[1, 5, self._last]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            pixel_values=torch.zeros(1, 6, 8, 8),
        )


class FakeVLA:
    def __init__(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.logits = torch.randn(7, V, generator=gen)
        self.logits[:, 31744:32000] += 4 * torch.randn(7, 256, generator=gen)
        self.generate_calls = []
        self.predict_calls = []
        self.norm_stats = {
            UNNORM: {
                "action": {k: v.tolist() for k, v in LIBERO_10_ACTION_STATS.items()},
            }
        }

    def get_action_dim(self, key):
        return 7

    def get_action_stats(self, key):
        return self.norm_stats[key]["action"]

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.generate_calls.append((input_ids, max_new_tokens, kwargs))
        chosen = self.logits.argmax(-1)
        return SimpleNamespace(
            sequences=torch.cat([input_ids, chosen[None]], dim=1),
            logits=tuple(self.logits[i][None] for i in range(7)),
        )

    def predict_action(self, **kwargs):
        self.predict_calls.append(kwargs)
        return np.arange(7, dtype=np.float64)


def _policy(center_crop=True, last_prompt_token=2):
    vla, processor = FakeVLA(), FakeProcessor(last_prompt_token)
    cfg = op.PolicyConfig(center_crop=center_crop)
    return op.OpenVLAPolicy(vla, processor, cfg), vla, processor


def _image(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (224, 224, 3), dtype=np.uint8)


def test_predict_returns_the_last_seven_generated_tokens_and_their_logits():
    policy, vla, _ = _policy()
    out = policy.predict(_image(), "put the bowl in the drawer")

    expected = vla.logits.argmax(-1).numpy()
    assert np.array_equal(out.chosen_token_id, expected)
    assert np.array_equal(out.argmax_full_id, expected)
    assert np.array_equal(out.slice_logits, vla.logits[:, 31744:32000].half().numpy())
    assert out.inference_latency_ms > 0


def test_generation_is_greedy_with_logits_returned_and_extra_inputs_forwarded():
    policy, vla, _ = _policy()
    policy.predict(_image(), "task")

    _, max_new_tokens, kwargs = vla.generate_calls[0]
    assert max_new_tokens == 7
    assert kwargs["do_sample"] is False
    assert kwargs["output_logits"] is True
    assert kwargs["return_dict_in_generate"] is True
    assert set(kwargs) >= {"attention_mask", "pixel_values"}


def test_the_empty_token_is_appended_only_when_the_prompt_lacks_it():
    policy, vla, _ = _policy(last_prompt_token=2)
    policy.predict(_image(), "task")
    assert vla.generate_calls[0][0][0].tolist() == [1, 5, 2, 29871]

    policy, vla, _ = _policy(last_prompt_token=29871)
    policy.predict(_image(), "task")
    assert vla.generate_calls[0][0][0].tolist() == [1, 5, 29871]


@pytest.mark.parametrize("center_crop", [True, False])
def test_preprocessing_matches_the_vendored_get_vla_action(center_crop):
    policy, _, mine = _policy(center_crop=center_crop)
    img = _image(3)
    policy.predict(img, "Pick Up The Book")

    vendored_processor = FakeProcessor()
    openvla_utils.get_vla_action(
        FakeVLA(), vendored_processor, "openvla-7b", {"full_image": img}, "Pick Up The Book",
        UNNORM, center_crop=center_crop,
    )  # fmt: skip
    assert mine.calls[0][0] == vendored_processor.calls[0][0]
    assert np.array_equal(mine.calls[0][1], vendored_processor.calls[0][1])


def test_reference_action_is_the_vendored_predict_action():
    policy, vla, _ = _policy()
    action = policy.reference_action(_image(), "task")

    assert np.array_equal(action, np.arange(7))
    assert vla.predict_calls[0]["unnorm_key"] == UNNORM
    assert vla.predict_calls[0]["do_sample"] is False


def test_action_stats_come_from_the_model_with_a_full_mask_by_default():
    policy, vla, _ = _policy()
    stats = policy.action_stats
    assert np.array_equal(stats["q01"], LIBERO_10_ACTION_STATS["q01"])
    assert stats["mask"].tolist() == [True] * 6 + [False]

    del vla.norm_stats[UNNORM]["action"]["mask"]
    assert policy.action_stats["mask"].tolist() == [True] * 7


def test_hub_commit_is_parsed_from_the_remote_module_path():
    commit = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
    name = f"transformers_modules.openvla.openvla-7b.{commit}.modeling_prismatic"
    assert op.hub_commit_from_module(name) == commit
    assert op.hub_commit_from_module("prismatic.extern.hf.modeling_prismatic") is None
