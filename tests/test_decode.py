import numpy as np
import pytest
from harness import decode

Q01 = np.array([-0.63, -0.77, -0.76, -0.097, -0.148, -0.274, 0.0])
Q99 = np.array([0.77, 0.846, 0.9375, 0.139, 0.159, 0.324, 1.0])
MASK = np.array([True] * 6 + [False])


@pytest.mark.parametrize(
    ("token_id", "expected_bin"),
    [(31999, 0), (31998, 1), (31872, 127), (31746, 253), (31745, 254)],
)
def test_action_slice_tokens_map_to_bins_without_clipping(token_id, expected_bin):
    bins, clipped = decode.token_to_bin(np.array([token_id]))
    assert bins[0] == expected_bin
    assert not clipped[0]


def test_first_slice_token_is_clipped_to_the_same_bin_as_its_neighbour():
    bins, clipped = decode.token_to_bin(np.array([31744, 31745]))
    assert list(bins) == [254, 254]
    assert list(clipped) == [True, False]


def test_text_tokens_clip_to_the_top_bin():
    bins, clipped = decode.token_to_bin(np.array([0, 29871, 31743]))
    assert list(bins) == [254, 254, 254]
    assert clipped.all()


def test_padding_tokens_clip_to_the_bottom_bin():
    bins, clipped = decode.token_to_bin(np.array([32000, 32030, 32063]))
    assert list(bins) == [0, 0, 0]
    assert clipped.all()


def test_bin_centers_are_the_255_midpoints_of_256_edges():
    centers = decode.bin_centers()
    assert centers.shape == (255,)
    assert centers[0] == pytest.approx(-1 + 1 / 255)
    assert centers[127] == pytest.approx(0.0, abs=1e-12)
    assert centers[254] == pytest.approx(1 - 1 / 255)


def test_unnormalize_leaves_masked_dims_untouched():
    normalized = np.full(7, 0.5)
    out = decode.unnormalize(normalized, Q01, Q99, MASK)
    assert out[:6] == pytest.approx(0.75 * (Q99[:6] - Q01[:6]) + Q01[:6])
    assert out[6] == 0.5


@pytest.mark.parametrize(
    ("decoded_gripper", "env_gripper"),
    [(0.996, -1.0), (0.0, 1.0), (0.51, -1.0), (0.49, 1.0)],
)
def test_gripper_is_binarized_at_one_half_and_sign_inverted(decoded_gripper, env_gripper):
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, decoded_gripper])
    executed = decode.to_env_action(action)
    assert executed[6] == env_gripper
    assert executed[:6].tolist() == action[:6].tolist()


def test_to_env_action_does_not_mutate_its_input():
    action = np.array([0.0] * 6 + [0.9])
    decode.to_env_action(action)
    assert action[6] == 0.9


def test_decode_action_returns_every_stage_for_a_mixed_token_vector():
    tokens = np.array([31999, 31744, 5, 32010, 31872, 31872, 31745])
    stages = decode.decode_action(tokens, Q01, Q99, MASK)
    assert list(stages.bin_idx) == [0, 254, 254, 0, 127, 127, 254]
    assert list(stages.clipped_flag) == [False, True, True, True, False, False, False]
    assert stages.normalized_action.shape == (7,)
    assert stages.unnormalized_action.shape == (7,)
    assert stages.executed_action.shape == (7,)


def _vendored_predict_action():
    modeling = pytest.importorskip("prismatic.extern.hf.modeling_prismatic")
    cls = modeling.OpenVLAForActionPrediction

    class Stub:
        predict_action = cls.predict_action
        get_action_dim = cls.get_action_dim
        get_action_stats = cls.get_action_stats
        _check_unnorm_key = staticmethod(cls._check_unnorm_key)
        vocab_size = 32000

        def __init__(self, tokens):
            self._tokens = tokens
            self.bin_centers = decode.bin_centers()
            self.norm_stats = {
                "suite": {
                    "action": {"q01": Q01.tolist(), "q99": Q99.tolist(), "mask": MASK.tolist()}
                }
            }

        def generate(self, input_ids, max_new_tokens):
            import torch

            return torch.cat([input_ids, torch.tensor([self._tokens])], dim=1)

    return Stub


def test_decode_matches_the_vendored_predict_action_on_random_tokens():
    import torch

    stub_cls = _vendored_predict_action()
    rng = np.random.default_rng(0)
    for _ in range(200):
        tokens = rng.integers(0, 32064, size=7)
        vendored = stub_cls(tokens.tolist()).predict_action(
            input_ids=torch.tensor([[1, 2, 3, 29871]]), unnorm_key="suite"
        )
        ours = decode.decode_action(tokens, Q01, Q99, MASK).unnormalized_action
        assert np.array_equal(vendored, ours)


def test_to_env_action_matches_the_vendored_gripper_functions():
    robot_utils = pytest.importorskip("experiments.robot.robot_utils")
    rng = np.random.default_rng(1)
    for _ in range(200):
        action = np.concatenate([rng.uniform(-1, 1, 6), rng.choice([0.0, 0.996, 0.3, 0.7], 1)])
        vendored = robot_utils.invert_gripper_action(
            robot_utils.normalize_gripper_action(action.copy(), binarize=True)
        )
        assert np.array_equal(decode.to_env_action(action), vendored)
