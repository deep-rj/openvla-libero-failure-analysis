"""OpenVLA action decoding: token id -> bin -> normalized -> unnormalized -> env action.

Mirrors `OpenVLAForActionPrediction.predict_action` and the LIBERO gripper post-processing
as pure functions; no model, env or I/O.
"""

from dataclasses import dataclass

import numpy as np

# text_config.vocab_size (32064) minus pad_to_multiple_of (64).
VOCAB_SIZE = 32000
N_ACTION_BINS = 256
# Bin edges number 256, so there are 255 centers.
N_BINS = N_ACTION_BINS - 1
# Action tokens are the last 256 ids below VOCAB_SIZE; higher id means lower bin.
ACTION_SLICE_START = VOCAB_SIZE - N_ACTION_BINS


@dataclass(frozen=True)
class DecodedAction:
    bin_idx: np.ndarray
    clipped_flag: np.ndarray
    normalized_action: np.ndarray
    unnormalized_action: np.ndarray
    executed_action: np.ndarray


def bin_centers(n_action_bins: int = N_ACTION_BINS) -> np.ndarray:
    edges = np.linspace(-1, 1, n_action_bins)
    return (edges[:-1] + edges[1:]) / 2.0


def token_to_bin(token_ids) -> tuple[np.ndarray, np.ndarray]:
    """Returns the clipped bin index and whether the raw bin fell outside [0, N_BINS - 1]."""
    raw = VOCAB_SIZE - np.asarray(token_ids) - 1
    return np.clip(raw, 0, N_BINS - 1), (raw < 0) | (raw > N_BINS - 1)


def unnormalize(normalized, q01, q99, mask) -> np.ndarray:
    return np.where(mask, 0.5 * (normalized + 1) * (q99 - q01) + q01, normalized)


def to_env_action(action) -> np.ndarray:
    """Binarizes the gripper at 0.5 and flips its sign (LIBERO: -1 open, +1 close)."""
    out = np.array(action, dtype=np.float64)
    out[..., -1] = -np.sign(2 * out[..., -1] - 1)
    return out


def decode_action(token_ids, q01, q99, mask) -> DecodedAction:
    bins, clipped = token_to_bin(token_ids)
    normalized = bin_centers()[bins]
    unnormalized = unnormalize(normalized, np.asarray(q01), np.asarray(q99), np.asarray(mask))
    return DecodedAction(
        bin_idx=bins,
        clipped_flag=clipped,
        normalized_action=normalized,
        unnormalized_action=unnormalized,
        executed_action=to_env_action(unnormalized),
    )
