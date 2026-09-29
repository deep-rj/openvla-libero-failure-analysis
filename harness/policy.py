"""Policy interface, logit summarization, and a seeded stub that needs no GPU."""

from typing import Protocol

import numpy as np

from harness.decode import ACTION_SLICE_START, N_ACTION_BINS, VOCAB_SIZE
from harness.schema import N_DIMS, ModelOutput

HEAD_WIDTH = 32064

# libero_10 statistics from the checkpoint's config.json.
LIBERO_10_ACTION_STATS = {
    "q01": np.array(
        [-0.6348214149475098, -0.7741071581840515, -0.7633928656578064,
         -0.09749999642372131, -0.14819999992847435, -0.2742857038974762, 0.0]
    ),
    "q99": np.array(
        [0.7714285850524902, 0.8464285731315613, 0.9375,
         0.13928571343421936, 0.15964286029338837, 0.3246428668498993, 1.0]
    ),
    "mask": np.array([True] * 6 + [False]),
}  # fmt: skip


class Policy(Protocol):
    action_stats: dict[str, np.ndarray]

    def predict(self, img: np.ndarray, task_label: str) -> ModelOutput:
        """Runs one greedy generation of 7 action tokens and returns its raw logits."""
        ...


def summarize_logits(logits: np.ndarray, chosen: np.ndarray, latency_ms: float) -> ModelOutput:
    """Reduces per-position logits [7, head width] to what the episode file stores."""
    logits = np.asarray(logits, dtype=np.float32)
    peak = logits.max(axis=-1, keepdims=True)
    lse = (np.log(np.exp(logits - peak).sum(axis=-1, dtype=np.float64)) + peak[:, 0]).astype(
        np.float32
    )
    return ModelOutput(
        slice_logits=logits[:, ACTION_SLICE_START : ACTION_SLICE_START + N_ACTION_BINS].astype(
            np.float16
        ),
        lse_full=lse,
        argmax_full_id=logits.argmax(axis=-1).astype(np.int32),
        chosen_token_id=np.asarray(chosen).astype(np.int32),
        inference_latency_ms=float(latency_ms),
    )


class StubPolicy:
    """Seeded synthetic logits with greedy decoding.

    Token 31744 (the untrained edge of the action slice) is never emitted. With
    `out_of_slice_rate` > 0, some positions instead put their mass on a text or padding
    token, so the clipping paths are exercised.
    """

    def __init__(self, seed: int = 0, out_of_slice_rate: float = 0.0):
        self.action_stats = LIBERO_10_ACTION_STATS
        self._rng = np.random.default_rng(seed)
        self._out_of_slice_rate = out_of_slice_rate

    def predict(self, img, task_label) -> ModelOutput:
        rng = self._rng
        logits = rng.normal(size=(N_DIMS, HEAD_WIDTH)).astype(np.float32)
        logits[:, ACTION_SLICE_START:VOCAB_SIZE] = rng.normal(scale=4, size=(N_DIMS, N_ACTION_BINS))
        logits[:, ACTION_SLICE_START] = -20.0
        for dim in np.flatnonzero(rng.random(N_DIMS) < self._out_of_slice_rate):
            outside = np.concatenate(
                [np.arange(ACTION_SLICE_START), np.arange(VOCAB_SIZE, HEAD_WIDTH)]
            )
            logits[dim, rng.choice(outside)] = 100.0
        return summarize_logits(logits, chosen=logits.argmax(axis=-1), latency_ms=0.0)
