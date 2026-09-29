"""Self-consistency checks over a logged episode (arrays from `schema.read_episode`)."""

import numpy as np

from harness import decode

_INT_MODEL_FIELDS = ("argmax_full_id", "chosen_token_id", "bin_idx")
_FLOAT_MODEL_FIELDS = (
    "slice_logits",
    "lse_full",
    "normalized_action",
    "unnormalized_action",
    "inference_latency_ms",
)


def check_self_consistency(arrays: dict, stats: dict, sent_actions) -> list[str]:
    """Returns one message per violation; an empty list means every check passed."""
    n = len(arrays["step_idx"])
    if len(sent_actions) != n:
        return [f"{len(sent_actions)} sent actions for {n} logged steps"]

    problems = []
    for k in np.flatnonzero(~arrays["is_wait_step"]):
        chosen = arrays["chosen_token_id"][k]
        differing = np.flatnonzero(chosen != arrays["argmax_full_id"][k]).tolist()
        if differing:
            problems.append(
                f"step {k}: chosen_token_id differs from argmax_full_id at dims {differing}"
            )

        expected = decode.decode_action(chosen, stats["q01"], stats["q99"], stats["mask"])
        logged = {
            "clipped_flag": arrays["clipped_flag"][k],
            "bin_idx": arrays["bin_idx"][k],
            "normalized_action": arrays["normalized_action"][k],
            "unnormalized_action": arrays["unnormalized_action"][k],
            "executed_action": arrays["executed_action"][k],
        }
        for name, value in logged.items():
            reference = getattr(expected, name)
            if reference.dtype == np.float64 and name != "executed_action":
                reference = reference.astype(np.float32)
            if not np.array_equal(reference, value):
                problems.append(f"step {k}: {name} does not follow from chosen_token_id")

    for k, sent in enumerate(sent_actions):
        if not np.array_equal(arrays["executed_action"][k], sent):
            problems.append(f"step {k}: executed_action differs from the action sent to the env")
    return problems


def check_wait_alignment(arrays: dict, expected_wait_steps: int) -> list[str]:
    wait = arrays["is_wait_step"]
    problems = []
    first_action = int(np.argmax(~wait)) if (~wait).any() else len(wait)
    if first_action != expected_wait_steps:
        problems.append(f"first non-wait step is {first_action}, expected {expected_wait_steps}")
    if wait[first_action:].any():
        problems.append(f"wait step after step {first_action}")

    for k in np.flatnonzero(wait):
        ints_ok = all((arrays[f][k] == -1).all() for f in _INT_MODEL_FIELDS)
        floats_ok = all(
            np.isnan(arrays[f][k].astype(np.float32)).all() for f in _FLOAT_MODEL_FIELDS
        )
        if not (ints_ok and floats_ok):
            problems.append(f"step {k}: wait step carries model output")
    problems.extend(
        f"step {k}: missing model output on an action step"
        for k in np.flatnonzero(~wait)
        if (arrays["chosen_token_id"][k] < 0).any()
    )
    return problems
