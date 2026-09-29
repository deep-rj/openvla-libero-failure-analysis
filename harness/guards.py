"""Load-time checks that the code and stats a run uses are the ones the results assume.

`from_pretrained(..., trust_remote_code=True)` executes the hub's model, config and
processor files in preference to the sibling repo's classes, and the hub revision is not
pinned, so the executed files are hashed against pinned values.
"""

import hashlib
import inspect
from pathlib import Path

HUB_CODE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
# Keyed by role: modeling_, configuration_ and processing_prismatic.py at HUB_CODE_REVISION.
EXPECTED_HUB_SHA256 = {
    "model": "9ce241c5ca09a4bed73654d0ca509baaff0fc5887d0bdd62e360ca8254f7794a",
    "config": "68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68",
    "processor": "06563586bbc73d36501a14a231a3174904ede830d33174d0c625ece8e931ef1e",
}
LM_HEAD_WIDTH = 32064

# Generation settings that would alter logits before they are scored under greedy decoding.
_LOGITS_PROCESSOR_DEFAULTS = {
    "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_decoder_ids": None,
    "min_length": 0,
    "min_new_tokens": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "remove_invalid_values": False,
    "renormalize_logits": False,
    "sequence_bias": None,
    "exponential_decay_length_penalty": None,
    "diversity_penalty": 0.0,
}
_NORM_STAT_FIELDS = ("q01", "q99", "mask")


class GuardError(RuntimeError):
    pass


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source_hashes(paths: dict[str, Path], expected: dict[str, str]) -> None:
    if set(paths) != set(expected):
        raise GuardError(f"source files {sorted(paths)} do not match expected {sorted(expected)}")
    bad = {
        name: (expected[name], actual)
        for name, path in paths.items()
        if (actual := sha256_file(path)) != expected[name]
    }
    if bad:
        lines = [f"  {name}: expected {exp}, got {act}" for name, (exp, act) in bad.items()]
        raise GuardError("executed source hash mismatch:\n" + "\n".join(lines))


def executed_source_paths(objects: dict[str, object]) -> dict[str, Path]:
    """Maps each name to the source file its object's class was loaded from."""
    paths = {}
    for name, obj in objects.items():
        cls = type(obj)
        if not cls.__module__.startswith("transformers_modules"):
            raise GuardError(
                f"{name}: class {cls.__module__}.{cls.__qualname__} is not hub remote code "
                "(expected a transformers_modules.* module)"
            )
        paths[name] = Path(inspect.getsourcefile(cls))
    return paths


def check_norm_stats(model_stats: dict, config_stats: dict, unnorm_key: str) -> None:
    for label, stats in (("model", model_stats), ("config", config_stats)):
        if unnorm_key not in stats:
            raise GuardError(f"unnorm_key {unnorm_key!r} missing from {label} norm_stats")
    model, config = model_stats[unnorm_key]["action"], config_stats[unnorm_key]["action"]
    for field in _NORM_STAT_FIELDS:
        if list(model[field]) != list(config[field]):
            raise GuardError(
                f"norm_stats[{unnorm_key!r}].{field} differs: {model[field]} vs {config[field]}"
            )


def check_generation_config(generation_config) -> None:
    changed = {
        key: getattr(generation_config, key, default)
        for key, default in _LOGITS_PROCESSOR_DEFAULTS.items()
        if getattr(generation_config, key, default) != default
    }
    if changed:
        raise GuardError(f"generation config changes logits under greedy decoding: {changed}")


def check_head_width(width: int) -> None:
    if width != LM_HEAD_WIDTH:
        raise GuardError(f"lm_head width is {width}, expected {LM_HEAD_WIDTH}")


def verify_loaded_model(vla, processor, unnorm_key: str, config_norm_stats: dict) -> None:
    """Runs every guard against a loaded model; hub code must match the pinned hashes."""
    check_source_hashes(
        executed_source_paths({"model": vla, "config": vla.config, "processor": processor}),
        EXPECTED_HUB_SHA256,
    )
    check_norm_stats(vla.norm_stats, config_norm_stats, unnorm_key)
    check_generation_config(vla.generation_config)
    check_head_width(vla.language_model.lm_head.out_features)
