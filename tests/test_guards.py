import hashlib

import pytest
from transformers import GenerationConfig

from harness import guards


def _write(path, text):
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_matching_source_hashes_pass(tmp_path):
    expected = {
        "model": _write(tmp_path / "m.py", "a = 1\n"),
        "config": _write(tmp_path / "c.py", "b = 2\n"),
    }
    guards.check_source_hashes({"model": tmp_path / "m.py", "config": tmp_path / "c.py"}, expected)


def test_changed_source_names_the_file_and_both_hashes(tmp_path):
    expected = {"model": _write(tmp_path / "m.py", "a = 1\n")}
    (tmp_path / "m.py").write_text("a = 2\n")
    with pytest.raises(guards.GuardError) as err:
        guards.check_source_hashes({"model": tmp_path / "m.py"}, expected)
    message = str(err.value)
    assert "model" in message
    assert expected["model"] in message
    assert hashlib.sha256(b"a = 2\n").hexdigest() in message


def test_unexpected_or_missing_files_fail(tmp_path):
    expected = {"model": _write(tmp_path / "m.py", "a = 1\n")}
    with pytest.raises(guards.GuardError):
        guards.check_source_hashes({}, expected)
    with pytest.raises(guards.GuardError):
        guards.check_source_hashes(
            {"model": tmp_path / "m.py", "extra": tmp_path / "m.py"}, expected
        )


STATS = {"q01": [0.0, -1.0], "q99": [1.0, 1.0], "mask": [True, False]}


def test_equal_norm_stats_pass():
    guards.check_norm_stats(
        {"suite": {"action": STATS}}, {"suite": {"action": dict(STATS)}}, "suite"
    )


def test_differing_q99_fails_and_names_the_field():
    other = {**STATS, "q99": [1.0, 0.5]}
    with pytest.raises(guards.GuardError, match="q99"):
        guards.check_norm_stats({"suite": {"action": STATS}}, {"suite": {"action": other}}, "suite")


def test_missing_unnorm_key_fails():
    with pytest.raises(guards.GuardError, match="suite"):
        guards.check_norm_stats({"other": {"action": STATS}}, {"suite": {"action": STATS}}, "suite")


def test_default_generation_config_passes():
    guards.check_generation_config(GenerationConfig())


@pytest.mark.parametrize(
    "override",
    [
        {"repetition_penalty": 1.2},
        {"no_repeat_ngram_size": 3},
        {"bad_words_ids": [[1]]},
        {"suppress_tokens": [1]},
        {"min_new_tokens": 2},
    ],
)
def test_logits_processing_generation_settings_fail(override):
    with pytest.raises(guards.GuardError):
        guards.check_generation_config(GenerationConfig(**override))


def test_head_width_must_be_the_padded_vocab():
    guards.check_head_width(32064)
    with pytest.raises(guards.GuardError):
        guards.check_head_width(32000)


def test_executed_source_rejects_a_class_that_is_not_hub_remote_code():
    class Local:
        pass

    with pytest.raises(guards.GuardError, match="transformers_modules"):
        guards.executed_source_paths({"model": Local()})


def test_executed_source_returns_the_file_the_class_was_loaded_from(tmp_path, monkeypatch):
    import sys
    import types

    source = tmp_path / "modeling_prismatic.py"
    source.write_text("class Remote: pass\n")
    module = types.ModuleType("transformers_modules.fake.modeling_prismatic")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    remote = type("Remote", (), {"__module__": module.__name__})()

    assert guards.executed_source_paths({"model": remote}) == {"model": source}
