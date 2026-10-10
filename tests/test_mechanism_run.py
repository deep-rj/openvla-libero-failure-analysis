import csv
import json

import pytest

from harness import mechanism_run as mr
from harness.mechanism import Prediction


def test_held_out_episodes_are_skipped_unless_included():
    failures = ["a", "b", "c"]
    assert mr.select_episodes(failures, heldout={"b"}, include_heldout=False) == ["a", "c"]
    assert mr.select_episodes(failures, heldout={"b"}, include_heldout=True) == failures


def test_a_missing_held_out_list_refuses_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="held-out"):
        mr.load_heldout(tmp_path / "heldout_label_selection.json", include_heldout=False)
    assert mr.load_heldout(tmp_path / "missing.json", include_heldout=True) == set()


def test_the_held_out_list_is_read_from_its_episode_ids(tmp_path):
    path = tmp_path / "heldout_label_selection.json"
    path.write_text(json.dumps({"episode_ids": ["x", "y"]}))
    assert mr.load_heldout(path, include_heldout=False) == {"x", "y"}


def test_predictions_are_written_one_row_per_episode(tmp_path):
    out = tmp_path / "labeling" / "mechanism_predicted.csv"
    mr.write_predictions(
        out,
        {
            "e1": Prediction("perception", 52, False, "grasp attempt"),
            "e2": Prediction("", None, True, "no unrecovered error found"),
        },
    )
    rows = list(csv.DictReader(out.open()))
    assert [r["episode_id"] for r in rows] == ["e1", "e2"]
    assert rows[0] == {
        "episode_id": "e1",
        "mechanism": "perception",
        "onset_step": "52",
        "abstained": "0",
        "reason": "grasp attempt",
    }
    assert (rows[1]["onset_step"], rows[1]["abstained"]) == ("", "1")
    assert mr.read_predictions(out)["e2"] == Prediction(
        "", None, True, "no unrecovered error found"
    )
    assert mr.read_predictions(out)["e1"].onset_step == 52


def test_dev_report_counts_agreement_abstentions_and_signed_onset_offsets():
    predicted = {
        "e1": Prediction("perception", 52, False, ""),
        "e2": Prediction("control: grasp", 90, False, ""),
        "e3": Prediction("", None, True, "no unrecovered error found"),
    }
    labels = {
        "e1": {"mechanism": "perception", "first_error_step": "40"},
        "e2": {"mechanism": "perception", "first_error_step": "100"},
        "e3": {"mechanism": "control: place", "first_error_step": "70"},
    }
    report = mr.dev_report(predicted, labels)
    assert report["n"] == 3
    assert report["decided"] == 2
    assert report["agree"] == 1
    assert report["abstained"] == ["e3"]
    assert report["confusion"][("perception", "perception")] == 1
    assert report["confusion"][("perception", "control: grasp")] == 1
    assert report["onset_offsets"] == {"e1": 12, "e2": -10}


def test_the_rendered_report_states_agreement_with_n_and_the_dev_set_caveat():
    report = {
        "n": 3,
        "decided": 2,
        "agree": 1,
        "abstained": ["e3"],
        "confusion": {("perception", "perception"): 1, ("perception", "control: grasp"): 1},
        "onset_offsets": {"e1": 12, "e2": -10},
    }
    text = mr.render_dev_report(report)
    assert "1/2 decided" in text
    assert "1/3 abstained" in text
    assert "development set" in text
    assert "| perception | control: grasp | 1 |" in text


def test_predictions_round_trip_through_the_csv(tmp_path):
    out = tmp_path / "mechanism_predicted.csv"
    predictions = {
        "e1": Prediction("perception", 52, False, "grasp attempt"),
        "e2": Prediction("", None, True, "no unrecovered error found"),
    }
    mr.write_predictions(out, predictions)
    assert mr.read_predictions(out) == predictions
