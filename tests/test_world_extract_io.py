import numpy as np
import pytest

from harness import world_extract


def _write(path, version):
    info = {"extraction_version": version}
    world_extract.write_extracted(path, {"eef_pos": np.zeros((2, 3))}, info)


def test_read_extracted_round_trips_the_current_version(tmp_path):
    _write(tmp_path / "ep", world_extract.EXTRACTION_VERSION)
    x, _ = world_extract.read_extracted(tmp_path / "ep")
    assert x["eef_pos"].shape == (2, 3)


def test_read_extracted_rejects_version_3(tmp_path):
    _write(tmp_path / "ep", 3)
    with pytest.raises(ValueError, match="stale"):
        world_extract.read_extracted(tmp_path / "ep")
