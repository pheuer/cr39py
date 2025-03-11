"""
Tests for cpsa.py
"""

from pathlib import Path

import numpy as np
import pytest

from cr39py.core.data import data_dir
from cr39py.scan.cpsa import extract_etch_time, read_cpsa


@pytest.mark.parametrize(
    "etch_time_str,time",
    [("not_valid", None), ("2h", 120), ("3hr", 180), ("40m", 40), ("120min", 120)],
)
def test_extract_etch_time(etch_time_str, time):
    path = Path(f"{etch_time_str}.cpsa")
    assert extract_etch_time(path) == time


def test_read_cpsa():
    path = data_dir / Path("test/test_alphas.cpsa")
    tracks, metadata = read_cpsa(path)

    assert isinstance(tracks, np.ndarray)
    assert tracks.shape[-1] == 6

    assert isinstance(metadata, dict)
    assert "nframes" in metadata
