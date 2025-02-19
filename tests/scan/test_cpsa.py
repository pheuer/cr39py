"""
Tests for cpsa.py
"""

from pathlib import Path

import pytest

from cr39py.scan.cpsa import extract_etch_time


@pytest.mark.parametrize(
    "etch_time_str,time",
    [("not_valid", None), ("2h", 120), ("3hr", 180), ("40m", 40), ("120min", 120)],
)
def test_extract_etch_time(etch_time_str, time):
    path = Path(f"{etch_time_str}.cpsa")
    assert extract_etch_time(path) == time
