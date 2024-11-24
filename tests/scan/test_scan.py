import importlib.resources
from pathlib import Path

import numpy as np
import pytest

from cr39py.core.ci import SilentPlotting
from cr39py.core.units import unit_registry as u
from cr39py.scan.base_scan import Scan
from cr39py.scan.cut import Cut
from cr39py.scan.subset import Subset


@pytest.fixture
def cr39scan():
    data_dir = Path(importlib.resources.files("cr39py")).parent.parent / Path("data")
    cpsa_path = data_dir / Path("test_alphas.cpsa")
    return Scan.from_cpsa(cpsa_path, etch_time=120)


def test_framesize(cr39scan):
    cr39scan.set_framesize("X", 500 * u.um)


def test_optimize_xy_framesize(cr39scan):
    cr39scan.optimize_xy_framesize()


def test_get_selected_tracks(cr39scan):
    cr39scan.nselected_tracks

    cr39scan.current_subset.add_cut(Cut(xmin=0))
    cr39scan.current_subset.add_cut(Cut(cmin=30))

    # Test with all cuts
    x = cr39scan.current_subset.apply_cuts(cr39scan.tracks)

    # Test with subset of cuts
    x = cr39scan.current_subset.apply_cuts(cr39scan.tracks, use_cuts=[0])

    # Test invert
    x = cr39scan.current_subset.apply_cuts(cr39scan.tracks, invert=True)

    # Test with ndslices
    cr39scan.current_subset.set_ndslices(5)
    cr39scan.current_subset.select_dslice(0)
    x = cr39scan.current_subset.apply_cuts(cr39scan.tracks)


def test_subset(cr39scan):

    cr39scan.add_subset()
    cr39scan.add_subset(Subset())

    # Test removing nonexistent subset
    with pytest.raises(ValueError):
        cr39scan.remove_subset(200)

    # Cannot remove current subset
    cr39scan.select_subset(0)
    with pytest.raises(ValueError):
        cr39scan.remove_subset(0)

    cr39scan.remove_subset(2)


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_track_energy(cr39scan, statistic):
    cr39scan.track_energy("D", statistic)


def test_rotate(cr39scan):
    cr39scan.rotate(45)


def test_histogram(cr39scan):
    cr39scan.histogram()


def test_plot(cr39scan):
    with SilentPlotting():
        cr39scan.cutplot()
