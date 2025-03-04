from pathlib import Path

import numpy as np
import pytest

from cr39py.core.ci import SilentPlotting
from cr39py.core.data import data_dir
from cr39py.core.units import u
from cr39py.scan.wrf import WedgeRangeFilter


@pytest.fixture
def wrf():
    cpsa_path = data_dir / Path("test/test_wrf_G093_6hr.cpsa")
    wrf = WedgeRangeFilter.from_cpsa(cpsa_path)

    wrf.set_framesize("X", 200 * u.um)
    wrf.set_framesize("D", 0.1 * u.um)
    return wrf


def test_init_with_keywords():
    # Not actually a WRF, but it exists and doesn't have any etch time or ID code in the filename
    cpsa_path = data_dir / Path("test/test_alphas.cpsa")

    wrf = WedgeRangeFilter.from_cpsa(cpsa_path, etch_time=360, wrf="G093")

    # Explicitly provide the calibration
    wrf = WedgeRangeFilter.from_cpsa(cpsa_path, etch_time=360, wrf=(800, 1200))

    # Test invalid calibration
    with pytest.raises(ValueError):
        wrf = WedgeRangeFilter.from_cpsa(cpsa_path, etch_time=360, wrf=(1, 2, 3, 4))

    # Test unknown ID code
    with pytest.raises(KeyError):
        wrf = WedgeRangeFilter.from_cpsa(
            cpsa_path, etch_time=360, wrf="not a valid WRF ID code"
        )


def test_wrf_analysis(wrf):
    wrf.set_limits(trange=(100, 1800), drange=(10, 16), crange=(0, 10), erange=(0, 15))

    guess = (
        15,
        0.1,
        1.5,
        16,
    )  # Mean energy, energy standard deviation, c-parameter, dmax parameter
    bounds = [(12, 17), (0.05, 2), (0.4, 2), (16, 20)]
    with SilentPlotting():
        result = wrf.fit(guess, bounds=bounds, plot=True)

    keys = ["Emean", "Estd", "C", "Dmax"]
    tolerances = (0.1, 0.05, 0.1, 1)
    expected = (15.331, 0.291, 1.022, 20.3)
    for i, x in enumerate(result):
        assert np.isclose(
            x, expected[i], atol=tolerances[i]
        ), f"{keys[i]} out of tolerance: |{x:.2f} - {expected[i]:.2f}| > {tolerances[i]}"
