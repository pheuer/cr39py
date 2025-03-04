from pathlib import Path

import numpy as np
import pytest

from cr39py.core.ci import SilentPlotting
from cr39py.core.data import data_dir
from cr39py.core.units import u
from cr39py.scan.wrf import WedgeRangeFilter, wrf_objective_function


@pytest.fixture
def wrf():
    cpsa_path = data_dir / Path("test/test_wrf_G093_6hr.cpsa")
    wrf = WedgeRangeFilter.from_cpsa(cpsa_path)

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


@pytest.mark.parametrize("attribute", ["background_region", "dmax", "wrf_calibration"])
def test_access_attributes(wrf, attribute):
    assert hasattr(wrf, attribute)
    getattr(wrf, attribute)


def test_set_limits(wrf):
    wrf.set_limits()
    wrf.set_limits(xrange=(-1, 1))
    wrf.set_limits(trange=(100, 1800), drange=(10, 22), crange=(0, 10), erange=(0, 15))


def test_wrf_objective_fcn():
    # Create some data with nan, 0 in it
    x1 = np.random.rand(10, 10)
    x1[0, 0] = 0
    x1[0, 2] = np.nan

    x2 = np.random.rand(10, 10)
    x2[1, 0] = 0
    x2[1, 2] = np.nan

    s1 = wrf_objective_function(x1, x2, return_sum=False)
    s2 = wrf_objective_function(x1, x2, return_sum=True)

    # They won't be exactly equal because of the difference in the mask for data=0


def test_wrf_plot(wrf):
    with SilentPlotting():
        wrf.plot_diameter_histogram()


def test_wrf_analysis(wrf):

    wrf.set_framesize("X", 200 * u.um)
    wrf.set_framesize("D", 0.1 * u.um)
    wrf.dmax = 20.5
    wrf.set_limits()

    wrf.set_limits(trange=(100, 1800), drange=(10, 22), crange=(0, 10), erange=(0, 15))

    # Mean energy, energy standard deviation, c-parameter, dmax parameter
    guess = (15.3, 0.3, 1)
    bounds = [(12, 17), (0.05, 2), (0.4, 2)]
    with SilentPlotting():
        result = wrf.fit(guess, bounds=bounds, plot=True)

    print(result)

    keys = ["Emean", "Estd", "C"]
    tolerances = (0.1, 0.1, 0.1)
    expected = (15.331, 0.291, 1.022)
    for i, x in enumerate(result.x):
        assert np.isclose(
            x, expected[i], atol=tolerances[i]
        ), f"{keys[i]} out of tolerance: |{x:.2f} - {expected[i]:.2f}| > {tolerances[i]}"
