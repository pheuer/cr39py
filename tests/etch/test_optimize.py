"""_
This file contains tests for the etch/optimize.py file.
"""

import numpy as np
import pytest

from cr39py import Stack, u
from cr39py.core.ci import SilentPlotting
from cr39py.etch.optimize import EtchTimeOptimizer


@pytest.fixture
def optimizer():
    mean_energy = 15.2 * u.MeV  # Mean energy in MeV
    std_dev_energy = 0.4 * u.MeV  # Standard deviation in MeV
    filters = Stack.from_string("15 um Ta, 1500 um CR-39, 180 um Al")

    opt = EtchTimeOptimizer(mean_energy, std_dev_energy, filters)
    return opt


def test_run_optimizer(optimizer):
    with SilentPlotting():
        optimum = optimizer.optimal_etch_time(3e4, plot=True, overlap_max=0.1)

    assert np.isclose(optimum / 60, 6.41, atol=0.25)


def test_no_optimal_time_exists(optimizer):
    """
    No possible etch time can meet this criteria - so a value error should be raised
    """
    with pytest.raises(ValueError):
        optimum = optimizer.optimal_etch_time(1e6, plot=False, overlap_max=0.005)
