import numpy as np
import pytest

from cr39py.core.units import unit_registry as u
from cr39py.response import TwoParameterModel


@pytest.mark.parametrize("diameter", [0.5, 1, 2, 4, 6, 8, 10])
@pytest.mark.parametrize("particle", ["p", "d", "t", "a"])
def test_track_energy(diameter, particle):
    model = TwoParameterModel()
    model.track_energy(diameter, particle, 120)


@pytest.mark.parametrize("energy", [0.2, 0.5, 1, 1.5, 2, 3, 5, 8, 10, 12, 15])
@pytest.mark.parametrize("particle", ["p", "d", "t", "a"])
def test_track_energy(energy, particle):
    model = TwoParameterModel()
    model.track_diameter(energy, particle, 120)
