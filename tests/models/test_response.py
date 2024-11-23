import numpy as np
import pytest

from cr39py.core.units import unit_registry as u
from cr39py.models.response import TwoParameterModel


@pytest.mark.parametrize("diameter", [0.5, 1, 2, 4, 6, 8, 10])
@pytest.mark.parametrize("particle", ["p", "d", "t", "a"])
@pytest.mark.parametrize("etch_time", [30, 60, 120, 180])
def test_track_energy_and_diameter(diameter, particle, etch_time):
    model = TwoParameterModel(particle)
    energy = model.track_energy(diameter, etch_time)

    if np.isnan(energy):
        return

    assert energy > 0

    d2 = model.track_diameter(energy, etch_time)

    if np.isnan(d2):
        return

    assert d2 > 0
    assert np.isclose(diameter, d2)

    e2 = model.etch_time(energy, diameter)

    assert np.isclose(e2, etch_time)
