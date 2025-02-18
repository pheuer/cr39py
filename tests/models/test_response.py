import numpy as np
import pytest

from cr39py.core.units import unit_registry as u
from cr39py.models.response import BulkEtchModel, CParameterModel, TwoParameterModel


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


def test_bulk_etch_response():
    model = BulkEtchModel()
    time = 1 * u.hr
    removal = model.removal(time)
    time2 = model.time_to_remove(removal)
    assert np.isclose(time.m_as(u.s), time2.m_as(u.s))


def test_cparameter():

    model = CParameterModel(0.509, 8.4)

    # Change the c and dmax params
    model.c = 0.5
    model.dmax = 8.5

    # Calculate the track diameter
    energy = 1
    diameter = model.track_diameter(energy)
    energy2 = model.track_energy(diameter)
    assert np.isclose(energy, energy2, rtol=0.05)

    # Test dconvert functions
    D_raw = 6
    D_scaled = model.D_scaled(D_raw)
    D_raw2 = model.D_raw(D_scaled)
    assert np.isclose(D_raw, D_raw2, rtol=0.05)
