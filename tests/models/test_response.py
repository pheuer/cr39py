import numpy as np
import pytest

from cr39py.core.units import unit_registry as u
from cr39py.models.response import BulkEtchModel, CParameterModel, TwoParameterModel


def test_two_parameter_model_attrs():
    model = TwoParameterModel("p")
    for attr in ["n", "k", "vB"]:
        assert hasattr(model, attr), f"Model does not have attribute {attr}"
        setattr(model, attr, 1.0)  # Set the attribute to a test value
        assert getattr(model, attr) == 1.0, f"Model attribute {attr} not set correctly"


@pytest.mark.parametrize("diameter", [0.5, 1, 6, 10])
@pytest.mark.parametrize("particle", ["p", "d", "t", "a"])
@pytest.mark.parametrize("etch_time", [30, 60, 180])
def test_two_parameter_model_consistency(diameter, particle, etch_time):
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


response_cases = [
    ("p", 1, 15),
    ("p", 2, 8),
    ("p", 5, 3),
    ("d", 1, 20),
    ("d", 2, 14.5),
    ("d", 6, 5.5),
    ("t", 1, 22.5),
    ("t", 2, 18),
    ("t", 10, 4.75),
]


@pytest.mark.parametrize("particle,energy,diameter", response_cases)
def test_two_parameter_model_vs_lahmann_paper(particle, energy, diameter):
    """
    Check the Two parameter class reproduces values from Fig. 4 of the
    B. Lahmann 2020 RSI.
    """
    # Create a model with parameters matching the caption of Fig. 4
    model = TwoParameterModel(particle)
    etch_time = 5 * 60

    predicted_diameter = model.track_diameter(energy, etch_time)
    assert np.isclose(predicted_diameter, diameter, atol=0.5)

    predicted_energy = model.track_energy(diameter, etch_time)
    assert np.isclose(predicted_energy, energy, atol=0.25)


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


@pytest.mark.parametrize("c,dmax", [(0.509, 0.84), (1.2, 21), (0.6, 10), (0.6, 15)])
def test_cparameter_model_different_values(c, dmax):
    model = CParameterModel(c, dmax)
    diameter = model.track_diameter(2)
