"""
Tests for layer.py
"""

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from cr39py.core.units import unit_registry as u
from cr39py.filtration.layer import Layer


def test_create_layer():
    l1 = Layer.from_properties(
        thickness=50 * u.um, material="Ta", active=True, name="testname"
    )
    str(l1)


def test_layer_from_string():
    l1 = Layer.from_properties(100 * u.um, "Ta")
    l2 = Layer.from_string("100 um ta")
    assert l1 == l2

    # Test invalid string
    with pytest.raises(ValueError):
        Layer.from_string("Not a valid layer string")


def test_layer_equality():
    l1 = Layer.from_string("100 um Ta")
    l2 = Layer.from_string("50 um Ta")
    l3 = Layer.from_string("100 um Ta")

    assert l1 != l2
    assert l1 == l3


def test_save_layer(tmpdir):

    tmppath = Path(tmpdir) / Path("layer.h5")
    l1 = Layer.from_properties(100 * u.um, "Ta")
    l1.to_hdf5(tmppath)

    l2 = Layer.from_hdf5(tmppath)

    assert l1 == l2


# These test cases are used for a bunch of validations of the
# ranging calculations
# the expected values are from the MIT AnalyzeCR39 calculator
# Values are Layer, Particle, Ein, Expected Eout,
# TODO: Add more cases with more SRIM data
cases = [
    ("1000 um Al", "Proton", 14.7 * u.MeV, 5.567 * u.MeV),
    ("15 um Ta", "Proton", 5 * u.MeV, 4.246 * u.MeV),
    ("25 um Ta", "Deuteron", 5 * u.MeV, 2.967 * u.MeV),
    ("25 um Al", "Triton", 3 * u.MeV, 1.642 * u.MeV),
]

# TODO add tests for projected range


@pytest.mark.parametrize("layer,particle,Ein,expected,", cases)
def test_layer_ion_ranging(
    layer,
    particle,
    Ein,
    expected,
):
    """Compare the calculated ranged-down energies to values
    from MIT's AnalyzeCR39 calculator.
    """
    l = Layer.from_string(layer)
    eout = l.range_down(particle, Ein)
    assert np.isclose(eout, expected, rtol=0.03)


@pytest.mark.parametrize("layer,particle,expected,Eout", cases)
def test_layer_remove_ranging(layer, particle, expected, Eout):
    """Compare the calculated reverse-ranging energies to values
    from MIT's AnalyzeCR39 calculator.
    """
    l = Layer.from_string(layer)
    ein = l.reverse_ranging(particle, Eout)
    assert np.isclose(ein, expected, rtol=0.03)


@pytest.mark.parametrize("layer,particle,Ein,ignore", cases)
def test_layer_reverse_ranging_self_consistency(layer, particle, Ein, ignore):
    l = Layer.from_string(layer)
    Eout = l.range_down(particle, Ein)
    Ein2 = l.reverse_ranging(particle, Eout)
    assert np.isclose(Ein, Ein2, rtol=0.01)


def test_reverse_ranging_particle_stops():
    """
    Test that ranging cannot be reversed if the particle would have stopped
    """
    l = Layer.from_string("1 m Ta")
    with pytest.raises(ValueError):
        l.reverse_ranging("Proton", 0 * u.MeV)

    with pytest.raises(ValueError):
        l.reverse_ranging("Proton", -1 * u.MeV)


@pytest.mark.parametrize("layer,particle,Ein,ignore", cases)
def test_ranging_energy_loss(layer, particle, Ein, ignore):
    l = Layer.from_string(layer)
    deltaE = l.ranging_energy_loss(particle, Ein)
    deltaE_actual = Ein - l.range_down(particle, Ein)
    assert np.isclose(deltaE, deltaE_actual, rtol=0.03)


# These cases test the projected range functionality
# Again the expected values are from MIT AnalyzeCR39
# Values are Material, Particle, Ein, Expected projected range, Expected Straggle
cases = [
    ("Al", "Proton", 14.7 * u.MeV, 1225 * u.um, 51.1 * u.um),
    ("Ta", "Proton", 12 * u.MeV, 261.98 * u.um, 17.06 * u.um),
    ("CR-39", "Proton", 2.5 * u.MeV, 86.52 * u.um, 3.63 * u.um),
    ("Ta", "Deuteron", 12 * u.MeV, 521 * u.um, 22.4 * u.um),
]


@pytest.mark.parametrize("material,particle,Ein,projrange,straggle", cases)
def test_projected_range(material, particle, Ein, projrange, straggle):
    l = Layer.from_string(f"1000 um {material}")
    pr = l.projected_range(particle, Ein)
    assert np.isclose(pr, projrange, rtol=0.03)


# TODO: Add stragle interpolator and test here


cases = [
    ("CR-39", "Proton", 4 * u.MeV, 2.5 * u.MeV),
    ("Al", "Deuteron", 12 * u.MeV, 4 * u.MeV),
]


@pytest.mark.parametrize("material,particle,Ein,Eat", cases)
def test_projected_depth_for_energy(material, particle, Ein, Eat):

    # Calculate the projected depth for the particle to be at E_at
    l = Layer.from_string(f"1000 um {material}")
    pd = l.projected_depth_for_energy(particle, Ein, Eat)

    # Calculate the actual energy at the projected depth to check
    l2 = Layer.from_string(f"{pd.m_as(u.um)} um {material}")
    Eat2 = l2.range_down(particle, Ein)

    assert np.isclose(Eat, Eat2, rtol=0.03)


def test_particle_stops_when_energy_goes_negative():
    l = Layer.from_string("1 m Ta")
    Eout = l.range_down("Proton", 2 * u.MeV, dx=0.1 * u.um)
    assert Eout.m == 0


def test_dx_too_large_in_ranging():
    l = Layer.from_string("100 um Ta")
    with pytest.raises(ValueError):
        l.range_down("Proton", 4 * u.MeV, dx=20 * u.um)
