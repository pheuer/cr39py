# -*- coding: utf-8 -*-
"""
Tests for stack.py
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from cr39py.core.units import unit_registry as u
from cr39py.filtration.stack import Layer, Stack


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
# TODO: Add more cases with more SRIM data
cases = [
    ("1000 um Al", "Proton", 14.7 * u.MeV, 5.567 * u.MeV),
    ("15 um Ta", "Proton", 5 * u.MeV, 4.246 * u.MeV),
    ("25 um Ta", "Deuteron", 5 * u.MeV, 2.967 * u.MeV),
    ("25 um Al", "Triton", 3 * u.MeV, 1.642 * u.MeV),
]


@pytest.mark.parametrize("layer,particle,Ein,expected", cases)
def test_layer_ion_ranging(layer, particle, Ein, expected):
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


def test_particle_stops_when_energy_goes_negative():
    l = Layer.from_string("1 m Ta")
    Eout = l.range_down("Proton", 2 * u.MeV, dx=0.1 * u.um)
    assert Eout.m == 0


def test_dx_too_large_in_ranging():
    l = Layer.from_string("100 um Ta")
    with pytest.raises(ValueError):
        l.range_down("Proton", 4 * u.MeV, dx=20 * u.um)


def test_create_stack_from_list_of_layers():
    layers = [
        Layer.from_properties(thickness=20 * u.um, material="W"),
        Layer.from_properties(thickness=150 * u.um, material="Ta", name="test2"),
    ]

    s1 = Stack.from_layers(*layers)


def test_create_stack_from_string():

    s1 = Stack.from_layers(
        Layer.from_properties(20 * u.um, "Ta"), Layer.from_properties(100 * u.um, "Al")
    )
    s2 = Stack.from_string("20 um Ta, 100 um Al")

    assert s1 == s2


def test_stackproperties():
    s = Stack.from_string("20 um Ta, 100 um Al")
    str(s)
    assert s.nactive == 2
    assert np.isclose(s.thickness, 120 * u.um)


def test_stack_ion_ranging():
    s = Stack.from_string("100 um Ta, 100 um Al")

    Ein = 15 * u.MeV
    Eout = s.range_down("Deuteron", Ein)
    Ein2 = s.reverse_ranging("Deuteron", Eout)
    assert np.isclose(Ein, Ein2, rtol=0.01)


def test_stack_ranging_energy_loss():
    s = Stack.from_string("100 um Ta, 100 um Al")
    Ein = 12 * u.MeV
    Eout = s.range_down("Deuteron", Ein)
    Elost = s.ranging_energy_loss("Deuteron", Ein)

    assert np.isclose(Elost, Ein - Eout, rtol=0.01)
