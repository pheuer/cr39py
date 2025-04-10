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
from cr39py.filtration.layer import Layer
from cr39py.filtration.stack import Stack


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


def test_add_layer():
    s = Stack.from_string("20 um Ta, 100 um Al")
    s.add_layer(Layer.from_string("10 um Ta"))
    s.add_layer("20 um Ta")


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


cases = [
    ("100 um Ta, 100 um Al", "Proton", 12 * u.MeV, 10.39 * u.um),
    # Case where particle will stop in the stack
    ("100 um Ta, 2000 um Al", "Proton", 1 * u.MeV, 1.12 * u.um),
]


@pytest.mark.parametrize("stack,particle,energy,expected", cases)
def test_stack_lateral_straggle(stack, particle, energy, expected):
    s = Stack.from_string(stack)
    straggle = s.lateral_straggle(particle, energy)
    assert np.isclose(straggle, expected, rtol=0.01)
