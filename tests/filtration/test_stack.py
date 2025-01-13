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
from cr39py.filtration.stack import FilterPack, Layer, Stack


def test_create_layer():
    l1 = Layer.from_properties(
        thickness=50 * u.um, material="Tantalum", active=True, name="testname"
    )
    str(l1)


def test_layer_from_string():
    l1 = Layer.from_properties(100 * u.um, "Ta")
    l2 = Layer.from_string("100 um tantalum")
    assert l1 == l2

    # Test invalid string
    with pytest.raises(ValueError):
        Layer.from_string("Not a valid layer string")


def test_layer_equality():
    l1 = Layer.from_string("100 um tantalum")
    l2 = Layer.from_string("50 um tantalum")
    l3 = Layer.from_string("100 um aluminum")

    assert l1 != l2
    assert l1 != l3


def test_save_layer(tmpdir):

    tmppath = Path(tmpdir) / Path("layer.h5")
    l1 = Layer.from_properties(100 * u.um, "Ta")
    l1.to_hdf5(tmppath)

    l2 = Layer.from_hdf5(tmppath)

    assert l1 == l2


def test_create_stack_from_list_of_layers():
    layers = [
        Layer.from_properties(thickness=20 * u.um, material="Tungsten"),
        Layer.from_properties(thickness=150 * u.um, material="Tantalum", name="test2"),
    ]

    s1 = Stack.from_layers(*layers)


def create_stack_from_string():

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


def create_stack_mixed_strings_and_layers():

    s1 = Stack.from_layers(
        Layer.from_properties(20 * u.um, "Ta"), Layer.from_properties(100 * u.um, "Al")
    )
    s2 = Stack.from_string("20 um Ta", Layer.from_properties(100 * u.um, "Al"))

    assert s1 == s2


def test_ion_ranging():
    s = Stack.from_string("100 um Ta")

    einit = 9 * u.MeV
    efinal = s.range_down_ion("d", einit)
    assert np.isclose(efinal, 2.35 * u.MeV, rtol=0.01)

    elost = s.ion_ranging_energy_loss("d", einit)

    assert np.isclose(elost, einit - efinal, rtol=0.01)


def test_ion_reverse_ranging():
    s = Stack.from_string("100 um Ta, 100 um Al")

    efinal = 1.5 * u.MeV
    einit = s.reverse_ion_ranging("p", efinal)

    efinal2 = s.range_down_ion("p", einit)

    assert np.isclose(efinal, efinal2, atol=0.1)


def test_create_filterpack_from_mixed_strings_and_stacks():
    s1 = Stack.from_string("100 um Ta, 100 um Al")
    s2 = Stack.from_string("10 um Au, 100 um polycarbonate")
    pack1 = FilterPack.from_stacks(s1, s2)

    str(pack1)

    pack2 = FilterPack.from_stacks(
        Stack.from_string("100 um Ta, 100 um Al"),
        "10 um Au, 100 um polycarbonate",
    )

    assert pack1 == pack2


def test_save_filter_pack(tmpdir):

    tmppath = Path(tmpdir) / Path("filterpack.h5")

    s1 = Stack.from_string("100 um Ta, 100 um Al")
    s2 = Stack.from_string("10 um Au, 100 um polycarbonate")
    pack1 = FilterPack.from_stacks(s1, s2)

    pack1.to_hdf5(tmppath)

    pack2 = FilterPack.from_hdf5(tmppath)

    assert pack1 == pack2
