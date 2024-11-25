import numpy as np
import pytest

from cr39py.core.units import u
from cr39py.models.fusion import (
    cross_section,
    d3hep_yield,
    reactions,
    reactivity,
    reduced_mass,
)


@pytest.mark.parametrize(
    "reaction,energy,expected",
    [
        ("D(D,n)", 10 * u.keV, 8e-31 * u.m**2),
        ("D(D,n)", 100 * u.keV, 7e-30 * u.m**2),
        ("3He(D,p)", 20 * u.keV, 1e-32 * u.m**2),
        ("3He(D,p)", 100 * u.keV, 9e-28 * u.m**2),
    ],
)
def test_cross_section_single_values(reaction, energy, expected):
    """
    Test against single values read off of the xs curves here
    https://scipython.com/blog/nuclear-fusion-cross-sections/#rating-177
    """
    xs = cross_section(reaction, energy)
    assert np.abs(xs - expected) / expected - 1 < 0.1


def test_cross_section_different_inputs():
    """
    This just ensures that the function runs for different inputs
    """
    energies = np.arange(1, 20, 1) * u.keV

    e, xs = cross_section("D(D,n)")
    assert isinstance(xs, u.Quantity)

    e, xs2 = cross_section("D(D,n)", energies=energies)
    assert isinstance(xs2, u.Quantity)


def test_cross_section_data_availability():
    """
    Tests to make sure a xs can be retrieved for each reaction in the
    reactions list.

    If the data file isn't available, this will fail  because it can't
    retrieve the HDF5 file.
    """
    for r in reactions:
        xs = cross_section(r, energies=10 * u.keV)
        assert isinstance(xs, u.Quantity)


@pytest.mark.parametrize(
    "reaction,tion,expected",
    [
        ("D(D,n)", 10 * u.keV, 1e-18 * u.cm**3 / u.s),
        ("D(D,n)", 100 * u.keV, 5e-17 * u.cm**3 / u.s),
        ("3He(D,p)", 10 * u.keV, 2e-19 * u.cm**3 / u.s),
        ("3He(D,p)", 100 * u.keV, 2e-16 * u.cm**3 / u.s),
    ],
)
def test_reactivity_single_values(reaction, tion, expected):
    """
    Test against single values read off of the reactivity curves here
    https://scipython.com/blog/nuclear-fusion-cross-sections/#rating-177
    """
    r = reactivity(reaction, tion)
    assert isinstance(r, u.Quantity)
    assert np.abs(r - expected) / expected - 1 < 0.1


def test_d3hep_yield():
    y = d3hep_yield(1e8, 6.5, 13, 11 * u.keV) * 1e-7
    assert np.isclose(y, 4.48, rtol=0.05)
