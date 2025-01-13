from pathlib import Path

import numpy as np
import pytest

from cr39py.core.units import u
from cr39py.filtration.material import Material


def test_saveable(tmpdir):
    tmppath = Path(tmpdir) / Path("material.h5")

    m1 = Material.from_string("Al")
    m1.to_hdf5(tmppath)

    m2 = Material.from_hdf5(tmppath)

    assert m1 == m2


opt = [("H", "h"), ("hydrogen", "h")]


@pytest.mark.parametrize("test_input,expected", opt)
def test_equivalencies(test_input, expected):
    assert Material.from_string(test_input) == Material.from_string(expected)


opt = [("Au", 19.3 * u.g / u.cm**3)]


@pytest.mark.parametrize("test_input,expected", opt)
def test_density(test_input, expected):
    assert np.isclose(Material.from_string(test_input).density, expected)


def test_ion_stopping_power_and_ranging():
    m = Material.from_string("Al")
    energies = np.linspace(2, 10, num=10) * u.MeV

    mu_rho = m.ion_stopping_power("d", energies=energies)
    assert mu_rho.u == u.keV / u.um

    mu_rho_interp = m.ion_stopping_power("d", return_interpolator=True)
    assert np.allclose(mu_rho_interp(energies), mu_rho)

    projected_range = m.ion_projected_range("d", energies=energies)
    assert projected_range.u == u.m

    projected_range_interp = m.ion_projected_range("d", return_interpolator=True)
    assert np.allclose(projected_range_interp(energies), projected_range)


def test_optimal_ion_filter_thickness():
    m = Material.from_string("Al")
    optimal = m.optimal_ion_ranging_thickness("p", 6 * u.MeV, 2 * u.MeV)
    assert np.isclose(optimal, 218 * u.um)
