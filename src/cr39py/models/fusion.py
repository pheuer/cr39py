"""
The `~cr39py.models.fusion` module contains functions relating to the fusion reactions that commonly create
the charged particles detected by CR39.
"""

from pathlib import Path

import h5py
import numpy as np

from cr39py.core.data import data_dir
from cr39py.core.units import u

_reaction_data = {
    "D(D,n)": "D(D,n)He-3.h5",
    "D(D,p)": "D(D,p)T.h5",
    "3He(D,p)": "He-3(D,p)A.h5",
}

reactions: list[str] = list(_reaction_data.keys())


def reduced_mass(reaction: str) -> float:
    """
    The reactant reduced mass for a nuclear reaction.

    Reaction string should be in the format r1(r2,p1)p2

    Parameters
    ----------

    reaction : str
        The nuclear reaction. Supported strings are listed in
        `~cr39py.models.fusion.reactions`. Products,
        if present, are ignored.

    Returns
    -------

    reduced_mass : u.Quantity (kg)
        The reduced mass of the reactants.


    """
    masses = {"p": 1, "D": 2, "T": 3, "3He": 3, "4He": 4}

    reactants = reaction.split(",")[0]
    r1, r2 = reactants.split("(")
    m1, m2 = masses[r1], masses[r2]

    return m1 * m2 / (m1 + m2) * 1.67e-27 * u.kg


def cross_section(
    reaction: str, energies: u.Quantity | None = None
) -> tuple[u.Quantity]:
    """
    The fusion cross section for a given nuclear reaction.

    Cross-section data is scraped from the ENDF database.

    Parameters
    ----------

    reaction : str
        The nuclear reaction. Supported strings are listed in
        `~cr39py.models.fusion.reactions`.

    energies : u.Quantity, optional
        Energy axis (in the center of mass frame) over which to
        interpolate the cross section. The default goes from 50-20,000 eV
        in 50 eV steps.

    Returns
    -------

    energies : u.Quantity
        Energy axis

    xs : u.Quantity
        Cross-section

    """

    if energies is None:
        energies = np.arange(10, 1e5, 50) * u.eV

    if energies.ndim == 0:
        energies = np.array([energies.m]) * energies.u

    if reaction not in reactions:
        raise ValueError(
            f"Reaction {reaction} not recognized. Valid inputs are " f"{reactions}"
        )

    path = data_dir / Path(f"nuclear/{_reaction_data[reaction]}")
    with h5py.File(path, "r") as f:
        _energies = f["energy"][:]  # eV
        xs = f["SIG"][:]  # m^2

    xs = np.interp(energies.m_as(u.eV), _energies, xs) * u.m**2

    if energies.size == 1:
        return xs[0]
    else:
        return energies, xs


def reactivity(reaction: str, tion: u.Quantity) -> tuple[u.Quantity]:
    """
    The fusion reactivity for a nuclear reaction.

    Parameters
    ----------

    reaction : str
        The nuclear reaction. Supported strings are listed in
        `~cr39py.models.fusion.reactions`.

    tion : u.Quantity
        Ion temperatures  over which to calculate the
        reactivity.

    Returns
    -------

    xs : u.Quantity
        Cross-section


    Notes
    -----

    This is quite a nice example notebook on fusion reactivities in python
    https://scipython.com/blog/nuclear-fusion-cross-sections/

    """
    mu = reduced_mass(reaction)

    # Get cross section
    # The energy axis here is important - it needs go high enough to make the
    # integral effectively 0 to infinity, and the spacing needs to be
    # fine enough for the integral to have good resolution.
    energies, xs = cross_section(reaction, energies=np.logspace(0, 5, 1000) * u.keV)

    if tion.ndim == 0:
        tion = np.array([tion.m]) * tion.u

    _tion = tion[None, :]
    _E = energies[:, None]
    _xs = xs[:, None]

    const = 4 / np.sqrt(2 * np.pi * mu) / (_tion**1.5)
    integrand = _xs * _E * np.exp(-_E / _tion)

    r = const * np.trapezoid(integrand, x=_E, axis=0)
    r = r[0, :].to(u.m**3 / u.s)

    if r.size == 1:
        return r[0]
    else:
        return r


def d3hep_yield(
    DDn_yield: float,
    D2_pressure: u.Quantity | float,
    He_pressure: u.Quantity | float,
    tion: u.Quantity,
):
    """
    The ratio of D3He protons to DD neutrons produced for specified fill pressures and ion temperature.

    D3He exploding pushers are a common backlighter for proton radiography experiments. They produce the three
    reactions

    - D(D,n)
    - D(D,p) (~3 MeV)
    - 3He(D,p) (~15 MeV)

    The neutron yield from the first reaction, and the deuterium ion temperature, are measured by the neutron
    time-of-flight detectors. The branching ratio between the D(D,n) and D(D,p) reactions is 50/50, so the
    neutron yield also gives the D(D,p) proton yield. This function calculates the expected 3He(D,p) yield for
    an expected D(D,n) yield and ion temperature (which influences the relative reactivities). This may be used
    when designing an experiment, or when estimating fluence of 3He(D,p) protons on the CR39 prior to etching.

    Parameters
    ----------

    DDn_yield : float
        The D(D,n) neutron yield

    D2_pressure: u.Quantity or float
        The D2 fill pressure. Can be a float as long as the units
        match the He pressure, since only the ratio enters into
        the calculation.

    He_pressure: u.Quantity or float
        The 3He fill pressure. Can be a float as long as the units
        match the D2 pressure, since only the ratio enters into
        the calculation.

    tion : u.Quantity, eV
        The deuterium ion temperature.


    Returns
    -------

    estimated_yield : float
        The estimated 3He(D,p) proton yield.

    """

    dd_reactivity = reactivity("D(D,n)", tion)
    d3he_reactivity = reactivity("3He(D,p)", tion)

    # Factor of 2 represents that D2 has two atoms per molecule, while 3He is monoatomic
    return (
        DDn_yield * d3he_reactivity / dd_reactivity * He_pressure / (2 * D2_pressure)
    ).m_as(u.dimensionless)


def ddp_energy(d3he_energy: float):
    """
    Estimates the final energy of a D(D,p) proton given the energy of a 3He(D,p) proton.

    The D(D,p) reaction produces protons with an energy of 3.02 MeV, while the 3He(D,p) reaction produces
    protons with an energy of 14.64 MeV. However, charging of the capsule causes a shift (generally an up shift)
    in the particle energy that is somewhat unpredictable. Typically the energy of the 3He(D,p) protons is
    directly measured using a wedge-range filter.  This function infers the energy of the D(D,p) protons based
    on this measurement, assuming that the energy upshift is the same for each species.

    Parameters
    ----------
    d3he_energy : float
        The energy of the 3He(D,p) proton on the detector, in MeV.

    Returns
    -------
    ddp_energy :float
        The estimated energy of the D(D,p) proton, in MeV.
    """

    return d3he_energy - 14.64 + 3.02
