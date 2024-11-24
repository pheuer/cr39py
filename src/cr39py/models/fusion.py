"""
The `~cr39py.models.fusion` module contains functions relating to the fusion reactions that commonly create
the charged particles detected by CR39.
"""

import h5py
import numpy as np

from cr39py.core.data import get_resource_path
from cr39py.core.units import u


def cross_section(reaction: str) -> tuple[u.Quantity]:
    """
    The fusion cross section for a given nuclear reaction.

    Cross-section data is scraped from the ENDF database.

    Parameters
    ----------

    reaction : str
        The nuclear reaction. Supported strings are:
        - 'D(D,n)'
        - 'D(D,p)'
        - '3He(D,p)'

    Returns
    -------

    energies : u.Quantity
        Energy axis

    xs : u.Quantity
        Cross-section

    """
    files = {
        "D(D,n)": "D(D,n)He-3.h5",
        "D(D,p)": "D(D,p)T.h5",
        "3He(D,p)": "He-3(D,p)A.h5",
    }

    if reaction not in files:
        raise ValueError(
            f"Reaction {reaction} not recognized. Valid inputs are " f"{list(files)}"
        )

    path = get_resource_path(files[reaction])
    with h5py.File(path, "r") as f:
        energies = f["energy"][:] * u.eV
        xs = f["SIG"][:] * u.m**2

    return energies, xs


def reduced_mass(reaction: str) -> float:
    """
    The reactant reduced mass for a nuclear reaction.

    Reaction string should be in the format r1(r2,p1)p2

    Valid reactants are [p,D,T,3He,4He]
    """
    masses = {"p": 1, "D": 2, "T": 3, "3He": 3, "4He": 4}

    reactants = reaction.split(",")[0]
    r1, r2 = reactants.split("(")
    m1, m2 = masses[r1], masses[r2]

    return m1 * m2 / (m1 + m2)


def reactivity(reaction: str, tion: u.Quantity) -> tuple[u.Quantity]:
    """
    The fusion reactivity for a nuclear reaction at a given ion temperature.

    Parameters
    ----------

    reaction : str
        The nuclear reaction. See valid reactions on
        `~cr39py.models.fusion.cross_section`.

    tion : u.Quantity, eV
        Ion temperature.

    Returns
    -------

    energies : u.Quantity
        Energy axis

    xs : u.Quantity
        Cross-section


    Notes
    -----

    This is quite a nice example notebook on fusion reactivities in python
    https://scipython.com/blog/nuclear-fusion-cross-sections/

    """
    mu = reduced_mass(reaction)
    energies, xs = cross_section(reaction)
    k_B = 1.38e-23 * u.J / u.K

    # TODO: do the calculation: https://scipython.com/blog/nuclear-fusion-cross-sections/


if __name__ == "__main__":
    print(reduced_mass("3He(D,p)"))
