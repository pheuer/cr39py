"""
Detector response functions for CR39

References:

N. Sinenian et al. 2011 RSI 82(10) https://doi.org/10.1063/1.3653549
B. Lahmann et al. 2020 RSI 91(5) https://doi.org/10.1063/5.0004129


"""

import numpy as np

from cr39py.core.units import unit_registry as u

__all__ = ["CParameterModel", "TwoParameterModel"]


class CParameterModel:
    """
    The C-parameter model of B. Lahmann et al. 2020.

    Only suitable for protons, but for that application this model
    is more accurate than the two-parameter model.
    """

    # Not implemented yet
    pass


class TwoParameterModel:
    """
    The Two-parameter model of B. Lahmann et al. 2020 provides
    a response function for protons, deuterons, tritons, and alphas.
    """

    # Response coefficents for protons, deuterons, tritions, and alphas
    # From Table 1 of Lahmann et al. 2020 RSI
    _data = {
        "p": {"Z": 1, "A": 1, "k": 0.7609, "n": 1.497},
        "d": {"Z": 1, "A": 2, "k": 0.8389, "n": 1.415},
        "t": {"Z": 1, "A": 3, "k": 0.8689, "n": 1.383},
        "a": {"Z": 2, "A": 4, "k": 0.3938, "n": 1.676},
    }

    # Bulk etch velocity is constant
    vB = 2.66  # km/s

    def __init__(self, particle, k=None, n=None):
        self.particle = str(particle).lower()

        self._k = k
        self._n = n

    @property
    def Z(self):
        return self._data[self.particle]["Z"]

    @property
    def A(self):
        return self._data[self.particle]["A"]

    @property
    def k(self):
        return self._data[self.particle]["k"] if self._k is None else self._k

    @k.setter
    def k(self, k):
        self._k = k

    @property
    def n(self):
        return self._data[self.particle]["n"] if self._n is None else self._n

    @n.setter
    def n(self, n):
        self._n = n

    def track_energy(self, diameter, etch_time, k=None, n=None):
        """
        The energy corresponding to a track of a given diameter.

        Parameters
        ----------
        diameter : float
            Track diameter in um

        etch_time : float
            Etch time in minutes.

        Returns
        -------

        energy : float | `~numpy.nan`
            Energy of track in MeV, or `~numpy.nan` if there is no
            valid energy that could have created a track of this diameter
            at this etch time.
        """
        k = self.k if k is None else k
        n = self.n if n is None else n

        etch_time_hrs = etch_time / 60
        energy = (
            self.Z**2
            * self.A
            * ((2 * etch_time_hrs * self.vB / diameter - 1) / self.k) ** (1 / self.n)
        )
        return energy if not np.iscomplex(energy) else np.nan

    def track_diameter(self, energy, etch_time, k=None, n=None):
        """
        The diameter for a track after a given etch time.

        Eq. 5 of B. Lahmann et al. 2020 RSI

        Parameters
        ----------
        energy : float
            Particle energy in MeV

        etch_time : float
            Etch time in minutes.

        Returns
        -------

        diameter : float
            Track diameter in um.
        """
        k = self.k if k is None else k
        n = self.n if n is None else n

        etch_time_hrs = etch_time / 60

        return (
            2
            * etch_time_hrs
            * self.vB
            / (1 + self.k * (energy / (self.Z**2 * self.A)) ** self.n)
        )

    def etch_time(self, energy, desired_diameter, k=None, n=None):
        """
        The etch time required to bring a track to the desired diameter.

        Parameters
        ----------
        energy : float
            Particle energy in MeV

        desired_diameter : float
            Desired final track diameter

        Returns
        -------

        etch_time : float
            Total etch time, in minutes
        """
        k = self.k if k is None else k
        n = self.n if n is None else n

        etch_time_hrs = (
            desired_diameter
            * (1 + k * (energy / (self.Z**2 * self.A)) ** n)
            / 2
            / self.vB
        )

        return etch_time_hrs * 60
