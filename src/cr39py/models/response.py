"""
Detector response functions for CR39.

CR-39 can be either "bulk etched" to remove material uniformly, or "track etched"
to develop tracks in the surface. Both etching processes are done in a sodium hydroxide (NaOH) bath.

Bulk Etch
---------
Bulk etching is performed in a mixture of 25% 10 normal NaOH and 75% methanol at 55 degrees C. This
rapidly and uniformly removes surface material. The ``BulkEtchModel`` class provides a simple model
for the amount of material removed given the bulk etch velocity.


Track Etch
----------
Track etching is done in a 6 normal (6 g/l) NaOH solution at 80 degrees C. During track etching,
about 2 um/hr of material is removed uniformly from the surface. The ``CParameterModel`` and
``TwoParameterModel`` classes provide response functions that can be used to estimate the energy
of a particle that created a track of a given diameter after a given track etch.


References
----------
N. Sinenian et al. 2011 RSI 82(10) https://doi.org/10.1063/1.3653549
B. Lahmann et al. 2020 RSI 91(5) https://doi.org/10.1063/5.0004129


"""

import numpy as np

from cr39py.core.units import unit_registry as u

__all__ = ["BulkEtchModel", "CParameterModel", "TwoParameterModel"]


class BulkEtchModel:
    """
    A simple fixed-velocity model for bulk etching CR-39.

    Parameters
    ----------

    bulk_etch_velocity : `~astropy.units.Quantity`
        The velocity at which material is removed during bulk etching.
        The default values is 63 um/hr, which is based on measurements
        at LLE.
    """

    def __init__(self, bulk_etch_velocity=63 * u.um / u.hr):
        self._bulk_etch_velocity = bulk_etch_velocity

    def removal(self, etch_time: u.Quantity):
        """
        Amount (depth) of CR-39 removed in a given time.

        This is the amount removed from a single surface: the piece is etched
        on both sides, so the total decrease in thickness will be 2x this value.

        Parameters
        ----------

        etch_time : u.Quantity
            Etch time

        Returns
        -------
        depth : u.Quantity
            Depth of material to remove

        """
        return (self._bulk_etch_velocity * etch_time).to(u.um)

    def time_to_remove(self, depth: u.Quantity):
        """
        Etch time to remove a given amount of material.

        This is the amount removed from a single surface: the piece is etched
        on both sides, so the total decrease in thickness will be 2x this value.

        Parameters
        ----------

        depth : u.Quantity
            Depth of material to remove


        Returns
        -------

        etch_time : u.Quantity
            Etch time

        """
        return (depth / self._bulk_etch_velocity).to(u.hr)


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

    # Response coefficients for protons, deuterons, tritions, and alphas
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
