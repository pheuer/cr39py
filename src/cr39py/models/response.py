"""
Detector response functions for CR39
====================================

CR-39 can be either "bulk etched" to remove material uniformly, or "track etched"
to develop tracks in the surface. Both etching processes are done in a sodium hydroxide (NaOH) bath.

Bulk Etch
---------
Bulk etching is performed in a mixture of 25% 10 normal NaOH and 75% methanol at 55 degrees C. This
rapidly and uniformly removes surface material. The `~cr39py.models.response.BulkEtchModel` class provides a simple model
for the amount of material removed given the bulk etch velocity.


Track Etch
----------
Track etching is done in a 6 normal (6 g/l) NaOH solution at 80 degrees C. During track etching,
about 2 um/hr of material is removed uniformly from the surface. The `~cr39py.models.response.CParameterModel` and
`~cr39py.models.response.TwoParameterModel` classes provide response functions that can be used to estimate the energy
of a particle that created a track of a given diameter after a given track etch.


References
----------
The following papers are useful references for CR-39 response to charged particles

* :cite:t:`Cartwright1978nuclear` is one of the original papers on the use of CR-39 for detecting charged particles.
* :cite:t:`Sinenian2011response` presents initial measurements of track diameter vs. etch time, and discusses the impact of factors such as etch bath temperature, time to exposure, and CR-39 aging.
* :cite:t:`Lahmann2020cr39` presents the c-parameter and two-parameter models for CR-39 response that are currently the most widely used response models.

These papers consider the response to other particles

* :cite:t:`Frenje2002absolute` measures the response of CR-39 to neutrons.

And these papers consider experimental factors that can modify the response of CR-39

* :cite:t:`Manuel2011changes` discusses the way that CR-39 response is modified by prolonged vacuum exposure.
* :cite:t:`Rinderknecht2015impact` and :cite:t:`RojasHerrera2015impact` discuss how x-ray exposure modifies the response of CR-39 to charged particles.

"""

import numpy as np
from scipy.interpolate import interp1d

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

    def __init__(self, bulk_etch_velocity=31.5 * u.um / u.hr):
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
    The C-parameter model of :cite:t:`Lahmann2020cr39`

    Only suitable for protons, but for that application this model
    is more accurate than the two-parameter model :cite:p:`Lahmann2020cr39`.

    The C-parameter model specifically fits the diameter of tracks in the 'flat' region
    of the 'hockey-stick' curve in diameter-contrast space, where tracks of increasing energy/etch time
    move approximately horizontally, so E(D) with no contrast dependence.

    References
    ----------
    When using this model, please cite :cite:t:`Lahmann2020cr39`.

    """

    def __init__(self, c, dmax):
        self._c = c
        self._dmax = dmax

    @property
    def c(self):
        """
        The C parameter of the model.
        """
        return self._c

    @c.setter
    def c(self, c):
        self._c = c

    @property
    def dmax(self):
        """
        The Dmax parameter of the model.

        The Dmax parameter is intended to correspond to the maximum real
        track diameter in the data, but this is not always exactly right. See
        the discussion in Appendix C of :cite:t:`Lahmann2020cr39`.
        """
        return self._dmax

    @dmax.setter
    def dmax(self, dmax):
        self._dmax = dmax

    def scaled_diameter_curve(self, E: np.ndarray) -> np.ndarray:
        """
        Scaled diameter as a function of incident particle energy.

        Parameters
        ----------

        E : np.ndarray
            Incident particle energy in MeV.

        Returns
        -------
        D : np.ndarray
            Scaled track diameter.


        Eq. B1-B3 of :cite:t:`Lahmann2020cr39`.

        Note
        ----
        There is a typo in the Lahmann paper that neglects to specify
        how to choose between B2 and B3. The correct form is shown below.
        """
        alphas = [1, 2, 11.3, 4.8]
        betas = [0.3, 3.0, 8.0]
        E = np.atleast_1d(E)
        D = np.zeros(E.shape)
        for alpha, beta in zip(alphas, betas):
            D += alpha * np.exp(-(E - 1) / beta)

        # Eq. B2 of Lahmann et al.
        if self.c <= 1:
            mask = D <= 20
            D[mask] = 20 * np.exp(-self.c * np.abs(np.log(D[mask] / 20)))
            mask = D > 20
            D[mask] = 40 - 20 * np.exp(-self.c * np.abs(np.log(D[mask] / 20)))

        # Eq. B3 of Lahmann et al.
        if self.c > 1:
            mask = D <= 10
            D[mask] = ((20 - D[mask]) ** 2 / (20 - 2 * D[mask])) * (
                np.exp(self.c / 2 * np.log(D[mask] ** 2 / (20 - D[mask]) ** 2)) - 1
            ) + 20
            mask = D > 10
            D[mask] = 20 - self.c * (20 - D[mask])

        return D

    @property
    def _M(self):
        """
        A parameter used along with dmax to scale diameters.
        """
        # Eq. B6 of Lahmann et al.
        if self.dmax < 12.5:
            f = 0
        elif self.dmax > 20:
            f = 1
        else:
            f = (self.dmax - 12.5) / (20 - 12.5)

        # Eq. B5 of Lahmann et al.
        M = (
            (20 - self.dmax)
            / (20 * self.dmax)
            * (7 / 10 * (1 - self.dmax / 23) * (1 - f) + f / 4)
        )

        return M

    def D_raw(self, D_scaled: np.ndarray) -> np.ndarray:
        """
        Convert's scaled diameter to raw diameters in um.

        Eq. B4 of :cite:t:`Lahmann2020cr39`, inverted for D_raw.

        Parameters
        ----------
        D_scaled : np.ndarray
            Scaled track diameters.

        Returns
        -------
        D_Raw : np.ndarray
            Raw track diameters, in um.
        """

        return self.dmax / (20 / D_scaled + self._M * self.dmax)

    def D_scaled(self, D_raw: np.ndarray) -> np.ndarray:
        """
        Convert's raw diameters in um to scaled diameters.

        Eq. B4 of :cite:t:`Lahmann2020cr39`.

        Parameters
        ----------
        D_Raw : np.ndarray
            Raw track diameters, in um.

        Returns
        -------
        D_scaled : np.ndarray
            Scaled track diameters.
        """

        return 20 * (D_raw / self.dmax) / (1 - self._M * D_raw)

    def track_diameter(self, energy: np.ndarray):
        """
        Track diameter as a function of incident particle energy.

        Evaluates Eq. B1-B6 of :cite:t:`Lahmann2020cr39`.

        Parameters
        ----------
        energy : np.ndarray
            Incident particle energy in MeV.

        Returns
        -------
        diameter : np.ndarray
            Track diameters in um.
        """

        D_scaled = self.scaled_diameter_curve(energy)
        D_raw = self.D_raw(D_scaled)
        return D_raw

    def track_energy(self, diameter: np.ndarray, eaxis=None):
        """
        Incident particle energy for a given track diameter.

        Inverts Eq. B1-B6 of :cite:t:`Lahmann2020cr39` by interpolating
        the scaled diameter curve.

        Parameters
        ----------
        diameter : np.ndarray
            Track diameters in um.

        eaxis : np.ndarray, optional
            Energy axis for interpolation. If not provided, a default
            axis of 200 points from 0.1 to 50 MeV is used.

        Returns
        -------
        energy : np.ndarray
            Incident particle energy in MeV.
        """

        if eaxis is None:
            eaxis = np.linspace(0.1, 50, 200)

        # First find the scaled diameter
        D_scaled = self.D_scaled(diameter)

        DE_curve = self.scaled_diameter_curve(eaxis)
        interp = interp1d(
            DE_curve, eaxis, kind="cubic", bounds_error=False, fill_value=np.nan
        )

        E = interp(D_scaled)

        if np.max(E) > np.max(eaxis):  # pragma: no cover
            raise ValueError(
                "Energy exceeds the maximum of the energy axis for interpolation - increase the maximum energy."
            )

        return E


class TwoParameterModel:
    """
    A CR-39 response model for protons, deuterons, tritons, and alphas.

    This model was initially described in :cite:t:`Lahmann2020cr39`.

    The two parameter model specifically predicts the diameter of tracks in the 'flat' region
    of the 'hockey-stick' curve in diameter-contrast space, where tracks of increasing energy/etch time
    move approximately horizontally, so E(D) with no contrast dependence.

    References
    ----------
    When using this model, please cite :cite:t:`Lahmann2020cr39`.
    """

    # Response coefficients for protons, deuterons, tritions, and alphas
    # From Table 1 of Lahmann et al. 2020 RSI
    _data = {
        "p": {"Z": 1, "A": 1, "k": 0.7609, "n": 1.497},
        "d": {"Z": 1, "A": 2, "k": 0.8389, "n": 1.415},
        "t": {"Z": 1, "A": 3, "k": 0.8689, "n": 1.383},
        "a": {"Z": 2, "A": 4, "k": 0.3938, "n": 1.676},
    }

    def __init__(self, particle, k=None, n=None, vB=2.66):
        self.particle = str(particle).lower()

        self._k = k
        self._n = n
        self._vB = vB  # um/hr

    @property
    def Z(self):
        """
        The selected particle's charge number.
        """
        return self._data[self.particle]["Z"]

    @property
    def A(self):
        """
        The selected particle's atomic mass number.
        """
        return self._data[self.particle]["A"]

    @property
    def k(self):
        """
        The ``k`` constant for the currently selected particle, as defined in
        Table 1 of :cite:t:`Lahmann2020cr39`.
        """
        return self._data[self.particle]["k"] if self._k is None else self._k

    @k.setter
    def k(self, k):
        self._k = k

    @property
    def n(self):
        """
        The ``n`` constant for the currently selected particle, as defined in
        Table 1 of :cite:t:`Lahmann2020cr39`.
        """
        return self._data[self.particle]["n"] if self._n is None else self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def vB(self):
        """
        The bulk etch velocity ``vB``  as defined in :cite:t:`Lahmann2020cr39`.
        """
        return self._vB

    @vB.setter
    def vB(self, vB):
        self._vB = vB

    def track_energy(self, diameter, etch_time, k=None, n=None):
        """
        The energy corresponding to a track of a given diameter.

        Equivalent to Eq. 5 of :cite:t:`Lahmann2020cr39`.

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

        Eq. 5 of :cite:t:`Lahmann2020cr39`.

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

        Equivalent to Eq. 5 of :cite:t:`Lahmann2020cr39`.

        Parameters
        ----------
        energy : float
            Particle energy in MeV

        desired_diameter : float
            Desired final track diameter in um.

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
