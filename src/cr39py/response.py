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

    vB = 2.66  # km/s

    def _particle(self, p):
        return str(p).lower()

    def track_energy(self, diameter, particle, etch_time):
        """
        The energy corresponding to a track of a given diameter.

        Parameters
        ----------
        diameter : float
            Track diameter in um

        particle : str
            Particle type: "p", "d", "t", "a"

        etch_time : float
            Etch time in minutes.

        Returns
        -------

        energy : float
            Energy of track in MeV

        """

        etch_time_hrs = etch_time / 60

        key = self._particle(particle)
        Z = self._data[key]["Z"]
        A = self._data[key]["A"]
        k = self._data[key]["k"]
        n = self._data[key]["n"]

        return Z**2 * A * ((2 * etch_time_hrs * self.vB / diameter - 1) / k) ** (1 / n)

    def track_diameter(self, energy, particle, etch_time):
        """
        The diameter for a track after a given etch time.

        Parameters
        ----------
        energy : float
            Particle energy in MeV

        particle : str
            Particle type: "p", "d", "t", "a"

        etch_time : float
            Etch time in minutes.

        Returns
        -------

        diameter : float
            Track diameter in um
        """

        etch_time_hrs = etch_time / 60

        key = self._particle(particle)
        Z = self._data[key]["Z"]
        A = self._data[key]["A"]
        k = self._data[key]["k"]
        n = self._data[key]["n"]

        return np.where(
            energy > 0,
            2 * etch_time_hrs * self.vB / (1 + k * (energy / (Z**2 * A)) ** n),
            np.nan,
        )
