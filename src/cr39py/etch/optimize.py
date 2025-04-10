"""The `~cr39py.etch.optimize` module contains tools for calculating optimal etch times."""

from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from cr39py import Layer, Stack, u
from cr39py.etch.overlap import mrn_distribution, mrn_overlap_fraction
from cr39py.models.response import TwoParameterModel


class EtchTimeOptimizer:
    """
    Simulates the development of the MRN diameter distribution with etch time to identify the optimal etch time.

    Parameters
    ----------
    energy_mean : `~cr39py.core.units.Quantity`
        Mean energy of the input energy distribution in MeV.

    energy_std : `~cr39py.core.units.Quantity`
        Standard deviation of the input energy distribution in MeV.

    filterstack : `~cr39py.filtration.stack.Stack` or `~cr39py.filtration.layer.Layer`
        The Stack or Layer that the particles filter through, up to the scan surface.

    particle : str, optional
        The type of particle to range. Default is 'proton'.

    """

    def __init__(
        self,
        energy_mean: u.Quantity,
        energy_std: u.Quantity,
        filterstack: Stack | Layer,
        particle: str = "proton",
    ) -> None:

        self.energy_mean = energy_mean.m_as(u.MeV)
        self.energy_std = energy_std.m_as(u.MeV)
        self.filterstack = filterstack
        self.particle = particle

        self._ein_axis = np.linspace(
            self.energy_mean - 3 * self.energy_std,
            self.energy_mean + 3 * self.energy_std,
            num=100,
        )  # MeV

    @cached_property
    def reduced_ranging_model(self):
        """
        A `~cr39py.filtration.stack.Stack.reduced_ranging_model` for the filter stack.
        """
        coeffs, model = self.filterstack.reduced_ranging_model(
            particle=self.particle, eout_cutoff=1 * u.MeV, plot=False
        )
        return model

    @property
    def ein_axis(self):
        """
        Returns the input energy axis.
        """
        return self._ein_axis

    @cached_property
    def _eout_axis_all(self):
        """
        Eout axis including NaN for stopped energies
        """
        return self.reduced_ranging_model(self.ein_axis)

    @cached_property
    def eout_axis(self):
        """
        Returns the energy out axis for the given energy in axis
        """
        e_out = self._eout_axis_all
        e_out = e_out[~np.isnan(e_out)]
        return e_out

    @cached_property
    def ein_distribution(self):
        """
        Input energy Gaussian distribution.
        """
        e_dist = np.exp(
            -((self.ein_axis - self.energy_mean) ** 2) / (2 * self.energy_std**2)
        )
        e_dist /= np.sum(e_dist)

        return e_dist

    @cached_property
    def eout_distribution(self):
        """
        Energy distribution after filter stack.
        """
        ind = ~np.isnan(self._eout_axis_all)
        return self.ein_distribution[ind]

    def diameter_axis(self, etch_time):
        """
        Returns the diameters for a list of given input energies at a given etch time.

        Parameters
        ----------
        etch_time : float
            Etch time in minutes.
        """
        # TODO: get rid of this hack by using a standard format for particle strings...
        pstr = self.particle[0]

        model = TwoParameterModel(particle=pstr)
        diameters = model.track_diameter(self.eout_axis, etch_time)

        return diameters

    def optimal_etch_time(
        self,
        fluence,
        etch_time_range=(1 * 60, 10 * 60),
        diameter_zone=(3, 8),
        overlap_max=0.1,
        plot=False,
    ):
        """
        Finds the optimal etch time to get as much signal as possible in
        the diameter zone.

        Parameters
        ----------
        etch_time_range : tuple, optional
            Etch time range in minutes.

        diameter_zone : tuple, optional
            Diameter range in which to maximize the signal, in um.

        overlap_max : float, optional
            Maximum track overlap fraction (1-F1) allowed.

        plot : bool, optional
            If True, plot the signal vs etch time.
        """

        def signal(diameters, diameter_zone):
            """
            Returns the signal in the diameter zone for a given etch time
            """
            # Get the number of tracks in the diameter zone
            ind = np.logical_and(
                diameters >= diameter_zone[0], diameters <= diameter_zone[1]
            )

            return np.sum(self.eout_distribution[ind])

        # Calculate for etch times in 5 min increments
        etch_times = np.arange(etch_time_range[0], etch_time_range[1], 5)

        signals = np.zeros(etch_times.shape)
        overlap = np.zeros(etch_times.shape)
        for i, etch_time in enumerate(etch_times):

            diameters = self.diameter_axis(etch_time)

            # Calculate the signal in the diameter zone
            signals[i] = signal(diameters, diameter_zone)

            # Fit the diameter distribution with the model
            popt, pcov = curve_fit(
                mrn_distribution, diameters, self.eout_distribution, p0=[1, 1]
            )
            maxd, sigma = popt

            # Interpolate the 1-F1 value
            # TODO: Set  bounds so sigma doesn't go negative? Or this works fine...
            overlap[i] = mrn_overlap_fraction(-1, fluence, maxd, np.abs(sigma))

        # Find the etch time for the most signal
        max_signal_time = etch_times[np.argmax(signals)]

        # Check latest etch time that satistifies the overlap condition
        overlap_ok = etch_times[overlap <= overlap_max]
        if overlap_ok.size == 0:
            raise ValueError("No etch times are within the overlap condition")

        max_time_overlap = overlap_ok[-1]

        # If the max signal time is more than the overlap time, etch to the
        # overlap time. Otherwise, etch to the max signal time
        optimal_etch_time = (
            max_signal_time if max_signal_time < max_time_overlap else max_time_overlap
        )

        if plot:
            fig, ax = plt.subplots()
            ax.set_title("Fluence: {:.2e} tracks/cm^2".format(fluence))
            ax.set_ylim(0, 100)
            ax.plot(etch_times, 100 * signals, label="Percent Signal in diameter zone")

            ax.axvline(
                optimal_etch_time,
                color="r",
                linestyle="solid",
                label=f"Optimal etch time: {optimal_etch_time/60:.1f} hr",
            )
            ax.axvline(
                max_signal_time, color="b", linestyle="--", label="Max signal etch time"
            )
            ax.axvline(
                max_time_overlap,
                color="orange",
                linestyle="--",
                label="Max overlap etch time",
            )

            ax.axhline(
                overlap_max * 100,
                color="gray",
                linestyle="--",
                label=f"Overlap max: {overlap_max*100:.0f}%",
            )

            # ax.axvspan(etch_time_range[0], etch_time_range[1], color='lime', alpha=0.2, label='>95% of max signal in diameter zone')
            ax.set_xlabel("Etch time (min)")
            ax.set_ylabel(f"% Signal")

            ax.plot(etch_times, overlap * 100, color="orange", label="1-F1")

            ax.legend()

        return optimal_etch_time
