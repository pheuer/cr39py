"""
The `~cr39py.filtration.layerlike` module contains the `~cr39py.filtration.layer.LayerLike` class,
which is a base class for ``Layer`` and ``Stack``. Methods in ``LayerLike`` utilize the shared
methods of the ``Layer`` and ``Stack`` classes.
The `~cr39py.filtration.layerlike` module is not intended to be used directly.
"""

from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from cr39py.core.units import u


def _eout_model(ein, estart, a, n):
    res = np.zeros(ein.shape)
    nonzero = ein >= estart
    res[nonzero] = a * (ein[nonzero] - estart) ** n + 1
    res[~nonzero] = np.nan

    return res


class LayerLike:

    def reduced_ranging_model(
        self,
        particle: str = "proton",
        eout_cutoff: u.Quantity = 1 * u.MeV,
        ein_max: u.Quantity = 20 * u.MeV,
        plot=True,
    ):
        """
        A reduced model for the ranging of a particle through the stack or layer.

        The model is:

        .. math::

            E_{out}(E_{in})= \begin{cases}
                a(E_{in}-E_0)^b & \text{if } E_{in} \geq E_0 \\
                \text{NaN} & \text{if } E_{in} < E_0 \\
            \end{cases}

        This model is accurate for particles that exit the layer or stack with energies
        close to eout_cutoff.

        Parameters
        ----------
        particle : str, optional
            The type of particle to range. Default is 'proton'.

        eout_cutoff : `~cr39py.core.units.Quantity`, optional
            The low-end cutoff output energy below which the model
            should not be fit. This is generally set to the CR-39
            detection sensitivity threshold, which is ~1 MeV.

        ein_max : `~cr39py.core.units.Quantity`, optional
            The maximum incoming particle energy to fit the ranging
            model to.

        plot : bool
            If True, plot the sample points and the fitted model.

        Returns
        -------

        coeffs : list[3]
            The coefficients of the fitted model.
            - The first element is the minimum input energy corresponding to
                the eout_cutoff. When eout_cutoff is set to the CR-39 detection
                threshold, this represents the lowest input  energy that will
                be detectable
            - The second and third elements are the scaling factor and exponent
                of the fitted model, respectively.

        model : callable
            A function that takes input energy in MeV and returns the output
            energy in MeV. The function returns NaN for output energies below the
            eout_cutoff.


        """
        ein_max = ein_max.m_as(u.MeV)

        # Find the zero energy by ranging up a 1 MeV proton through the stack
        emin = self.reverse_ranging(particle, 1 * u.MeV).m_as(u.MeV)[0]

        # Range down a few points across the selected range
        e_in = np.linspace(emin, ein_max, 10)  # MeV
        e_out = self.range_down(particle, e_in * u.MeV).m_as(u.MeV)  # MeV

        # TODO: to make this model generalize better further from emin,
        # fit the farther away part with a separate linear model?

        _model = lambda x, a, n: _eout_model(x, emin, a, n)
        popt, pcov = curve_fit(_model, e_in, e_out, p0=[1, 0.6])

        if plot:
            fig, ax = plt.subplots()
            ein_axis = np.linspace(0, ein_max, num=200)
            ax.scatter(e_in, e_out, label="Data")
            ax.set_title(f"Eout={popt[0]:.2f}(Ein - {emin:.2f})^{popt[1]:.2f}")
            ax.plot(ein_axis, _model(ein_axis, *popt), label="Fitted curve")

        coeff = [emin, *popt]
        eout_model = lambda e: _model(e, *popt)
        return coeff, eout_model
