from functools import cache
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

from cr39py.core.data import data_dir
from cr39py.core.units import u
from cr39py.models.response import CParameterModel
from cr39py.scan.base_scan import Scan


@cache
def _remove_ranging_interpolator():
    """
    Returns an interpolator for incident proton energy as a function of
    filter thickness and output energy.

    Currently the only file included is for protons in aluminum - could generate
    more for other particles if needed?

    Returns
    -------

    E_in_interp : RegularGridInterpolator
        Interpolator that takes (thickness(um), E_out(MeV)) and returns E_in(MeV)

    """

    data_file = data_dir / Path("srim/proton_Al_remove_ranging.h5")
    with h5py.File(data_file, "r") as f:
        thickness = f["thickness"][:]  # um
        E_out = f["E_out"][:]  # MeV
        E_in = f["E_in"][:, :]  # MeV

    E_in_interp = RegularGridInterpolator(
        (thickness, E_out), E_in, bounds_error=False, method="cubic"
    )

    return E_in_interp


def synthetic_wrf_data(
    params: np.ndarray,
    xaxis: np.ndarray,
    daxis: np.ndarray,
    wrf_calib: np.ndarray,
    remove_ranging_interpolator=None,
) -> np.ndarray:
    """
    Creates synthetic 2D WRF data (in X,D space).

    Parameters
    ----------
    params : Sequence[float] len(4)
        The four fittable parameters:
        - Mean energy of Gaussian particle energy distribution.
        - Standard deviation of the Gaussian particle energy distribution.
        - C-parameter for the C-parameter response model.
        - dmax parameter for the C-parameter response model.

    xaxis : np.ndarray
        X-axis, in cm

    daxis : np.ndarray
        Diameter axis, in um

    wrf_calib: tuple(float)
        WRF slope and offset calibration coefficients (m,b).

    remove_ranging_interpolator : RegularGridInterpolator, optional
        Interpolator for translating E_in( (E_out, wrf_thickness)).
        If an interpolator is not provided, one will be created.
        Providing an already-loaded interpolator allows greatly speeds
        up calls to this function, necessary for fitting.

    Returns
    -------
    synthetic_data : np.ndarray
        Synthetic data (arbitrary units) in X,D space.
    """

    if remove_ranging_interpolator is None:
        remove_ranging_interpolator = _remove_ranging_interpolator()

    emean, estd, c, dmax = params

    # Convert the track diameter axis from the scan to track energy
    model = CParameterModel(c, dmax)

    # Calculate energy incident on CR-39 from diameters
    ein_axis = model.track_energy(daxis)

    # Calculate the WRF thickness
    m, b = wrf_calib
    wrf_thickness = m * xaxis + b

    # Translate that E_in axis to an E_out value at every thickness
    T, E = np.meshgrid(wrf_thickness, ein_axis, indexing="ij")
    eout_axis = remove_ranging_interpolator((T, E))

    # Use those E_in values with the distribution to create a synthetic image
    synthetic_data = np.exp(-((emean - eout_axis) ** 2) / 2 / estd**2)

    return synthetic_data


def wrf_objective_function(synthetic: np.ndarray, data: np.ndarray, return_sum=True):
    """
    Calculates the chi2 error between synthetic and real WRF data in X,D space.

    Parameters
    ----------
    synthetic: np.ndarray [nx, nd]
        Synthetic WRF data.

    data : np.ndarray [nx, nd]
        Actual WRF data to compare with synthetic data.

    return_sum : bool, optional
        If True, returns just the nansum of the chi2 over
        the entire image. Default is True.

    Returns
    -------
    chi2 : np.ndarray[nx,nd] | float
        Chi2 map or summed single value, depending on ``return_sum`` keyword.

    """
    # Create a mask to only compare the values where they are finite and where
    # the data, which is in the denominator, is non-zero
    mask = np.isfinite(synthetic) * np.isfinite(data) * (data > 0)

    if return_sum:
        _data = data[mask]
        _data /= np.nansum(_data)
        _synthetic = synthetic[mask]
        _synthetic /= np.nansum(synthetic)
        return np.nansum((_data - _synthetic) ** 2 / _data)

    # If returning the entire chi2 array, we set the unused pixels to NaN
    # to retain the shape of the original data
    else:
        _data = np.copy(data)
        _data[~mask] = np.nan
        _synthetic = np.copy(synthetic)
        _synthetic[~mask] = np.nan

        _data = _data / np.nansum(_data)
        _synthetic_data = _synthetic_data / np.nansum(_synthetic_data)

        chi2 = (_data - _synthetic_data) ** 2 / _data
        return chi2


class WedgeRangeFilter(Scan):

    _calib_file = data_dir / Path("calibration/wrf_calibrations.yml")

    def __init__(self) -> None:
        super().__init__()

        # WRF calibration coefficients
        self._m = None
        self._b = None

    @property
    def _wrf_calib_data(self):
        """
        WRF calibration data as dictionary.

        WRF id codes (lowercase) are the keys, then contents is another
        dictionary containing 'm' and 'b'.
        """

        with open(self._calib_file, "r") as f:
            data = yaml.safe_load(f)

        return data

    def _get_wrf_calib_from_file(self, id: str):
        """
        Looks up the m,b calibration coefficients for a WRF
        from the calibration file.

        Parameters
        ----------
        id : str
            WRF id code, e.g. "g034". Lowercase.

        Returns
        -------
        m,b : float
            Slope and offset fit coefficients.
        """

        data = self._wrf_calib_data
        if id.lower() not in data:
            raise KeyError(f"No calibration data found for {id} in {self._calib_file}")

        entry = data[id]
        m, b = entry["m"], entry["b"]
        return m, b

    def _get_wrf_id_from_filename(self, path: Path):
        """
        See if the filename contains a valid WRF id.
        """
        data = self._wrf_calib_data

        # Split filename by underscores, then try
        # to see if any segment is a valid WRF id
        segments = str(path.stem).split("_")
        for s in segments:
            if s.lower() in data:
                return s.lower()

        # If nothing was found, raise an exception
        raise ValueError(
            f"No valid WRF ID was found in the filename {path.stem} that matches an entry in the calibration file {self._calib_file}"
        )

    @classmethod
    def from_cpsa(
        cls,
        path: Path,
        etch_time: float | None = None,
        wrf: str | tuple[float] | None = None,
    ):
        """
        Initialize a WedgeRangeFilter (WRF) object from a MIT CPSA file.

        The etch_time can be automatically extracted from the filename
        if it is included in a format like  ``_#m_``, ``_#min_``, ``_#h_``,
        ``_#hr_``, etc.

        The thickness profile is uniquely calibrated for each WRF. In cr39py, this
        calibration takes the form of fit coefficients to the equation

        .. math::
            \\text{thickness} = m*x + b

        Where :math:`x` is the horizontal position of the CR-39 scan, and :math:`(m,b)` are the
        slope and offset calibration coefficients (with units of um/cm and um, respectively).
        WRFs imprint fiducials (via holes in the filter) onto the CR-39 which are used to align the
        piece precisely prior to scanning, so the x-axis in the scan should always be identical
        to :math:`x` in the fit.


        Parameters
        ---------
        path : `~pathlib.Path`
            Path to the CPSA file.

        etch_time : float
            Etch time in minutes.

        wrf : str | tuple[float] | None
            The wedge range filter used. Valid options include
            - A string WRF ID code, e.g. "g034".
            - A tuple of (slope, offset) defining the WRF profile (see description).
            If no value is supplied, the filename will be searched for a valid WRF ID code.

        """
        obj = super().from_cpsa(path, etch_time=etch_time)

        if wrf is None:
            # Try to find WRF ID from filename
            wrf = obj._get_wrf_id_from_filename(path)

        if isinstance(wrf, str):
            # Try to find calibration data for the provided ID
            m, b = obj._get_wrf_calib_from_file(wrf)
        else:
            # Otherwise try to get calibration constants from the keyword itself
            try:
                m, b = wrf
            except IndexError:
                raise ValueError(f"Invalid value for wrf keyword: {wrf}")

        obj._m = m
        obj._b = b

        return obj

    @property
    def xaxis(self) -> u.Quantity:
        """
        X-axis for the WRF scan.
        """
        return self._axes["X"].axis

    @property
    def wrf_thickness(self):
        return (self._m * self.xaxis.m_as(u.cm) + self._b) * u.um

    # TODO: this really only makes sense if you don't need to visualize the
    # interim steps...

    # TODO: Framesizes also need to be set based on fluences...
    def set_limits(
        self,
        trange: tuple[float] = (400, 1800),
        xrange: tuple[float | None] | None = None,
        yrange: tuple[float | None] | None = None,
        drange: tuple[float | None] | None = (10, 20),
        crange: tuple[float | None] = (0, 10),
        erange: tuple[float | None] | None = (0, 15),
        plot=False,
    ) -> None:
        """
        Set limits on the tracks that will be included in the analysis.

        Parameters
        ----------
        trange : tuple[float], optional
            Range of x values to include, specified as a range in
            the thickness of the filter wedge. Defaults to (100, 1800)

        xrange : tuple[float], optional
            Range of x values to include. Defaults to None, in which
            case the ``trange`` will be used. This keyword overrides
            ``trange``.

        yrange : tuple[float], optional
            Range of y values to include. The default is to include all
            y-values.

        drange : tuple[float], optional
            Range of diameters to include. The default range is (10,20).

        crange : tuple[float], optional
            Range of contrasts to include. The default range is (0,10).

        erange : tuple[float], optional
            Range of eccentricities to include. The default range is (0,15).

        plot : bool
            If true, make summary plots as cuts are applied

        """
        # Clear the current cuts and domain prior to setting new bounds
        self.current_subset.clear_domain()
        self.current_subset.clear_cuts()

        if xrange is not None:
            xmin, xmax = xrange
        elif trange is not None:
            xrange = (np.array(trange) - self._b) / self._m
            xmin, xmax = xrange

        if yrange is not None:
            ymin, ymax = yrange
        else:
            ymin, ymax = None, None

        if crange is not None:
            cmin, cmax = crange
        else:
            cmin, cmax = None, None

        if drange is not None:
            dmin, dmax = drange
        else:
            dmin, dmax = None, None

        if erange is not None:
            emin, emax = erange
        else:
            emin, emax = None, None

        self.current_subset.set_domain(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            cmin=cmin,
            cmax=cmax,
            dmin=dmin,
            dmax=dmax,
            emin=emin,
            emax=emax,
        )

        """
        # First set the contrast range to eliminate noise

        if cmax is not None:
            self.current_subset.add_cut(cmin=cmax)
        if cmin is not None:
            self.current_subset.add_cut(cmax=cmin)
        self._crange = (
            np.min(self._tracks[:, 3]) if cmin is None else cmin,
            np.max(self._tracks[:, 3]) if cmax is None else cmax,
        )

        # Set the x range

        # Save the bounds for plotting
        self._xrange = (
            np.min(self._tracks[:, 0]) if xmin is None else xmin,
            np.max(self._tracks[:, 0]) if xmax is None else xmax,
        )


        self._yrange = (
            np.min(self._tracks[:, 1]) if ymin is None else ymin,
            np.max(self._tracks[:, 1]) if ymax is None else ymax,
        )



        # Now create and add the cuts
        # Since cuts EXCLUDE tracks, two cuts are required (to exclude tracks above and below the range)
        # and the min/max of the range are the max/min of those respective cuts.


        if dmax is not None:
            self.current_subset.add_cut(dmin=dmax)
        if dmin is not None:
            self.current_subset.add_cut(dmax=dmin)
        self._drange = (
            np.min(self._tracks[:, 2]) if dmin is None else dmin,
            np.max(self._tracks[:, 2]) if dmax is None else dmax,
        )


        if emax is not None:
            self.current_subset.add_cut(emin=emax)
        if emin is not None:
            self.current_subset.add_cut(emax=emin)
        self._erange = (
            np.min(self._tracks[:, 4]) if emin is None else emin,
            np.max(self._tracks[:, 4]) if emax is None else emax,
        )
        """

    def fit(
        self,
        guess: tuple[float] = (15, 0.1, 1, 20),
        bounds: list[tuple[float]] = [(12, 17), (0.05, 2), (0.4, 1.6), (14, 24)],
        plot=True,
    ) -> np.ndarray:
        """
        Fit the selected WRF data with the synthetic WRF model.

        Fitting is done with a differential evolution algorithm.

        The model fits the data with a four parameter model:
        - Mean energy of Gaussian particle energy distribution.
        - Standard deviation of the Gaussian particle energy distribution.
        - C-parameter for the C-parameter response model.
        - dmax parameter for the C-parameter response model.


        Parameters
        ----------
        guess : tuple[float], optional
            Initial guess for the four fit parameters. The default is (15, 0.1, 1, 20)
        bounds : list[tuple[float]], optional
            (min,max) bounds for the four fit parameters. The default is [(12,17), (0.05, 2), (0.4,1.6), (14,24)]
        plot : bool, optional
            If True, plot a comparison between the data and best fit at the end. Default is True.

        Returns
        -------
        best_fit : np.ndarray
            Best fit results for each parameter
        """

        remove_ranging_interp = _remove_ranging_interpolator()
        xax, dax, data = self.histogram(axes=("X", "D"))

        print(dax)

        def minimization_fcn(params):
            synthetic = synthetic_wrf_data(
                params,
                xax.m,
                dax.m,
                (self._m, self._b),
                remove_ranging_interpolator=remove_ranging_interp,
            )
            return wrf_objective_function(synthetic, data, return_sum=True)

        res = differential_evolution(minimization_fcn, bounds, x0=guess)

        if plot:
            synthetic_data = synthetic_wrf_data(
                res.x,
                xax.m,
                dax.m,
                (self._m, self._b),
                remove_ranging_interpolator=remove_ranging_interp,
            )

            emean, estd, c, dmax = res.x
            fig, ax = plt.subplots()
            ax.set_title(
                f"Emean={emean:.2f} MeV, Estd={estd:.2f} MeV, c={c:.2f}, dmax={dmax:.2f} um"
            )
            ax.pcolormesh(xax.m, dax.m, data.T, cmap="binary_r")
            ax.contour(xax.m, dax.m, synthetic_data.T, [0.1, 0.3, 0.5, 0.7, 0.9])
        return res.x
