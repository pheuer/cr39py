"""
The `~cr39py.scan.wrf` module contains functionality for analyzing scans of
CR=39 fielded with a wedge range filter, and for forward-modeling synthetic
WRF data.

A WRF is a wedge of material (usually aluminum) with different thicknesses
along the x-axis of the CR-39. Combined with the diameter-energy response
relationship of the CR-39, the WRF can be used to determine the energy
distribution of incident particles.


References
----------
The following papers are important references for WRFs.

* :cite:t:`Seguin2003spectrometry` is the foundational paper on charged particle spectroscopy with CR-39.
* :cite:t:`Seguin2012advances` presents improvements on the design and describes some of the analysis of WRFs.
* :cite:t:`Zylstra2012charged` describes the implementation of WRFs on the National Ignition Facility.
* :cite:t:`Sio2014technique` presents an specialty technique for analyzing extremely saturated WRF data - this technique is currently not implemented in cr39py.

"""

from functools import cache
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

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
    dmax: float,
    wrf_calibration: np.ndarray,
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

    xaxis : np.ndarray
        X-axis, in cm

    daxis : np.ndarray
        Diameter axis, in um

    dmax : float
        Dmax parameter in um

    wrf_calibration: tuple(float)
        WRF slope and offset calibration coefficients (m,b).

    Returns
    -------
    synthetic_data : np.ndarray
        Synthetic data (arbitrary units) in X,D space.
    """

    remove_ranging_interpolator = _remove_ranging_interpolator()

    emean, estd, c = params

    # Convert the track diameter axis from the scan to track energy
    model = CParameterModel(c, dmax)

    # Calculate energy incident on CR-39 from diameters
    ein_axis = model.track_energy(daxis)

    # Calculate the WRF thickness
    m, b = wrf_calibration
    wrf_thickness = m * xaxis + b

    # Translate that E_in axis to an E_out value at every thickness
    T, E = np.meshgrid(wrf_thickness, ein_axis, indexing="ij")
    eout_axis = remove_ranging_interpolator((T, E))

    # Use those E_in values with the distribution to create a synthetic image
    synthetic_data = np.exp(-((emean - eout_axis) ** 2) / 2 / estd**2)
    # synthetic_data = synthetic_data /np.nanmax(synthetic_data)

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
        _synthetic = _synthetic / np.nansum(_synthetic)

        chi2 = (_data - _synthetic) ** 2 / _data
        return chi2


class WedgeRangeFilter(Scan):
    """
    A scan of a piece of CR-39 fielded behind a Wedge Range Filter (WRF).

    The thickness profile is uniquely calibrated for each WRF. In cr39py, this
    calibration takes the form of fit coefficients to the equation

    .. math::
        \\text{thickness} = m*x + b

    Where :math:`x` is the horizontal position of the CR-39 scan, and :math:`(m,b)` are the
    slope and offset calibration coefficients (with units of um/cm and um, respectively).
    WRFs imprint fiducials (via holes in the filter) onto the CR-39 which are used to align the
    piece precisely prior to scanning, so the x-axis in the scan should always be identical
    to :math:`x` in the fit.

    """

    _calib_file = data_dir / Path("calibration/wrf_calibrations.yml")

    def __init__(self) -> None:
        super().__init__()

        # WRF calibration coefficients
        self._m = None
        self._b = None

        # Dmax parameter for the C-parameter model
        self._dmax = None

        self._background_region = None

    @property
    def wrf_calibration(self) -> tuple[float]:
        """
        WRF calibration coefficients (m,b).
        """
        return self._m, self._b

    @property
    def dmax(self) -> float:
        """
        Dmax parameter for the C-parameter model.
        """
        return self._dmax

    @property
    def background_region(self):
        """
        Background region for the WRF scan in Thickness,Diameter space.

        [[tmin, tmax], [dmin, dmax]]

        Returns
        -------
        background, tuple[tuple[float,float]]
            Background region
        """
        if self._background_region is None:
            return ((0, 800), (0, 20))
        else:
            return self._background_region

    @dmax.setter
    def dmax(self, value: float):
        self._dmax = value

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
        id = id.lower()
        data = self._wrf_calib_data
        if id not in data:
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

        If a WRF ID code matching a calibration saved in the WRF calibration file
        is found in the filename, the calibration will automatically be retrieved.

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
            except ValueError as e:
                raise ValueError(f"Invalid value for wrf keyword: {wrf}") from e

        obj._m = m
        obj._b = b

        return obj

    @property
    def xaxis(self) -> u.Quantity:
        """
        X-axis for the WRF scan.
        """
        return self._axes["X"].axis(tracks=self.selected_tracks)

    @property
    def wrf_thickness(self) -> u.Quantity:
        """
        Thickness of the WRF as a function of the x-axis.
        """
        return (self._m * self.xaxis.m_as(u.cm) + self._b) * u.um

    def plot_diameter_histogram(self, dlim=None):
        """
        Plot a histogram of the track diameters.

        Useful for identifying dmax.
        """
        bins = self._axes["D"].axis(tracks=self.selected_tracks).m_as(u.um)
        fig, ax = plt.subplots()
        ax.hist(self.selected_tracks[:, 2], bins=bins)
        if dlim is not None:
            ax.set_xlim(*dlim)
        ax.set_yscale("log")
        return fig, ax

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

        These limits are implemented on the domain of the subset.

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

    def fit(
        self,
        guess: tuple[float] = (15, 0.1, 1),
        bounds: list[tuple[float]] = [(12, 17), (0.05, 2), (0.4, 1.6)],
        plot=True,
    ) -> np.ndarray:
        """
        Fit the selected WRF data with the synthetic WRF model.

        Fitting is done with a nonlinear least squares algorithm.

        The model fits the data with a three parameter model:
        - Mean energy of Gaussian particle energy distribution.
        - Standard deviation of the Gaussian particle energy distribution.
        - C-parameter for the C-parameter response model.

        The `self.dmax` parameter is used as a fixed parameter in the
        C-paramter model in the fit.

        Background subtraction is automatically performed, using the
        values stored in the `self.background_region` attribute.

        Parameters
        ----------
        guess : tuple[float], optional
            Initial guess for the fit parameters.
        bounds : list[tuple[float]], optional
            (min,max) bounds for the fit parameters.
        plot : bool, optional
            If True, plot a comparison between the data and best fit at the end. Default is True.

        Returns
        -------
        best_fit : np.ndarray
            Best fit results for each parameter


        References
        ----------
        When using this method to fit WRF data, please cite :cite:t:`Seguin2012advances`

        """

        # FInd the background and remove from the data
        xax, dax, data = self.histogram(axes="XD")

        thickness = self.wrf_thickness.m
        ta = np.argmin(np.abs(thickness - self.background_region[0][0]))
        tb = np.argmin(np.abs(thickness - self.background_region[0][1]))
        da = np.argmin(np.abs(dax.m_as(u.um) - self.background_region[1][0]))
        db = np.argmin(np.abs(dax.m_as(u.um) - self.background_region[1][1]))
        bkg = np.nanmean(data[ta:tb, da:db].m)
        data = data - bkg

        def minimization_fcn(params):
            synthetic = synthetic_wrf_data(
                params,
                xax.m,
                dax.m,
                self.dmax,
                self.wrf_calibration,
            )
            return wrf_objective_function(synthetic, data, return_sum=True)

        # Least squares takes bounds in a weird format: this re-organizes the list
        _bounds = ([x[0] for x in bounds], [x[1] for x in bounds])

        res = least_squares(minimization_fcn, guess, bounds=_bounds)
        # res = differential_evolution(minimization_fcn, bounds, x0=guess)

        if plot:
            synthetic_data = synthetic_wrf_data(
                res.x,
                xax.m,
                dax.m,
                self.dmax,
                self.wrf_calibration,
            )

            emean, estd, c = res.x
            fig, ax = plt.subplots()
            ax.set_title(
                f"Emean={emean:.2f} MeV, Estd={estd:.2f} MeV, c={c:.2f}, bkg={bkg:.2e}, dmax={self.dmax:.2f} um"
            )
            ax.pcolormesh(xax.m, dax.m, data.T, cmap="binary_r")
            ax.contour(xax.m, dax.m, synthetic_data.T, 5)
        return res
