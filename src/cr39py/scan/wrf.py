from pathlib import Path

import h5py
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution

from cr39py.core.data import data_dir
from cr39py.core.units import u
from cr39py.scan.base_scan import Scan


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

    def _remove_ranging_interpolator(self):
        """
        Returns an interpolator for incident proton energy as a function of
        filter thickness and output energy.

        Currently the only file included is for protons in aluminum - could generate
        more for other particles if needed?

        Returns
        -------

        interp : RegularGridInterpolator
            Interpolator that takes (thickness(um), E_out(MeV)) and returns E_in(MeV)

        """

        data_file = data_dir / Path("proton_Al_remove_ranging.h5")
        with h5py.File(data_file, "r") as f:
            thickness = f["thickness"][:]  # um
            E_out = f["E_out"][:]  # MeV
            E_in = f["E_in"][:, :]

        E_in_interp = RegularGridInterpolator(
            (thickness, E_out), E_in, bounds_error=False, method="cubic"
        )

        return E_in_interp

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
    def wrf_thickness(self):
        x = self._axes["X"].axis
        print(x)
        return (self._m * x.m_as(u.cm) + self._b) * u.um


if __name__ == "__main__":
    test_file = Path(
        r"C:\Users\pheu\Box\pheuer\Research\Experiments\CR39\2024_PRadMagRecon-25A\O113530_NDI_WA1769_G156_5.25h_s3_40x.cpsa"
    )
    wrf = WedgeRangeFilter.from_cpsa(test_file)
