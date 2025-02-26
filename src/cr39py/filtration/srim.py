"""
The `~cr39py.filtration.srim` module contains functionality for reading particle stopping and scattering datafiles
generated by the `SRIM <http://www.srim.org/>`__ code :cite:p:`SRIM`.
"""

import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from cr39py.core.data import data_dir


class SRIMData:
    """
    Represents an output file from the `SRIM <http://www.srim.org/>`__ code.


    Definitions
    -----------
    In SRIM, particles have initial velocities parallel to the x-axis, and
    the y and z axes are parallel to the target surface. If the total number
    of particles is denoted :math:`N` and :math:`(x_i,y_i,z_i)` is the final position of the deposited ion, then


    .. math::
        \\text{Projected (logitudinal) Range} = R_p = \sum_i x_i / N = \langle x \\rangle

    .. math::
        \\text{Radial Range} = R_r = \sum_i \sqrt{y_i^2 + z_i^2} / N

    .. math::
        \\text{Longitudinal Straggle}  = \sigma = \sqrt{\sum_i x_i^2/N - R_p^2} = \sqrt{ \langle (\Delta x_i)^2 \\rangle }

    .. math::
        \\text{Radial Straggle}  = \sigma_r = \sqrt{\sum_i (y_i^2 + z_i^2)/N - R_r^2} = \sqrt{ \langle (\Delta r_i)^2 \\rangle }

    .. math::
        \\text{Lateral Straggle}  = \sigma_y = \\bigg ( \sum_i (  [(|y_i| + |z_i|)/2]^2 /N \\bigg )^{1/2}

    These definitions are taken from the SRIM/TRIM user manual.

    References
    ----------

    When using this code, please cite :cite:t:`SRIM`.

    """

    @classmethod
    def from_file(cls, file: Path | str) -> None:
        """
        Loads a SRIM output file and stores the data as attributes.

        The SRIM file must have stopping power saved in units of keV/um.

        SRIM data files must be placed in the directory
        ``cr39py/data/srim``.

        The filename must follow the format ``[particle] in [target material].txt``

        Parameters
        ----------
        file : str
            Filepath to a SRIM output file
        """

        obj = cls()

        file = Path(file)
        basename = file.stem
        particle, material = basename.split(" in ")
        obj._particle = particle.lower()
        obj._material = material.lower()

        _srim_data_dir = data_dir / Path(f"srim")
        print(file)
        if not file.exists():
            raise FileNotFoundError(f"File {file.stem} not found in {_srim_data_dir}.")

        obj._read_srim(file)

        return obj

    @classmethod
    def from_strings(
        cls,
        particle: str,
        material: str,
    ) -> None:
        """Loads SRIM data from strings.

        Parameters
        ----------
        particle : str
            The type of particle in the SRIM simulation. Use the full particle name, e.g.
            "Deuteron", "Proton", "Alpha", etc.
        material : str
            The target material in the SRIM simulation.
        """

        _srim_data_dir = data_dir / Path(f"srim")
        files = []
        for path in _srim_data_dir.iterdir():
            match = re.search(
                rf".*(?=.*\b{particle}\b)(?=.*\b{material}\b).*",
                str(path),
                re.IGNORECASE,
            )
            if match is not None:
                files.append(path)

        if len(files) > 1:
            raise FileNotFoundError(
                f"Multiple SRIM files found matching {particle} and {material}."
            )
        elif len(files) == 0:
            raise FileNotFoundError(
                f"No SRIM files found matching {particle} and {material}."
            )
        else:
            file = files[0]

        obj = cls()

        obj._particle = particle.lower()
        obj._material = material.lower()
        obj._read_srim(file)

        return obj

    @property
    def particle(self) -> str:
        """The type of particle in the SRIM simulation."""
        return self._particle

    @property
    def material(self) -> str:
        """The target material in the SRIM simulation."""
        return self._material

    def _read_srim(self, file: str | Path) -> None:
        """
        Reads a SRIM output file and stores the values as attributes
        of the class.

        Parameters
        ----------
        file : str
            Filepath to a SRIM output file
        """

        ion_energy = []
        energy_convert = {"eV": 1, "keV": 1e3, "MeV": 1e6, "GeV": 1e9}
        range_convert = {"A": 1e-10, "nm": 1e-9, "um": 1e-6, "mm": 1e-3, "cm": 1e-2}
        dEdx_electronic = []
        dEdx_nuclear = []
        projected_range = []
        longitudinal_straggle = []
        lateral_straggle = []

        with open(file, "r") as f:
            # Read in the file contents
            c = f.readlines()

        # Process/skip through the header
        # Number of lines is unknown, so follow the procedure
        # 1) Find the column header row by looking for a known string
        # 2) Skip another 3 rows from there
        while True:
            s = c[0]
            # Grab the stopping power unit from the header
            if "Stopping Units" in s:
                if "keV / micron" not in s:
                    raise ValueError("SRIM files must use SP units keV/um")

            # Recongnize the end of the header
            if "dE/dx" in s:
                break
            else:
                c = c[1:]
        # Skip the remaining lines
        c = c[3:]

        while True:
            s = c[0].split(" ")
            # Discard any single space strings
            s = [st for st in s if st != ""]
            # Strip white space
            s = [st.strip() for st in s]

            # The first column is the ion energy, along with a unit
            energy, unit = s[0], s[1]
            if unit not in energy_convert.keys():
                raise ValueError(
                    f"Unrecognized energy unit: {unit}"
                )  # pragma: no cover
            ion_energy.append(float(energy) * energy_convert[unit])

            # Read the dEdx electronic and nuclear
            # These get read in as a pair, along with another awkward empty string
            # because of the spacing...
            electronic, nuclear = s[2], s[3]
            dEdx_electronic.append(float(electronic))
            dEdx_nuclear.append(float(nuclear))

            # Read the projected range
            rng, unit = s[4], s[5]
            if unit not in range_convert.keys():
                raise ValueError(
                    f"Unrecognized range unit: `{unit}`"
                )  # pragma: no cover
            projected_range.append(float(rng) * range_convert[unit])

            rng, unit = s[6], s[7]
            if unit not in range_convert.keys():
                raise ValueError(
                    f"Unrecognized range unit: `{unit}`"
                )  # pragma: no cover
            longitudinal_straggle.append(float(rng) * range_convert[unit])

            rng, unit = s[8], s[9]
            if unit not in range_convert.keys():
                raise ValueError(
                    f"Unrecognized range unit: `{unit}`"
                )  # pragma: no cover
            lateral_straggle.append(float(rng) * range_convert[unit])

            # If the next line contains the dotted line at the end of the file,
            # terminate the loop
            if "--" in c[1]:
                break
            # Else remove the line we just read and start again
            else:
                c = c[1:]

        self._ion_energy = np.array(ion_energy)
        self._dEdx_electronic = np.array(dEdx_electronic)
        self._dEdx_nuclear = np.array(dEdx_nuclear)
        self._dEdx_total = self.dEdx_electronic + self.dEdx_nuclear
        self._projected_range = np.array(projected_range)
        self._longitudinal_straggle = np.array(longitudinal_straggle)
        self._lateral_straggle = np.array(lateral_straggle)

    @property
    def ion_energy(self) -> np.ndarray:
        """
        Ion energy axis for other data.
        """
        return self._ion_energy

    @property
    def dEdx_electronic(self) -> np.ndarray:
        """
        Electronic stopping power in keV/um.
        """
        return self._dEdx_electronic

    @property
    def dEdx_nuclear(self) -> np.ndarray:
        """
        Nuclear stopping power in keV/um.
        """
        return self._dEdx_nuclear

    @property
    def dEdx_total(self) -> np.ndarray:
        """
        Total stopping power in keV/um.
        """
        return self._dEdx_total

    @property
    def dEdx_total_interpolator(self) -> callable:
        """A cubic spline interpolator for the total stopping power data.

        Returns
        -------
        interp_fcn : callable
            Takes an ion energy value in eV and returns the total
            stopping power, dEdx_total, at that energy in keV/um.
        """

        # Uses log-log scale fed into a cubic spline.
        cs = CubicSpline(x=np.log(self.ion_energy), y=np.log(self.dEdx_total))
        interp_fcn = lambda e: np.exp(cs(np.log(e)))

        return interp_fcn

    @property
    def projected_range(self) -> np.ndarray:
        """
        Projected range in m.
        """
        return self._projected_range

    @property
    def projected_range_interpolator(self) -> callable:
        """A cubic spline interpolator for the projected range data.

        Returns
        -------

        interp_fcn : callable
            Takes an ion energy value in eV and returns the projected
            range at that energy in meters.
        """

        interp_fcn = CubicSpline(x=self.ion_energy, y=self.projected_range)
        return interp_fcn

    @property
    def longitudinal_straggle(self) -> np.ndarray:
        """
        Longitudinal straggle in m.
        """
        return self._longitudinal_straggle

    @property
    def longitudinal_straggle_interpolator(self) -> callable:
        """A cubic spline interpolator for the longitudinal (parallel to the incident particle velocity)
        straggle data.

        Returns
        -------

        interp_fcn : callable
            Takes an ion energy value in eV and returns the expected longitudinal stragle at that energy in meters.
        """
        interp_fcn = CubicSpline(x=self.ion_energy, y=self.longitudinal_straggle)
        return interp_fcn

    @property
    def lateral_straggle(self) -> np.ndarray:
        """
        Lateral straggle in m.
        """
        return self._lateral_straggle

    @property
    def lateral_straggle_interpolator(self) -> callable:
        """A cubic spline interpolator for the lateral (perpendicular to the incident particle velocity)
        straggle data.

        Returns
        -------

        interp_fcn : callable
            Takes an ion energy value in eV and returns the expected lateral stragle at that energy in meters.
        """
        interp_fcn = CubicSpline(x=self.ion_energy, y=self.lateral_straggle)
        return interp_fcn
