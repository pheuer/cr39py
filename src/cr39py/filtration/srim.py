import re
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from cr39py.core.data import data_dir


class SRIMData:

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
        longitudinal_straggling = []
        lateral_straggling = []

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
                raise ValueError(f"Unrecognized energy unit: {unit}")
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
                raise ValueError(f"Unrecognized range unit: `{unit}`")
            projected_range.append(float(rng) * range_convert[unit])

            rng, unit = s[6], s[7]
            if unit not in range_convert.keys():
                raise ValueError(f"Unrecognized range unit: `{unit}`")
            longitudinal_straggling.append(float(rng) * range_convert[unit])

            rng, unit = s[8], s[9]
            if unit not in range_convert.keys():
                raise ValueError(f"Unrecognized range unit: `{unit}`")
            lateral_straggling.append(float(rng) * range_convert[unit])

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
        self._longitudinal_straggling = np.array(longitudinal_straggling)
        self._lateral_straggling = np.array(lateral_straggling)

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
    def projected_range(self) -> np.ndarray:
        """
        Projected range in m.
        """
        return self._projected_range

    @property
    def longitudinal_straggling(self) -> np.ndarray:
        """
        Longitudinal straggling in m.
        """
        return self._longitudinal_straggling

    @property
    def lateral_straggling(self) -> np.ndarray:
        """
        Lateral straggling in m.
        """
        return self._lateral_straggling

    @property
    def projected_range_interpolator(self) -> callable:
        """A cubic spline interpolator for the projected range data.

        Returns
        -------

        interp_fcn : callable
            Takes an ion energy value in eV and returns the projected
            range at that energy in meters.
        """

        # Uses log-log scale fed into a cubic spline.
        cs = CubicSpline(x=np.log(self.ion_energy), y=np.log(self.projected_range))
        interp_fcn = lambda e: np.exp(cs(np.log(e)))

        return interp_fcn

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
