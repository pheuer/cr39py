import importlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from cr39py.core.exportable_class import ExportableClassMixin, saveable_class
from cr39py.core.units import u


@saveable_class()
class Material(ExportableClassMixin):
    _matfile = importlib.resources.files("cr39py.core.resources") / Path("materials.h5")

    # Save no attributes - see overwritten
    _exportable_attributes = ["name"]

    @classmethod
    def from_string(cls, name):
        obj = cls()

        with h5py.File(obj._matfile, "r") as f:
            # Read in aliases: translate str->dict with json.loads
            # Need to replace single quotes with double for json.loads
            aliases = json.loads(f.attrs["aliases"].replace("'", '"'))

            materials = f.attrs["materials"]

        # By convention, use lower-case for material names
        name = name.lower()

        if name in aliases.keys():
            name = aliases[name]

        if name not in materials:
            raise ValueError(f"No material {name}")

        obj.name = name

        return obj

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name.title()

    @property
    def density(self):
        """
        Material mass density
        """
        with h5py.File(self._matfile, "r") as f:
            grp = f[self.name]
            if "density" not in grp:
                raise ValueError("No density data for {self.name}")
            density = grp["density"][...] * u(grp["density"].attrs["unit"])
        return density

    def xray_mass_attenuation_coefficient(
        self, energies=None, return_interpolator=False
    ):
        """
        X-ray mass attenuation coefficient from the NIST database.

        https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients

        Parameters
        ----------
        energies : u.Quantity
            Energies at which to interpolate the value.

        Raises
        ------
        ValueError
            If data is not available or if energies are outside of the
            range of the data.

        Returns
        -------
        X-ray mass attenuation coefficients @ energies : u.Quantity


        """

        with h5py.File(self._matfile, "r") as f:
            grp = f[f"{self.name}"]
            if "xray_mass_attenuation_coefficients" not in grp:
                raise ValueError(
                    f"No x-ray mass attenuation coefficient data for {self.name}"
                )

            grp = grp["xray_mass_attenuation_coefficients"]
            eaxis = grp["energy"][:] * u(grp["energy"].attrs["unit"])
            mu_rho = grp["mu_rho"][:] * u(grp["mu_rho"].attrs["unit"])

        if np.min(energies) <= np.min(eaxis) or np.max(energies) >= np.max(eaxis):
            raise ValueError(
                "Provided energies outside the data range: "
                f"{np.min(eaxis)}-{np.max(eaxis)}"
            )

        # Interpolate data to the user-provided energy values.
        # Uses log-log scale fed into a cubic spline.
        cs = CubicSpline(
            x=np.log(eaxis.m_as(u.eV)), y=np.log(mu_rho.m), extrapolate=False
        )
        interp_fcn = lambda e: np.exp(cs(np.log(e.m_as(u.eV)))) * mu_rho.u

        if energies is not None:
            return interp_fcn(energies)
        elif return_interpolator:
            return interp_fcn
        else:
            raise ValueError("Set either `energies` or `return_interpolator` kwargs")

    def _ion_stopping_power(self, incident_particle):
        """
        Reads ion stopping power data

        """

        incident_particle = incident_particle.lower()

        if incident_particle not in ["p", "d", "t"]:
            raise ValueError(f"Invalid particle: {incident_particle}")

        with h5py.File(self._matfile, "r") as f:

            grp = f[f"{self.name}"]
            try:
                grp = grp["ion_stopping_power"]

            except KeyError as err:
                raise ValueError(f"No ion stopping power data for {self.name}") from err

            try:
                grp = grp[incident_particle]

            except KeyError as err:
                raise ValueError(
                    f"No ion stopping power data for {self.name} "
                    f"for {incident_particle}"
                ) from err

            return grp.name

    def ion_stopping_power(
        self, incident_particle, energies=None, return_interpolator=False
    ):

        name = self._ion_stopping_power(incident_particle)

        with h5py.File(self._matfile, "r") as f:
            grp = f[name]
            eaxis = grp["energy"][:] * u(grp["energy"].attrs["unit"])
            dEdx_total = grp["dEdx_total"][:] * u(grp["dEdx_total"].attrs["unit"])

        # Interpolate data to the user-provided energy values.
        # Uses log-log scale fed into a cubic spline.
        cs = CubicSpline(x=np.log(eaxis.m_as(u.MeV)), y=np.log(dEdx_total.m))
        interp_fcn = lambda e: np.exp(cs(np.log(e.m_as(u.MeV)))) * dEdx_total.u

        if energies is not None:
            return interp_fcn(energies)
        elif return_interpolator:
            return interp_fcn
        else:
            raise ValueError("Set either `energies` or `return_interpolator` kwargs")

    def ion_projected_range(
        self, incident_particle, energies=None, return_interpolator=False
    ):

        name = self._ion_stopping_power(incident_particle)

        with h5py.File(self._matfile, "r") as f:
            grp = f[name]
            eaxis = grp["energy"][:] * u(grp["energy"].attrs["unit"])
            proj_rng = grp["projected_range"][:] * u(
                grp["projected_range"].attrs["unit"]
            )

        # Interpolate data to the user-provided energy values.
        # Uses log-log scale fed into a cubic spline.
        cs = CubicSpline(x=np.log(eaxis.m_as(u.MeV)), y=np.log(proj_rng.m))
        interp_fcn = lambda e: np.exp(cs(np.log(e.m_as(u.MeV)))) * proj_rng.u

        if energies is not None:
            return interp_fcn(energies)
        elif return_interpolator:
            return interp_fcn
        else:
            raise ValueError("Set either `energies` or `return_interpolator` kwargs")

    def optimal_ion_ranging_thickness(
        self, incident_particle, E_start, E_end, dx=1 * u.um
    ):
        """
        Find the optimal thickness for a filter of this material to range the particle
        down from E_start to E_end

        """

        sp_fcn = self.ion_stopping_power(incident_particle, return_interpolator=True)
        Ep = E_start
        thickness = 0 * dx.u
        while True:
            if Ep <= E_end:
                return thickness

            thickness += dx
            Ep -= sp_fcn(Ep) * dx
