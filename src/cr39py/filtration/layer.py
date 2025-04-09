"""
The `~cr39py.filtration.layer` module contains the `~cr39py.filtration.layer.Layer` class, which
is used to represent a piece of filtration material.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from cr39py.core.exportable_class import ExportableClassMixin, saveable_class
from cr39py.core.units import unit_registry as u
from cr39py.filtration._layerlike import LayerLike
from cr39py.filtration.srim import SRIMData


@saveable_class()
class Layer(ExportableClassMixin, LayerLike):
    r"""
    A layer in a detector stack stack. The layer could either be an active
    layer (a piece of film or other recording media)
    or an inactive layer (a filter or inactive part of the film, such as
    a substrate )

    References
    ----------
    The ion stopping and ranging calculations performed by this clase use
    data from the `SRIM <http://www.srim.org/>`__ code :cite:p:`SRIM`.
    If you use these features, please cite SRIM.

    """

    _exportable_attributes = ["thickness", "material", "active", "name"]

    def __init__(self):
        """A layer of a filtration stack."""
        # Cache of SRIM data tables: keys are particle names (lowercase)
        # and values are SRIMData objects for each key
        self._srim_data = {}

    @classmethod
    def from_properties(
        cls,
        thickness: u.Quantity,
        material: str,
        active: bool = True,
        name: str = "",
    ):
        r"""
        Creates a layer with explicitly provided parameters.

        Parameters
        ----------

        thickness : pint
            The thickness of the layer, in units convertible to meters.

        material : `Material`
            Material of the layer: should correspond to the name of the materials
            in filenames of the stopping power data in the data/srim directory.

        active : `bool`, optional
            If `True`, this layer is marked as an active layer. The default is `True`.

        name : `str`, optional
            An optional name for the layer.

        """
        obj = cls()
        obj.thickness = thickness
        obj.material = material
        obj.name = name
        obj.active = active
        return obj

    @classmethod
    def from_string(cls, s):
        """
        Create a layer from a string of the following form

        [Thickness] [unit string] [material string]
        """

        # Split the string by whitespace
        s = s.split()
        if len(s) != 3:
            raise ValueError("Invalid string code for Material")

        unit = u(s[1])
        thickness = float(s[0]) * unit
        material = s[2]

        # TODO: support active/inactive and name here as optional
        # additional entries

        return cls.from_properties(thickness, material)

    def __eq__(self, other):

        return (
            self.thickness == other.thickness
            and self.material.lower() == other.material.lower()
            and self.name == other.name
            and self.active == other.active
        )

    def __str__(self):
        return f"{self.thickness.m_as(u.um):.1f} um {self.material}"

    def srim_data(self, particle: str):
        """`~cr39py.filtration.srim.SRIMData` object for this layer and given particle.

        Parameters
        ----------

        particle: str
            One of the valid particle names, e.g. "proton", "deuteron", "alpha", etc.
        """
        key = particle.lower()

        if key not in self._srim_data:
            self._srim_data[key] = SRIMData.from_strings(particle, self.material)
        return self._srim_data[key]

    def _range_ion(
        self,
        particle: str,
        E: u.Quantity,
        dx: u.Quantity = 1 * u.um,
        reverse: bool = False,
    ) -> u.Quantity:
        """
        Calculate the energy a particle will be ranged down to through the layer.

        Used in the ``range_down`` and ``reverse_ranging`` methods below.

        Parameters
        ----------
        particles : str
            Incident particle

        E : u.Quantity
            If ``reverse`` is ``False``, energy of the particle before ranging in the layer.
            If ``reverse`` is ``True``, energy of the particle after ranging in the layer.

        dx : u.Quantity, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.

        reverse : bool
            If True, reverse the process to find the starting energy of
            a particle given the final energy. Used in `reverse_ion_ranging`

        Returns
        -------

        E : u.Quantity
            If ``reverse`` is ``False``, energy of the particle after ranging in the layer.
            If ``reverse`` is ``True``, energy of the particle before ranging in the layer.

        """

        # Strip units for faster computation - return results in eV
        E = np.atleast_1d(E.m_as(u.eV))

        # Get a cubic splines interpolator for the stopping power
        # in this layer
        sp_fcn = self.srim_data(particle).dEdx_total_interpolator

        # Slice the layer into sublayer dx thick
        nsublayers = int(np.floor(self.thickness.m_as(u.um) / dx.m_as(u.um)))
        sublayers = np.ones(nsublayers) * dx.m_as(u.um)
        # Include any remainder in the last sublayer
        sublayers[-1] += self.thickness.m_as(u.um) % dx.m_as(u.um)

        # Calculate the energy deposited in each sublayer
        # This is essentially numerically integrating the stopping power
        for ds in sublayers:
            # Interpolate the stopping power at the current energy
            interpolated_stopping_power = sp_fcn(
                E
            )  # Interpolated stopping power is in kV/um

            if reverse:
                interpolated_stopping_power *= -1

            dE = interpolated_stopping_power * ds * 1e3  # dE in keV -> eV

            # TODO: Find a better way of detecting if dx is too large, or automatically determining
            # an appropriate dx

            E -= dE

            # If energy is at or below zero, return 0.
            # The particle has stopped.
            E[E < 0] = 0
            if np.all(E == 0):
                return E * u.eV

        return E * u.eV

    def projected_range(self, particle: str, E_in: u.Quantity) -> u.Quantity:
        """
        Calculate the projected range of a particle in the layer.

        See the `~cr39py.filtration.srim.SRIMData` class for formal definition of this quantity.

        Parameters
        ----------
        particle : str
            Incident particle

        E_in : u.Quantity
            Energy of the particle before ranging in the layer.

        Returns
        -------
        R : u.Quantity
            Projected range of the particle in the layer.
        """
        prjrng_interp = self.srim_data(particle).projected_range_interpolator
        return prjrng_interp(E_in.m_as(u.eV)) * u.m

    def lateral_straggle(self, particle: str, E_in: u.Quantity) -> u.Quantity:
        """
        Calculate the lateral straggle of a particle in the layer.

        If the particle passes through the layer, this is the straggle experienced by the particle in the layer.
        If the particle stops in the layer, this is the total straggle up to the stopping point.

        See the `~cr39py.filtration.srim.SRIMData` class for formal definition of this quantity.

        Parameters
        ----------
        particle : str
            Incident particle

        E_in : u.Quantity
            Energy of the particle before ranging in the layer.

        Returns
        -------
        straggle : u.Quantity
            Lateral straggle of the particle in the stack.
        """
        E_in = np.atleast_1d(E_in)

        straggle_interp = self.srim_data(particle).lateral_straggle_interpolator

        E_out = self.range_down(particle, E_in)

        stopped_i = E_out.m <= 0
        straggle = np.empty(E_out.shape) * u.um

        # The SRIM files include the full straggle assuming the particle stops
        # So, if the particle stops, we can just use the full straggle
        full_straggle = straggle_interp(E_in.m_as(u.eV)) * u.m
        straggle[stopped_i] = full_straggle[stopped_i]

        # If the particle doesn't stop, we need to subtract the straggle
        # the SRIM file gives for the remaining energy
        remaining_straggle = straggle_interp(E_out[~stopped_i].m_as(u.eV)) * u.m
        straggle[~stopped_i] = full_straggle[~stopped_i] - remaining_straggle

        return straggle

    def projected_depth_for_energy(
        self, particle: str, E_in: u.Quantity, E_at: u.Quantity
    ) -> u.Quantity:
        """
        Calculate the depth in the layer where a particle will have a given energy.

        Parameters
        ----------

        particle : str
            Incident particle

        E_in : u.Quantity
            Energy of the particle before ranging in the layer.

        E_at : u.Quantity
            Energy of the particle at the depth to be calculated.

        Returns
        -------

        depth : u.Quantity
            Depth of the particle into the layer when the desired energy
            is achieved.

        """
        prjrng_interp = self.srim_data(particle).projected_range_interpolator
        total_range = prjrng_interp(E_in.m_as(u.eV))
        range_at_desired_energy = prjrng_interp(E_at.m_as(u.eV))
        return (total_range - range_at_desired_energy) * u.m

    def range_down(
        self,
        particle: str,
        E_in: u.Quantity,
        dx: u.Quantity = 1 * u.um,
    ) -> u.Quantity:
        """
        Calculate the energy a particle will be ranged down to through the layer.

        Parameters
        ----------
        particles : str
            Incident particle

        E_in : u.Quantity
            Energy of the particle before ranging in the layer.

        dx : u.Quantity, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.

        Returns
        -------

        E_out : u.Quantity
            Energy of the particle after ranging in the layer. If zero, the
            particle stopped in the stack.

        """
        return self._range_ion(particle, E_in, dx=dx, reverse=False)

    def reverse_ranging(
        self,
        particle: str,
        E_out: u.Quantity,
        dx: u.Quantity = 1 * u.um,
    ) -> u.Quantity:
        """
        Calculate the energy a particle would have had before ranging in
        the layer.

        Parameters
        ----------
        particles : str
            Incident particle

        E_out : u.Quantity
            Energy of the particle after exiting the layer.

        dx : u.Quantity, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.


        Returns
        -------

        E_in: u.Quantity
            Energy of the particle before ranging in the layer.

        """
        if E_out.m <= 0:
            raise ValueError("Cannot reverse ranging if particle stopped in the layer.")

        return self._range_ion(particle, E_out, dx=dx, reverse=True)

    def ranging_energy_loss(self, particle: str, E_in: u.Quantity) -> u.Quantity:
        """
        Calculate the energy a particle will lose in the layer.

        Parameters
        ----------

        particles : str
            Incident particle

        E_in : u.Quantity
            Energy of the particle before the layer.

        Returns
        -------

        E_in_stack : u.Quantity
            Energy the particle leaves in the layer.

        """
        return E_in - self.range_down(particle, E_in)
