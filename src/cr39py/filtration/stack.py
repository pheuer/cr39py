"""
The `~cr39py.filtration.stack` module contains the Layer and Stack classes, which
are used to represent filtration stacks composed of multiple layers of material.

These classes can be used to calculate particle ranging in detector filters.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from cr39py.core.exportable_class import ExportableClassMixin, saveable_class
from cr39py.core.units import unit_registry as u
from cr39py.filtration.srim import SRIMData


@saveable_class()
class Layer(ExportableClassMixin):
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
        A layer in a detector stack stack. The layer could either be an active
        layer (a piece of film or other recording media)
        or an inactive layer (a filter or inactive part of the film, such as
        a substrate )

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
        if len(s) < 3:
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

        # TODO: strip units within this calculation to make it faster?

        # Find the peak of the stopping power curve
        sp_peak = (
            self.srim_data(particle).ion_energy[
                np.argmax(self.srim_data(particle).dEdx_total)
            ]
            * u.eV
        )

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
            interpolated_stopping_power = sp_fcn(E.m_as(u.eV))

            if reverse:
                interpolated_stopping_power *= -1

            dE = interpolated_stopping_power * u.keV / u.um * (ds * u.um)

            # Compute the fractional error in the stopping power across the sublayer
            # Use this to raise an exception if the numerical integration is too coarse
            # Only do this calculation above the Bragg peak, because the SP error
            # is always large when you traverse the Bragg peak
            if E > sp_peak:
                sp_err = (
                    np.abs(sp_fcn((E - dE).m_as(u.eV)) - interpolated_stopping_power)
                    / interpolated_stopping_power
                )
                if sp_err > 0.05:
                    print(sp_err)
                    raise ValueError(
                        "|sp(E-dE)-sp(E)|/sp(E)={sp_err*100:.2f}% exceeds recommended threshold: use a smaller `dx`."
                    )

            E -= dE

            # If energy is at or below zero, return 0.
            # The particle has stopped.
            if E <= 0 * E.u:
                return 0 * E.u
        return E

    def projected_range(self, particle: str, E_in: u.Quantity) -> u.Quantity:
        """
        Calculate the projected range of a particle in the layer.

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


@saveable_class()
class Stack(ExportableClassMixin):
    r"""
    An ordered list of `~cr39py.filtration.stack.Layer` objects.
    """

    _exportable_attributes = ["layers"]

    @classmethod
    def from_layers(cls, *args):
        """Creates a stack from a sequence of Layers.

        Each layer should be provided as a separate argument.
        """

        obj = cls()

        # Replace any strings with Layer objects
        _args = []
        for arg in args:
            _arg = Layer.from_string(arg) if isinstance(arg, str) else arg
            _args.append(_arg)

        obj.layers = list(_args)
        return obj

    @classmethod
    def from_string(cls, s):
        """
        Create a stack from a comma separated list of Layer strings
        """
        s = s.split(",")
        layers = [Layer.from_string(si) for si in s]
        return cls.from_layers(*layers)

    def __str__(self):
        s = "Stack:\n"
        for l in self.layers:
            s += str(l) + "\n"
        return s

    def __eq__(self, other):
        if self.nlayers != other.nlayers:
            return False

        for i in range(self.nlayers):
            if self.layers[i] != other.layers[i]:
                return False
        return True

    @property
    def nlayers(self):
        return len(self.layers)

    @property
    def nactive(self):
        r"""
        The number of layers in the stack marked 'active'
        """
        return len([layer for layer in self.layers if layer.active])

    @property
    def thickness(self):
        r"""
        The total thickness of the stack.
        """
        thickness = np.array([layer.thickness.m_as(u.mm) for layer in self.layers])
        return np.sum(thickness) * u.mm

    def range_down(
        self,
        particle,
        E_in,
        dx=1 * u.um,
    ):
        """
        Calculate the energy a particle will be ranged down to through the stack.

        Parameters
        ----------
        particles : Particle
            Incident particle

        E_in : u.Quantity
            Initial energy of incident particle

        dx : u.Quantity, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.

        Returns
        -------

        E_out : u.Quantity
            Energy of the particle after leaving the stack. If zero, the
            particle stopped in the stack.

        """
        E = E_in

        for l in self.layers:

            E = l.range_down(particle, E, dx=dx)

            if E <= 0 * E.u:
                return 0 * E.u
        return E

    def reverse_ranging(
        self,
        particle,
        E_out,
        dx=1 * u.um,
        max_nsublayers=None,
    ):
        """
        Calculate the energy of a particle before ranging in the stack
        from its energy after the stack.

        Parameters
        ----------
        particle: str
            Incident particle

        E_out : u.Quantity
            Energy of the particle after the stack.

        dx : u.Quantity, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.

        Returns
        -------

        E_in : u.Quantity
            Energy of the particle before entering the stack.

        """
        E = E_out

        for l in self.layers[::-1]:

            E = l.reverse_ranging(particle, E, dx=dx)

        return E

    def ranging_energy_loss(
        self, particle: str, E_in: u.Quantity, dx=1 * u.um
    ) -> u.Quantity:
        """
        Calculate the energy a particle will lose in the stack.

        Parameters
        ----------

        particles : str
            Incident particle

        E_in : u.Quantity
            Energy of the particle before the stack.


        dx : u.Quantity
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.


        Returns
        -------

        E_in_stack : u.Quantity
            Energy the particle leaves in the stack.

        """
        return E_in - self.range_down(particle, E_in, dx=dx)
