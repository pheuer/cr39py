import numpy as np
from scipy.integrate import cumulative_trapezoid

from cr39py.core.exportable_class import ExportableClassMixin, saveable_class
from cr39py.core.units import unit_registry as u
from cr39py.filtration.material import Material


@saveable_class()
class Layer(ExportableClassMixin):
    _exportable_attributes = ["thickness", "material", "active", "name"]

    @classmethod
    def from_properties(
        cls,
        thickness: u.Quantity,
        material: str | Material,
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
            Material of the layer

        active : `bool`, optional
            If `True`, this layer is marked as an active layer. The default is `True`.

        name : `str`, optional
            An optional name for the layer.

        """
        obj = cls()

        if isinstance(material, str):
            material = Material.from_string(material)

        obj.thickness = thickness
        obj.material = material
        obj.name = name
        obj.active = active
        return obj

    @classmethod
    def from_string(cls, s):
        """
        Create a layer from a string of the following form

        ##### UNIT MATERIAL
        """

        # Split the string by whitespace
        s = s.split()
        if len(s) < 3:
            raise ValueError("Invalid string code for Material")

        unit = u(s[1])
        thickness = float(s[0]) * unit
        material = Material.from_string(s[2])

        # TODO: support active/inactive and name here as optional
        # additional entries

        return cls.from_properties(thickness, material)

    def __eq__(self, other):

        return (
            self.thickness == other.thickness
            and self.material == other.material
            and self.name == other.name
            and self.active == other.active
        )

    def __str__(self):
        return f"{self.thickness.m_as(u.um):.1f} um {self.material}"


@saveable_class()
class Stack(ExportableClassMixin):
    r"""
    An ordered list of Layers

    """

    _exportable_attributes = ["layers"]

    @classmethod
    def from_layers(cls, *args):
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

    def range_down_ion(
        self, particle, Ep, dx=1 * u.um, max_nsublayers=None, reverse=False
    ):
        """
        Calculate the energy a particle will be ranged down to through the stack

        Parameters
        ----------
        particles : Particle
            Incident particle

        Ep : u.Quantity
            Initial energy of incident particle

        dx : u.Quantity, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.

        max_nsublayers: int
            Maximum number of sublayers to allow. If None,
            all sublayers will be included as determined by dx.

        reverse : bool
            If True, reverse the process to find the starting energy of
            a particle given the final energy. Used in `reverse_ion_ranging`

        """
        if reverse:
            layers = self.layers[::-1]
        else:
            layers = self.layers

        for l in layers:

            # Get a cubic splines interpolator for the stopping power
            # in this layer
            sp_fcn = l.material.ion_stopping_power(particle, return_interpolator=True)

            # Slice the layer into sublayer dx thick
            nsublayers = int(np.floor(l.thickness.m_as(u.um) / dx.m_as(u.um)))
            if max_nsublayers is not None and nsublayers > max_nsublayers:
                nsublayers = max_nsublayers
                dx = l.thickness / nsublayers

            sublayers = np.ones(nsublayers) * dx.m_as(u.um)
            # Include any remainder in the last sublayer
            sublayers[-1] += l.thickness.m_as(u.um) % dx.m_as(u.um)

            # Calculate the energy deposited in each sublayer
            # This is essentially numerically integrating the stopping power
            for ds in sublayers:
                # Interpolate the stopping power at the current energy
                interpolated_stopping_power = sp_fcn(Ep)

                if reverse:
                    interpolated_stopping_power *= -1

                Ep -= interpolated_stopping_power * (ds * u.um)

                if Ep < 0 * Ep.u:
                    return 0 * Ep.u
        return Ep

    def ion_ranging_energy_loss(self, particle, Ep):
        """
        Calculate the energy a particle will lose in the stack
        """
        return Ep - self.range_down_ion(particle, Ep)

    def reverse_ion_ranging(self, particle, Ep, dx=1 * u.um, max_nsublayers=None):
        """
        Given the final energy of a particle, calculate the initial energy
        it would have had prior to ranging through the stack.
        """
        return self.range_down_ion(
            particle, Ep, dx=dx, max_nsublayers=max_nsublayers, reverse=True
        )

    def transmitted_xray_spectrum(self, energies, input_spectrum):
        """
        Calculate transmitted x-ray spectrum
        """
        spectrum = np.copy(input_spectrum)

        # Get transmitted spectrum
        for layer in self.layers:
            mat = layer.material
            mu = mat.xray_mass_attenuation_coefficent(energies=energies) * mat.density
            # Apply decrease in intensity over this layer
            spectrum *= np.exp(-mu * layer.thickness)

        return spectrum

    def transmitted_xray_band(self, energies, spectrum, threshold=0.90):
        """
        Calculate the energy range mean and HWHM points through this filter

        Parameters
        ----------

        Threshold: float
            Percentage of the signal to include as ``in the bounds``.
            Defaults to 95%

        """

        spectrum = self.transmitted_xray_spectrum(energies, spectrum)

        emax = energies[np.argmax(spectrum)]

        # Calculate the cumulative spectrum and locate the 5% and 95%
        # levels to find the points that bound 95% of the signal
        ispect = cumulative_trapezoid(spectrum.m, x=energies.m)
        ispect /= np.max(ispect)

        dthreshold = (1 - threshold) / 2
        e1 = energies[np.argmin(np.abs(ispect - dthreshold))]
        e2 = energies[np.argmin(np.abs(ispect - (1 - dthreshold)))]

        return emax, e1, e2


@saveable_class()
class FilterPack(ExportableClassMixin):
    """
    A FilterPack contains one or more regions of filtration, each represented
    by a different Stack of Layers.
    """

    _exportable_attributes = ["regions"]

    @classmethod
    def from_stacks(cls, *args):
        obj = cls()

        # Replace any strings with Stack objects
        _args = []
        for arg in args:
            _arg = Stack.from_string(arg) if isinstance(arg, str) else arg
            _args.append(_arg)

        obj.regions = list(_args)
        return obj

    @property
    def nregions(self):
        return len(self.regions)

    def __str__(self):
        s = "FilterPack:\n"
        for r in self.regions:
            s += str(r) + "\n"
        return s

    def __eq__(self, other):
        if self.nregions != other.nregions:
            return False

        for i in range(self.nregions):
            if self.regions[i] != other.regions[i]:
                return False
        return True
