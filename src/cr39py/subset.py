from cr39py.core.exportable_class import ExportableClassMixin
from cr39py.cut import Cut


class Subset(ExportableClassMixin):
    """
    A subset of the track data.

    The subset is defined by a domain, a list
    of cuts, and a number of diameter slices (Dslices).

    Parameters
    ----------

    domain : `~cr39py.cut.Cut`
        A cut that defines the domain of the subset. The domain is the area in parameter
        space the subset encompasses. This could limit the subset to a region in space
        (e.g. x=[-5, 0]) or another parameter (e.g. D=[0,20]). The domain is represented
        by a Cut object, but it is inclusive rather than exclusive.

    ndslices : int
        Number of bins in the diameter axis to slice the data into.
        The default is 1.


    Notes
    -----



    The subset includes a list of cuts that are used to exclude tracks that
    would otherwise be included in the domain.

    Track diameter is proportional to particle energy, so slicing the subset
    into bins sorted by diameter sorts the tracks by diameter. Slices are
    created by equally partitioining the tracks in the subset into some
    number of dslices.
    """

    _exportable_attributes = ["cuts", "domain", "ndslices", "current_dslice_index"]

    def __init__(self, domain=None, ndslices=None):

        self.current_dslice_index = 0

        self.cuts = []
        self.ndslices = ndslices
        self.current_dslice_index = 0

        if domain is not None:
            self.set_domain(domain)
        # If no domain is set, set it with an empty cut
        else:
            self.domain = Cut()

        # By default, set the number of dslices to be 1
        if ndslices is None:
            self.set_ndslices(1)
        else:
            self.set_ndslices(ndslices)

    def __eq__(self, other):
        """
        Determines whether two subsets are equal.

        Two subsets are defined to be equal if they have equal
        domains, cuts, and numbers of dslices.
        """

        if not isinstance(other, Subset):
            return False

        if set(self.cuts) != set(other.cuts):
            return False

        if self.domain != other.domain:
            return False

        if self.ndslices != other.ndslices:
            return False

        return True

    def __str__(self) -> str:
        """String representation of the Subset"""
        s = ""
        s += "Domain:" + str(self.domain) + "\n"
        s += "Current cuts:\n"
        if len(self.cuts) == 0:
            s += "No cuts set yet\n"
        else:
            for i, cut in enumerate(self.cuts):
                s += f"Cut {i}: {str(cut)}\n"
        s += f"Num. dslices: {self.ndslices} "
        s += f"[Selected dslice index: {self.current_dslice_index}]\n"

        return s

    def __hash__(self):
        """Hash of the Subset"""
        s = "domain:" + str(hash(self.domain))
        s += "ndslices:" + str(self.ndslices)
        s += "current_dslice_index" + str(self.current_dslice_index)
        for i, c in enumerate(self.cuts):
            s += f"cut{i}:" + str(hash(c))

        return hash(s)

    def set_domain(self, domain: Cut | None) -> None:
        """
        Sets the domain cut

        Parameters
        ----------
        domain : `~cr39py.cut.Cut` | None
            The domain, represented as a Cut object.
            If None, the domain is set to an empty
            domain.

        """
        if domain is None:
            self.domain = Cut()

        self.domain = domain

    def select_dslice(self, dslice: int | None) -> None:
        """Set the currently selected dslice.

        Parameters
        ----------

        dslice: int|None
            The dslice to select. If None, then all dslices
            will be selected.

        """
        if dslice is None:
            self.current_dslice_index = None
        elif dslice > self.ndslices - 1:
            raise ValueError(
                f"Cannot select the {dslice} dslice, there are only "
                f"{self.ndslices} dslices."
            )
        else:
            self.current_dslice_index = dslice

    def set_ndslices(self, ndslices):
        """
        Sets the number of ndslices

        Parameters
        ----------

        ndslices : int
            Number of dslices

        """
        if not isinstance(ndslices, int) or ndslices < 0:
            raise ValueError(
                "ndslices must be an integer > 0, but the provided value"
                f"was {ndslices}"
            )

        self.ndslices = int(ndslices)

        # Ensure that, when changing the number of dslices,
        # you don't end up with a current_dslice_index that exceeds
        # the new number of dslices
        if self.current_dslice_index > self.ndslices - 1:
            self.current_dslice_index = self.ndslices - 1

    # ************************************************************************
    # Methods for managing cut list
    # ************************************************************************
    @property
    def ncuts(self):
        """Number of current cuts on this subset"""
        return len(self.cuts)

    def add_cut(self, *args, **kwargs) -> None:
        """
        Adds a new Cut to the Subset.

        Either provide a single Cut (as an argument), or any combination of the
        cut keywords to create a new cut.

        Parameters
        ----------

        cut : `~cr39py.cut.Cut`
            A single cut to be added, as an argument.


        xmin, xmax, ymin, ymax, cmin, cmax, dmin, dmax, emin, emax : float
            Keywords defining a new cut. Default for all keywords is None.

        Examples
        --------

        Create a cut, then add it to the subset

        >>> cut = Cut(cmin=30)
        >>> subset.add_cut(cut)

        Or create a new cut on the subset automatically.
        >>> subset.add_cut(cmin=30)

        """

        if len(args) == 1:
            c = args[0]
        else:
            c = Cut(**kwargs)

        self.cuts.append(c)

    def remove_cut(self, i: int) -> None:
        """Remove an existing cut from the subset.

        The cut will be removed from the list, so the index of latter
        cuts in the list will be decremented.

        Parameters
        ----------

        i : int
            Index of the cut to remove

        """

        if i > len(self.cuts) - 1:
            print(
                f"Cannot remove the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        else:
            self.cuts.pop(i)

    def replace_cut(self, i: int, cut: Cut):
        """Replace the ith cut in the Subset list with a new cut

        Parameters
        ----------
        i : int
            Index of the Cut to replace.
        cut : `cr39py.cut.Cut`
            New cut to insert.
        """
        if i > len(self.cuts) - 1:
            print(
                f"Cannot replace the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        else:
            self.cuts[i] = cut
