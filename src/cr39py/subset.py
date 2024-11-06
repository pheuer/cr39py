from cr39py.core.exportable_class import ExportableClassMixin
from cr39py.cut import Cut


class Subset(ExportableClassMixin):
    """
    A subset of the track data. The subset is defined by a domain, a list
    of cuts, and a number of diameter slices (Dslices).

    Paramters
    ---------

    grp : h5py.Group or string file path
        An h5py Group or file path to an h5py file from which to
        load the subset

    Notes
    -----

    Domain
    ------
    The domain is the area in parameter space the subset encompasses. This
    could limit the subset to a region in space (e.g. x=[-5, 0]) or another
    parameter (e.g. D=[0,20]). The domain is represented by a cut, but it is
    an inclusive rather than an exclusive cut.

    List of Cuts
    ------------
    The subset includes a list of cuts that are used to exclude tracks that
    would otherwise be included in the domain.

    DSlices
    ------
    Track diameter is proportional to particle energy, so slicing the subset
    into bins sorted by diameter sorts the tracks by diameter. Slices are
    created by equally partitioining the tracks in the subset into some
    number of dslices.
    """

    _exportable_attributes = ["cuts", "domain", "ndslices", "current_dslice_index"]

    def __init__(self, *args, domain=None, ndslices=None):

        self.cuts = []
        self.ndslices = ndslices

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

        self.current_dslice_index = 0

    def __eq__(self, other):

        if not isinstance(other, Subset):
            return False

        if set(self.cuts) != set(other.cuts):
            return False

        if self.domain != other.domain:
            return False

        if self.ndslices != other.ndslices:
            return False

        return True

    def __str__(self):
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
        s = "domain:" + str(hash(self.domain))
        s += "ndslices:" + str(self.ndslices)
        s += "current_dslice_index" + str(self.current_dslice_index)
        for i, c in enumerate(self.cuts):
            s += f"cut{i}:" + str(hash(c))

        return hash(s)

    def set_domain(self, cut):
        """
        Sets the domain cut: an inclusive cut that will not be inverted
        """
        self.domain = cut

    def select_dslice(self, i):
        if i is None:
            self.current_dslice_index = None
        elif i > self.ndslices - 1:
            raise ValueError(
                f"Cannot select the {i} dslice, there are only "
                f"{self.ndslices} dslices."
            )
        else:
            self.current_dslice_index = i

    def set_ndslices(self, ndslices):
        """
        Sets the number of ndslices
        """
        if not isinstance(ndslices, int) or ndslices < 0:
            raise ValueError(
                "ndslices must be an integer > 0, but the provided value"
                f"was {ndslices}"
            )

        else:
            self.ndslices = int(ndslices)

    # ************************************************************************
    # Methods for managing cut list
    # ************************************************************************
    @property
    def ncuts(self):
        return len(self.cuts)

    def add_cut(self, *args, **kwargs):

        if len(args) == 1:
            c = args[0]
        else:
            c = Cut(**kwargs)

        self.cuts.append(c)

    def remove_cut(self, i):
        if i > len(self.cuts) - 1:
            print(
                f"Cannot remove the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        else:
            self.cuts.pop(i)

    def replace_cut(self, i, c):
        if i > len(self.cuts) - 1:
            print(
                f"Cannot replace the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        else:
            self.cuts[i] = c
