"""
The `~cr39py.subset` module contains the `~cr39py.subset.Subset` class, which represents a subset of the tracks in a CR39 dataset.

A subset consists of a list of cuts, all of which are applied to exclude tracks. The remaining tracks are in the subset. The
subset also includes a domain, which is an initial cut. The only difference between the domain and the cuts is that the domain is always applied,
while the other cuts may sometimes be inverted to plot the excluded tracks during analysis.

Subsets can also be divided into bins along the diameter axis (which corresponds to the energy of the particles). This provides an easy
way to examine the histograms of tracks made by different energy particles. Each of these bins is called a ``dslice``, and they can be used
by using `~cr39py.subset.Subset.set_ndslices` to set ``ndslices`` to an integer > 1, then using `~cr39py.subset.Subset.select_dslice` to select
the index of a particular dslice to show.

"""

import numpy as np

from cr39py.core.exportable_class import ExportableClassMixin
from cr39py.core.types import TrackData
from cr39py.scan.cut import Cut


class Subset(ExportableClassMixin):
    """
    A subset of the track data.

    The subset is defined by a domain, a list
    of cuts, and a number of diameter slices (Dslices).

    The domain is a cut that defines the area in parameter space that the subset. Unlike regular cuts,
    the domain is inclusive, and the ``apply_cuts`` method will not invert the domain when it inverts other cuts.

    Parameters
    ----------

    domain : `~cr39py.cut.Cut`
        A cut that defines the domain of the subset. The domain is the area in parameter
        space the subset encompasses. This could limit the subset to a region in space
        (e.g. x=[-5, 0]) or another parameter (e.g. D=[0,20]). The domain is represented
        by a `~cr39py.cut.Cut`.

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

    def set_domain(self, *args, **kwargs) -> None:
        """
        Sets the domain cut

        Parameters
        ----------
        domain : `~cr39py.cut.Cut` | None
            The domain, represented as a Cut object.
            If None, the domain is set to an empty
            domain.

        """
        if len(args) == 1 and isinstance(args[0], Cut):
            c = args[0]
        else:
            c = Cut(**kwargs)

        self.domain = c

    def clear_domain(self) -> None:
        """
        Reset the domain to an empty cut.
        """
        self.domain = Cut()

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

    def set_ndslices(self, ndslices: int) -> None:
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

        Or create a new cut on the subset automatically

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

        Raises
        ------

        ValueError
            If the index is out of bounds.

        """

        if i > len(self.cuts) - 1:
            raise ValueError(
                f"Cannot remove the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        self.cuts.pop(i)

    def replace_cut(self, i: int, cut: Cut):
        """Replace the ith cut in the Subset list with a new cut

        Parameters
        ----------
        i : int
            Index of the Cut to replace.
        cut : `~cr39py.cut.Cut`
            New cut to insert.

        Raises
        ------

        ValueError
            If the index is out of bounds.

        """
        if i > len(self.cuts) - 1:
            raise ValueError(
                f"Cannot replace the {i} cut, there are only " f"{len(self.cuts)} cuts."
            )
        self.cuts[i] = cut

    def clear_cuts(self) -> None:
        """
        Remove all cuts.
        """
        self.cuts = []

    def apply_cuts(
        self, tracks: TrackData, use_cuts: list[int] | None = None, invert: bool = False
    ) -> TrackData:
        """
        Applies the cuts to the provided track array.

        Parameters
        ----------

        tracks : `~np.ndarray` (ntracks,6)
            Tracks to which cuts will be applied.

        use_cuts : int, list of ints (optional)
            If provided, only the cuts corresponding to the int or ints
            provided will be applied. The default is to apply all cuts

        invert : bool (optional)
            If True, return the inverse of the cuts selected, i.e. the
            tracks that would otherwise be excluded. The domain will not be inerted.
            The default is False.

        Returns
        -------

        selected_tracks : `~numpy.ndarray` (ntracks,6)
            The selected track array.

        """

        # Valid cut indices based on the current number of cuts
        valid_cuts = list(np.arange(self.ncuts))

        # If use_cuts is set, make sure all of the indices are valid
        if use_cuts is None:
            use_cuts = valid_cuts
        else:
            use_cuts = list(use_cuts)
            for s in use_cuts:
                if s not in valid_cuts:
                    raise ValueError(
                        f"Specified cut index is invalid: {s}. "
                        f"Valid cuts are {valid_cuts}"
                    )

        # boolean mask of tracks to include in the selected tracks
        ntracks = tracks.shape[0]
        include = np.ones(ntracks).astype(bool)

        for i, cut in enumerate(self.cuts):
            if i in use_cuts:
                # Get a boolean array of tracks that are inside this cut
                x = cut.test(tracks)

                # negate to get a list of tracks that are NOT
                # in the excluded region
                include *= np.logical_not(x)

        if self.domain is not None:
            domain_include = self.domain.test(tracks)

        # Select only these these tracks
        # If inverting, do not invert the domain tracks
        if invert:
            selected_tracks = tracks[(~include) * domain_include, :]
        else:
            selected_tracks = tracks[include * domain_include, :]

        # Calculate the bin edges for each dslice
        # !! note that the tracks are already sorted into order by diameter
        # when the CR39 data is read in
        #
        # Skip if ndslices is 1 (nothing to cut) or if ndslices is None
        # which indicates to use all of the available ndslices
        if self.ndslices != 1 and self.current_dslice_index is not None:
            # Figure out the dslice width
            dbin = int(selected_tracks.shape[0] / self.ndslices)
            # Extract the appropriate portion of the tracks
            b0 = self.current_dslice_index * dbin
            b1 = b0 + dbin
            selected_tracks = selected_tracks[b0:b1, :]

        return selected_tracks
