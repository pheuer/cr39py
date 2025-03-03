"""
The `~cr39py.scan.base_scan` module contains the `~cr39py.scan.base_scan.Scan` class, which represents
a scan of an etched piece of CR39.
"""

import copy
import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from fast_histogram import histogram2d

from cr39py.core.exportable_class import ExportableClassMixin
from cr39py.core.types import TrackData
from cr39py.core.units import unit_registry as u
from cr39py.models.response import TwoParameterModel
from cr39py.scan.cpsa import extract_etch_time, read_cpsa
from cr39py.scan.cut import Cut
from cr39py.scan.subset import Subset

__all__ = ["Axis", "Scan"]


class Axis(ExportableClassMixin):
    """An axis of a CR-39 scan.

    Parameters
    ----------

    ind : int
        Index of the track data array corresponding to this axis.

    unit : u.Quantity
        Unit for this axis.

    default_range : tuple[float]
        Defaults for the (min, max, framesize) of this axis. Values are
        floats in the unit set by the unit keyword. Any of the values can
        be None, in which case the range will be automatically determined
        from the track data.

    """

    _exportable_attributes = ["ind", "_unit", "_default_range", "framesize"]

    def __init__(
        self,
        ind: int = None,
        unit: u.Quantity = None,
        default_range: tuple[float | None] = (None, None, None),
    ) -> None:

        if ind is None:  # pragma: no cover
            raise ValueError("ind argument is required")

        if unit is None:  # pragma: no cover
            raise ValueError("unit argument is required")

        # These parameters are intended to not be mutable
        self._ind = ind
        self._unit = unit
        self._default_range = default_range

        # Framesize is mutable
        self._framesize = None

    @property
    def ind(self) -> int:
        """The array index for this axis.

        Returns
        -------
        index : int
        """
        return self._ind

    @property
    def name(self) -> str:
        """
        Name of this axis.

        Returns
        -------
        name : str
        """
        axes = "XYDCEZ"
        return axes[self.ind]

    @property
    def unit(self):
        """Unit of this axis.

        Returns
        -------
        unit : u.Quantity
        """
        return self._unit

    @property
    def default_range(self):
        """
        Default range (min, max, framesize) for this axis.
        Any values that are set to None will be estimated based
        on the track data automatically.
        """
        return self._default_range

    @property
    def framesize(self) -> float:
        """
        Frame (bin) size for this axis.

        Returns
        -------
        framesize : float
            Frame size
        """
        return self._framesize

    @framesize.setter
    def framesize(self, framesize: float) -> None:
        """
        Set the framesize for this axis.

        Parameters
        ----------
        framesize : float
            New framesize
        """
        self._framesize = framesize

    def _set_default_framesize(self, tracks: TrackData) -> None:
        """
        Calculates an initial framesize based on the selected tracks.

        This function is called when a scan is first loaded to generate
        a reasonable first guess frame size for each axis.
        """
        # If a default framesize was specified, return that
        default_framesize = self.default_range[2]
        if default_framesize is not None:
            framesize = default_framesize
        else:
            # Otherwise, determine a framesize that will result in about
            # 20 tracks per frame
            ntracks = tracks.shape[0]
            nbins = int(np.clip(np.sqrt(ntracks) / 20, 20, 200))
            minval = np.min(tracks[:, self.ind])
            maxval = np.max(tracks[:, self.ind])
            framesize = (maxval - minval) / nbins

        self.framesize = framesize

    def axis(
        self,
        range: tuple[float | None, float | None] | None = None,
        tracks: TrackData | None = None,
    ) -> u.Quantity:
        """
        Calculate an axis over a given range using the set framesize.

        Keywords have the following priority

        1) Range
        2) tracks: range will be calculated as the min and max of the tracks.
        3) Default range

        Parameters
        ----------
        range : tuple(float|None, float|None), optional
            A tuple of (min, max) values for the axis.

        tracks : np.ndarray, optional
            Tracks data to use for the axis.

        Returns
        -------
        axis : u.Quantity
            Axis array
        """
        if range is None:
            range = (None, None)

        if range[0] is not None:
            minval = range[0]
        elif tracks is not None:
            minval = np.min(tracks[:, self.ind])
        else:
            raise ValueError(
                f"No range provided, no tracks, and no default range set for {self.name} min."
            )

        if range[1] is not None:
            maxval = range[1]
        elif tracks is not None:
            maxval = np.max(tracks[:, self.ind])
        else:
            raise ValueError(
                f"No range provided, no tracks, and no default range set for {self.name} max."
            )

        ax = np.arange(minval, maxval, self.framesize)
        ax *= self.unit

        return ax


class Scan(ExportableClassMixin):
    """
    A representation of a scan of a piece of CR-39 data.

    A Scan object contains an array of tracks and an axis for each
    dimension of the track data: X,Y,D,C,E,Z. A Scan object also
    contains a list of Subset objects, each of which contains
    an arbitrary number of Cut objects. One Subset is selected
    at a time, and the cuts from that subset are applied to create
    the selected_tracks object. Selected_tracks can be written out
    as a histogram for further data analysis.
    """

    _exportable_attributes = [
        "_tracks",
        "_axes",
        "_subsets",
        "_current_subset_index",
        "_etch_time",
    ]

    def __init__(self) -> None:
        self._axes = {
            "X": Axis(ind=0, unit=u.cm, default_range=(None, None, None)),
            "Y": Axis(ind=1, unit=u.cm, default_range=(None, None, None)),
            "D": Axis(ind=2, unit=u.um, default_range=(0, 20, 0.5)),
            "C": Axis(ind=3, unit=u.dimensionless, default_range=(0, 80, 1)),
            "E": Axis(ind=4, unit=u.dimensionless, default_range=(0, 50, 1)),
            "Z": Axis(ind=5, unit=u.um, default_range=(None, None, None)),
        }

        self._current_subset_index = 0
        self._subsets = [Subset()]

        self._tracks = None

        # Etch time, u.Quantity
        self._etch_time = None

        self._filepath = None

        self.metadata = {}

    @property
    def tracks(self) -> TrackData:
        """
        The array of tracks in the scan, without any cuts applied.

        For the tracks after cuts, use the ``selected_tracks`` property.

        The track array has shape (ntracks,6), where each track has the
        values [X,Y,D,C,E,Z].

        Returns
        -------
        tracks : `~numpy.ndarray` (ntracks,6)
            Track array
        """
        return self._tracks

    @property
    def etch_time(self) -> u.Quantity:
        """ "
        The cumulative track etch time for the scan.

        Returns
        -------
        etch_time : u.Quantity
            Etch time
        """
        return self._etch_time

    @property
    def filepath(self) -> Path:
        """
        Path to the file from which the scan was loaded.

        If the scan was not loaded from a file, e.g. if it was
        created directly from a track array, this will be ``None``.
        """
        return self._filepath

    @property
    def axes(self) -> dict[Axis]:
        """
        A dictionary of `~cr39py.scan.base_scan.Axis` objects.

        Keys to the dictionary are the axis names, "X","Y","D","C","E","Z".
        """
        return self._axes

    # **********************************
    # Class Methods for initialization
    # **********************************

    @classmethod
    def from_tracks(
        cls, tracks: TrackData, etch_time: float, metadata: dict | None = None
    ):
        """
        Initialize a Scan object from an array of tracks.

        Parameters
        ---------
        tracks : np.ndarray (ntracks,6)
            Array of tracks with [X,Y,D,C,E,Z] values.

        etch_time : float
            Etch time in minutes.

        metadata : dict
            Dictionary of metadata to attach to the Scan object.
        """

        obj = cls()

        if metadata is None:
            metadata = {}

        obj._etch_time = etch_time * u.min
        obj._tracks = tracks
        obj.metadata = metadata

        # Find an initial framesize for each axis based on the provided tracks
        for name in obj._axes:
            obj._axes[name]._set_default_framesize(tracks)

        return obj

    @classmethod
    def from_cpsa(cls, path: Path, etch_time: float | None = None):
        """
        Initialize a Scan object from an MIT CPSA file.

        The etch_time can be automatically extracted from the filename
        if it is included in a format like  ``_#m_``, ``_#min_``, ``_#h_``,
        ``_#hr_``, etc.

        Parameters
        ---------
        path : `~pathlib.Path`
            Path to the CPSA file.

        etch_time : float
            Etch time in minutes.

        """
        # If the etch time is not provided, attempt to automatically extract
        # from the CPSA filename
        if etch_time is None:
            etch_time = extract_etch_time(path)
        if etch_time is None:
            raise ValueError(
                "Etch time not provided or successfully extracted from CPSA filename."
            )

        tracks, metadata = read_cpsa(path)

        obj = cls.from_tracks(tracks, etch_time, metadata=metadata)
        obj._filepath = path

        return obj

    # **********************************
    # Framesize setup
    # **********************************
    def set_framesize(self, ax_key: str, framesize: float | u.Quantity) -> None:
        """
        Sets the bin width for a given axis.

        If axs is 'X' or 'Y', update the framesize
        for both so that the frames remain square.

        Parameters
        ----------

        ax_key : str
            Name of the axis to change.

        framesize : float | u.Quantity
            New framesize. If no units are provided, units
            are assume to be those of the axis being changed.

        """

        if isinstance(framesize, u.Quantity):
            framesize = float(framesize.m_as(self._axes[ax_key].unit))
        # If no unit is supplied, assume the
        # default units for this axis

        if str(ax_key).upper() in ["X", "Y", "XY"]:
            self._axes["X"].framesize = framesize
            self._axes["Y"].framesize = framesize
        else:
            self._axes[ax_key].framesize = framesize

    def framesize(self, ax_key: str = "XY") -> u.Quantity:
        """
        The frame size for a given axis.

        Parameters
        ----------
        ax_key : str, optional
            The axis for the frame size. The default is 'XY',
            which returns the framesize for the X and Y axes,
            which are always the same.

        Returns
        -------
        framesize: u.Quantity
            Framesize of the requested axis.
        """
        if ax_key == "XY":
            return self._axes["X"].framesize * self._axes[ax_key].unit
        elif ax_key in self._axes:
            return self._axes[ax_key].framesize * self._axes[ax_key].unit
        else:
            raise KeyError(f"Axis name not recognized: {ax_key}")

    def optimize_xy_framesize(self, tracks_per_frame_goal: int = 10) -> None:
        """
        Optimizes XY framesize for a given tracks per frame.

        Creates square frames.

        Parameters
        ----------

        tracks_per_frame_goal: int (optional)
            Number of tracks per bin to optimize for.
            Default is 10.
        """

        # initialize with current framesize
        framesize = self._axes["X"].framesize

        # Estimate the ideal framesize so that the median bin has some
        # number of tracks in it
        goal_met = False
        ntries = 0
        while not goal_met:
            _, _, image = self.histogram()
            median_tracks = np.median(image)

            print(f"framesize: {framesize:.1e}, median_tracks: {median_tracks:.2f}")

            # If many tries have happened, you may be in a loop and need
            # to relax the requirement
            if ntries > 25:
                atol = 3 + (ntries - 25) / 10
            else:
                atol = 3

            # Accept the framesize if within 5% of the goal value
            if np.isclose(median_tracks, tracks_per_frame_goal, atol=atol):
                print("Goal met")
                goal_met = True
            else:
                print("Trying a different framesize")
                # Amount by which to change the framesize side length
                # to try and capture the right number of tracks
                framesize_change = np.sqrt(tracks_per_frame_goal / median_tracks)

                # If the bin is too small, shrink by a bit less than
                # the calculated amount
                if median_tracks > tracks_per_frame_goal:
                    framesize_change *= 0.95
                else:
                    framesize_change *= 1.05

                # TODO: Move in steps smaller than the calculated optimum
                # to avoid overshooting
                framesize = framesize * framesize_change

                ntries += 1

            self.set_framesize("XY", framesize)

    # ************************************************************************
    # Manipulate Subsets
    # ************************************************************************

    @property
    def current_subset(self) -> Subset:
        """
        The currently selected subset object.
        """
        return self._subsets[self._current_subset_index]

    @property
    def nsubsets(self) -> int:
        """
        The current number of subsets.
        """
        return len(self._subsets)

    def select_subset(self, i: int) -> None:
        """
        Select a subset based on its index.

        Parameters
        ----------
        i : int
            Index of the subset to select
        """
        if i > self.nsubsets - 1 or i < -self.nsubsets:
            raise ValueError(
                f"Cannot select subset {i}, there are only " f"{self.nsubsets} subsets."
            )
        else:
            # Handle negative indexing
            if i < 0:
                i = self.nsubsets + i
            self._current_subset_index = i

    def add_subset(self, *args: Subset) -> None:
        """
        Adds a subset to the list.

        If no argument is provided, an empty subset
        will be created and added.

        Parameters
        ----------
        subset : `~cr39py.scan.subset.Subset`
            Subset to add.
        """
        if len(args) == 1:
            subset = args[0]
        elif len(args) == 0:
            subset = Subset()
        self._subsets.append(subset)

    def remove_subset(self, i: int) -> None:
        """
        Remove a subset based on its index.

        Parameters
        ----------
        i : int
            Index of the subset to remove.

        Raises
        ------
        ValueError
            If index exceeds the number of subsets defined,
            or if the index corresponds to the currently selected
            subset, which cannot be removed.
        """
        if i > self.nsubsets - 1:
            raise ValueError(
                f"Cannot remove the {i} subset, there are only "
                f"{self._subsets} subsets."
            )

        elif i == self._current_subset_index:
            raise ValueError("Cannot remove the currently selected subset.")

        else:
            self._subsets.pop(i)

    # ************************************************************************
    # Manipulate Cuts
    # These methods are all wrapers for methods on the current selected Subset
    # ************************************************************************
    def set_domain(self, *args, **kwargs) -> None:
        """
        Sets the domain cut on the currently selected subset.

        See docstring for
        `~cr39py.subset.Subset.set_domain`
        """
        self.current_subset.set_domain(*args, **kwargs)

    def select_dslice(self, dslice: int | None) -> None:
        """
        Select a new dslice by index.

        See docstring for
        `~cr39py.subset.Subset.select_dslice`
        """
        self.current_subset.select_dslice(dslice)

    def set_ndslices(self, ndslices: int) -> None:
        """
        Sets the number of ndslices on the current subset.

        See docstring for
        `~cr39py.subset.Subset.set_ndslices`
        """
        self.current_subset.set_ndslices(ndslices)

    # ************************************************************************
    # Methods for managing cut list
    # ************************************************************************
    def add_cut(self, *args, **kwargs) -> None:
        """
        Add a cut to the currently selected subset.

        Takes the same arguments as
        `~cr39py.subset.Subset.add_cut`
        """
        self.current_subset.add_cut(*args, **kwargs)

    def remove_cut(self, *args, **kwargs) -> None:
        """
        Remove a cut from the currently selected subset.

        Takes the same arguments as
        `~cr39py.subset.Subset.remove_cut`
        """
        self.current_subset.remove_cut(*args, **kwargs)

    def replace_cut(self, *args, **kwargs) -> None:
        """
        Replace a cut on the currently selected subset.

        Takes the same arguments as
        `~cr39py.subset.Subset.replace_cut`
        """
        self.current_subset.replace_cut(*args, **kwargs)

    # *************************************************************************
    # Track Manipulation
    # *************************************************************************

    @property
    def ntracks(self) -> int:
        """
        Number of tracks.
        """
        return self._tracks.shape[0]

    def _reset_selected_tracks(self):
        """Reset the cached properties associated with _selected_tracks."""
        if hasattr(self, "_selected_tracks"):
            del self._selected_tracks

    @cached_property
    def _selected_tracks(self) -> TrackData:
        """
        Cached TrackData array containing the tracks selected
        by the currently selected subset.
        """
        # Save hash of the current subset, only reset tracks
        # property if the subset has changed, or if the framesize has
        # changed
        self._cached_subset_hash = hash(self.current_subset)

        tracks = self.current_subset.apply_cuts(self._tracks)

        return tracks

    @property
    def selected_tracks(self) -> TrackData:
        """
        Tracks array for currently selected tracks.

        This property wraps a cached property
        `_selected_tracks` that is reset whenever anything is done that could
        change the selected tracks.
        """
        if hasattr(self, "_selected_tracks"):

            # If the subset matches the copy cached the last time
            # _selected_tracks was updated, the property is still up to date
            if hash(self.current_subset) == self._cached_subset_hash:
                pass
            # If not, delete the properties so they will be created again
            else:
                # Set the selected tracks to be re-generated.
                self._reset_selected_tracks()

        return self._selected_tracks

    @property
    def nselected_tracks(self) -> int:
        """
        Number of currently selected tracks.
        """
        return self.selected_tracks.shape[0]

    def rotate(self, angle: float, center: tuple[float] = (0, 0)) -> None:
        """
        Rotates the tracks in the XY plane by `rot` around a point.

        Parameters
        ---------

        angle: float
            Rotation angle in degrees

        center : tuple[float]
            Center of rotation. The default is (0,0).
        """

        x = self._tracks[:, 0] - center[0]
        y = self._tracks[:, 1] - center[1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta += np.deg2rad(angle)
        self._tracks[:, 0] = r * np.cos(theta) + center[0]
        self._tracks[:, 1] = r * np.sin(theta) + center[1]

        self._reset_selected_tracks()

    def track_energy(self, particle: str, statistic: str = "mean") -> u.Quantity:
        """
        The energy of the currently selected tracks.

        This function uses the `~cr39py.models.response.TwoParameterModel`
        of :cite:t:`Lahmann2020cr39` for the CR-39 response

        Parameters
        ----------
        particle : str
            One of ['p', 'd', 't', 'a']

        statistic : str
            One of ['mean', 'median']

        Returns
        -------
        energy : float
            Energy in MeV


        References
        ----------
        Please cite :cite:t:`Lahmann2020cr39` for the two parameter model if
        you use this method.

        """

        d = self.selected_tracks[:, 2]
        if statistic == "mean":
            d = np.mean(d)
        elif statistic == "median":
            d = np.median(d)
        else:  # pragma: no cover
            raise ValueError(f"Statistic keyword not recognized: {statistic}")

        model = TwoParameterModel(particle)
        energy = model.track_energy(d, self._etch_time.m_as(u.min))

        return energy

    # *************************************************************************
    # Data output
    # *************************************************************************

    def histogram(
        self,
        axes: tuple[str] = "XY",
        quantity: str | None = None,
        tracks: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray]:
        """
        Create a histogram of the currently selected track data

        The following quantities can be used as axes or quantities:
        - 'X': x position
        - 'Y': y position
        - 'D': diameter
        - 'C': contrast
        - 'E': ecentricity
        - 'Z' : z position/lens position during scan


        Histograms of the following composite quantities can also be made
        - CHI : The ``chi`` track overlap parameter from :cite:t:`Zylstra2012new`
        - F2 : The ``F2`` track overlap parameter from :cite:t:`Zylstra2012new`
        - 'TRACK DENSITY' : The number of tracks per cm^2 in each cell

        Parameters
        ---------

        axes : str, optional
            The axes of the histogram, as a two character string.
            The default is "XY".

        quantity: str, optional
            The quantity to plot. Default is to plot the number
            of particles per cell.

        tracks : np.ndarray (optional)
            Tracks data from which to make the histogram. Default
            is the currently selected track data.

        xlim : tuple (min,max), None
            Range of the horizontal axis. Defaults to the
            default range for this axis.

        ylim : tuple (min,max), None
            Range of the vertical axis. Defaults to the
            default range for this axis.

        Returns
        -------

        hax  : `~np.ndarray`
            Horizontal axis

        vax : `~np.ndarray`
            Vertical axis

        histogram : `~np.ndarray`
            Histogram array

        """

        # TODO: There is currently no way to make histograms
        # of the custom quantities with other tracks
        # because they are properties so no keywords can be passed
        # down into histogram...
        #
        # If the quantity is on the custom quantity list,
        # return the custom quantity
        if quantity == "CHI":
            return self.chi
        elif quantity == "F2":
            return self.F2
        elif quantity == "TRACK DENSITY":
            return self.track_density

        if tracks is None:
            tracks = self.selected_tracks

        # Make copies of the track objects
        ax0 = self._axes[axes[0]]
        ax1 = self._axes[axes[1]]

        hax = ax0.axis(range=xlim, tracks=tracks)
        vax = ax1.axis(range=ylim, tracks=tracks)

        # If creating a histogram like the X,Y,D plots
        if quantity is not None:
            ax2 = self._axes[quantity]
            weights = tracks[:, ax2.ind]
        else:
            weights = None

        rng = [
            (np.min(hax.m), np.max(hax.m)),
            (np.min(vax.m), np.max(vax.m)),
        ]
        bins = [hax.size, vax.size]

        arr = (
            histogram2d(
                tracks[:, ax0.ind],
                tracks[:, ax1.ind],
                bins=bins,
                range=rng,
                weights=weights,
            )
            * u.dimensionless
        )

        # Create the unweighted histogram and divide by it (sans zeros)
        if quantity is not None:
            arr_uw = histogram2d(
                tracks[:, ax0.ind],
                tracks[:, ax1.ind],
                bins=bins,
                range=rng,
            )
            nz = np.nonzero(arr_uw)
            arr[nz] = arr[nz] / arr_uw[nz]
            arr = arr * ax2.unit

        return hax, vax, arr

    @property
    def chi(self) -> tuple[np.ndarray]:
        """
        The Zylstra overlap parameter ``chi`` for each cell.

        As defined in :cite:t:`Zylstra2012new`.

        Only includes currently selected tracks.

        Returns
        -------

        hax  : `~np.ndarray`
            Horizontal axis

        vax : `~np.ndarray`
            Vertical axis

        chi : `~np.ndarray`
            Histogram of chi for each cell
        """
        x, y, ntracks = self.histogram(axes=("X", "Y"))
        x, y, D = self.histogram(axes=("X", "Y"), quantity="D")

        area = self._axes["X"].framesize * self._axes["Y"].framesize

        chi = (ntracks / area * np.pi * D**2).to(u.dimensionless)

        return x, y, chi

    @property
    def F2(self) -> tuple[np.ndarray]:
        """
        The Zylstra overlap parameter ``F2`` for each cell.

        As defined in :cite:t:`Zylstra2012new`.

        F2 is the fraction of tracks that overlap one other track, and
        is a reasonable approximation of the number of tracks that will
        be lost due to track overlap.

        .. math::
           F_2 = \chi (1 - 2\chi/3)

        As shown in the paper, this analytical model starts to fail
        when the ``chi`` parameter excceeds about 25%. Above this
        threshold, the analytical model over-estimtates the true
        number of track overlaps (according to Monte-Carlo simulations).

        Only includes currently selected tracks.

        Returns
        -------

        hax  : `~np.ndarray`
            Horizontal axis

        vax : `~np.ndarray`
            Vertical axis

        F2 : `~np.ndarray`
            Histogram of F2 for each cell

        """

        x, y, chi = self.chi

        F2 = chi * (1 - 2 * chi / 3)

        return x, y, F2

    @property
    def track_density(self) -> tuple[np.ndarray]:
        """Track density in tracks/cm^2 for each bin of the histogram.

        Only includes currently selected tracks.

        Returns
        -------

        hax  : `~np.ndarray`
            Horizontal axis

        vax : `~np.ndarray`
            Vertical axis

        track_density : `~np.ndarray`
            Histogram of track density for each cell.
        """

        x, y, ntracks = self.histogram(axes=("X", "Y"))

        cell_area = (self.axes["X"].framesize * self.axes["Y"].framesize).m_as(u.cm**2)
        track_density = ntracks / cell_area

        return x, y, track_density

    def save_histogram(self, path: Path, *args, **kwargs) -> None:
        """
        Save a track histogram to a file.

        The file extension will be used to determine the save format. Supported formats are
        - .h5,.hdf5 : HDF5 file
        - .csv : CSV file

        The HDF5 interface will also include the axes and some metadata, but the CSV interface
        will only save the histogram 2D array.

        Parameters
        ----------
        path : `~pathlib.Path`
            Path to save the histogram to.

        *args, **kwargs
            Additional arguments to pass to the histogram method.
        """

        hax, vax, arr = self.histogram(*args, **kwargs)

        ext = path.suffix

        if ext.lower() in [".h5", ".hdf5"]:
            with h5py.File(path, "w") as f:
                f["x"] = hax.m
                f["x"].attrs["unit"] = str(hax.u)
                f["y"] = vax.m
                f["y"].attrs["unit"] = str(vax.u)
                f["data"] = arr.m
                f["data"].attrs["unit"] = str(arr.u)

        elif ext.lower() == ".csv":
            np.savetxt(path, arr.m, delimiter=",")

        elif ext.lower() == ".png":
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            if self.filepath is not None:
                ax.set_title(self.filepath.stem, fontsize=9)
            ax.pcolormesh(hax.m, vax.m, arr.m.T)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.savefig(path, dpi=200)

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # *************************************************************************
    # Track Manipulation
    # *************************************************************************

    def plot(
        self,
        axes: str | None = None,
        quantity: str | None = None,
        tracks: TrackData | None = None,
        xlim: Sequence[float, None] | None = None,
        ylim: Sequence[float, None] | None = None,
        zlim: Sequence[float, None] | None = None,
        use_default_ranges=False,
        log: bool = False,
        figax=None,
        show=True,
    ):
        """
        Plots a histogram of the track data.

        Parameters
        ----------

        axes: str, optional
            Sets which axes to plot, as a two-character string.
            Default is "XY".

        quantity: str | None
            Sets which quantity to plot. The default is None, which will
            result in plotting an unweighted histogram of the number
            of tracks in each frame. Any of the track quantities are
            valid, as are the list of custom quantities listed
            in the docstring for the histogram method.

        tracks: `~numpy.ndarray` (ntracks,6) (optional)
            Array of tracks to plot. Defaults to the
            currently selected tracks.

        xlim: Sequence[float,None] (optional)
            Limits for the horizontal axis. Setting either value to
            None will use the minimum or maximum of the data range
            for that value. Default is to plot the full data range.

        ylim: Sequence[float,None] (optional)
            Limits for the vertical axis. Setting either value to
            None will use the minimum or maximum of the data range
            for that value. Default is to plot the full data range.

        zlim: Sequence[float,None] (optional)
            Limits for the plotted quantity. Setting either value to
            None will use the minimum or maximum of the data range
            for that value. Default is to plot the full data range.

        use_default_ranges: bool, optional
            If True, if no ranges are provided for an axis, use
            default ranges (if not None) before using the full range.
            Default is False.

        log : bool (optional)
            If ``True``, plot the log of the quantity.

        figax : tuple(Fig,Ax), optional
            Tuple of (Figure, Axes) onto which the plot will
            be put. If none is provided, a new figure will be
            created.

        show : bool, optional
            If True, call plt.show() at the end to display the
            plot. Default is True. Pass False if this plot is
            being made as a subplot of another figure.

        Returns
        -------

        fig, ax : Figure, Axes
            The matplotlib figure and axes objects with
            the plot.



        """
        if tracks is None:
            tracks = self.selected_tracks

        fontsize = 16

        # If a figure and axis are provided, use those
        if figax is not None:
            fig, ax = figax
        else:
            fig = plt.figure()
            ax = fig.add_subplot()

        if axes is None:
            axes = "XY"

        xax, yax, arr = self.histogram(
            axes=axes, quantity=quantity, tracks=tracks, xlim=xlim, ylim=ylim
        )

        # Set all 0's in the histogram to NaN so they appear as
        # blank white space on the plot
        arr[arr == 0] = np.nan

        if quantity is None:
            ztitle = "# Tracks"
            title = f"{axes[0]}, {axes[1]}"
        else:
            ztitle = quantity
            title = f"{axes[0]}, {axes[1]}, {quantity}"

        # Iterate through the axes and make decisions about the limits
        limits = [xlim, ylim, zlim]
        for i in range(len(limits)):
            # Skip zlim if quantity is None
            if i == 2 and quantity is None:
                continue

            axname = axes[i] if i < 2 else quantity
            axobj = self._axes[axname]
            axis = [xax, yax, arr][i]
            if limits[i] is None:
                limits[i] = [None, None]

            if limits[i][0] is None and use_default_ranges:
                limits[i][0] = axobj.default_range[0]
            limits[i][0] = np.nanmin(axis.m) if limits[i][0] is None else limits[i][0]

            if limits[i][1] is None and use_default_ranges:
                limits[i][1] = axobj.default_range[1]
            limits[i][1] = np.nanmax(axis.m) if limits[i][1] is None else limits[i][1]
        xlim, ylim, zlim = limits

        # Apply log transform if requested
        if log:
            title += " (log)"
            nonzero = np.nonzero(arr)
            arr[nonzero] = np.log10(arr[nonzero])
        else:
            title += " (lin)"

        if axes == ("X", "Y"):
            ax.set_aspect("equal")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(axes[0], fontsize=fontsize)
        ax.set_ylabel(axes[1], fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)

        try:
            p = ax.pcolorfast(xax.m, yax.m, arr.m.T)

            cb_kwargs = {
                "orientation": "vertical",
                "pad": 0.07,
                "shrink": 0.8,
                "aspect": 16,
            }
            cbar = fig.colorbar(p, ax=ax, **cb_kwargs)
            cbar.set_label(ztitle, fontsize=fontsize)

        except ValueError:  # raised if one of the arrays is empty
            pass

        return fig, ax

    def cutplot(self, tracks: TrackData | None = None, show: bool = True):
        """
        Makes a standard figure useful for applying cuts.

        Subplots are:
        - (X,Y,Num. Tracks (lin)) (simple histogram)
        - (D,C, Num. Tracks (log))
        - (X, Y, D (lin)) (average diameter per frame)
        - (D, E, Num. Tracks (log))

        Parameters
        ----------

        tracks : `~numpy.ndarray` (ntracks, 6), optional
            Array of tracks to plot. Defaults to the
            currently selected tracks.

        show : bool, optional
            If True, call plt.show() at the end to display the
            plot. Default is True. Pass False if this plot is
            being made as a subplot of another figure.

        """

        if tracks is None:
            tracks = self.selected_tracks

        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        title = (
            f"Subset {self._current_subset_index}, "
            f"dslice {self.current_subset.current_dslice_index} of "
            f"{self.current_subset.ndslices} selected, "
            f"\nEtch time: {self._etch_time.m_as(u.min):.1f} min."
        )

        fig.suptitle(title)

        # X, Y
        ax = axarr[0][0]
        self.plot(
            axes=("X", "Y"),
            show=False,
            figax=(fig, ax),
            xlim=self.current_subset.domain.xrange,
            ylim=self.current_subset.domain.yrange,
            tracks=tracks,
        )

        # D, C
        ax = axarr[0][1]
        self.plot(
            axes=("D", "C"),
            show=False,
            figax=(fig, ax),
            log=True,
            xlim=self.current_subset.domain.drange,
            ylim=self.current_subset.domain.crange,
            tracks=tracks,
        )

        # X, Y, D
        ax = axarr[1][0]
        self.plot(
            axes=("X", "Y"),
            quantity="D",
            show=False,
            figax=(fig, ax),
            xlim=self.current_subset.domain.xrange,
            ylim=self.current_subset.domain.yrange,
            zlim=self.current_subset.domain.drange,
            tracks=tracks,
        )

        # D, E
        ax = axarr[1][1]
        self.plot(
            axes=("D", "E"),
            show=False,
            figax=(fig, ax),
            log=True,
            xlim=self.current_subset.domain.drange,
            ylim=self.current_subset.domain.erange,
            tracks=tracks,
        )

        if show:
            plt.show()

        return fig, ax

    def focus_plot(self, show: bool = True):
        """
        Plot of the microscope focus (Z) across the XY plane of the scan.

        Used to look for abnormalities that may indicate a failed scan.

        Parameters
        ----------

        show : bool, optional
            If True, call plt.show() at the end to display the
            plot. Default is True. Pass False if this plot is
            being made as a subplot of another figure.
        """

        fig, ax = plt.subplots()

        self.plot(
            axes=("X", "Y", "Z"),
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
        )

        if show:
            plt.show()
        return fig, ax

    # *******************************************************
    # Command line interface
    # *******************************************************

    def cli(self) -> None:  # pragma: no cover
        """
        Command line interface for interactively setting up cuts.
        """
        # Import here to avoid circular import
        from cr39py.scan._cli import scan_cli

        return scan_cli(self)
