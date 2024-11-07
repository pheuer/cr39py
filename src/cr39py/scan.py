import copy
import os
from functools import cached_property
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from fast_histogram import histogram2d

from cr39py.cli import _cli_input
from cr39py.core.exportable_class import ExportableClassMixin
from cr39py.core.units import unit_registry as u
from cr39py.cpsa import read_cpsa
from cr39py.cut import Cut
from cr39py.response import TwoParameterModel
from cr39py.subset import Subset


class Scan(ExportableClassMixin):
    """
    A representation of a piece of CR39 data.

    A Scan object contains an array of tracks and an axis for each
    dimension of the track data: X,Y,D,C,E,Z. A Scan object also
    contains a list of Subset objects, each of which contains
    an arbitrary number of Cut objects. One Subset is selected
    at a time, and the cuts from that subset are applied to create
    the selected_tracks object. Selected_tracks can be written out
    as a histogram for further data analysis.
    """

    # Axes dictionary for trackdata
    axes_ind = {"X": 0, "Y": 1, "D": 2, "C": 3, "E": 5, "Z": 6}

    axes_units = {
        "X": u.cm,
        "Y": u.cm,
        "D": u.um,
        "C": u.dimensionless,
        "E": u.dimensionless,
        "Z": u.um,
    }

    # Set default ranges for some of the parametres
    # min, max, binsize
    default_ranges = {
        "X": (None, None, None),
        "Y": (None, None, None),
        "D": (0, 20, 0.5),
        "C": (0, 80, 1),
        "E": (0, 50, 1),
        "Z": (None, None, None),
    }

    _exportable_attributes = [
        "tracks",
        "axes",
        "binsize",
        "subsets",
        "current_subset_index",
        "etch_time",
    ]

    def __init__(self):

        # Store figures once created for blitting
        self.plotfig = None
        self.cutplotfig = None

        self.current_subset_index = 0
        self.subsets = []

        self.tracks = None
        self.binsize = {
            "X": None,
            "Y": None,
            "D": None,
            "C": None,
            "E": None,
            "Z": None,
        }

        # Etch time, u.Quantity
        self.etch_time = None

    # **********************************
    # Class Methods for initialization
    # **********************************

    @classmethod
    def from_tracks(cls, tracks, etch_time: float):
        """
        Initialize a Scan object from an array of tracks.

        Paramters
        ---------
        tracks : np.ndarray (ntracks,6)
            Array of tracks with [X,Y,D,C,E,Z] values.

        etch_time : float
            Etch time in minutes.
        """
        obj = cls()

        obj.etch_time = etch_time * u.min
        obj.tracks = tracks

        # Estimate a good binsize for each dimension
        # Aim for some # of tracks per bin, ~25-50 is about right?
        for i, ax in enumerate(obj.axes_ind.keys()):

            binsize = obj.default_ranges[ax][2]
            if binsize is None:
                nbins = int(np.clip(np.sqrt(obj.ntracks) / 20, 20, 200))
                minval = np.min(obj.tracks[:, i])
                maxval = np.max(obj.tracks[:, i])
                binsize = (maxval - minval) / nbins

            obj.binsize[ax] = binsize * obj.axes_units[ax]

        # Initialize the list of subsets with a single subset to start.
        obj.subsets = [
            Subset(),
        ]

        return obj

    @classmethod
    def from_cpsa(cls, path: Path, etch_time: float):
        """
        Initialize a Scan object from a CPSA file.

        Paramters
        ---------
        path : `~pathlib.Path`
            Path to the CPSA file.

        etch_time : float
            Etch time in minutes.

        """
        tracks = read_cpsa(path)

        return cls.from_tracks(tracks, etch_time)

    # **********************************
    # Axes + bin sizes
    # **********************************

    def set_binsize(self, ax, binsize):
        """
        Sets the bin width for a given axis
        """
        if ax in ["X", "Y"]:
            self.set_binsize("XY", binsize)
        elif ax == "XY":
            self.binsize["X"] = binsize
            self.binsize["Y"] = binsize
        else:
            self.binsize[ax] = binsize

    def optimize_binsize(self, tracks_per_bin_goal=10):
        """
        Optimizes binsize for a given tracks per bin.

        Creates square bins.

        Parameters
        ----------

        tracks_per_bin_goal: int (optional)
            Number of tracks per bin to optimize for.
            Default is 10.
        """

        # initialize with current binsize
        binsize = self.binsize["X"]

        # Estimate the ideal binsize so that the median bin has some
        # number of tracks in it
        goal_met = False
        ntries = 0
        while not goal_met:
            _, _, image = self.frames()
            median_tracks = np.median(image)

            print(f"binsize: {binsize:.1e}, median_tracks: {median_tracks:.2f}")

            # If many tries have happened, you may be in a loop and need
            # to relax the requirement
            if ntries > 25:
                atol = 3 + (ntries - 25) / 10
            else:
                atol = 3

            # Accept the binsize if within 5% of the goal value
            if np.isclose(median_tracks, tracks_per_bin_goal, atol=atol):
                print("Goal met")
                goal_met = True
            else:
                print("Trying a different binsize")
                # Amount by which to change the binsize side length
                # to try and capture the right number of tracks
                binsize_change = np.sqrt(tracks_per_bin_goal / median_tracks)

                # If the bin is too small, shrink by a bit less than
                # the calculated amount
                if median_tracks > tracks_per_bin_goal:
                    binsize_change *= 0.95
                else:
                    binsize_change *= 1.05

                # TODO: Move in steps smaller than the calculated optimum
                # to avoid overshooting
                binsize = binsize * binsize_change

                ntries += 1

            self.set_binsize("XY", binsize)

    def _axes(self, tracks):
        """
        Calculate axes for a given track array.

        Iterates over the axes
        """
        axes = {}
        for ax, ind in self.axes_ind.items():
            # Calculate a min and max value for the axis
            minval = self.default_ranges[ax][0]
            if minval is None:
                minval = np.min(tracks[:, ind])

            maxval = self.default_ranges[ax][1]
            if maxval is None:
                maxval = np.max(tracks[:, ind])

            # Create the new axis
            axes[ax] = (
                np.arange(minval, maxval, self.binsize[ax].m_as(self.axes_units[ax]))
                * self.axes_units[ax]
            )
        return axes

    @property
    def axes(self):
        """
        Axes for the currently selected tracks.
        """
        return self._axes(self.selected_tracks)

    # ************************************************************************
    # Manipulate Subsets
    # ************************************************************************

    @property
    def current_subset(self):
        return self.subsets[self.current_subset_index]

    @property
    def nsubsets(self):
        return len(self.subsets)

    def select_subset(self, i):
        if i > self.nsubsets - 1 or i < -self.nsubsets:
            raise ValueError(
                f"Cannot select subset {i}, there are only " f"{self.nsubsets} subsets."
            )
        else:
            # Handle negative indexing
            if i < 0:
                i = self.nsubsets + i
            self.current_subset_index = i

    def add_subset(self, *args):
        if len(args) == 1:
            subset = args[0]
        elif len(args) == 0:
            subset = Subset()
        self.subsets.append(subset)

    def remove_subset(self, i):
        if i > self.nsubsets - 1:
            raise ValueError(
                f"Cannot remove the {i} subset, there are only "
                f"{self.subsets} subsets."
            )

        elif i == self.current_subset_index:
            raise ValueError("Cannot remove the currently selected subset.")

        else:
            self.subsets.pop(i)

    # ************************************************************************
    # Manipulate Cuts
    # These methods are all wrapers for methods on the current selected Subset
    # ************************************************************************
    def set_domain(self, *args, **kwargs):
        """
        Sets the domain cut on the currently selected subset.
        """
        self.current_subset.set_domain(*args, **kwargs)

    def select_dslice(self, *args, **kwargs):
        self.current_subset.select_dslice(*args, **kwargs)

    def set_ndslices(self, *args, **kwargs):
        """
        Sets the number of ndslices
        """
        self.current_subset.set_ndslices(*args, **kwargs)

    # ************************************************************************
    # Methods for managing cut list
    # ************************************************************************
    def add_cut(self, *args, **kwargs):
        """
        Add a cut to the currently selected subset.
        """
        self.current_subset.add_cut(*args, **kwargs)

    def remove_cut(self, *args, **kwargs):
        """
        Remove a cut from the currently selected subset.
        """
        self.current_subset.remove_cut(*args, **kwargs)

    def replace_cut(self, *args, **kwargs):
        """
        Replace a cut on the currently selected subset.
        """
        self.current_subset.replace_cut(*args, **kwargs)

    # *************************************************************************
    # Track Manipulation
    # *************************************************************************

    @property
    def ntracks(self):
        return self.tracks.shape[0]

    @cached_property
    def _selected_tracks(self):
        # Save hash of the current subset, only reset tracks
        # property if the subset has changed, or if the binsize has
        # changed
        self._cached_subset_hash = hash(self.current_subset)
        self._cached_binsize = copy.copy(self.binsize)
        return self._get_selected_tracks()

    def reset_selected_tracks(self):
        """Reset the cached selected tracks"""
        if hasattr(self, "_selected_tracks"):
            del self._selected_tracks

    def _get_selected_tracks(self, use_cuts: None | list[int] = None, invert=False):
        """
        Return tracks that meet the current set of specifications,
        and/or specifications made by the keywords.

        use_cuts : int, list of ints (optional)
            If provided, only the cuts corresponding to the int or ints
            provided will be applied. The default is to apply all cuts

        invert : bool (optional)
            If true, return the inverse of the cuts selected. Default is
            false.


        """

        valid_cuts = list(np.arange(len(self.current_subset.cuts)))
        if use_cuts is None:
            use_cuts = valid_cuts
        else:
            for s in use_cuts:
                if s not in valid_cuts:
                    raise ValueError(f"Specified cut index is invalid: {s}")
        use_cuts = list(use_cuts)

        keep = np.ones(self.ntracks).astype(bool)

        for i, cut in enumerate(self.current_subset.cuts):
            if i in use_cuts:
                # Get a boolean array of tracks that are inside this cut
                x = cut.test(self.tracks)

                # negate to get a list of tracks that are NOT
                # in the excluded region (unless we are inverting)
                if not invert:
                    x = np.logical_not(x)
                keep *= x

        # Regardless of anything else, only show tracks that are within
        # the domain
        if self.current_subset.domain is not None:
            keep *= self.current_subset.domain.test(self.tracks)

        # Select only these tracks
        selected_tracks = self.tracks[keep, :]

        # Calculate the bin edges for each dslice
        # !! note that the tracks are already sorted into order by diameter
        # when the CR39 data is read in
        #
        # Skip if ndslices is 1 (nothing to cut) or if ndslices is None
        # which indicates to use all of the available ndslices
        if (
            self.current_subset.ndslices != 1
            and self.current_subset.current_dslice_index is not None
        ):
            # Figure out the dslice width
            dbin = int(selected_tracks.shape[0] / self.current_subset.ndslices)
            # Extract the appropriate portion of the tracks
            b0 = self.current_subset.current_dslice_index * dbin
            b1 = b0 + dbin
            selected_tracks = selected_tracks[b0:b1, :]
        return selected_tracks

    @property
    def selected_tracks(self):
        """
        Tracks array for currently selected tracks.

        This property wraps a cached property
        `_selected_tracks` that is reset whenever anything is done that could
        change the selected tracks.
        """
        if hasattr(self, "_selected_tracks"):

            # If the subset matches the copy cached the last time
            # _selected_tracks was updated, the property is still up to date
            if (
                hash(self.current_subset) == self._cached_subset_hash
                and self.binsize == self._cached_binsize
            ):
                pass
            # If not, delete the properties so they will be created again
            else:
                self.reset_selected_tracks()

        return self._selected_tracks

    @property
    def nselected_tracks(self):
        """
        Number of currently selected tracks.
        """
        return self.selected_tracks.shape[0]

    def rotate(self, angle: float):
        """
        Rotates the tracks in the XY plane by `rot` around (0,0)

        Paramters
        ---------

        angle: float
            Rotation angle in degrees
        """

        x = self.tracks[:, 0]
        y = self.tracks[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta += np.deg2rad(angle)
        self.tracks[:, 0] = r * np.cos(theta)
        self.tracks[:, 1] = r * np.sin(theta)

        self.reset_selected_tracks()

    def track_energy(self, particle, statistic="mean"):
        """
        The energy of the tracks on the current subset + dslice

        statistic : str
            One of ['mean', 'min', 'max']
        """

        d = self.selected_tracks[:, 2]
        if statistic == "mean":
            d = np.mean(d)
        elif statistic == "min":
            d = np.min(d)
        elif statistic == "max":
            d = np.max(d)
        else:  # pragma: no cover
            raise ValueError(f"Statistic keyword not recognized: {statistic}")

        model = TwoParameterModel()
        energy = model.track_energy(d, particle, self.etch_time.m_as(u.min))

        return energy

    # *************************************************************************
    # Data output
    # *************************************************************************

    def frames(self, axes=("X", "Y"), tracks=None):
        """
        Create a histogram of the currently selected track data

        Paramters
        ---------

        axes : tuple of 2 or 3 str
            The first two values represent the axes of the histogram. If no
            third value is included, then the resulting histogram is of the
            number of hits in each  bin. If a third value is included,
            the histogram will be of that value in each bin

            Chose from the following:
            'X': x position
            'Y': y position
            'D': diameter
            'C': contrast
            'E': ecentricity
            'Z' : z position/lens position during scan

            The default is ('X', 'Y')


        tracks : np.ndarray (optional)
            Tracks data from which to make the histogram. Default
            is the currently selected track data.

        """

        i0 = self.axes_ind[axes[0]]
        i1 = self.axes_ind[axes[1]]

        if tracks is None:
            tracks = self.selected_tracks

        _axes = self.axes
        ax0 = _axes[axes[0]].m
        ax1 = _axes[axes[1]].m

        # If creating a histogram like the X,Y,D plots
        if len(axes) == 3:
            i2 = self.axes_ind[axes[2]]
            weights = tracks[:, i2]
        else:
            weights = None

        rng = [(np.min(ax0), np.max(ax0)), (np.min(ax1), np.max(ax1))]
        bins = [ax0.size, ax1.size]

        arr = histogram2d(
            tracks[:, i0],
            tracks[:, i1],
            bins=bins,
            range=rng,
            weights=weights,
        )

        # Create the unweighted histogram and divide by it (sans zeros)
        if len(axes) == 3:
            arr_uw = histogram2d(
                tracks[:, i0],
                tracks[:, i1],
                bins=bins,
                range=rng,
            )
            nz = np.nonzero(arr_uw)
            arr[nz] = arr[nz] / arr_uw[nz]

        return ax0, ax1, arr

    def cli(self):  # pragma: no cover
        """
        Command line interface for interactively setting up cuts.
        """
        self.cutplot(show=True)

        # This flag keeps track of whether any changes have been made
        # by the CLI, and will be returned when it exits
        changed = False

        while True:

            print("*********************************************************")
            print(
                f"Current subset index: {self.current_subset_index} of {np.arange(len(self.subsets))}"
            )
            # Print a summary of the current subset
            print(self.current_subset)
            print(
                f"ntracks selected: {self.nselected_tracks:.1e} "
                f"(of {self.ntracks:.1e})"
            )

            print(
                "add (a), edit (e), edit the domain (d), remove (r), plot (p), "
                "plot inverse (pi), switch subsets (subset), change dslices (dslice), "
                "change the number of dslices (ndslices), end (end), help (help)"
            )

            split = _cli_input(mode="alpha-integer list", always_pass=[])
            x = split[0]

            if x == "help":
                print(
                    "Enter commands, followed by any additional arugments "
                    "separated by commas.\n"
                    " ** Commands ** \n"
                    "'a' -> create a new cut\n"
                    "'c' -> Select a new dslice\n"
                    "Argument (one int) is the index of the dslice to select"
                    "Enter 'all' to select all"
                    "'d' -> edit the domain\n"
                    "'e' -> edit a cut\n"
                    "Argument (one int) is the cut to edit\n"
                    "'ndslices' -> Change the number of dslices on this subset."
                    "'p' -> plot the image with current cuts\n"
                    "'pi' -> plot the image with INVERSE of the cuts\n"
                    "'r' -> remove an existing cut\n"
                    "Arguments are numbers of cuts to remove\n"
                    "'subset' -> switch subsets or create a new subset\n"
                    "Argument is the index of the subset to switch to, or"
                    "'new' to create a new subset"
                    "'help' -> print this documentation\n"
                    "'end' -> accept the current values\n"
                    "'binsize` -> Change the binsize on an axis\n"
                    " ** Cut keywords ** \n"
                    "xmin, xmax, ymin, ymax, dmin, dmax, cmin, cmax, emin, emax\n"
                    "e.g. 'xmin:0,xmax:5,dmax=15'\n"
                )

            elif x == "end":
                self.cutplot(show=True)
                break

            elif x == "a":
                print("Enter new cut parameters as key:value pairs separated by commas")
                kwargs = _cli_input(mode="key:value list")

                # validate the keys are all valid dictionary keys
                valid = True
                for key in kwargs.keys():
                    if key not in list(Cut.defaults.keys()):
                        print(f"Unrecognized key: {key}")
                        valid = False

                if valid:
                    c = Cut(**kwargs)
                    self.current_subset.add_cut(c)

                self.cutplot(show=True)
                changed = True

            elif x == "binsize":
                print("Enter the name of the axis to change")
                ax_name = _cli_input(mode="alpha-integer")
                ax_name = ax_name.upper()
                print(f"Selected axis {ax_name}")
                print(f"Current binsize is {self.binsize[ax_name]:.1e}")
                print("Enter new binsize")
                binsize = _cli_input(mode="float")
                self.set_binsize(ax_name, binsize)
                self.cutplot(show=True)
                changed = True

            elif x == "dslice":
                if len(split) < 2:
                    print(
                        "Select the index of the dslice to switch to, or"
                        "enter 'all' to select all dslices"
                    )
                    ind = _cli_input(mode="alpha-integer")
                else:
                    ind = split[1]

                if ind == "all":
                    self.select_dslice(None)
                else:
                    self.select_dslice(int(ind))
                self.cutplot(show=True)
                changed = True

            elif x == "d":
                print("Current domain: " + str(self.current_subset.domain))
                print(
                    "Enter a list key:value pairs with which to modify the domain"
                    "(set a key to 'None' to remove it)"
                )
                kwargs = _cli_input(mode="key:value list")
                self.current_subset.domain.update(**kwargs)
                self.cutplot(show=True)
                changed = True

            elif x == "e":
                if len(split) > 1:
                    ind = int(split[1])

                    if ind >= len(self.current_subset.cuts):
                        print("Invalid subset number")

                    else:
                        print(
                            f"Selected cut ({ind}) : "
                            + str(self.current_subset.cuts[ind])
                        )
                        print(
                            "Enter a list key:value pairs with which to modify this cut"
                            "(set a key to 'None' to remove it)"
                        )

                        kwargs = _cli_input(mode="key:value list")
                        self.current_subset.cuts[ind].update(**kwargs)
                        self.cutplot(show=True)
                        changed = True
                else:
                    print(
                        "Specify the number of the cut you want to modify "
                        "as an argument after the command."
                    )

            elif x == "ndslices":
                if len(split) < 2:
                    print("Enter the requested number of dslices")
                    ind = _cli_input(mode="alpha-integer")
                else:
                    ind = split[1]
                self.set_ndslices(int(ind))
                self.cutplot(show=True)

                changed = True

            elif x in ["p", "pi"]:
                if x == "pi":
                    deselected_tracks = self._get_selected_tracks(invert=True)
                    self.cutplot(show=True, tracks=deselected_tracks)
                else:
                    self.cutplot(show=True)

            elif x == "r":
                if len(split) < 2:
                    print("Select the index of the cut to remove")
                    ind = _cli_input(mode="integer")
                else:
                    ind = split[1]
                print(f"Removing cut {int(ind)}")
                self.current_subset.remove_cut(int(ind))
                self.cutplot(show=True)

                changed = True

            elif x == "subset":
                if len(split) < 2:
                    print(
                        "Select the index of the subset to switch to, or "
                        "enter 'new' to create a new subset."
                    )
                    ind = _cli_input(mode="alpha-integer")
                else:
                    ind = split[1]

                if ind == "new":
                    ind = len(self.subsets)
                    print(f"Creating a new subset, index {ind}")
                    subset = Subset()
                    self.add_subset(subset)

                print(f"Selecting subset {ind}")
                self.select_subset(int(ind))
                self.cutplot(show=True)
                changed = True

            else:
                print(f"Invalid input: {x}")

        return changed

    # *************************************************************************
    # Track Manipulation
    # *************************************************************************

    def plot(
        self,
        axes=("X", "Y"),
        log=False,
        clear=False,
        xrange=None,
        yrange=None,
        zrange=None,
        show=True,
        figax=None,
        tracks=None,
    ):
        """
        Plots a histogram of the track data

        Parameters
        ----------

        axes: tuple of str
            Indicates which axes to plot. If two axes are provided,
            a histogram of tracks will be made. If three axes are
            provided,

        tracks : tracks array to plot

        """

        if xrange is None:
            xrange = [None, None]
        if yrange is None:
            yrange = [None, None]
        if zrange is None:
            zrange = [None, None]

        fontsize = 16

        xax, yax, arr = self.frames(axes=axes, tracks=tracks)

        # If a figure and axis are provided, use those
        if figax is not None:
            fig, ax = figax
        elif self.plotfig is None or clear:
            fig = plt.figure()
            ax = fig.add_subplot()
            self.plotfig = [fig, ax]
        else:
            fig, ax = self.plotfig

        if axes[0:2] == ("X", "Y"):
            ax.set_aspect("equal")

        if len(axes) == 3:
            ztitle = axes[2]
            title = f"{axes[0]}, {axes[1]}, {axes[2]}"
        else:
            ztitle = "# Tracks"
            title = f"{axes[0]}, {axes[1]}"

        arr[arr == 0] = np.nan

        # Calculate bounds
        if xrange[0] is None:
            xrange[0] = np.nanmin(xax)
        if xrange[1] is None:
            xrange[1] = np.nanmax(xax)
        if yrange[0] is None:
            yrange[0] = np.nanmin(yax)
        if yrange[1] is None:
            yrange[1] = np.nanmax(yax)

        if log:
            title += " (log)"
            nonzero = np.nonzero(arr)
            arr[nonzero] = np.log10(arr[nonzero])
        else:
            title += " (lin)"

        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)

        ax.set_xlabel(axes[0], fontsize=fontsize)
        ax.set_ylabel(axes[1], fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)

        try:
            p = ax.pcolorfast(xax, yax, arr.T)

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

        if show:
            plt.show()

        return fig, ax

    def cutplot(self, tracks=None):
        """
        Makes a standard figure with several views of the track data.
        """

        if tracks is None:
            tracks = self.selected_tracks

        self.cutplotfig = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
        self.cutplotfig[0].subplots_adjust(hspace=0.3, wspace=0.3)

        # Figure tuple contains:
        # (fig, axarr, bkg)
        fig, axarr = self.cutplotfig

        title = f"Subset {self.current_subset_index}, "

        title += (
            f"dslice {self.current_subset.current_dslice_index} of "
            f"{self.current_subset.ndslices} selected."
        )
        fig.suptitle(title)

        # X, Y
        ax = axarr[0][0]
        self.plot(
            axes=("X", "Y"),
            show=False,
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
            tracks=tracks,
        )

        # D, C
        ax = axarr[0][1]
        self.plot(
            axes=("D", "C"),
            show=False,
            figax=(fig, ax),
            log=True,
            xrange=self.current_subset.domain.drange,
            yrange=self.current_subset.domain.crange,
            tracks=tracks,
        )

        # X, Y, D
        ax = axarr[1][0]
        self.plot(
            axes=("X", "Y", "D"),
            show=False,
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
            zrange=self.current_subset.domain.drange,
            tracks=tracks,
        )

        # D, E
        ax = axarr[1][1]
        self.plot(
            axes=("D", "E"),
            show=False,
            figax=(fig, ax),
            log=True,
            xrange=self.current_subset.domain.drange,
            yrange=self.current_subset.domain.erange,
            tracks=tracks,
        )

        return fig, ax

    def focus_plot(self):
        """
        Plot the focus (z coordinate) over the scan. Used to look for
        abnormalities that may indicate a failed scan.
        """

        fig, ax = plt.subplots()

        self.plot(
            axes=("X", "Y", "Z"),
            figax=(fig, ax),
            xrange=self.current_subset.domain.xrange,
            yrange=self.current_subset.domain.yrange,
        )
