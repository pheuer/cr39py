"""
The `~cr39py.scan` module contains the `~cr39py.scan.Scan` class, which represents a scan of an etched piece of CR39.
"""

import copy
import os
from functools import cached_property
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from fast_histogram import histogram2d

from collections.abc import Sequence

from cr39py.scan.cli import _cli_input
from cr39py.core.exportable_class import ExportableClassMixin
from cr39py.core.units import unit_registry as u
from cr39py.core.types import TrackData
from cr39py.scan.cpsa import read_cpsa
from cr39py.scan.cut import Cut
from cr39py.models.response import TwoParameterModel
from cr39py.scan.subset import Subset

from IPython import display

__all__  = ['Scan']

class _Axis(ExportableClassMixin):

    _exportable_attributes = ["ind", 
        '_unit', '_default_range', 'framesize']

    def __init__(self, ind=None, unit=None, 
                default_range=(None,None,None))->None:
        
        # These parameters are intended to not be mutable
        self._ind = ind
        self._unit = unit
        self._default_range = default_range
        self.framesize = None

    @property
    def ind(self)->int:
        return self._ind
    @property
    def unit(self):
        return self._unit
    @property
    def default_range(self):
        """
        Default range is (min, max, framesize).
        None means set based on data bounds
        """
        return self._default_range

    def setup(self, tracks:TrackData)->None:
        """
        Setup the axes for the provided track array.

        Parameters
        ----------
        tracks : `~numpy.ndarray` (ntracks,6)
            Tracks for which the axis should be initialized.

        """
        self._init_framesize(tracks)

    def _init_framesize(self, tracks:TrackData )->None:
        """
        Calculates an initial framesize.
        """
        framesize = self.default_range[2]
        ntracks = tracks.shape[0]
        
        if framesize is None:
            nbins = int(np.clip(np.sqrt(ntracks) / 20, 20, 200))
            minval = np.min(tracks[:, self.ind])
            maxval = np.max(tracks[:, self.ind])
            framesize = (maxval - minval) / nbins
        
        self.framesize = framesize * self.unit

    def axis(self, tracks:TrackData, units:bool=True)->np.ndarray | u.Quantity:
        """
        Axis calculated for the provided array of tracks.

        Parameters
        ----------

        tracks : `~numpy.ndarray` (ntracks,6)
            Tracks for which the axis should be created.

        units : bool
            If True, return axis as a Quantity. Otherwise
            return as a `~numpy.ndarray` in the base units
            for this axis.

        """

        # Calculate a min and max value for the axis
        minval = self.default_range[0]
        if minval is None:
            minval = np.min(tracks[:, self.ind])

        maxval = self.default_range[1]
        if maxval is None:
            maxval = np.max(tracks[:, self.ind])

        ax =  np.arange(minval, maxval, 
                        self.framesize.m_as(self.unit))
        
        if units:
            ax *= self.unit
        
        return ax
            


    


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

    
    axes = {'X': _Axis(ind=0, unit=u.cm, default_range=(None, None, None)),
               'Y': _Axis(ind=1, unit=u.cm, default_range=(None, None, None)),
               'D': _Axis(ind=2, unit=u.um, default_range=(0, 20, 0.5)),
               'C': _Axis(ind=3, unit=u.dimensionless, default_range=(0, 80, 1)),
               'E': _Axis(ind=4, unit=u.dimensionless, default_range=(0, 50, 1)),
               'Z': _Axis(ind=5, unit=u.um, default_range=(None, None, None)),
               }

    _exportable_attributes = [
        "tracks",
        "axes",
        "framesize",
        "subsets",
        "current_subset_index",
        "etch_time",
    ]

    def __init__(self)->None:
        self.current_subset_index = 0
        self.subsets = []

        self.tracks = None

        # Etch time, u.Quantity
        self.etch_time = None

    # **********************************
    # Class Methods for initialization
    # **********************************

    @classmethod
    def from_tracks(cls, tracks:TrackData, etch_time: float):
        """
        Initialize a Scan object from an array of tracks.

        Parameters
        ---------
        tracks : np.ndarray (ntracks,6)
            Array of tracks with [X,Y,D,C,E,Z] values.

        etch_time : float
            Etch time in minutes.
        """
        obj = cls()

        obj.etch_time = etch_time * u.min
        obj.tracks = tracks

        # Initialize the axes based on the provided tracks
        for ax in obj.axes.values():
            ax.setup(obj.tracks)

        # Initialize the list of subsets with a single subset to start.
        obj.subsets = [Subset(),]

        return obj

    @classmethod
    def from_cpsa(cls, path: Path, etch_time: float):
        """
        Initialize a Scan object from a CPSA file.

        Parameters
        ---------
        path : `~pathlib.Path`
            Path to the CPSA file.

        etch_time : float
            Etch time in minutes.

        """
        tracks = read_cpsa(path)

        return cls.from_tracks(tracks, etch_time)

    # **********************************
    # Framesize setup
    # **********************************
    def set_framesize(self, ax_key:str, framesize: float | u.Quantity)->None:
        """
        Sets the bin width for a given axis.

        If axs is 'X' or 'Y', update the framesize
        for both so that the frames remain square.

        Parameters
        ----------

        ax_key : str
            Name of the axis to change.

        framesize : float | u.Quantity
            New framesize

        """

        # If no unit is supplied, assume the
        # default units for this axis
        if not isinstance(framesize, u.Quantity):
            framesize *= self.axes[ax_key].unit

        if ax_key in ["X", "Y"]:
            self.set_framesize("XY", framesize)
        elif ax_key == "XY":
            self.axes["X"].framesize = framesize
            self.axes["Y"].framesize = framesize
        else:
            self.axes[ax_key].framesize = framesize

    def optimize_xy_framesize(self, tracks_per_frame_goal:int=10)->None:
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
        framesize = self.axes['X'].framesize

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
    def current_subset(self)->None:
        return self.subsets[self.current_subset_index]

    @property
    def nsubsets(self)->None:
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

    def add_subset(self, *args:Subset)->None:
        if len(args) == 1:
            subset = args[0]
        elif len(args) == 0:
            subset = Subset()
        self.subsets.append(subset)

    def remove_subset(self, i:int)->None:
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
    def set_domain(self, *args, **kwargs)->None:
        """
        Sets the domain cut on the currently selected subset.

        See docstring for
        `~cr39py.subset.Subset.set_domain`
        """
        self.current_subset.set_domain(*args, **kwargs)

    def select_dslice(self, dslice: int | None)->None:
        """
        Select a new dslice by index. 

        See docstring for
        `~cr39py.subset.Subset.select_dslice`
        """
        self.current_subset.select_dslice(dslice)

    def set_ndslices(self, ndslices:int)->None:
        """
        Sets the number of ndslices on the current subset.

        See docstring for
        `~cr39py.subset.Subset.set_ndslices`
        """
        self.current_subset.set_ndslices(ndslices)

    # ************************************************************************
    # Methods for managing cut list
    # ************************************************************************
    def add_cut(self, *args, **kwargs)->None:
        """
        Add a cut to the currently selected subset.

        Takes the same arguments as 
        `~cr39py.subset.Subset.add_cut`
        """
        self.current_subset.add_cut(*args, **kwargs)

    def remove_cut(self, *args, **kwargs)->None:
        """
        Remove a cut from the currently selected subset.

        Takes the same arguments as 
        `~cr39py.subset.Subset.remove_cut`
        """
        self.current_subset.remove_cut(*args, **kwargs)

    def replace_cut(self, *args, **kwargs)->None:
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
    def ntracks(self)->int:
        """
        Number of tracks.
        """
        return self.tracks.shape[0]

    @cached_property
    def _selected_tracks(self)->TrackData:
        # Save hash of the current subset, only reset tracks
        # property if the subset has changed, or if the framesize has
        # changed
        self._cached_subset_hash = hash(self.current_subset)
        return self.current_subset.apply_cuts(self.tracks)

    def reset_selected_tracks(self):
        """Reset the cached selected tracks"""
        if hasattr(self, "_selected_tracks"):
            del self._selected_tracks

    @property
    def selected_tracks(self)->TrackData:
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
            ):
                pass
            # If not, delete the properties so they will be created again
            else:
                self.reset_selected_tracks()

        return self._selected_tracks

    @property
    def nselected_tracks(self)->int:
        """
        Number of currently selected tracks.
        """
        return self.selected_tracks.shape[0]

    def rotate(self, angle: float, center:tuple[float]=(0,0))->None:
        """
        Rotates the tracks in the XY plane by `rot` around
        a point

        Parameters
        ---------

        angle: float
            Rotation angle in degrees

        center : tuple[float]
            Center of rotation. The default is (0,0).

        """

        x = self.tracks[:, 0] - center[0]
        y = self.tracks[:, 1] - center[1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta += np.deg2rad(angle)
        self.tracks[:, 0] = r * np.cos(theta) + center[0]
        self.tracks[:, 1] = r * np.sin(theta) + center[1]

        self.reset_selected_tracks()

    def track_energy(self, particle, statistic="mean")->u.Quantity:
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

        model = TwoParameterModel(particle)
        energy = model.track_energy(d, self.etch_time.m_as(u.min))

        return energy

    # *************************************************************************
    # Data output
    # *************************************************************************

    def histogram(self, axes:tuple[str]=("X", "Y"),
                quantity: str|None=None, 
                tracks:np.ndarray|None=None)->tuple[np.ndarray]:
        """
        Create a histogram of the currently selected track data

        The following quantities can be used as axes or quantities:
        - 'X': x position
        - 'Y': y position
        - 'D': diameter
        - 'C': contrast
        - 'E': ecentricity
        - 'Z' : z position/lens position during scan

        Parameters
        ---------

        axes : tuple(str), optional
            The axes of the histogram.  The default is ('X', 'Y')

        quantity: str, optional
            The quantity to plot. Default is to plot the number
            of particles per cell. 

        tracks : np.ndarray (optional)
            Tracks data from which to make the histogram. Default
            is the currently selected track data.

        Returns
        -------

        hax  : `~np.ndarray`
            Horizontal axis

        vax : `~np.ndarray`
            Vertical axis

        histogram : `~np.ndarray`
            Histogram array

        """
        if tracks is None:
            tracks = self.selected_tracks

        ax0= self.axes[axes[0]]
        ax1= self.axes[axes[1]]
        ax0_axis = ax0.axis(tracks, units=False)
        ax1_axis = ax1.axis(tracks, units=False)

        # If creating a histogram like the X,Y,D plots
        if quantity is not None:
            ax2 = self.axes[quantity]
            weights = tracks[:, ax2.ind]
        else:
            weights = None

        rng = [(np.min(ax0_axis), np.max(ax0_axis)),
                (np.min(ax1_axis), np.max(ax1_axis))]
        bins = [ax0_axis.size, ax1_axis.size]

        arr = histogram2d(
            tracks[:, ax0.ind],
            tracks[:, ax1.ind],
            bins=bins,
            range=rng,
            weights=weights,
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

        return ax0_axis, ax1_axis, arr

    def overlap_parameter_histogram(self)->tuple[np.ndarray]:
        """The Zylstra overlap parameter for each cell.

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

        chi = (
            ntracks / self.axes["X"].framesize / self.axes["Y"].framesize * np.pi * D**2
        ).m_as(u.dimensionless)

        return x, y, chi

    # *************************************************************************
    # Track Manipulation
    # *************************************************************************

    def plot(
        self,
        axes: tuple[str] | None =None,
        quantity: str|None = None,
        tracks:TrackData|None=None,
        xrange: Sequence[float,None] | None  =None,
        yrange:Sequence[float,None] | None  =None,
        zrange: Sequence[float,None] | None  =None, 
        log:bool=False,
        figax=None,
        show=True,
        
    ):
        """
        Plots a histogram of the track data.

        In addition to the track quantities [X,Y,D,C,E,Z], the following
        custom quantities can also be plotted: 

        - CHI : The track overlap parameter from Zylstra et al. 2012

        Parameters
        ----------

        axes: tuple of str, optional
            Sets which axes to plot. If two axes are provided,
            a histogram of tracks will be made. Default is ('X','Y')

        quantity: str | None
            Sets which quantity to plot. The default is None, which will
            result in plotting an unweighted histogram of the number
            of tracks in each frame. Any of the track quantities are 
            valid, as are the list of custom quantities above. 

        tracks: `~numpy.ndarray` (ntracks,6) (optional)
            Array of tracks to plot. Defaults to the 
            currently selected tracks. 

        xrange: Sequence[float,None] (optional)
            Limits for the horizontal axis. Setting either value to
            None will use the minimum or maximum of the data range
            for that value. Default is to plot the full data range.

        yrange: Sequence[float,None] (optional)
            Limits for the vertical axis. Setting either value to
            None will use the minimum or maximum of the data range
            for that value. Default is to plot the full data range.

        zrange: Sequence[float,None] (optional)
            Limits for the plotted quantity. Setting either value to
            None will use the minimum or maximum of the data range
            for that value. Default is to plot the full data range.

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
        fontsize = 16

        # If a figure and axis are provided, use those
        if figax is not None:
            fig, ax = figax
        else:
            fig = plt.figure()
            ax = fig.add_subplot()

        if axes is None:
            axes = ("X", "Y")

        if xrange is None:
            xrange = [None, None]
        if yrange is None:
            yrange = [None, None]
        if zrange is None:
            zrange = [None, None]

        # Get the requested histogram
        if quantity == 'CHI':
            xax, yax, arr = self.overlap_parameter_histogram()
        else:
            xax, yax, arr = self.histogram(axes=axes, tracks=tracks)

        # Set all 0's in the histogram to NaN so they appear as 
        # blank white space on the plot
        arr[arr == 0] = np.nan

        if quantity is None:
            ztitle = "# Tracks"
            title = f"{axes[0]}, {axes[1]}"
        else:
            ztitle = quantity
            title = f"{axes[0]}, {axes[1]}, {quantity}"

        # Set any None bounds to the extrema of the ranges
        xrange[0] = np.nanmin(xax) if xrange[0] is None else xrange[0]
        xrange[1] = np.nanmax(xax) if xrange[1] is None else xrange[1]
        yrange[0] = np.nanmin(yax) if yrange[0] is None else yrange[0]
        yrange[1] = np.nanmax(yax) if yrange[1] is None else yrange[1]
        zrange[0] = np.nanmin(arr) if zrange[0] is None else zrange[0]
        zrange[1] = np.nanmax(arr) if zrange[1] is None else zrange[1]
        
        # Apply log transform if requested
        if log:
            title += " (log)"
            nonzero = np.nonzero(arr)
            arr[nonzero] = np.log10(arr[nonzero])
        else:
            title += " (lin)"


        if axes == ("X", "Y"):
            ax.set_aspect("equal")

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

        return fig, ax

    def cutplot(self, tracks:TrackData|None=None, show:bool=True):
        """
        Makes a standard figure useful for applying cuts.

        Subplots are:
        - (X,Y,Number of particles) (simple histogram)
        - 

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
            f"Subset {self.current_subset_index}, "
            f"dslice {self.current_subset.current_dslice_index} of "
            f"{self.current_subset.ndslices} selected, "
            f'\nEtch time: {self.etch_time.m_as(u.min):.1f} min.'
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
            axes=("X", "Y"),
            quantity='D',
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

        if show:
            plt.show()

        return fig, ax

    def focus_plot(self, show:bool=True):
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

        if show:
            plt.show()
        return fig,ax

    # *******************************************************
    # Command line interface
    # *******************************************************

    def cli(self):  # pragma: no cover
        """
        Command line interface for interactively setting up cuts.
        """

        # This flag keeps track of whether any changes have been made
        # by the CLI, and will be returned when it exits
        changed = False

        while True:
            # Clear IPython output to avoid piling up plots
            display.clear_output(wait=False)

            # Create a cut plot
            self.cutplot(show=True)
            
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
                    "'framesize` -> Change the framesize on an axis\n"
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

            elif x == "framesize":
                print("Enter the name of the axis to change")
                ax_name = _cli_input(mode="alpha-integer")
                ax_name = ax_name.upper()
                print(f"Selected axis {ax_name}")
                print(f"Current framesize is {self.axes[ax_name].framesize:.1e}")
                print("Enter new framesize")
                framesize = _cli_input(mode="float")
                self.set_framesize(ax_name, framesize)
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
                    deselected_tracks = self.current_subset.apply_cuts(self.tracks, invert=True)
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
