"""
The `~cr39py.cut` module contains the `~cr39py.cut.Cut` class, which represents a CR39 track cut.

A cut defines a region in which tracks should be excluded from the analysis. For example, a cut defined by

>>> Cut(cmin=30)

will exclude all tracks with contrast greater than 30%, while

>>> Cut(dmax=2, xmin=0)

will exclude all tracks with diameters less than 2 um and x positions greater than x=0.
"""

import numpy as np

from cr39py.core.exportable_class import ExportableClassMixin


class Cut(ExportableClassMixin):
    """
    Represents a track cut.

    A cut is series of upper and lower bounds on tracks that should be
    excluded from the analyzed tracks.

    Parameters
    ----------

    xmin: float (None)
        Minimum x value

    xmax: float (None),
        Maximum x value

    ymin: float (None)
        Minimum y value

    ymax: float (None),
        Maximum y value

    dmin: float (None)
        Minimum d value

    dmax: float (None),
        Maximum d value

    cmin: float (None)
        Minimum c value

    cmax: float (None),
        Maximum c value

    emin: float (None)
        Minimum e value

    emax: float (None),
        Maximum e value

    zmin: float (None)
        Minimum z value

    zmax: float (None),
        Maximum z value


    """

    _exportable_attributes = [
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "dmin",
        "dmax",
        "cmin",
        "cmax",
        "emin",
        "emax",
        "zmin",
        "zmax",
    ]

    defaults = {
        "xmin": -1e6,
        "xmax": 1e6,
        "ymin": -1e6,
        "ymax": 1e6,
        "dmin": 0,
        "dmax": 1e6,
        "cmin": 0,
        "cmax": 1e6,
        "emin": 0,
        "emax": 1e6,
        "zmin": 0,
        "zmax": 1000,
    }

    indices = {
        "xmin": 0,
        "xmax": 0,
        "ymin": 1,
        "ymax": 1,
        "dmin": 2,
        "dmax": 2,
        "cmin": 3,
        "cmax": 3,
        "emin": 4,
        "emax": 4,
        "zmin": 5,
        "zmax": 5,
    }

    def __init__(
        self,
        *args,
        xmin: float = None,
        xmax: float = None,
        ymin: float = None,
        ymax: float = None,
        dmin: float = None,
        dmax: float = None,
        cmin: float = None,
        cmax: float = None,
        emin: float = None,
        emax: float = None,
        zmin: float = None,
        zmax: float = None,
    ):

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dmin = dmin
        self.dmax = dmax
        self.cmin = cmin
        self.cmax = cmax
        self.emin = emin
        self.emax = emax
        self.zmin = zmin
        self.zmax = zmax

    @property
    def bounds(self):
        """
        A dictionary of the current bounds.
        """
        bounds = {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "dmin": self.dmin,
            "dmax": self.dmax,
            "cmin": self.cmin,
            "cmax": self.cmax,
            "emin": self.emin,
            "emax": self.emax,
            "zmin": self.zmin,
            "zmax": self.zmax,
        }
        return bounds

    def __eq__(self, other):
        """Determines whether two cuts are equal

        Two cuts are defined to be equal if they have the same bounds.
        """
        if not isinstance(other, Cut):
            return False

        for key in self.bounds.keys():
            if self.bounds[key] != other.bounds[key]:
                return False

        return True

    def __hash__(self):
        """Generates a hash for comparing cuts

        The hash is generated by hashing a string representation of the
        bounds of the cut, separated by underscores.
        """
        s = ""
        for val in self.bounds.items():
            s += f"{val}_"
        return hash(s)

    def __str__(self):
        """
        String representation of the cut.
        """
        s = [f"{key}:{val}" for key, val in self.bounds.items() if val is not None]
        s = ", ".join(s)
        if s == "":
            return "[Empty cut]"
        else:
            return s

    # These range properties are used to set the range for plotting
    @property
    def xrange(self):
        "Range of cut in X"
        return [self.bounds["xmin"], self.bounds["xmax"]]

    @property
    def yrange(self):
        "Range of cut in Y"
        return [self.bounds["ymin"], self.bounds["ymax"]]

    @property
    def drange(self):
        "Range of cut in D"
        return [self.bounds["dmin"], self.bounds["dmax"]]

    @property
    def crange(self):
        "Range of cut in C"
        return [self.bounds["cmin"], self.bounds["cmax"]]

    @property
    def erange(self):
        "Range of cut in E"
        return [self.bounds["emin"], self.bounds["emax"]]

    @property
    def zrange(self):
        "Range of cut in Z"
        return [self.bounds["zmin"], self.bounds["zmax"]]

    def update(self, **bounds):
        """
        Updates the cut from a list of provided keywords.

        Accepted keywords: xmin, xmax, ymin, ymax, cmin, cmax,
        dmin, dmax, emin, emax, zmin, zmax

        Examples
        --------

        >>> cut.update(xmin=-1, cmax=20)
        """
        for key, val in bounds.items():
            _key = key.lower()

            # Raise an exception if an invalid key is provided
            if _key not in self.bounds:
                raise KeyError(f"Unrecognized key for cut bounds: {key}")

            # If the string `none` is provided as a value, encode that as
            # python None
            # This is used in the interface for Scan's CLI
            if val is None:
                self.bounds[_key] = None

            elif isinstance(val, str) and val.strip().lower() == "none":
                self.bounds[_key] = None

            # Otherwise update the bounds value with the new value
            else:
                self.bounds[_key] = float(val)

    def test(self, trackdata):
        """
        Returns a boolean array representing which tracks fall within this cut.

        Note that the mask returned by this function is tracks that are inside
        the cut, e.g. tracks that should be excluded by the cut.

        Parameters
        ----------
        trackdata : np.ndarray (ntracks, 6)
            Track data array.

        Returns
        -------
        in_cut : np.ndarray, bool (ntracks,)
            A boolean array indicating whether or not each track in the
            trackdata array fits within the cut.

        """
        ntracks, _ = trackdata.shape
        in_cut = np.ones(ntracks).astype("bool")

        for key in self.bounds.keys():
            if self.bounds[key] is not None:
                i = self.indices[key]
                if "min" in key:
                    in_cut *= np.greater(trackdata[:, i], getattr(self, key))
                else:
                    in_cut *= np.less(trackdata[:, i], getattr(self, key))

        # Return a 1 for every track that is in the cut
        return in_cut.astype(bool)
