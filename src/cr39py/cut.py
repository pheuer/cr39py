import numpy as np

from cr39py.core.exportable_class import ExportableClassMixin


class Cut(ExportableClassMixin):
    """
    A cut is series of upper and lower bounds on tracks that should be
    excluded.
    """

    _exportable_attributes = ["bounds"]

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
        "emin": 5,
        "emax": 5,
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
    ):

        self.bounds = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "dmin": dmin,
            "dmax": dmax,
            "cmin": cmin,
            "cmax": cmax,
            "emin": emin,
            "emax": emax,
        }

    def __eq__(self, other):
        if not isinstance(other, Cut):
            return False

        for key in self.bounds.keys():
            if self.bounds[key] != other.bounds[key]:
                return False

        return True

    def __hash__(self):
        s = ""
        for val in self.bounds.items():
            s += f"{val}_"
        return hash(s)

    def __getattr__(self, key):

        if key in self.bounds.keys():
            if self.bounds[key] is None:
                return self.defaults[key]
            else:
                return self.bounds[key]
        else:
            raise ValueError(f"Unknown attribute for Cut: {key}")

    # These range properties are used to set the range for plotting
    @property
    def xrange(self):
        return [self.bounds["xmin"], self.bounds["xmax"]]

    @property
    def yrange(self):
        return [self.bounds["ymin"], self.bounds["ymax"]]

    @property
    def drange(self):
        return [self.bounds["dmin"], self.bounds["dmax"]]

    @property
    def crange(self):
        return [self.bounds["cmin"], self.bounds["cmax"]]

    @property
    def erange(self):
        return [self.bounds["emin"], self.bounds["emax"]]

    def __str__(self):
        s = [f"{key}:{val}" for key, val in self.bounds.items() if val is not None]
        s = ", ".join(s)
        if s == "":
            return "[Empty cut]"
        else:
            return s

    def update(self, **bounds):
        """
        Updates based off of provided keywords, e.g.

        cut.update(xmin=-1, cmax=20)
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
        Given tracks, return a boolean array representing which tracks
        fall within this cut
        """
        ntracks, _ = trackdata.shape
        keep = np.ones(ntracks).astype("bool")

        for key in self.bounds.keys():
            if self.bounds[key] is not None:
                i = self.indices[key]
                if "min" in key:
                    keep *= np.greater(trackdata[:, i], getattr(self, key))
                else:
                    keep *= np.less(trackdata[:, i], getattr(self, key))

        # Return a 1 for every track that is in the cut
        return keep.astype(bool)
