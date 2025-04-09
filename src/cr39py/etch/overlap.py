"""The `~cr39py.etch.overlap` module contains tools for calculating track overlap probabilities."""

from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interpn

from cr39py.core.data import data_dir


def mrn_distribution(
    diameter: np.ndarray | float, maxd: float, sigma: float
) -> np.ndarray:
    """
    The Modified Reciprocal Normal (MRN) track probability distribution function.

    This distribution function is a good fit for the diameter of tracks formed in CR-39 by
    a population of particles with an initially Gaussian energy distribution that have
    passed through some filtration prior to the CR-39.
    The sum of the distribution is normalized to 1.

    Parameters
    ----------
    diameter : np.ndarray|float
        Track diameter in um.
    maxd : float
        Most probable diameter of the track diameter distribution, in um.
    sigma : float
        Width parameter of the track diameter distribution, in um.

    Returns
    -------
    pdf : np.ndarray
        Probability for each track diameter.
    """

    dist = np.exp(-((maxd / diameter - 1) ** 2) / 2 / sigma**2)
    dist /= np.sum(dist)
    return dist


def single_diameter_overlap_fraction(
    Fnum: int,
    chi: np.ndarray | None = None,
    Fn: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Interpolate the overlap fraction curves from Monte-Carlo results.

    These curves are calculated for tracks with a single diameter, but :cite:t:`Zylstra2012new`
    shows that this distribution is applicable for Gaussian distributions of track diameters with
    modest standard deviations.

    If `chi` is provided, it will be used to interpolate the overlap fraction curve Fn(chi).
    If `Fn` is provided, it will be used to interpolate the chi value for the given overlap fraction, chi(Fn).

    :math:`\chi=\eta \pi \bar{D}^2` is defined in :cite:t:`Zylstra2012new` for fluence :math:`\eta` and
    average track diameter :math:`\bar{D}`.

    Fn is the track overlap curve for Fnum ``n``, where F1 is the number of tracks with no overlaps, F2 is the number of tracks with one overlap, etc.
    This definition is slightly different from the one in :cite:t:`Zylstra2012new`, where tracks in a cluster of three tracks,
    each of which overlap one of the others, are assigned F=3, rather than F=2.


    Parameters
    ----------
    Fnum : int
        Fnum determines which curve will be interpolated. Options are:
            - Fnum=1: Returns F1
            - Fnum=2: Returns F2
            - Fnum=3: Returns F3
            - Fnum=4: Returns F4+
            - Fnum=-1 (default): Returns 1-F1

    chi : np.ndarray, optional
        Normalized fluence :math:`\chi`. Provide this to interpolate the overlap fraction curve Fn(chi).

    Fn : np.ndarray, optional
        Overlap fraction. Provide this to interpolate the chi value for the given overlap fraction, chi(Fn).


    Returns
    -------
    fcurve : np.ndarray
        Overlap fraction curve Fn(chi) or chi(Fn), depending on the input provided.
    """
    if chi is None and Fn is None:
        raise ValueError("Either chi or Fn must be provided.")
    elif chi is not None and Fn is not None:
        raise ValueError("Only one of chi or Fn can be provided.")

    if Fn is not None and Fnum not in [-1, 1, 4]:
        raise ValueError(
            "Chi can only be interpolated from Fn for monotonic curves, which are Fnum=-1, 1, or 4."
        )

    file = data_dir / Path("cr39/F1-F4+.txt")
    arr = np.loadtxt(file, delimiter="\t", skiprows=1)
    _x = arr[:, 1]

    if Fnum == -1:
        _y = 1 - arr[:, 2]
    elif Fnum in [1, 2, 3, 4]:
        _y = arr[:, Fnum + 1]
    else:
        raise ValueError("Fnum must be one of [-1, 1, 2, 3, 4]")

    if chi is not None:
        curve = np.interp(chi, _x, _y)
    elif Fn is not None:
        # Make sure _y is sorted for interpolation
        if Fnum in [1]:
            _x, _y = np.flip(_x), np.flip(_y)
        curve = np.interp(Fn, _y, _x)

    return curve


def mrn_overlap_fraction(
    Fnum: int,
    fluence: np.ndarray,
    maxd: float,
    sigma: float,
) -> np.ndarray:
    r"""
    Interpolate the MRN overlap fraction curves from Monte-Carlo results.

    These curves are calculated for tracks with a the Modified Reciprocal Normal (MRN) track
    distribution, characterized by parameters ``maxd`` and ``sigma``.

    Fn is the track overlap curve for Fnum ``n``, where F1 is the number of tracks with no overlaps, F2 is the number of tracks with one overlap, etc.
    This definition is slightly different from the one in :cite:t:`Zylstra2012new`, where tracks in a cluster of three tracks,
    each of which overlap one of the others, are assigned F=3, rather than F=2.

    Parameters
    ----------
    Fnum : int
        Fnum determines which curve will be interpolated. Options are:
            - Fnum=1: Returns F1
            - Fnum=2: Returns F2
            - Fnum=3: Returns F3
            - Fnum=4: Returns F4+
            - Fnum=-1 (default): Returns 1-F1

    fluence : float|np.ndarray
       Track fluence in tracks/cm^2.

    maxd : float
        Maximum diameter of the track diameter distribution, in um.

    sigma : float
        Standard deviation of the track diameter distribution, in um.

    Returns
    -------
    fcurve : np.ndarray
        Overlap fraction curve Fn(fluence).
    """

    path = data_dir / Path("cr39/F1-F4+ special_distribution.h5")

    # Choose the index of the Fnum to interpolate
    # Do this first so we only load the part of the file we need
    if Fnum in [1, -1]:
        fi = 0
    elif Fnum in [2, 3, 4]:
        fi = Fnum - 1
    else:
        raise ValueError("Fnum must be one of [-1, 1, 2, 3, 4]")

    with h5py.File(path, "r") as f:
        data = f["data"][..., fi]
        dmax_arr = f["dmax_arr"][:]
        sigma_arr = f["sigma_arr"][:]
        track_densities = f["track_densities"][...]

    # If Fnum=-1, we want to return 1- F1
    if Fnum == -1:
        data = 1 - data

    # Cast Fluence as 1d even if its a float, for uniform output
    fluence = np.atleast_1d(fluence)

    # Assemble array of locations to sample grids at
    xi = np.array(
        [
            fluence,
            np.full(fill_value=maxd, shape=fluence.shape),
            np.full(fill_value=sigma, shape=fluence.shape),
        ]
    ).T

    Farr = interpn(
        (track_densities, dmax_arr, sigma_arr),
        data,
        xi,
        method="linear",
        bounds_error=True,
    )

    return Farr
