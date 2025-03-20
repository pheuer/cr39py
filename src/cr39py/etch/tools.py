"""The `~cr39py.etch.tools` module contains tools for calculating etch times."""

from pathlib import Path

import numpy as np

from cr39py.core.data import data_dir


def overlap_fraction(
    Fnum: int,
    chi: np.ndarray | None = None,
    Fn: np.ndarray | None = None,
) -> np.ndarray:
    """
    Interpolate the overlap fraction curves from Monte-Carlo results.

    If `chi` is provided, it will be used to interpolate the overlap fraction curve Fn(chi).
    If `Fn` is provided, it will be used to interpolate the chi value for the given overlap fraction, chi(Fn).

    :math:`$\chi=\eta \pi \bar{D}^2$` is defined in :cite:t:`Zylstra2012new` for fluence :math:`$\eta$` and
    average track diameter :math:`$\bar{D}$`.

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
        Normalized fluence :math:`$\chi$`. Provide this to interpolate the overlap fraction curve Fn(chi).

    Fn : np.ndarray, optional
        Overlap fraction. Provide this to interpolate the chi value for the given overlap fraction, chi(Fn).


    Returns
    -------
    fcurve : np.ndarray
        Overlap fraction curve Fn(chi) or chi(Fn), depending on the input provided.
    """
    if chi is None and F is None:
        raise ValueError("Either chi or F must be provided.")
    elif chi is not None and F is not None:
        raise ValueError("Only one of chi or F can be provided.")

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
        curve = np.interp(Fn, _y, _x)

    return curve


def goal_diameter(fluence, desired_overlap_percentage=5, max_goal=10):
    """
    Calculates the ideal track diameter in um to achieve a given overlap percentage at a given fluence on CR-39.

    Parameters
    ----------
    fluence : float
        The fluence on the CR-39 in 1/cm^2

    desired_overlap_percentage : float, optional
        The desired maximum percentage of overlapping tracks. Default is 5, in which
        case 5% of tracks will suffer some level of overlap with another track.

    max_goal : float, optional
        A maximum at which the goal track diameter will be clipped, in um.
        Prevents extremely large goal diameters being suggested at low fluences.

    Returns
    -------
    goal_diameter: float
        The goal track diameter in um.
    """
    goal_chi = overlap_fraction(Fnum=-1, Fn=desired_overlap_percentage * 1e-2)
    goal_d = np.sqrt(goal_chi / np.pi / (fluence * 1e-8))
    return np.clip(goal_d, a_min=0, a_max=max_goal)
