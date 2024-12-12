import matplotlib.pyplot as plt
import numpy as np

from cr39py.scan.base_scan import Scan


def _get_alignment_point_shift(pre_scan: Scan, post_scan: Scan, plots=True):
    """_summary_

    Parameters
    ----------
    pre_scan : `~cr39py.scan.base_scan.Scan`
        The scan prior to bulk etching.
    post_scan : `~cr39py.scan.base_scan.Scan`
        The scan after bulk etching.

    plots : bool, optional
        Whether or not to make a plot summarizing the shift of each
        alignment point


    Returns
    -------
    dx, dy : float, float
        mean dx, dy shift in cm

    """
    # Load the metadata
    points = ["UL-m", "UR-m", "UL-FE", "UL-NE", "UR-FE", "UR-NE"]
    x1, x2, y1, y2 = [], [], [], []
    dx, dy = [], []

    for i, point in enumerate(points):
        try:
            _x1, _y1 = pre_scan.metadata[point]
        except KeyError as e:
            raise KeyError(
                f"Pre-etch scan file missing required alignment point {point}"
            ) from e
        x1.append(_x1)
        y1.append(_y1)

        try:
            _x2, _y2 = post_scan.metadata[point]
        except KeyError as e:
            raise KeyError(
                f"Post-etch scan file missing required alignment point {point}"
            ) from e
        x2.append(_x2)
        y2.append(_y2)

        # Calculate the shift of the second scan relative to the first in um
        # Points are saved in cm
        dx.append((_x2 - _x1))
        dy.append((_y2 - _y1))

    dx = np.array(dx)
    dy = np.array(dy)
    theta = theta = np.rad2deg(np.arctan2(dy, dx))
    shift = np.hypot(dx, dy)

    if plots:
        fig, ax = plt.subplots()
        ax.set_title("Alignment point movement")

        for i, point in enumerate(points):
            ax.quiver(
                x1[i],
                y1[i],
                x2[i] - x1[i],
                y2[i] - y1[i],
                label=f"{point}: (theta, shift): {theta[i]:.1f} deg, {shift[i]*1e4:.1f} um",
                color=f"C{i}",
            )

        ax.legend(loc="lower right")

    # Remove any points falling more than a standard deviation from the median shift
    keep = np.array(
        [np.abs(s - np.median(shift)) < np.std(shift) for s in shift]
    ).astype(bool)
    nreject = np.sum(~keep)
    if np.sum(nreject) > 1:
        raise ValueError(
            "Rejecting an unacceptable number of alignment points "
            f"({nreject}/{dx.size})"
        )
    dx = dx[keep]
    dy = dy[keep]

    return np.mean(dx), np.mean(dy)


def coincident_tracks(pre_scan: Scan, post_scan: Scan, plots=True):
    """Finds coincident tracks between two scans.

    Used to de-noise data by finding coincident tracks
    between two scans of the same piece of CR-39 before
    and after a bulk etch.

    Alignment of the two scans is done using a series of alignment fiducials on the
    CR-39 whose locations are manually identified during the scan setup. These points
    are labeled

    - UL-NE (Upper left, near edge)
    - UL-FE (Upper left, far from edge)
    - UR-NE (Upper right, near edge)
    - UR-FE (Upper right, far from edge)

    This function requires that these four points be saved in the metadata of each
    `~cr39py.scan.base_scan.Scan` object, and will raise an exception if they are
    not found.

    Parameters
    ----------
    pre_scan : `~cr39py.scan.base_scan.Scan`
        The scan prior to bulk etching.
    post_scan : `~cr39py.scan.base_scan.Scan`
        The scan after bulk etching.

    Returns
    -------

    coincident_scan : `~cr39py.scan.base_scan.Scan`
        A `~cr39py.scan.base_scan.scan` object with only
        the coincident tracks.


    Raises
    ------
    ValueError
        If the four alignment points are not found in one of the
        scan files metadata.

    ValueError
        If more than one point alignment point is rejected for being
        and outlier.


    """

    # Get the alignment shift
    dx, dy = _get_alignment_point_shift(pre_scan, post_scan, plots=plots)

    pre_tracks = pre_scan._tracks
    post_tracks = post_scan._tracks

    # Shift the tracks in the post scan based on the alignment points
    post_tracks[:, 0] = post_tracks[:, 0] - dx
    post_tracks[:, 1] = post_tracks[:, 1] - dy
