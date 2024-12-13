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
    points = ["UL-M", "UR-M", "UL-FE", "UL-NE", "UR-FE", "UR-NE"]
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

    # Try fine-tuning the correction by looking at tracks within some distance of the alignment point

    point = 4

    center = (x1[point], y1[point])

    dx = 1000 * 1e-4  # cm
    _pre_tracks = np.copy(pre_scan._selected_tracks)
    _pre_tracks[:, 0] -= x1[point]
    _pre_tracks[:, 1] -= y1[point]
    pre_mask = (np.abs(_pre_tracks[:, 0]) < dx) * (np.abs(_pre_tracks[:, 1]) < dx)
    _pre_tracks = _pre_tracks[pre_mask, :2]

    dx = 600 * 1e-4
    _post_tracks = np.copy(post_scan._selected_tracks)
    _post_tracks[:, 0] -= x2[point]
    _post_tracks[:, 1] -= y2[point]
    post_mask = (np.abs(_post_tracks[:, 0]) < dx) * (np.abs(_post_tracks[:, 1]) < dx)
    _post_tracks = _post_tracks[post_mask, :2]

    fig, ax = plt.subplots()
    ax.set_title(points[point])
    ax.scatter(0, 0, marker="*", s=50)
    ax.scatter(_pre_tracks[:, 0] * 1e4, _pre_tracks[:, 1] * 1e4, s=25)
    ax.scatter(_post_tracks[:, 0] * 1e4, _post_tracks[:, 1] * 1e4, s=5)

    def matches_between_sets_of_points(a, b, tol=10e-4):
        """
        a [n,2]
        b [m, 2]

        returns
        m : int
            Matches between the two sets of points, within tol
        """
        na = a.shape[0]
        match = 0
        for i in range(na):
            dist = np.min(np.hypot(a[i, 0] - b[:, 0], a[i, 1] - b[:, 1]))
            if dist < tol:
                match += 1

        return match

    print(_pre_tracks.shape, _post_tracks.shape)

    npre = _pre_tracks.shape[0]
    npost = _post_tracks.shape[0]
    matches = []
    shifts = []

    p = 0
    for i in range(npre):
        # Shift the post tracks so that
        xshift, yshift = _post_tracks[p, :] - _pre_tracks[i, :]
        shifts.append((xshift, yshift))
        post_shifted = _post_tracks - np.array([xshift, yshift])
        m = matches_between_sets_of_points(post_shifted, _pre_tracks)
        matches.append(m)

    print(np.max(matches))
    xshift, yshift = shifts[np.argmax(matches)]

    print(xshift * 1e4, yshift * 1e4)

    fig, ax = plt.subplots()
    ax.set_title(points[point])
    ax.scatter(0, 0, marker="*", s=50)
    ax.scatter(_pre_tracks[:, 0] * 1e4, _pre_tracks[:, 1] * 1e4, s=25)
    ax.scatter(
        _post_tracks[:, 0] * 1e4 - xshift * 1e4,
        _post_tracks[:, 1] * 1e4 - yshift * 1e4,
        s=10,
    )

    """
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
    """
    return 0, 0
    # return np.mean(dx), np.mean(dy)


def coincident_tracks(
    pre_scan: Scan,
    post_scan: Scan,
    tolerance=5,
    plots=True,
):
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

    tolerance : float
        Tolerance required to count a track as being coincident. In um.

    plots: bool, optional
        Make summary plots. Defaults to True.

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


    Notes
    -----

    Papers on coincidence counting CR-39 tracks

    D. T. Casey et al. RSI 2011 https://doi.org/10.1063/1.3605483
    B. Lahmann et al. RSI 2016 https://doi.org/10.1063/1.4958910


    """

    # Get the alignment shift
    dx, dy = _get_alignment_point_shift(pre_scan, post_scan, plots=plots)

    pre_tracks = pre_scan._tracks
    post_tracks = post_scan._tracks

    # Shift the tracks in the post scan based on the alignment points
    post_tracks[:, 0] = post_tracks[:, 0] - dx
    post_tracks[:, 1] = post_tracks[:, 1] - dy

    # Make sure both scans framesizes are set to the actual microscope
    # framesize of the first scan
    pre_scan.set_framesize("XY", pre_scan.metadata["frame_size_x"] * 1e-4)
    post_scan.set_framesize("XY", pre_scan.metadata["frame_size_x"] * 1e-4)

    """

    X, Y = pre_scan._axes["X"].axis(pre_tracks, units=False), pre_scan._axes["Y"].axis(
        pre_tracks, units=False
    )
    for i in range(X.size):
        for j in range(Y.size):
            # Select tracks in or near this frame
            _pre_mask = (
                (pre_tracks[:, 0] > X[i])
                * (pre_tracks[:, 0] < X[i + 1])
                * (pre_tracks[:, 1] > Y[j])
                * (pre_tracks[:, 1] < Y[j + 1])
            )
            _pre_tracks = pre_tracks[_pre_mask, :2]

            # Select tracks in or near this frame
            _post_mask = (
                (post_tracks[:, 0] > X[i])
                * (post_tracks[:, 0] < X[i + 1])
                * (post_tracks[:, 1] > Y[j])
                * (post_tracks[:, 1] < Y[j + 1])
            )
            _post_tracks = post_tracks[_post_mask, :2]

            if j == 12:
                fig, ax = plt.subplots()
                ax.scatter(_pre_tracks[:, 0], _pre_tracks[:, 1])
                ax.scatter(_post_tracks[:, 0], _post_tracks[:, 1])
                plt.show()

                raise ValueError

    """

    # Loop through the frames

    # For each point in the frame, compare position to all tracks in that frame + the adjacent 8 frames
    # on the previous scan. Tag tracks as accept or reject in a boolean mask array (of the pre scan tracks)
    # based on their max distance from another track compared to the tolerance
