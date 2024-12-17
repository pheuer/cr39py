import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from cr39py.scan.base_scan import Scan


def _get_rough_alignment_from_fiducials(
    pre_scan: Scan, post_scan: Scan, plots: bool = True
) -> tuple[float]:
    """Roughly align two CR-39 scans based on fiducials recorded in the metadata.

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
    rot, dx, dy : tuple[float]
        rotation in rad, dx, dy shift in cm

    """
    # Load the metadata
    points = ["UL-FE", "UL-NE", "UR-FE", "UR-NE"]
    x = {}

    for i, point in enumerate(points):
        try:
            x1, y1 = pre_scan.metadata[point]
        except KeyError as e:
            raise KeyError(
                f"Pre-etch scan file missing required alignment point {point}"
            ) from e

        try:
            x2, y2 = post_scan.metadata[point]
        except KeyError as e:
            raise KeyError(
                f"Post-etch scan file missing required alignment point {point}"
            ) from e

        x[point] = np.array([x1, y1, x2, y2])

    # Calculate the vectors between each set of points in the pre and post scans
    pre_vecs = []
    post_vecs = []
    weights = []
    for p1 in points:
        for p2 in points:
            if p1 == p2:
                continue

            x1, y1, x2, y2 = x[p1]
            _x1, _y1, _x2, _y2 = x[p2]

            # Scipy function is for 3D vectors, so add a third component for Z
            pre_vec = np.array([_x1 - x1, _y1 - y1, 0])
            post_vec = np.array([_x2 - x2, _y2 - y2, 0])
            # Calculate the mean separation of the fiducial pair
            weight = np.mean(
                np.array([np.linalg.norm(pre_vec), np.linalg.norm(post_vec)])
            )

            weights.append(weight)
            pre_vecs.append(pre_vec)
            post_vecs.append(post_vec)

    # Find the rotation that best aligns the vectors
    # Weight the vectors by their lengths: farther separated points should
    # be given more weight
    rot, rssd, sens = Rotation.align_vectors(
        pre_vecs, post_vecs, weights=weights, return_sensitivity=True
    )
    rot = -np.arccos(rot.as_matrix()[0, 0])

    center = x["UL-FE"][:2]

    # Create a 2D rotation matrix and rotate the points accordingly
    rmatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    for point in points:
        p = x[point][2:] - center
        x[point][2:] = np.matmul(rmatrix, p) + center

    # Now calculate dx, dy for each set of points and select the translation
    # dx, dy are the vector FROM the post scan TO the pre scan, so that ADDING
    # them to the post-scan should reset the positions
    dx, dy = [], []
    for p in points:
        x1, y1, x2, y2 = x[p]
        dx.append(x1 - x2)
        dy.append(y1 - y2)

    dx = np.array(dx)
    dy = np.array(dy)
    theta = np.rad2deg(np.arctan2(dy, dx))
    shift = np.hypot(dx, dy)

    if plots:
        fig, ax = plt.subplots()
        ax.set_title("Alignment point movement")

        for i, p in enumerate(points):
            x1, y1, x2, y2 = x[p]
            ax.quiver(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                label=f"{p}: (rot, dx, dy): {np.rad2deg(rot):.3f} deg, {dx[i]*1e4:.1f} um, {dy[i]*1e4:.1f} um",
                color=f"C{i}",
            )

        ax.legend(loc="lower right")

    return center, rot, np.mean(dx), np.mean(dy)


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

    I. Lengar et al. NIMB 2002 https://doi.org/10.1016/S0168-583X(02)00483-4
    - Coincidence counting between two foils, just says two fiducials were used
    for alignment. Coincidence radius was 15 um


    """

    def makeplot(*args, w=400 * 1e-4, center=(0, 0), figax=None, labels=None):

        if figax is not None:
            fig, ax = figax
        else:
            fig, ax = plt.subplots()

        for i, tracks in enumerate(args):

            mask = (np.abs(tracks[:, 0] - center[0]) < w) * (
                np.abs(tracks[:, 1] - center[1]) < w
            )

            tracks = tracks[mask, :]

            if labels is not None:
                lbl = labels[i]
            else:
                lbl = None
            ax.scatter(tracks[:, 0], tracks[:, 1], s=15, label=lbl)

        ax.legend(loc="upper left")

    # Get the alignment shift
    center, rot, dx, dy = _get_rough_alignment_from_fiducials(
        pre_scan, post_scan, plots=plots
    )

    # Get the track data, rotate and translate the second scan
    _pre_tracks = np.copy(pre_scan._tracks[:, :2])
    _post_tracks = np.copy(post_scan._tracks[:, :2])

    fig_center = center - np.array([400, 400]) * 1e-4
    figax = plt.subplots()
    makeplot(_pre_tracks, center=fig_center, figax=figax, labels=["Pre"])

    rmatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    p = _post_tracks[:, :2] - center
    _post_tracks[:, :2] = np.matmul(rmatrix, p.T).T + center

    _post_tracks[:, 0] = _post_tracks[:, 0] + dx
    _post_tracks[:, 1] = _post_tracks[:, 1] + dy

    makeplot(_post_tracks, center=fig_center, figax=figax, labels=["Post shift"])

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

    # Loop through the frames

    # For each point in the frame, compare position to all tracks in that frame + the adjacent 8 frames
    # on the previous scan. Tag tracks as accept or reject in a boolean mask array (of the pre scan tracks)
    # based on their max distance from another track compared to the tolerance
