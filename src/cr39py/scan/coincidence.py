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

    x1, y1, x2, y2 = x["UL-NE"]
    _x1, _y1, _x2, _y2 = x["UR-NE"]

    # Scipy function is for 3D vectors, so add a third component for Z
    pre_vec = np.array([_x1 - x1, _y1 - y1, 0])
    post_vec = np.array([_x2 - x2, _y2 - y2, 0])

    # Find the rotation that best aligns the vectors
    # Weight the vectors by their lengths: farther separated points should
    # be given more weight
    rot, rssd = Rotation.align_vectors(pre_vec, post_vec)
    rot = -np.arccos(rot.as_matrix()[0, 0])

    center = x["UL-NE"][:2]

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

    dx = x["UL-NE"][0] - x["UL-NE"][2]
    dy = x["UL-NE"][1] - x["UL-NE"][3]

    return center, rot, dx, dy


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
    center, rot, dx, dy = _get_rough_alignment_from_fiducials(
        pre_scan, post_scan, plots=plots
    )

    # Get the track data, rotate and translate the second scan
    _pre_tracks = np.copy(pre_scan._tracks[:, :2])
    _post_tracks = np.copy(post_scan._tracks[:, :2])

    rmatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    p = _post_tracks[:, :2] - center
    _post_tracks[:, :2] = np.matmul(rmatrix, p.T).T + center

    _post_tracks[:, 0] = _post_tracks[:, 0] + dx
    _post_tracks[:, 1] = _post_tracks[:, 1] + dy

    w = 500 * 1e-4
    pre_mask = (np.abs(_pre_tracks[:, 0] - center[0]) < w) * (
        np.abs(_pre_tracks[:, 1] - center[1]) < w
    )
    w = 500 * 1e-4
    post_mask = (np.abs(_post_tracks[:, 0] - center[0]) < w) * (
        np.abs(_post_tracks[:, 1] - center[1]) < w
    )

    _pre_tracks = _pre_tracks[pre_mask, :]
    _post_tracks = _post_tracks[post_mask, :]

    fig, ax = plt.subplots()
    ax.scatter(_pre_tracks[:, 0], _pre_tracks[:, 1], s=15)
    ax.scatter(_post_tracks[:, 0], _post_tracks[:, 1], s=10)

    # Loop through the frames

    # For each point in the frame, compare position to all tracks in that frame + the adjacent 8 frames
    # on the previous scan. Tag tracks as accept or reject in a boolean mask array (of the pre scan tracks)
    # based on their max distance from another track compared to the tolerance
