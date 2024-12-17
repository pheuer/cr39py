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
    rot, rssd = Rotation.align_vectors(pre_vecs, post_vecs, weights=weights)
    rot = -np.arccos(rot.as_matrix()[0, 0])
    print(np.rad2deg(rot))

    # Create a 2D rotation matrix and rotate the points accordingly
    rmatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    for point in points:
        x[point][2:] = np.matmul(rmatrix, x[point][2:])

    print(x)

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

    return rot, np.mean(dx), np.mean(dy)


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
    rot, dx, dy = _get_rough_alignment_from_fiducials(pre_scan, post_scan, plots=plots)

    # Get the track data, rotate and translate the second scan

    rmatrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

    _pre_tracks = np.copy(pre_scan._tracks[:, :2])
    _post_tracks = np.copy(post_scan._tracks[:, :2])

    _post_tracks[:, :2] = np.matmul(rmatrix, _post_tracks[:, :2].T).T
    _post_tracks[:, 0] = _post_tracks[:, 0] + dx
    _post_tracks[:, 1] = _post_tracks[:, 1] + dy

    center = (1, 2)
    w = 300 * 1e-4
    pre_mask = (np.abs(_pre_tracks[:, 0] - center[0]) < w) * (
        np.abs(_pre_tracks[:, 1] - center[1]) < w
    )
    w = 200 * 1e-4
    post_mask = (np.abs(_post_tracks[:, 0] - center[0]) < w) * (
        np.abs(_post_tracks[:, 1] - center[1]) < w
    )

    _pre_tracks = _pre_tracks[pre_mask, :]
    _post_tracks = _post_tracks[post_mask, :]

    fig, ax = plt.subplots()
    ax.scatter(_pre_tracks[:, 0], _pre_tracks[:, 1], s=25)
    ax.scatter(_post_tracks[:, 0], _post_tracks[:, 1], s=10)

    pre_points = [list(_pre_tracks[i, :]) for i in range(_pre_tracks.shape[0])]
    post_points = [list(_post_tracks[i, :]) for i in range(_post_tracks.shape[0])]

    import itertools

    pre_combs = list(itertools.combinations(range(len(pre_points)), 3))
    post_combs = list(itertools.combinations(range(len(post_points)), 3))

    # https://stackoverflow.com/questions/43126580/match-set-of-x-y-points-to-another-set-that-is-scaled-rotated-translated-and
    def get_triangles(points, combinations):
        triangles = []
        for p0, p1, p2 in combinations:
            d1 = np.hypot(points[p0][0] - points[p1][0], points[p0][1] - points[p1][1])
            d2 = np.hypot(points[p0][0] - points[p2][0], points[p0][1] - points[p2][1])
            d3 = np.hypot(points[p1][0] - points[p2][0], points[p1][1] - points[p2][1])
            d_min = min(d1, d2, d3)
            d_unsort = [d1 / d_min, d2 / d_min, d3 / d_min]
            triangles.append(sorted(d_unsort))
        return triangles

    pre_triangles = get_triangles(pre_points, pre_combs)
    post_triangles = get_triangles(post_points, post_combs)

    def sumTriangles(A_triang, B_triang):
        tr_sum, tr_idx = [], []
        for i, A_tr in enumerate(A_triang):
            for j, B_tr in enumerate(B_triang):
                # Absolute value of lengths differences.
                tr_diff = abs(np.array(A_tr) - np.array(B_tr))
                # Sum the differences
                tr_sum.append(sum(tr_diff))
                tr_idx.append([i, j])

        # Index of the triangles in A and B with the smallest sum of absolute
        # length differences.
        tr_idx_min = tr_idx[tr_sum.index(min(tr_sum))]
        A_idx, B_idx = tr_idx_min[0], tr_idx_min[1]
        print("Smallest difference: {}".format(min(tr_sum)))

        return A_idx, B_idx

    # Index of the A and B triangles with the smallest difference.
    A_idx, B_idx = sumTriangles(pre_triangles, post_triangles)

    # Indexes of points in A and B of the best match triangles.
    A_idx_pts, B_idx_pts = pre_combs[A_idx], post_combs[B_idx]
    print(f"triangle A {A_idx_pts} matches triangle B {B_idx_pts}")

    # Loop through the frames

    # For each point in the frame, compare position to all tracks in that frame + the adjacent 8 frames
    # on the previous scan. Tag tracks as accept or reject in a boolean mask array (of the pre scan tracks)
    # based on their max distance from another track compared to the tolerance
