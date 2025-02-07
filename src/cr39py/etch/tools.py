"""The `~cr39py.etch.tools` module contains tools for calculating etch times."""

import numpy as np


def goal_diameter(fluence, desired_F2=0.05, max_goal=10):
    """Calculates the ideal track diameter in um to achieve a given overlap parameter F2.

    Parameters
    ----------
    fluence : float
        The fluence on the CR-39 in 1/cm^2

    desired_F2 : float, optional
        The desired track overlap parameter F2. The default
        value is 0.05, meaning that ~5% of tracks will suffer overlap with
        a neighbor.

    max_goal : float, optional
        A maximum at which the goal track diameter will be clipped, in um.
        Prevents extremely large goal diameters being suggested at low fluences.

    Returns
    -------
    goal_diameter: float
        The goal track diameter in um.
    """

    def chi(F2):
        """
        Zylstra Eq. 9 inverted to solve for chi given F2
        Works up to F2=0.375, then the sqrt becomes imaginary.
        """
        return 3 / 4 * (1 - np.sqrt(1 - 8 / 3 * F2))

    if desired_F2 > 0.3:
        raise ValueError(f"Model breaks down for F2~>0.3, provided F2={desired_F2}")

    goal_chi = chi(desired_F2)
    goal_d = np.sqrt(goal_chi / np.pi / (fluence * 1e-8))
    return np.clip(goal_d, a_min=0, a_max=max_goal)
