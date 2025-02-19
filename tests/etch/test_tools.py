"""_
This file contains tests for the etch/tools.py file.
"""

import numpy as np
import pytest

from cr39py.etch.tools import goal_diameter

cases = [
    (1e5, 0.05, 20, 4.06),
    # Extremely low fluence and high F2 to test max goal diameter
    (1e2, 0.2, 5, 5),
]


@pytest.mark.parametrize("fluence, desired_F2, max_goal, expected", cases)
def test_goal_diameter(fluence, desired_F2, max_goal, expected):
    assert np.isclose(goal_diameter(fluence, desired_F2, max_goal), expected, rtol=0.03)


def test_goal_diameter_raises():
    """
    Raise an exception if F2 > 0.3
    """
    with pytest.raises(ValueError):
        goal_diameter(1e5, 0.4, 20)
