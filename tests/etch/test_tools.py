"""_
This file contains tests for the etch/tools.py file.
"""

import numpy as np
import pytest

from cr39py.etch.tools import goal_diameter, overlap_fraction

cases = [
    # Moderate/high fluence with a known value
    (1e5, 5, 20, 4.06),
    # Extremely low fluence and high overlap percentage to test max goal diameter
    (1e2, 50, 5, 5),
]


@pytest.mark.parametrize(
    "fluence, desired_overlap_percentage, max_goal, expected", cases
)
def test_goal_diameter(fluence, desired_overlap_percentage, max_goal, expected):
    assert np.isclose(
        goal_diameter(fluence, desired_overlap_percentage, max_goal),
        expected,
        rtol=0.03,
    )


@pytest.mark.parametrize("Fnum", [1, 2, 3, 4, -1])
def test_overlap_fraction_Fnum(Fnum):
    # Check self-consistency by running the function both ways for each Fn value
    chi = 1
    _Fn = overlap_fraction(Fnum=Fnum, chi=chi)

    if Fnum in [1, -1, 4]:
        _chi = overlap_fraction(Fnum=Fnum, Fn=_Fn)
        assert np.isclose(chi, _chi, rtol=0.05)
    else:
        # F2 and F3 are not monotonic, so chi(Fn) is not defined
        with pytest.raises(ValueError):
            _chi = overlap_fraction(Fnum=Fnum, Fn=_Fn)


def test_overlap_fraction_inputs():

    # Running with one arg is no problem
    overlap_fraction(1, Fn=0.5)
    overlap_fraction(1, chi=0.5)

    # Invalid Fnum
    with pytest.raises(ValueError):
        overlap_fraction(12, Fn=0.5)

    # Both args
    with pytest.raises(ValueError):
        overlap_fraction(1, Fn=0.5, chi=0.5)

    # Neither arg
    with pytest.raises(ValueError):
        overlap_fraction(1)
