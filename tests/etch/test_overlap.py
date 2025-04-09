"""_
This file contains tests for the etch/overlap.py file.
"""

import numpy as np
import pytest

from cr39py.etch.overlap import mrn_overlap_fraction, single_diameter_overlap_fraction


@pytest.mark.parametrize("Fnum", [1, 2, 3, 4, -1])
def test_single_diameter_overlap_fraction_Fnum(Fnum):
    # Check self-consistency by running the function both ways for each Fn value
    chi = 1
    _Fn = single_diameter_overlap_fraction(Fnum=Fnum, chi=chi)

    if Fnum in [1, -1, 4]:
        _chi = single_diameter_overlap_fraction(Fnum=Fnum, Fn=_Fn)
        assert np.isclose(chi, _chi, rtol=0.05)
    else:
        # F2 and F3 are not monotonic, so chi(Fn) is not defined
        with pytest.raises(ValueError):
            _chi = single_diameter_overlap_fraction(Fnum=Fnum, Fn=_Fn)


def test_single_diameter_overlap_fraction_inputs():

    # Running with one arg is no problem
    single_diameter_overlap_fraction(1, Fn=0.5)
    single_diameter_overlap_fraction(1, chi=0.5)

    # Invalid Fnum: not on the list
    with pytest.raises(ValueError):
        single_diameter_overlap_fraction(12, chi=1)

    # Invalid Fnum to interpolate Chi(Fn) from
    with pytest.raises(ValueError):
        single_diameter_overlap_fraction(2, Fn=0.5)

    # Both args
    with pytest.raises(ValueError):
        single_diameter_overlap_fraction(1, Fn=0.5, chi=0.5)

    # Neither arg
    with pytest.raises(ValueError):
        single_diameter_overlap_fraction(1)


def test_mrn_overlap_fraction_inputs():

    # Running with one arg is no problem
    x = mrn_overlap_fraction(1, 1e5, 4, 0.3)
    assert x.shape == (1,)

    # Run with a vector of multiple fluences
    x = mrn_overlap_fraction(1, np.array([1e4, 1e5]), 4, 0.3)
    assert x.shape == (2,)

    # Run all the acceptable Fn's
    for i in [-1, 1, 2, 3, 4]:
        mrn_overlap_fraction(i, 1e5, 4, 0.3)

    # Invalid Fnum: not on the list
    with pytest.raises(ValueError):
        mrn_overlap_fraction(12, 1e5, 4, 0.3)
