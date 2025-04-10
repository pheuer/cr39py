import pytest

from cr39py.core.ci import SilentPlotting
from cr39py.core.units import u
from cr39py.filtration.layer import Layer
from cr39py.filtration.stack import Stack

# Create a list of objects that should all act
# as LayyerLike subclasses
l1 = Layer.from_string("100 um Al")
s1 = Stack.from_string("15 um Ta, 180 um Al")
test_objects = [l1, s1]


@pytest.mark.parametrize("obj", test_objects)
def test_reduced_ranging_model(obj):

    with SilentPlotting():
        coeffs, model = obj.reduced_ranging_model(plot=True)

    assert len(coeffs) == 3

    # Test model works with units as expected
    eout = model(15 * u.MeV)
    assert isinstance(eout, u.Quantity)
    assert eout.to(u.MeV).u == u.MeV
