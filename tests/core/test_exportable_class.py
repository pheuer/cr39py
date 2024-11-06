from pathlib import Path

import numpy as np
import pytest

from cr39py.core.exportable_class import ExportableClassMixin, saveable_class
from cr39py.core.units import unit_registry as u


@saveable_class(attributes=["a", "b"])
class AnObj(ExportableClassMixin):
    def __init__(self):
        self.a = 1
        self.b = "text"


@pytest.mark.parametrize(
    "x",
    [
        None,
        1,
        1.2,
        True,
        "2",
        np.arange(5).astype(np.int32),
        np.random.random((3, 3)).astype(np.float32),
        # Test numpy array with compression, turns on automatically
        # for larger arrays
        np.random.random((20, 20)).astype(np.float16),
        u.Quantity(2, u.cm),
        u.Measurement(2, 1, u.cm),
        AnObj(),
        # Dict
        {"a": 1, 1: "text", "2": True},
        # list
        [1, 2, "a", "blue"],
        # tuple
        (1, 2),
        # Dict of numpy arrays
        {"X": np.arange(1), "Y": np.random.random(1)},
    ],
)
def test_export_and_import(x, tmpdir):

    tmp_path = Path(tmpdir) / Path("tmp.h5")

    @saveable_class(attributes=["test"])
    class AClass(ExportableClassMixin):
        def __init__(self):
            self.test = x

    obj = AClass()

    obj.to_hdf5(tmp_path)

    obj2 = AClass.from_hdf5(tmp_path)

    if isinstance(x, ExportableClassMixin):
        for attr in x._exportable_attributes:
            assert getattr(x, attr) == getattr(obj2.test, attr)

    elif isinstance(x, u.Measurement):
        assert x.m.n == obj2.test.m.n
        assert x.m.s == obj2.test.m.s
        assert x.u == obj2.test.u

    elif isinstance(x, np.ndarray):
        assert np.allclose(obj2.test, x)

    else:
        assert obj2.test == x


def test_attribute_inheritence():

    @saveable_class(attributes=["a", "d"])
    class A(ExportableClassMixin):
        _exportable_attributes = ["x"]

        # attribute added in `saveable_class`
        a = "a"
        # attribute added in `_exportable_attributes`
        x = "x"

        # Test saving a property
        @property
        def z(self):
            return "z"

    # Class that inherits from A but also has its own attributes
    @saveable_class(attributes=["b"])
    class B(A, ExportableClassMixin):
        b = "b"

    @saveable_class(attributes=["c"])
    class C(B, ExportableClassMixin):
        c = "c"

    # Make an instance of the class
    c = C()

    # Check that all the attributes exist on the child class, and that
    # they have the expected values
    for attr in "abcxz":
        assert hasattr(c, attr)
        assert getattr(c, attr) == attr
