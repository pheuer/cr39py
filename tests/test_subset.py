from pathlib import Path

import pytest

from cr39py.cut import Cut
from cr39py.subset import Subset


def test_save_and_load(tmpdir):
    subset = Subset()
    subset.cuts.append(Cut(xmin=-1, ymin=-2))

    tmp_path = Path(tmpdir) / Path("tmp_file.h5")

    subset.to_hdf5(tmp_path)

    subset2 = Subset.from_hdf5(tmp_path)

    assert subset == subset2


def test_dslices():
    subset = Subset()
    with pytest.raises(ValueError):
        subset.set_ndslices("N")

    subset.set_ndslices(5)

    with pytest.raises(ValueError):
        subset.select_dslice(24)

    subset.select_dslice(2)


def test_str_subset():
    str(Subset())

    subset = Subset()
    subset.add_cut(Cut(xmin=0))
    str(subset)


def test_equality_subset():
    subset1 = Subset()
    subset1.add_cut(Cut(xmin=0))

    subset2 = Subset()
    subset2.add_cut(Cut(xmin=0))

    subset3 = Subset()
    subset3.add_cut(Cut(xmin=0))
    subset3.add_cut(cmax=20)

    assert subset1 == subset2
    assert subset1 != subset3


def test_add_remove_replace_cut():

    subset = Subset()
    subset.add_cut(Cut(xmin=0))
    subset.add_cut(cmax=20)
    subset.add_cut(dmin=5)

    subset.remove_cut(2)

    subset.replace_cut(0, Cut(xmin=5))
