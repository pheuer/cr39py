from pathlib import Path

import h5py
import pytest

from cr39py.cut import Cut


def test_save_and_load(tmpdir):

    cut = Cut(xmin=-1, xmax=2)

    tmp_path = Path(tmpdir) / Path("tmp_file.h5")

    cut.to_hdf5(tmp_path)

    cut2 = Cut.from_hdf5(tmp_path)

    assert cut2 == cut


def test_properties_cut():
    cut = Cut(xmin=-1, xmax=2)
    cut.xrange
    cut.yrange
    cut.drange
    cut.erange
    cut.crange

    cut.xmin
    cut.dmax

    str(cut)

    cut2 = Cut(xmin=-1, xmax=2)
    cut3 = Cut(xmin=-1, xmax=2, cmin=20)

    assert cut == cut2
    assert cut != cut3


def test_update_cut():
    cut = Cut(xmin=-1, xmax=2)
    cut.update(cmin=10)
