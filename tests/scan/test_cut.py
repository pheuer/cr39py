from pathlib import Path

import h5py
import numpy as np
import pytest

from cr39py.scan.cut import Cut


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

    with pytest.raises(KeyError):
        cut.update(not_a_valid_cut_key=10)

    cut.update(cmin="none")
    assert cut.bounds["cmin"] is None

    cut.update(cmin=10)
    cut.update(cmin=None)
    assert cut.bounds["cmin"] is None


def test_cut_test():
    # Each track is X,Y,D,C,E,Z
    # Create a tracks array with random uniform values in the ranges
    # X,Y = [-5,5]
    # D = [0,10]
    # C = [0,100]
    # E = [0,1]
    # Z = [0,1000]
    ntracks = 200
    tracks = np.zeros((ntracks, 6))
    tracks[:, :2] = np.random.uniform(low=-5, high=5, size=(ntracks, 2))
    tracks[:, 2] = np.random.uniform(low=0, high=10, size=ntracks)
    tracks[:, 3] = np.random.uniform(low=0, high=100, size=ntracks)
    tracks[:, 4] = np.random.uniform(low=0, high=1, size=ntracks)
    tracks[:, 5] = np.random.uniform(low=0, high=1000, size=ntracks)

    c = Cut()
    assert np.sum(c.test(tracks)) == ntracks

    c = Cut(cmin=30)
    assert np.sum(c.test(tracks)) == np.sum(tracks[:, 3] > 30)

    c = Cut(cmin=30, dmax=2)
    assert np.sum(c.test(tracks)) == np.sum((tracks[:, 3] > 30) & (tracks[:, 2] < 2))
