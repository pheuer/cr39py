from pathlib import Path

import numpy as np
import pytest

from cr39py.scan.cut import Cut
from cr39py.scan.subset import Subset


def test_save_and_load(tmpdir):
    subset = Subset()
    subset.cuts.append(Cut(xmin=-1, ymin=-2))

    tmp_path = Path(tmpdir) / Path("tmp_file.h5")

    subset.to_hdf5(tmp_path)

    subset2 = Subset.from_hdf5(tmp_path)

    assert subset == subset2


def test_set_domain():
    d = Cut(xmin=0)
    subset = Subset(domain=d)
    d2 = Cut(xmin=2)
    subset.set_domain(d2)

    # Empty cut
    subset.set_domain()

    # Create cut from keywords
    subset.set_domain(xmin=0, xmax=10)


def test_dslices():
    subset = Subset()
    with pytest.raises(ValueError):
        subset.set_ndslices("N")

    subset = Subset(ndslices=3)

    subset.set_ndslices(5)

    # Cannot be more than number of dslices
    with pytest.raises(ValueError):
        subset.select_dslice(24)

    subset.select_dslice(4)

    # Should now select last dslice
    subset.set_ndslices(2)
    assert subset.current_dslice_index == 1


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

    # Generate a hash explicitly
    hash(subset2)

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

    # Cannot replace a cut that does not exist
    with pytest.raises(ValueError):
        subset.replace_cut(1000, Cut(xmin=5))


def test_apply_cuts():

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

    # With no cuts applied, all tracks are selected
    subset = Subset()
    sel_tracks = subset.apply_cuts(tracks)
    assert sel_tracks.shape[0] == tracks.shape[0]

    subset = Subset()
    subset.add_cut(cmin=30)
    subset.add_cut(dmin=10)
    sel_tracks = subset.apply_cuts(tracks)
    assert sel_tracks.shape[0] == np.sum((tracks[:, 3] < 30) & (tracks[:, 2] < 10))

    # Try using only the first cut
    sel_tracks = subset.apply_cuts(tracks, use_cuts=[0])
    assert sel_tracks.shape[0] == np.sum((tracks[:, 3] < 30))

    # Try inverting the cuts
    sel_tracks = subset.apply_cuts(tracks, invert=True)
    assert sel_tracks.shape[0] == np.sum((tracks[:, 3] > 30) | (tracks[:, 2] > 10))

    # Test with a domain
    subset = Subset()
    subset.set_domain(xmin=-2.5, xmax=2.5)

    sel_tracks = subset.apply_cuts(tracks)
    assert sel_tracks.shape[0] == np.sum(np.abs(tracks[:, 0]) < 2.5)

    # Test with a domain and a cut
    subset = Subset()
    subset.set_domain(xmin=-2.5, xmax=2.5)
    subset.add_cut(cmin=30)

    sel_tracks = subset.apply_cuts(tracks)
    assert sel_tracks.shape[0] == np.sum(
        (np.abs(tracks[:, 0]) < 2.5) & (tracks[:, 3] < 30)
    )

    # Now inverting should only invert the cut, not the domain
    sel_tracks = subset.apply_cuts(tracks, invert=True)
    assert sel_tracks.shape[0] == np.sum(
        (np.abs(tracks[:, 0]) < 2.5) & (tracks[:, 3] > 30)
    )
