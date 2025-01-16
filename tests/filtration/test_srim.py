from pathlib import Path

import numpy as np
import pytest

from cr39py.core.data import data_dir
from cr39py.filtration.srim import SRIMData


def test_load_srim_from_file():
    file = data_dir / Path("srim/Proton in Al.txt")
    srim = SRIMData.from_file(file)
    assert srim.particle == "proton"
    assert srim.material == "al"
    assert np.isclose(srim.ion_energy[0], 999.999)


entries = [
    ("Proton", "Al", 2.551e1),
    ("proton", "AL", 2.551e1),
    ("TRITON", "Ta", 4.866e1),
]


@pytest.mark.parametrize("particle,material,dEdxelectronic0", entries)
def test_load_srim_from_strings(particle, material, dEdxelectronic0):
    srim = SRIMData.from_strings(particle, material)
    assert srim.particle == particle.lower()
    assert srim.material == material.lower()
    assert np.isclose(srim.dEdx_electronic[0], dEdxelectronic0)


def test_srimdata_attributes():
    attrs = [
        ("particle", str),
        ("material", str),
        ("ion_energy", np.ndarray),
        ("dEdx_total", np.ndarray),
        ("dEdx_electronic", np.ndarray),
        ("dEdx_nuclear", np.ndarray),
        ("projected_range", np.ndarray),
        ("longitudinal_straggling", np.ndarray),
        ("lateral_straggling", np.ndarray),
    ]
    file = data_dir / Path("srim/Proton in Al.txt")
    srim = SRIMData.from_file(file)

    for attr, type in attrs:
        assert hasattr(srim, attr)
        assert isinstance(getattr(srim, attr), type)
