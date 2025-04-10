import numpy as np
import pytest

from cr39py.core.ci import SilentPlotting
from cr39py.etch.track_overlap_mc import MonteCarloTrackOverlap


def test_plot_tracks():
    mc = MonteCarloTrackOverlap(random_seed=42)
    xyd = mc.draw_tracks(10)
    with SilentPlotting():
        mc.plot_tracks(xyd)


def test_run_monte_carlo():
    mc = MonteCarloTrackOverlap(framesize=300, random_seed=42)
    assert mc.frame_area == 300**2
    track_densities = np.array([1e3])
    Farr = mc.run_curve(track_densities, 2)


def test_different_diameter_distributions():

    daxis = np.arange(5, 10, 0.25)
    diameter_distribution = np.ones(daxis.size)
    diameter_distribution /= np.sum(diameter_distribution)

    mc = MonteCarloTrackOverlap(
        daxis=daxis,
        diameter_distribution=diameter_distribution,
        random_seed=42,
    )
    xyd = mc.draw_tracks(5000)
    hist, _ = np.histogram(xyd[:, 2], bins=5)
    print(hist)
    # Uniform distribution means the histogram should be about flat
    assert np.allclose(hist, np.mean(hist), rtol=0.1)


def test_Gaussian_diameter_distributions():

    diameters_mean, diameters_std = 8, 1
    mc = MonteCarloTrackOverlap(
        diameters_mean=diameters_mean, diameters_std=diameters_std, random_seed=42
    )
    xyd = mc.draw_tracks(500)
    assert np.isclose(np.mean(xyd[:, 2]), diameters_mean, rtol=0.05)
    assert np.isclose(np.std(xyd[:, 2]), diameters_std, rtol=0.1)
