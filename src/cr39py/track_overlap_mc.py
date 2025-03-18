from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.patches import Circle, Rectangle

rng = np.random.default_rng()


class MonteCarloTrackOverlap:
    def __init__(self):
        # All dimensions in um
        self.xy_range = (-300, 300)
        self.border = 25
        self.diameters = np.arange(0.5, 20, 0.025)
        self.diameters_mean = 5
        self.diameters_std = 2

    @property
    def frame_area(self):
        """
        Frame area in um**2
        """
        return (self.xy_range[1] - self.xy_range[0]) ** 2

    @property
    def frame_area_with_border(self):
        """
        Frame area in um**2
        This is includes the border, and is the area over which tracks are actually drawn.
        """
        return (self.xy_range[1] - self.xy_range[0] + 2 * self.border) ** 2

    @property
    def diameter_dist(self):
        dist = np.exp(
            -((self.diameters - self.diameters_mean) ** 2) / 2 / self.diameters_std**2
        )
        return dist / np.sum(dist)

    def draw_tracks(
        self,
        ntracks,
    ):
        xyd = np.empty((ntracks, 3))

        # Draw spatially uniform positions in the plane
        xyd[:, 0] = rng.uniform(
            low=self.xy_range[0] - self.border,
            high=self.xy_range[1] + self.border,
            size=ntracks,
        )
        xyd[:, 1] = rng.uniform(
            low=self.xy_range[0] - self.border,
            high=self.xy_range[1] + self.border,
            size=ntracks,
        )

        # Draw diameters from the diameter distribution
        if self.diameters_std == 0:
            xyd[:, 2] = self.diameters_mean
        else:
            xyd[:, 2] = rng.choice(
                self.diameters, size=ntracks, replace=True, p=self.diameter_dist
            )

        return xyd

    def compute_overlaps(self, xyd, return_overlaps=False):
        """
        Computes the number tracks overlapping each track.

        If one track overlaps another, each track is counted as overlapping the other.
        """

        n_all = xyd.shape[0]

        # Find only the tracks that are within the domain
        mask = (
            (xyd[:, 0] > self.xy_range[0])
            & (xyd[:, 0] < self.xy_range[1])
            & (xyd[:, 1] > self.xy_range[0])
            & (xyd[:, 1] < self.xy_range[1])
        )
        n_in_domain = np.sum(mask)
        xyd_in_domain = xyd[mask, :]

        # For each track, find all overlapping tracks
        num_overlaps = np.zeros(n_all)
        for i in range(n_all):

            # If track is not in the domain, set to NaN
            if not mask[i]:
                num_overlaps[i] = np.nan

            # For a track in the domain, find any overlaps (including tracks outside the domain)
            else:
                distance = np.hypot(xyd[:, 0] - xyd[i, 0], xyd[:, 1] - xyd[i, 1])

                # Tracks overlap if the distance between them is less than r1+r2
                overlaps = distance < (xyd[i, 2] + xyd[:, 2])

                # Includes self-overlap, because that's how the Zylstra paper defines the numbering
                num_overlaps[i] = np.sum(overlaps)

        F1 = np.nansum(num_overlaps == 1) / n_in_domain
        F2 = np.nansum(num_overlaps == 2) / n_in_domain
        F3 = np.nansum(num_overlaps == 3) / n_in_domain
        F4plus = np.nansum(num_overlaps > 4) / n_in_domain

        Farr = np.array([F1, F2, F3, F4plus])

        if return_overlaps:
            return Farr, num_overlaps
        return Farr

    def run_samples(self, ntracks, nsamples, nworkers=None):
        """
        Run the compute_overlap function with ntracks tracks nsamples times, then
        return the distribution of F1 and F2 values.


        Returns
        -------
        Farr : np.ndarray (4, nsamples)
            Array of F1, F2, F3, F4+ values for each sample

        """
        if nworkers is None:
            nworkers = cpu_count() - 1

        def run_sample(ntracks):
            xyd = self.draw_tracks(ntracks)
            return self.compute_overlaps(xyd)

        _results = Parallel(n_jobs=nworkers)(
            delayed(run_sample)(ntracks) for i in range(nsamples)
        )

        # Reformat results into 2D array
        results = np.array(_results).T
        return results

    def run_curve(self, track_density: np.ndarray, nsamples=100):
        """
        Generate F1-F4+ curves for an array of track densities (in tracks/cm^2)
        """
        Farr = np.zeros((4, track_density.size))
        for i, track_density in enumerate(track_density):
            ntracks = int(track_density * self.frame_area_with_border / 1e8)
            Farr[:, i] = self.run_samples(ntracks, nsamples).mean(axis=1)

        return Farr

    def plot_tracks(self, xyd):
        Farr, num_overlaps = self.compute_overlaps(xyd, return_overlaps=True)

        F1, F2, F3, F4plus = Farr

        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        ax.set_title(f"F1={F1:.2f}, F2={F2:.2f}")

        ax.set_xlim(self.xy_range[0] - self.border, self.xy_range[1] + self.border)
        ax.set_ylim(self.xy_range[0] - self.border, self.xy_range[1] + self.border)

        rect = Rectangle(
            (self.xy_range[0], self.xy_range[0]),
            self.xy_range[1] - self.xy_range[0],
            self.xy_range[1] - self.xy_range[0],
            fill=False,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        rect = Rectangle(
            (self.xy_range[0] - self.border, self.xy_range[0] - self.border),
            self.xy_range[1] - self.xy_range[0] + 2 * self.border,
            self.xy_range[1] - self.xy_range[0] + 2 * self.border,
            fill=False,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ntracks = xyd.shape[0]
        for i in range(ntracks):

            if np.isnan(num_overlaps[i]):
                color = "purple"
            elif num_overlaps[i] < 2:
                color = "black"
            else:
                color = "red"
            circle = Circle(
                (xyd[i, 0], xyd[i, 1]), xyd[i, 2], fill=False, edgecolor=color
            )
            ax.add_patch(circle)
