import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.patches import Circle, Rectangle

rng = np.random.default_rng()


class MonteCarloTrackOverlap:
    def __init__(
        self,
        framesize: float = 300,
        border: float = 25,
        diameters_mean: float = 10,
        diameters_std: float = 0,
        daxis=np.arange(0.5, 20, 0.025),
        diameter_distribution=None,
    ) -> None:
        """
        A Monte-Carlo (MC) simulation to calculate the fraction of overlapped tracks on CR-39.

        The simulation covers a single microscope 'frame' of the CR-39 scan, as well as a border region to avoid edge effects.
        Tracks in the border region can overlap tracks in the domain, but are otherwise not counted in the output.

        Parameters
        ----------
        framesize : float, optional
            Size of the CR-39 frame is (framesize, framesize) in um. The default is 300 um.
        border : float, optional
            The thickness of the border region in um. The default is 25 um.
        diameters_mean : float, optional
            The mean diameter of the tracks in um. The default is 10 um.
        diameters_std : float, optional
            The standard deviation of the track diameters in um. The default is 0 um, in which case all tracks
            are set to the mean diameter.
        daxis : np.ndarray, optional
            The array of diameters to use for the diameter distribution, in um. The default is np.arange(0.5, 20, 0.025).
        diameter_distribution : np.ndarray, optional
            The probability distribution of diameters to use for the simulation, over the diameters in ``daxis``.
            If not provided, a Gaussian distribution centered at ``diameters_mean`` with standard deviation ``diameters_std`` will be used.

        """

        self.framesize = framesize
        self.border = border
        self.diameters_mean = diameters_mean
        self.diameters_std = diameters_std
        self.daxis = daxis
        self.diameter_distribution = None

    @property
    def frame_area(self):
        """
        Frame area in um^2
        """
        return self.framesize**2

    @property
    def frame_area_with_border(self):
        """
        Frame area in um**2, including border.
        """
        return (self.framesize + 2 * self.border) ** 2

    @property
    def gaussian_diameter_distribution(self) -> np.ndarray:
        """
        Default diameter distribution is a Gaussian centered at ``diameters_mean`` with standard deviation ``diameters_std``.
        """
        dist = np.exp(
            -((self.daxis - self.diameters_mean) ** 2) / 2 / self.diameters_std**2
        )
        return dist / np.sum(dist)

    def draw_tracks(
        self,
        ntracks: int,
    ) -> np.ndarray:
        """
        Draws a set of tracks with random positions and diameters.

        Positions are chosen from a uniform distribution. Diameters are drawn from ``diameter_distribution`` if provided, otherwise a
        Gaussian distribution is used.

        Parameters
        ----------
        ntracks : int
            Number of tracks to draw

        Returns
        -------
        np.ndarray, (ntracks,3)
            (X,Y,Diameter) for each track
        """
        xyd = np.empty((ntracks, 3))

        # Draw spatially uniform positions in the plane
        xyd[:, 0] = rng.uniform(
            low=-self.framesize / 2 - self.border,
            high=self.framesize / 2 + self.border,
            size=ntracks,
        )
        xyd[:, 1] = rng.uniform(
            low=-self.framesize / 2 - self.border,
            high=self.framesize / 2 + self.border,
            size=ntracks,
        )

        # Draw diameters from the diameter distribution
        if self.diameters_std == 0:
            xyd[:, 2] = self.diameters_mean
        else:
            if self.diameter_distribution is None:
                diameter_dist = self.gaussian_diameter_distribution(self.daxis)
            else:
                diameter_dist = self.diameter_distribution

            xyd[:, 2] = rng.choice(
                self.diameters, size=ntracks, replace=True, p=diameter_dist
            )

        return xyd

    def compute_overlaps(self, xyd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the number tracks overlapping each track.

        If one track overlaps another, each track is counted as overlapping the other.
        Tracks in the border region can overlap tracks inside the domain, but are not counted in the F1-F4+ values.

        Parameters
        ----------
        xyd : np.ndarray (ntracks, 3)
            Array of (X,Y,Diameter) for each track.

        Returns
        -------
        Farr : np.ndarray (4,)
            Array of F1, F2, F3, F4+ values.

        num_overlaps : np.ndarray (ntracks,)
            Array of the number of overlaps detected for each track. Tracks outside the domain are set to NaN.
        """

        n_all = xyd.shape[0]

        # Find only the tracks that are within the domain
        mask = (
            (xyd[:, 0] >= -self.framesize / 2)
            & (xyd[:, 0] <= self.framesize / 2)
            & (xyd[:, 1] >= -self.framesize / 2)
            & (xyd[:, 1] <= self.framesize / 2)
        )
        n_in_domain = np.sum(mask)
        n_out_of_domain = n_all - n_in_domain

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
                overlaps = distance <= (xyd[i, 2] / 2 + xyd[:, 2] / 2)

                # Includes self-overlap, because that's how the Zylstra paper defines the numbering
                num_overlaps[i] = np.sum(overlaps)

        assert np.sum(~np.isnan(num_overlaps)) == n_in_domain

        F1 = np.nansum(num_overlaps == 1) / n_in_domain
        F2 = np.nansum(num_overlaps == 2) / n_in_domain
        F3 = np.nansum(num_overlaps == 3) / n_in_domain
        F4plus = np.nansum(num_overlaps >= 4) / n_in_domain

        Farr = np.array([F1, F2, F3, F4plus])

        return Farr, num_overlaps

    def run_samples(self, ntracks: int, nsamples: int, nworkers: int | None = None):
        """
        Run the compute_overlap function with ntracks tracks nsamples times, then
        return the distribution of F1 and F2 values.

        If the joblib package is installed, the samples will be run in parallel using the number of workers specified by nworkers.
        Otherwise, the samples will be run in serial.

        Parameters
        ----------
        ntracks : int
            Number of tracks to draw for each sample.

        nsamples : int
            Number of samples to run.

        nworkers : int, optional
            Number of parallel workers to use. If None, will use all available cores minus one.

        Returns
        -------
        Farr : np.ndarray (4, nsamples)
            Array of F1, F2, F3, F4+ values for each sample

        """

        # Define a function that runs a sample for a given number of tracks
        def run_sample(ntracks):
            xyd = self.draw_tracks(ntracks)
            Farr, _ = self.compute_overlaps(xyd)
            return Farr

        # Try importing joblib - if not found, run without parallelization
        try:  # pragma: no cover
            from multiprocessing import cpu_count

            from joblib import Parallel, delayed

            if nworkers is None:
                nworkers = cpu_count() - 1
        except ModuleNotFoundError:
            nworkers = 1

        # Run the samples in parallel or in serial as requested
        if nworkers > 1:  # pragma: no cover
            _results = Parallel(n_jobs=nworkers)(
                delayed(run_sample)(ntracks) for i in range(nsamples)
            )
        else:
            _results = [run_sample(ntracks) for i in range(nsamples)]

        # Reformat results into 2D array
        results = np.array(_results).T
        return results

    def run_curve(
        self, track_densities: np.ndarray, nsamples: int, nworkers: int | None = None
    ) -> np.ndarray:
        """
        Generate F1-F4+ curves for an array of track densities.

        If the joblib package is installed, the samples will be run in parallel using the number of workers specified by nworkers.
        Otherwise, the samples will be run in serial.

        Parameters
        ----------
        track_densities : np.ndarray
            Array of track densities in tracks/cm^2.

        nsamples : int
            Number of samples to run for each track density.

        nworkers : int, optional
            Number of parallel workers to use. If None, will use all available cores minus one.

        Returns
        -------
        Farr : np.ndarray (4, track_densities.size)
            Array of F1, F2, F3, F4+ values for each track density.
        """
        Farr = np.zeros((4, track_densities.size))

        for i in tqdm.tqdm(
            range(track_densities.size), desc="Running track density curve"
        ):
            track_density = track_densities[i]
            ntracks = int(track_density * self.frame_area_with_border / 1e8)
            Farr[:, i] = np.nanmean(self.run_samples(ntracks, nsamples), axis=1)

        return Farr

    def plot_tracks(self, xyd: np.ndarray) -> None:
        """
        Plots a set of tracks on the current frame.

        Parameters
        ----------
        xyd : np.ndarray (ntracks, 3)
            Tracks to plot.
        """
        Farr, num_overlaps = self.compute_overlaps(xyd)

        F1, F2, F3, F4plus = Farr

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal")

        ax.set_title(f"F1={F1:.2f}, F2={F2:.2f}, F3={F3:.2f}, F4+={F4plus:.2f}")

        ax.set_xlim(-self.framesize / 2 - self.border, self.framesize / 2 + self.border)
        ax.set_ylim(-self.framesize / 2 - self.border, self.framesize / 2 + self.border)

        rect = Rectangle(
            (-self.framesize / 2, -self.framesize / 2),
            self.framesize,
            self.framesize,
            fill=False,
            edgecolor="orange",
            facecolor="none",
            label="Domain",
        )
        ax.add_patch(rect)

        rect = Rectangle(
            (-self.framesize / 2 - self.border, -self.framesize / 2 - self.border),
            self.framesize + 2 * self.border,
            self.framesize + 2 * self.border,
            fill=False,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

        ntracks = xyd.shape[0]
        for i in range(ntracks):

            if np.isnan(num_overlaps[i]):
                color = "purple"
            # Color-code based on number of overlaps
            else:
                colors = ["black", "red", "lime"]
                ind = int(num_overlaps[i]) - 1
                if ind <= 2:
                    color = colors[ind]
                else:
                    color = "blue"

            circle = Circle(
                (xyd[i, 0], xyd[i, 1]), xyd[i, 2] / 2, fill=False, edgecolor=color
            )
            ax.add_patch(circle)

        ax.plot([], [], color="purple", label="Outside domain")
        ax.plot([], [], color="black", label="F1")
        ax.plot([], [], color="red", label="F2")
        ax.plot([], [], color="lime", label="F3")
        ax.plot([], [], color="blue", label="F4+")
        ax.legend(loc="upper left")
