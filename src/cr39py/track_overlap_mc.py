import numpy as np

rng = np.random.default_rng()


class MonteCarloTrackOverlap:
    def __init__(self):
        self.xy_range = (-5, 5)
        self.diameters = np.arange(0.5, 10, 0.1)
        self.diameters_mean = 7
        self.diameters_std = 2

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
        xyd[:, 0] = rng.uniform(low=self.xy_range[0], high=self.xy_range[1])
        xyd[:, 1] = rng.uniform(low=self.xy_range[0], high=self.xy_range[1])

        # Draw diameters from the diameter distribution
        xyd[:, 2] = rng.choice(
            self.diameters, size=ntracks, replace=True, p=self.diameter_dist
        )

        return xyd

    def compute_overlaps(xyd):
        ntracks = xyd.shape[0]

        # For each track, find all overlapping tracks
        num_overlaps = np.zeros(ntracks)
        for i in range(ntracks):
            distance = np.hypot(xyd[:, 0] - xyd[i, 0], xyd[:, 1] - xyd[i, 1])

            # Tracks overlap if the distance between them is less than r1+r2
            overlaps = distance < (xyd[i, 2] + xyd[:, 2])

            # Remove this point's self-overlap
            overlaps[i] = False

            num_overlaps = np.sum(overlaps)
