"""
This module contains code for reading the MIT CPSA format for CR-39
track data.
"""

import re
from collections import namedtuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cr39py.core.types import TrackData


def read_cpsa(path: Path) -> TrackData:
    """Reads a CPSA file.

    Parameters
    ----------
    path : `~pathlib.Path`
        Path to the CPSA file.

    Returns
    -------
    tracks: `~np.ndarray` (ntracks,6)
       Array of track data.

    Notes
    -----
    Adapted from code written by Hans Rinderknecht.

    """
    with open(path, "rb") as file:

        # First read in the header values
        print("***CPSA header***")
        version = -np.fromfile(file, count=1, dtype="int32")[0]
        print(f"...Version: {version}")

        # Number of microscope frames in the x and y directions of the scan
        # respectively
        nx = np.fromfile(file, count=1, dtype="int32")[0]
        ny = np.fromfile(file, count=1, dtype="int32")[0]
        nframes = nx * ny
        print(f"...nx, ny microscope bins: {nx}, {ny}")
        print(f"...Nframes: {nframes}")

        # h[3] is "Nbins" which is not used except with legacy data\
        np.fromfile(file, count=1, dtype="int32")[0]

        # Pixel size in microns: note that this is stored as a single
        # so needs to be read as such
        pix_size = np.fromfile(file, count=1, dtype="single")[0]
        print(f"...Pixel size: {pix_size:.1e} um")

        # h[5] is "ppb" which is not used except with legacy data
        np.fromfile(file, count=1, dtype="single")[0]

        # Thresholds for border, contrast, eccentricity,
        # and number of eccentricity moments
        threshold = np.fromfile(file, count=4, dtype="int32")[0]
        print(f"...Threshold: {threshold}")

        # Number of utilized camera image pixels in the x and y directions
        NFPx = np.fromfile(file, count=1, dtype="int32")[0]
        NFPy = np.fromfile(file, count=1, dtype="int32")[0]
        print(f"...Untilized camera image px NFPx, NFPy: {NFPx}, {NFPy}")

        # Microscope frame size in microns
        fx = pix_size * NFPx
        fy = pix_size * NFPy
        print(f"...Microscope frame size fx, fy: {fx:.1e} um, {fy:.1e} um")

        # Store the scan data in a dictionary, which will be returned along with
        # the track data
        metadata = {
            "version": version,
            "nx_frames": nx,
            "ny_frames": ny,
            "nframes": nframes,
            "pixel_size": pix_size,
            "threshold": threshold,
            "NFPx": NFPx,
            "NFPy": NFPy,
            "frame_size_x": fx,
            "frame_size_y": fy,
        }

        # Read the full datafile as int32 and separate out the track info
        # Represents metadata from a single frame of cr39
        # used when reading CPSA files
        FrameHeader = namedtuple(
            "FrameHeader",
            ["number", "xpos", "ypos", "hits", "BLENR", "zpos", "x_ind", "y_ind"],
        )

        # Frame headers stores the info from each frame header
        frame_headers = []
        # Tracks in each frame
        frame_tracks = []
        # Keep track of the total number of hits (tot_hits) and the
        # running total of hits (cum_hits) which will be used later
        # for assembling the trackdata array
        tot_hits = 0
        cum_hits = np.array([0], dtype="int32")

        # Collect x and y positions of frames in sets that, once sorted
        # will be the x and y axes of the dataset
        xax = np.zeros(nx)
        yax = np.zeros(ny)

        pbar = tqdm(range(nframes))
        pbar.set_description("Reading CPSA file")
        for i in pbar:
            # Read the bin header
            h = np.fromfile(file, count=10, dtype="int32")

            # Header contents are as follows
            # 0 -> frame number (starts at 1)
            # 1 -> xpos (frame center x position, in 1e-7 m)
            # 2 -> ypos (frame center y position, in 1e-7 m)
            # 3 -> hits (number of tracks in this frame)
            # 4,5,6 -> BLENR (something about rejected tracks?)
            # 7 -> zpos (microscope focus? Units? )
            # 8 -> x_ind (x index of the frame, staring at 0)
            # 9 -> y_ind (y index of the frame, staring at 0)

            fh = FrameHeader(
                number=h[0],
                xpos=h[1],
                ypos=h[2],
                hits=h[3],
                BLENR=h[4:7],
                zpos=h[7],
                x_ind=h[8],
                y_ind=h[9],
            )
            frame_headers.append(fh)

            # Put the bin x and y values in the appropriate place in the axes
            xax[fh.x_ind] = fh.xpos * 1e-5
            yax[fh.y_ind] = fh.ypos * 1e-5

            # Increment the counters for the number of hits
            tot_hits += fh.hits
            cum_hits = np.append(cum_hits, tot_hits)

            # Read the track data for this frame
            # Each frame entry contains a sequence for each hit, which
            # contains the following integers
            # 1) diameter (int16)  in units of 1e-2*pix_size (?)
            # 2) ecentricity (uint)
            # 3) contrast (uint)
            # 4) avg contrast (uint)
            # 5) x pos (int16) in units of 1e-4*pix_size
            # 6) y pos (int16) in units of 1e-4*pix_size
            # 7) z pos (int16) in units of 1e-2*pix_size (??)
            #
            # The x and y pos are relative to the upper right corner
            # of the current frame

            # TODO: What is the difference between contrast and avg. contrast?
            # Which should be the "C" in the analysis?

            t = np.zeros([fh.hits, 6])
            if fh.hits > 0:

                # Diameters (converting to um)
                t[:, 2] = (
                    np.fromfile(file, count=fh.hits, dtype="int16") * 1e-2 * pix_size
                )

                # Ecentricities
                t[:, 4] = np.fromfile(file, count=fh.hits, dtype="byte")

                # Contrast
                t[:, 3] = np.fromfile(file, count=fh.hits, dtype="byte")

                # Avg Contrast
                # Do not store
                # t[:, 4] = np.fromfile(file, count=fh.hits, dtype="byte")
                _ = np.fromfile(file, count=fh.hits, dtype="byte")

                # x position, cm
                # Positions are relative to the top right of the current
                # frame, so we need to adjust them accordingly
                t[:, 0] = (
                    -np.fromfile(file, count=fh.hits, dtype="int16") * pix_size * 1e-4
                    + fh.xpos * 1e-5
                    + (fx / 2) * 1e-4
                )

                # y position, cm
                t[:, 1] = (
                    np.fromfile(file, count=fh.hits, dtype="int16") * pix_size * 1e-4
                    + fh.ypos * 1e-5
                    - (fy / 2) * 1e-4
                )

                # z position, microns
                t[:, 5] = fh.zpos * pix_size * 1e-2

            frame_tracks.append(t)

        # Read the footer, save the whole string into the metadata
        # This contains a bunch of additional metadata
        footer = b""
        for line in file:
            footer += line
        footer = footer.decode("cp1250")
        metadata["cpsa_footer"] = footer

    # CPSA file now closed - post-process the data

    # The order of the quantities in track data is:
    # 0) x position (cm))
    # 1) y position (cm)
    # 2) diameter (um)
    # 3) contrast (dimless)
    # 4) avg contrast (dimless)
    # 5) ecentricity (dimless)
    # 6) z position/lens position (um)

    # Re-shape the track data into a list of every track
    tracks = np.zeros([tot_hits, 6], dtype=np.float32)
    for i in range(nframes):
        tracks[cum_hits[i] : cum_hits[i + 1], :] = frame_tracks[i]

    # Sort the tracks by diameter for later slicing into energy dslices
    isort = np.argsort(tracks[:, 2])
    tracks = tracks[isort, :]

    # Sort the yaxis (it's backwards...)
    yax = np.sort(yax)

    # Extract some quantities from the footer using regex
    # In scans that are part of a coincidence counting process, six marks will be made on the piece
    # This regex pulls out the lines in the footer that record where these images were taken, which
    # look like this
    # IMAGES recorded before and after this scan were at:
    #   0>    (4.22560, 2.76639)   [Notes: UL-m]
    #   1>    (0.52671, 2.75683)   [Notes: UR-m]
    #   2>    (4.27408, 2.76639)   [Notes: UL-FE]
    #   3>    (0.48003, 2.75683)   [Notes: UR-NE]
    #   4>    (4.53513, 2.76701)   [Notes: UL-NE]
    #   5>    (0.74559, 2.75503)   [Notes: UR-FE]
    note_fields = ["UL-m", "UR-m", "UL-FE", "UL-NE", "UR-FE", "UR-NE"]
    for note in note_fields:
        # This regex matches a line like "0>    (4.22560, 2.76639)   [Notes: UL-m]"
        # and extracts the pair of points (4.22560, 2.76639) as the match group
        pattern = f"\([+-]?([0-9]*[.][0-9]+),\s[+-]?([0-9]*[.][0-9]+)\)[\s]+\[Notes:[\s]+{note}\]\n"
        match = re.search(pattern, footer)
        if match is not None:
            point = np.array([float(x) for x in match.groups()])
            metadata[note] = point

    return tracks, metadata


def extract_etch_time(path: Path) -> float:
    """Attempts to extract the etch time from a CPSA filename.

    Expects an entry like ``_#m_``, ``_#h_``, ``_#hr_``, etc.

    Parameters
    ----------

    path : `~pathlib.Path`
        Filepath, from which only the file name will be searched.

    Returns
    -------

    etch_time : float|None
        Etch time in minutes. If no etch time is extracted,
        returns None.

    """
    path = Path(path)

    # Cast the filename as lowercase, so "H" and "h" will both match
    filename = str(path.name).lower()

    # Possible labels, and their conversion factors to minutes
    tags = {"m": 1, "min": 1, "h": 60, "hr": 60}
    for tag in tags:
        # Search the filename for the matching pattern
        m = re.search(f"([0-9]+){tag}", filename)
        # If any match is found, convert the matched numeric
        # group to a float, multiply by the conversion factor,
        # then return
        if m is not None:
            return float(m.group(1)) * tags[tag]

    return None
