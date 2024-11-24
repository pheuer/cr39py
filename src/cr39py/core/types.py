from typing import Annotated, TypeVar

import numpy as np

TrackData = Annotated[np.ndarray, "(ntracks,6)"]
