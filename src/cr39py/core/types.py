import numpy as np
from typing import Annotated, TypeVar
TrackData = Annotated[np.ndarray, "(ntracks,6)"]