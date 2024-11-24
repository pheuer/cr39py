import importlib.resources
from pathlib import Path

data_dir = Path(importlib.resources.files("cr39py")).parent.parent / Path("data")
