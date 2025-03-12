import importlib.resources
from pathlib import Path

# data_dir = Path(importlib.resources.files("cr39py")).parent.parent / Path("data")
data_dir = Path(__file__).parent.parent.parent.parent / Path("data")
print(data_dir)
