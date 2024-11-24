import importlib.resources
from pathlib import Path

data_dir = Path(importlib.resources.files("cr39py")).parent.parent / Path("data")


def get_resource_path(name):
    """
    Get the path to a resource in the data folder.
    """
    return data_dir / Path(name)
