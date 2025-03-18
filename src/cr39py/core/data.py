import importlib.resources
from pathlib import Path

# No good solution to providing paths to data files: https://discuss.python.org/t/easy-and-recommended-way-to-get-path-of-datafile-within-package/20581
# data files need to be bundled inside the src file
data_dir = Path(__file__).parent / Path("data")
