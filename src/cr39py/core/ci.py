import os
import warnings

import matplotlib.pyplot as plt
from matplotlib import get_backend
from matplotlib import use as matplotlib_use


def in_ci():  # pragma: no cover
    """
    GitLab CI sets the variable CI == 'true' on all pipelines,
    so the presence and value of this environment variable can be used
    to determine whether the code is being run in GitLab CI.
    """
    return "CI" in os.environ and os.environ["CI"] == "true"


def in_unit_test():  # pragma: no cover
    """
    Pytest sets this environment variable
    """
    # https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    return "PYTEST_CURRENT_TEST" in os.environ


class SilentPlotting:  # pragma: no cover
    """
    Context manager that allows matplotlib to create plots silently
    to ensure plotting functions run without actually having all the plots
    open.
    """

    def __enter__(self):
        warnings.filterwarnings(
            "ignore", "FigureCanvasAgg is non-interactive, and thus cannot be shown"
        )
        self.current_backend = get_backend()
        plt.close("all")
        matplotlib_use("Agg")

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close("all")
        matplotlib_use(self.current_backend)
