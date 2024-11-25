"""
pint units registry. To create quantities with units:

>> from cr39py.core.units import u
>> x = 1 * u.m
"""

import warnings

import numpy as np
import pint

unit_registry = pint.UnitRegistry()

# PSL as a dimensionless unit
unit_registry.define("PSL=")


unit_registry.define("thou=25400*nm=mil")

__all__ = []


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    unit_registry.Quantity([])
warnings.filterwarnings("ignore", category=pint.UnitStrippedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pint")
unit_registry.enable_contexts("boltzmann")
unit_registry.default_system = "cgs"

u = unit_registry
