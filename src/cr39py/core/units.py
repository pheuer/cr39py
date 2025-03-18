"""
pint units registry. To create quantities with units:

>> from cr39py.core.units import u
>> x = 1 * u.m
"""

import warnings

import numpy as np
import pint

__all__ = ["u"]

unit_registry = pint.UnitRegistry()
u = unit_registry
