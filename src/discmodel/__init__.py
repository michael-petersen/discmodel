"""
discmodel

basic exponential disc for testing imaging and kinematic modelling routines

See README.md for details
"""

from .optional_imports import _check_lintsampler, _check_flex
HAS_LINTSAMPLER = _check_lintsampler()
HAS_FLEX = _check_flex()

from .discmodel import DiscGalaxy
from importlib.metadata import version

__version__ = version("discmodel")
__all__ = ["DiscGalaxy"]

