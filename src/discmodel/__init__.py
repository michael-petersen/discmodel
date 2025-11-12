"""
discmodel

basic exponential disc for testing imaging and kinematic modelling routines

See README.md for details
"""

from .optional_imports import check_lintsampler, check_flex
HAS_LINTSAMPLER = check_lintsampler()
HAS_FLEX = check_flex()

from .discmodel import DiscGalaxy
from importlib.metadata import version

__version__ = version("discmodel")
__all__ = ["DiscGalaxy"]

