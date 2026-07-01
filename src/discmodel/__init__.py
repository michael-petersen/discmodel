"""
discmodel

basic exponential disc for testing imaging and kinematic modelling routines

See README.md for details
"""

from importlib.metadata import PackageNotFoundError, version

from .optional_imports import _check_lintsampler, _check_flex
HAS_LINTSAMPLER = _check_lintsampler()
HAS_FLEX = _check_flex()

from .discmodel import DiscGalaxy

try:
    __version__ = version("discmodel")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["DiscGalaxy"]
