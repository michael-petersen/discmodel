"""
discmodel

basic exponential disc for testing imaging and kinematic modelling routines

See README.md for details
"""
from .discmodel import DiscGalaxy
from importlib.metadata import version

__version__ = version("discmodel")
__all__ = ["DiscGalaxy"]