"""Processing and presentation of computed data

Result objects hold computed data and offer postprocessing and plotting functions
which are specifically adapted to the nature of the stored data.
"""
from ..support.pickle import save, load
from .bands import *
from .path import *
from .series import *
from .spatial import *
from .sweep import *
from .wavefuction import *

__all__ = ['save', 'load', 'make_path', 'make_area']
