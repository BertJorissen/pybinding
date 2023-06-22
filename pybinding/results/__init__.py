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
from .bands import __all__ as bands_all
from .path import __all__ as path_all
from .series import __all__ as series_all
from .spatial import __all__ as spatial_all
from .sweep import __all__ as sweep_all
from .wavefuction import __all__ as wavefuction_all

__all__ = ['save', 'load', 'make_path', 'make_area', 'Bands']
__all__ += bands_all
__all__ += path_all
__all__ += series_all
__all__ += spatial_all
__all__ += sweep_all
__all__ += wavefuction_all
