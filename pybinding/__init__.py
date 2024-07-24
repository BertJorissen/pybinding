"""Package for numerical tight-binding calculations in solid state physics"""
__author__ = "Dean Moldovan"
__copyright__ = "2015-2024, " + __author__
__version__ = "1.0.4"
import os
import sys
if sys.platform.startswith("linux"):
    # When the _pybinding C++ extension is compiled with MKL, it requires specific
    # dlopen flags on Linux: RTLD_GLOBAL. This will not play nice with some scipy
    # modules, i.e. it will produce segfaults. As a workaround, specific modules
    # are imported first with default dlopenflags.
    # After that, RTLD_GLOBAL must be set for MKL to load properly. It's not possible
    # to set RTLD_GLOBAL, import _pybinding and then reset to default flags. This is
    # fundamentally an MKL issue which makes it difficult to resolve. This workaround
    # is the best solution at the moment.
    import scipy  # As written above, not friendly to LAPACK in scipy>=1.13.0rc1, so import first, then set the flag
    sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)
try:
    import _pybinding as _cpp
except ImportError as e:
    if "GLIBCXX" in str(e):
        msg = ("The version of libstdc++.so found in this environment is older than "
               "the GCC which was used to compile pybinding. If you're using conda, "
               "its internal libstdc++ may be masking the system library. Switching "
               "to the conda-forge channel and removing the outdated library should "
               "fix the issue. You can use the following commands:                \n"
               "  conda config --add channels conda-forge                         \n"
               "  conda update --all                                              \n"
               "  conda uninstall libgcc                                            ")
        raise ImportError(msg).with_traceback(e.__traceback__)
    else:
        raise

from .model import *
from .lattice import *
from .shape import *
from .modifier import *
from .results import *

from .chebyshev import *
from .berry import *
from .disentangle import *
from .parallel import parallel_for, parallelize

from . import (constants, greens, parallel, results, solver, system, utils, wannier)
from .utils import pltutils


def tests(options=None, plugins=None):
    """Run the tests

    Parameters
    ----------
    options : list or str
        Command line options for pytest (excluding target file_or_dir).
    plugins : list
        Plugin objects to be auto-registered during initialization.
    """
    import pytest
    import pathlib
    from .utils.misc import cd

    args = options or []
    if isinstance(args, str):
        args = args.split()
    module_path = pathlib.Path(__file__).parent

    if (module_path / 'tests').exists():
        # tests are inside installed package -> use read-only mode
        args.append('--failpath=' + os.getcwd() + '/failed')
        with cd(module_path), pltutils.backend('Agg'):
            args += ['-c', str(module_path / 'tests/local.cfg'), str(module_path)]
            error_code = pytest.main(args, plugins)
    else:
        # tests are in dev environment -> use development mode
        with cd(module_path.parent), pltutils.backend('Agg'):
            error_code = pytest.main(args, plugins)

    return error_code or None
