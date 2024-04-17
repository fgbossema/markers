# -*- coding: utf-8 -*-

"""Top-level package for Markers."""

__author__ = """Francien Bossema"""
__email__ = 'bossema@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.

from .markers3 import *
from .phantoms import *
from .plotting import *
from .track_markers import *
from .reconstructions import *
#from .jacobian_new import *
from .blobdetector import *
from .simulated_measurements import *
from .BM_data import *
from .Getty_data import *
from .rijxray_data import *
