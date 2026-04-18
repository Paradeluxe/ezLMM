"""
Backward-compatibility re-export of the original ezlmm API.

New code should import directly from submodules::

    from ezlmm.data import DataLoader, read_data
    from ezlmm.model import LinearMixedModel, GeneralizedLinearMixedModel
    from ezlmm.report import extract_contrast
    from ezlmm.utils import r_object, r2p, p2r

This module exists so existing code that did ``from ezlmm.core import *``
continues to work without modification.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd

os.environ['RPY2_CFFI_MODE'] = 'ABI'
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
pd.set_option("display.max_columns", None)

# Utility helpers
from ezlmm.utils import r_object, r2p, p2r

# Data utilities
from ezlmm.data import read_data
from ezlmm.data import DataLoader

# Report generation
from ezlmm.report import extract_contrast

# Models
from ezlmm.model import LinearMixedModel
from ezlmm.model import GeneralizedLinearMixedModel

__all__ = [
    "r_object",
    "r2p",
    "p2r",
    "read_data",
    "extract_contrast",
    "DataLoader",
    "LinearMixedModel",
    "GeneralizedLinearMixedModel",
]
