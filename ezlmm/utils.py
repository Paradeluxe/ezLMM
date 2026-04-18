"""Utility helpers for R interoperability."""

import re

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import Formula, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr


# R package imports — loaded once at module level
lmerTest = importr('lmerTest')
emmeans = importr('emmeans')
stats = importr("stats")
Matrix = importr("Matrix")
lme4 = importr("lme4")
car = importr('car')
nlme = importr("nlme")


def r2p(r_obj):
    """Convert an R object to a Python object (DataFrame or scalar)."""
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(r_obj)


def p2r(p_obj):
    """Convert a Python object to an R object."""
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(p_obj)


class r_object:
    """
    Wrapper around an R named list, allowing dict-style access.

    If accessing a key returns a nested R object, wraps it in r_object
    recursively. Otherwise returns the plain Python value.
    """

    def __init__(self, r_obj):
        self.obj = dict(zip(r_obj.names, r_obj))

    def __getitem__(self, key):
        try:
            return r_object(self.obj[key])
        except TypeError:
            # Not a nested R object — return the scalar as-is,
            # stripping any non-alphabetic characters from the string repr.
            return re.sub(r'[^a-zA-Z\s]', '', str(self.obj).strip())
