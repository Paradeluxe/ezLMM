"""Utility helpers for R interoperability.

R imports are deferred — they only trigger when R-dependent functions
are actually called. This allows ezlmm to be imported (version check, etc.)
even when rpy2 / R is not installed.
"""

import re

import numpy as np
import pandas as pd


# ─── Lazy R package imports (loaded on first use) ────────────────────────────

def _get_r_packages():
    """Lazily import and cache R packages."""
    if not hasattr(_get_r_packages, "_cache"):
        import rpy2.robjects as ro
        from rpy2.robjects import Formula, pandas2ri, numpy2ri
        from rpy2.robjects.packages import importr

        _get_r_packages._cache = {
            "ro": ro,
            "Formula": Formula,
            "pandas2ri": pandas2ri,
            "numpy2ri": numpy2ri,
            "lmerTest": importr("lmerTest"),
            "emmeans": importr("emmeans"),
            "stats": importr("stats"),
            "Matrix": importr("Matrix"),
            "lme4": importr("lme4"),
            "car": importr("car"),
            "nlme": importr("nlme"),
        }
    return _get_r_packages._cache


def r2p(r_obj):
    """Convert an R object to a Python object (DataFrame or scalar)."""
    cache = _get_r_packages()
    ro = cache["ro"]
    pandas2ri = cache["pandas2ri"]
    numpy2ri = cache["numpy2ri"]
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(r_obj)


def p2r(p_obj):
    """Convert a Python object to an R object."""
    cache = _get_r_packages()
    ro = cache["ro"]
    pandas2ri = cache["pandas2ri"]
    numpy2ri = cache["numpy2ri"]
    with (ro.default_converter + pandas2ri.converter + numpy2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(p_obj)


class r_object:
    """Wrapper around an R named list, allowing dict-style access.

    If accessing a key returns a nested R object, wraps it in r_object
    recursively. Otherwise returns the plain Python value.
    """

    def __init__(self, r_obj):
        # Guard against bare R scalars (floats, ints, strings) that have no .names
        if hasattr(r_obj, 'names') and r_obj.names is not None:
            self.obj = dict(zip(r_obj.names, r_obj))
        else:
            self.obj = r_obj

    def __getitem__(self, key):
        try:
            val = self.obj[key]
            # If it's an R object with names, wrap it recursively
            if hasattr(val, 'names'):
                return r_object(val)
            # Scalar numeric → return as-is
            if isinstance(val, (int, float)):
                return val
            # String → strip trailing/leading whitespace only
            if isinstance(val, str):
                return val.strip()
            return val
        except (KeyError, TypeError):
            # Fallback for truly unrecognised types — strip but preserve numbers/dots
            s = str(self.obj)
            # Preserve floats and scientific notation; only strip truly garbage chars
        return re.sub(r'[^a-zA-Z0-9.+\-e]', '', s).strip()

    def __repr__(self):
        return f"r_object({self.obj})"

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def __getattr__(name):
    """Lazily expose R packages from the shared cache."""
    if name in ("emmeans", "lmerTest", "lme4", "car", "nlme", "stats", "Matrix"):
        return _get_r_packages()[name]
    raise AttributeError(f"module 'ezlmm.utils' has no attribute {name!r}")
