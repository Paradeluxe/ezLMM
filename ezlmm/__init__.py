"""
ezlmm: simplified linear and generalized linear mixed models.

Intended usage::

    from ezlmm import LinearMixedModel

    model = LinearMixedModel()
    model.read_data("data.csv")
    model.code_variables({"condition": {"old": "new"}})
    model.dep_var = "rt"
    model.indep_var = ["condition", "group"]
    model.random_var = ["subject"]
    model.fit()
    print(model.report)

For advanced use, import from submodules::

    from ezlmm.model import LinearMixedModel, GeneralizedLinearMixedModel
    from ezlmm.data import read_data, DataLoader
    from ezlmm.report import extract_contrast
    from ezlmm.utils import r_object, r2p, p2r
"""

# Import only the lightweight, R-independent modules at package init.
# R-dependent imports (rpy2) are deferred to submodules to avoid
# hard-blocking usage of data/report utilities when R is not installed.
from ezlmm.data import DataLoader, read_data
from ezlmm.report import extract_contrast

__version__ = "0.1.0"
__all__ = [
    "LinearMixedModel",
    "GeneralizedLinearMixedModel",
    "DataLoader",
    "read_data",
    "extract_contrast",
    "r_object",
    "r2p",
    "p2r",
]


def __getattr__(name):
    """Lazily import model classes to avoid loading rpy2 unless needed."""
    if name == "LinearMixedModel":
        from ezlmm.model import LinearMixedModel
        return LinearMixedModel
    if name == "GeneralizedLinearMixedModel":
        from ezlmm.model import GeneralizedLinearMixedModel
        return GeneralizedLinearMixedModel
    if name in ("r_object", "r2p", "p2r"):
        from ezlmm import utils
        return getattr(utils, name)
    raise AttributeError(f"module 'ezlmm' has no attribute {name!r}")
