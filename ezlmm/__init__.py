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

__version__ = "0.4.5"

__all__ = [
    "LinearMixedModel",
    "GeneralizedLinearMixedModel",
]


def __getattr__(name):
    """Lazily import model classes and utilities to avoid loading rpy2 until needed."""
    if name == "LinearMixedModel":
        from ezlmm.model import LinearMixedModel

        return LinearMixedModel
    if name == "GeneralizedLinearMixedModel":
        from ezlmm.model import GeneralizedLinearMixedModel

        return GeneralizedLinearMixedModel
    if name in ("DataLoader", "read_data"):
        from ezlmm.data import DataLoader, read_data

        return locals()[name]
    if name == "extract_contrast":
        from ezlmm.report import extract_contrast

        return extract_contrast
    if name in ("r_object", "r2p", "p2r"):
        from ezlmm import utils

        return getattr(utils, name)
    raise AttributeError(f"module 'ezlmm' has no attribute {name!r}")
