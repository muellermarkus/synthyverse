from importlib import import_module

from .base import (
    BaseGenerator,
    ConstraintEnforcer,
    DataProcessor,
    SynthyverseGenerator,
    TabularImputer,
    TabularSchema,
)
from ._optional import has_ctgan, has_tabpfn, require_ctgan, require_tabpfn

from .config import get_config

_BASE_GENERATORS = {
    "ARFGenerator": (".arf_generator", "arf"),
    "TabSynGenerator": (".tabsyn_generator", "tabsyn"),
    "CDTDGenerator": (".cdtd_generator", "cdtd"),
    "TabARGNGenerator": (".tabargn_generator", "tabargn"),
    "TabDDPMGenerator": (".tabddpm_generator", "tabddpm"),
    "TabDiffGenerator": (".tabdiff_generator", "tabdiff"),
    "TabCascadeGenerator": (".tabcascade_generator", "tabcascade"),
    "TabbyFlowGenerator": (".tabbyflow_generator", "tabbyflow"),
    "UnivariateGenerator": (".univariate_generator", "univariate"),
    "SMOTEGenerator": (".smote_generator", "smote"),
    "SynthpopGenerator": (".synthpop_generator", "synthpop"),
    "XGBDDPMGenerator": (".xgbddpm_generator", "xgbddpm"),
    "XGBDiffusionGenerator": (".xgbdiffusion_generator", "xgbdiffusion"),
    "UTreesGenerator": (".utrees_generator", "utrees"),
    "ForestDiffusionGenerator": (".forestdiffusion_generator", "forestdiffusion"),
}

_CTGAN_GENERATORS = {
    "CTGANGenerator": (".ctgan_generator", "ctgan"),
    "TVAEGenerator": (".tvae_generator", "tvae"),
}

_TABPFN_GENERATOR = {
    "TabPFNGenerator": (".tabpfn_generator", "tabpfn"),
}

_GENERATORS = {**_BASE_GENERATORS, **_CTGAN_GENERATORS, **_TABPFN_GENERATOR}
_GENERATOR_BY_NAME = {name: cls for cls, (_, name) in _GENERATORS.items()}


def _available_generators():
    available = dict(_BASE_GENERATORS)
    if has_ctgan():
        available.update(_CTGAN_GENERATORS)
    if has_tabpfn():
        available.update(_TABPFN_GENERATOR)
    return available


def _require_optional(class_name: str) -> None:
    if class_name in _CTGAN_GENERATORS and not has_ctgan():
        require_ctgan()
    if class_name in _TABPFN_GENERATOR and not has_tabpfn():
        require_tabpfn()


def __getattr__(name: str):
    if name == "all_generators":
        all_generators = [__getattr__(cls) for cls in _available_generators()]
        globals()[name] = all_generators
        return all_generators
    _require_optional(name)
    if name in _GENERATORS:
        module_name, _ = _GENERATORS[name]
        generator = getattr(import_module(module_name, __name__), name)
        globals()[name] = generator
        return generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_generator(generator_name: str):
    """Get a generator class by name."""
    class_name = _GENERATOR_BY_NAME.get(generator_name)
    if class_name is None:
        raise ValueError(f"Generator {generator_name} not found")
    _require_optional(class_name)
    return __getattr__(class_name)


__all__ = [
    "BaseGenerator",
    "ConstraintEnforcer",
    "DataProcessor",
    "SynthyverseGenerator",
    "TabularImputer",
    "TabularSchema",
    *list(_available_generators()),
    "all_generators",
    "get_generator",
]
