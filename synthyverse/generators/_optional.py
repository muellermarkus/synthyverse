from importlib.util import find_spec


CTGAN_EXTRA_MESSAGE = (
    "CTGANGenerator and TVAEGenerator require the optional ctgan dependency. "
    "Install it with `pip install synthyverse[ctgan]` and review the ctgan "
    "Business Source License before using these generators."
)

TABPFN_EXTRA_MESSAGE = (
    "TabPFNGenerator requires the optional tabpfn dependency. "
    "Install it with `pip install synthyverse[tabpfn]` and review the TabPFN "
    "License before using this generator."
)


def has_ctgan() -> bool:
    return find_spec("ctgan") is not None

def require_ctgan() -> None:
    if not has_ctgan():
        raise ImportError(CTGAN_EXTRA_MESSAGE)

def has_tabpfn() -> bool:
    return find_spec("tabpfn") is not None

def require_tabpfn() -> None:
    if not has_tabpfn():
        raise ImportError(TABPFN_EXTRA_MESSAGE)
