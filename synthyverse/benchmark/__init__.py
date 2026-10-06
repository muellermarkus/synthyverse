from importlib import import_module


def __getattr__(name: str):
    if name in globals():
        return globals()[name]
    if name == "TabularSynthesisBenchmark":
        benchmark = import_module(".synthesis", __name__).TabularSynthesisBenchmark
        globals()[name] = benchmark
        return benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TabularSynthesisBenchmark"]
