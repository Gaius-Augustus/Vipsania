from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cli, data, hub, model, util, xai
    from .annotate import annotate_genome
    from .train import Trainer, TrainerConfig
    from .util import create_model

__all__ = [
    "Trainer",
    "TrainerConfig",
    "annotate_genome",
    "cli",
    "create_model",
    "data",
    "hub",
    "model",
    "util",
    "xai",
]

_SUBMODULES = ("cli", "data", "hub", "model", "util", "xai")
_ATTRIBUTES = {
    "Trainer": ".train",
    "TrainerConfig": ".train",
    "annotate_genome": ".annotate",
    "create_model": ".util",
}


def __getattr__(name: str):
    """Import submodules and their contents on first use.

    Keeping the package namespace lazy delays the import of TensorFlow
    until something is actually requested from Vipsania. The command
    line interface can therefore still set the environment variables
    that TensorFlow reads while it is imported.
    """
    import importlib

    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _ATTRIBUTES:
        module = importlib.import_module(_ATTRIBUTES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
