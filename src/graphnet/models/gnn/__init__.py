"""GNN-specific modules, for performing the main learnable operations."""

from typing import Any

from .convnet import ConvNet
from .dynedge import DynEdge
from .dynedge_jinst import DynEdgeJINST
from .dynedge_kaggle_tito import DynEdgeTITO
from .RNN_tito import RNN_TITO
from .particlenet import ParticleNeT
from .grit import GRIT


def __getattr__(name: str) -> Any:
    """Resolve `DeepIce` lazily from `graphnet.models.transformer`.

    An eager import would tie this package's initialisation order to
    `transformer.icemix`, which itself imports submodules of this package;
    lazy resolution keeps the two packages initialisable in either order.
    """
    if name == "DeepIce":
        from graphnet.models.transformer.icemix import DeepIce

        return DeepIce
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
