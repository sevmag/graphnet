"""Compatibility alias: `DeepIce` lives in `graphnet.models.transformer`.

This module keeps the historical dotted path
``graphnet.models.gnn.icemix.DeepIce`` resolvable, so existing model
configs and checkpoints load unchanged.
"""

from graphnet.models.transformer.icemix import DeepIce

__all__ = ["DeepIce"]
