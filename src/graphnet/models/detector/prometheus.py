"""Prometheus-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import PROMETHEUS_GEOMETRY_TABLE_DIR


class ORCA150SuperDense(Detector):
    """`Detector` class for Prometheus ORCA150SuperDense."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca_150.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350) / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class TRIDENT1211(Detector):
    """`Detector` class for Prometheus TRIDENT1211."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "trident.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 1900

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 3000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeUpgrade7(Detector):
    """`Detector` class for Prometheus IceCubeUpgrade7."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_upgrade.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 2000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class WaterDemo81(Detector):
    """`Detector` class for Prometheus WaterDemo81."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "demo_water.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 2000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class BaikalGVD8(Detector):
    """`Detector` class for Prometheus BaikalGVD8."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "gvd.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceDemo81(Detector):
    """`Detector` class for Prometheus IceDemo81."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "demo_ice.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 3000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class ARCA115(Detector):
    """`Detector` class for Prometheus ARCA115."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "arca.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class ARCA115Realistic(Detector):
    """`Detector` class for ARCA115 at the KM3NeT readout level.

    `ARCA115` describes the photon tier, where a hit is a detected photon and
    the only observables are the module position and an arrival time. This
    class describes what the DAQ would actually deliver: L0 hits carrying a
    time-over-threshold and the pointing direction of the photocathode that
    fired.

    The multi-PMT module is the reason the direction matters. A KM3NeT module
    holds 31 photocathodes facing different ways, so the same photon recorded
    on an up- rather than down-facing PMT means something different, and a
    model given only the module position cannot tell the two apart.

    Expected features: `x`, `y`, `z`, `dir_x`, `dir_y`, `dir_z`, `t`, `tot`.
    Positions and directions are carried per hit rather than joined from
    `geometry_table`, which indexes modules by a 0-based string and a global
    sensor number where the simulation labels them 1-based per string.

    `npe`, `charge` and `is_noise` are simulation truth with no measurable
    counterpart, so they are deliberately absent from the feature map and
    passing them as inputs raises rather than silently training on truth.
    """

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "arca.parquet"
    )
    xyz = ["x", "y", "z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "x": self._xy,
            "y": self._xy,
            "z": self._z,
            "dir_x": self._identity,
            "dir_y": self._identity,
            "dir_z": self._identity,
            "t": self._t,
            "tot": self._tot,
        }
        return feature_map

    def _xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _z(self, x: torch.tensor) -> torch.tensor:
        # The instrumented depths span 2888-3500 m, so dividing alone would
        # leave a large offset carrying almost no variance.
        return (x + 3200.0) / 200.0

    def _t(self, x: torch.tensor) -> torch.tensor:
        # Scale only: no constant can remove the per-event injection offset,
        # so the origin is left to the data representation. The divisor is the
        # spread about an event's own time origin, 1012 ns rms.
        return x / 1.0e03

    def _tot(self, x: torch.tensor) -> torch.tensor:
        # Time-over-threshold is KM3NeT's only charge record. Hits whose
        # discriminator windows overlap are merged into one, which reaches
        # ~40x the single-photo-electron value, so the tail is compressed
        # rather than handed to the network raw.
        return torch.log10(torch.clamp(x, min=1.0))


class ORCA150(Detector):
    """`Detector` class for Prometheus ORCA150."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCube86Prometheus(Detector):
    """`Detector` class for Prometheus IceCube86."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeDeepCore8(Detector):
    """`Detector` class for Prometheus IceCubeDeepCore8."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_deepcore.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeGen2(Detector):
    """`Detector` class for Prometheus IceCubeGen2."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_gen2.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class PONETriangle(Detector):
    """`Detector` class for Prometheus PONE Triangle."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "pone_triangle.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class Prometheus(ORCA150SuperDense):
    """Reference to ORCA150SuperDense."""
