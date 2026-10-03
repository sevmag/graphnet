"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import NUBENCH_GEOMETRY_TABLE_DIR


class NuBenchDetector(Detector):
    """Generic Detector Class from the NuBench Paper (arXiv:2511.13111)."""

    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"
    sensor_time_column = "t"
    charge_column = "charge"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
            "charge": self._charge,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 10e5

    def _charge(self, x: torch.tensor) -> torch.tensor:
        # floor charge at 1e-2; perturbation/noise can make it non-positive
        return torch.log10(1 + torch.clamp(x, min=1e-2))


class NuBenchSpacetimeDetector(NuBenchDetector):
    """NuBench detector on a single length unit for all four coordinates.

    `NuBenchDetector` scales x and y by 100 m, z by 1000 m and time by 1 ms.
    Those choices keep every feature O(1), but they leave the four coordinates
    on three different scales, which matters wherever they are combined rather
    than embedded independently:

    - `dx^2 + dy^2 + dz^2` is not a Euclidean distance, since depth
      separations are compressed tenfold against horizontal ones.
    - A spacetime interval `dx^2 - (c dt)^2` needs time in length units. The
      conversion factor depends on both scales, so it silently changes with
      the normalisation; `DeepIce`'s default is the IceCube value and is
      ~167x too small here, which drives the time term to ~0.01% of the
      spatial one and makes essentially every pair read as spacelike.

    Here all three axes are scaled by 100 m -- the NuBench geometries span
    ~1 km in every direction, so one unit suits all of them -- and time is
    converted to the same unit as `c * t / 100`. The interval is then a true
    Minkowski interval with no conversion constant at all, and the four
    coordinates share a range of roughly +-5.

    Depth is centred on the sensor centroid before scaling. That does not
    affect any interval, where the offset cancels in a difference, but it
    keeps the absolute feature near zero: NuBench depths sit ~2 km below the
    surface, so scaling by 100 m without centring would hand the encoder a
    coordinate around -20.

    The parent class is unchanged, so models and checkpoints trained against
    the original normalisation keep working.
    """

    # Vacuum c in m/ns. Signal in a neutrino telescope is carried by a
    # near-luminal muon before it is carried by photons in ice (c/n), so the
    # vacuum value is the causal bound: it is the speed that decides whether
    # two hits could belong to the same event at all.
    C_M_PER_NS = 0.299792458
    POS_SCALE_M = 100.0

    @property
    def z_centre(self) -> float:
        """Depth of the sensor centroid, in metres."""
        if not hasattr(self, "_z_centre"):
            self._z_centre = float(self.geometry_table["sensor_pos_z"].mean())
        return self._z_centre

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return (x - self.z_centre) / self.POS_SCALE_M

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x * self.C_M_PER_NS / self.POS_SCALE_M


class HexagonSpacetime(NuBenchSpacetimeDetector):
    """`Detector` class for Hexagon on a single length unit."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "hexagon.parquet"
    )


class FlowerLSpacetime(NuBenchSpacetimeDetector):
    """`Detector` class for Flower L on a single length unit."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_l.parquet"
    )


class FlowerS(NuBenchDetector):
    """`Detector` class for Flower S."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_s.parquet"
    )


class FlowerL(NuBenchDetector):
    """`Detector` class for Flower L."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_l.parquet"
    )


class FlowerXL(NuBenchDetector):
    """`Detector` class for Flower XL."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "flower_xl.parquet"
    )


class Triangle(NuBenchDetector):
    """`Detector` class for Triangle."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "triangle.parquet"
    )


class Cluster(NuBenchDetector):
    """`Detector` class for Cluster."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "cluster.parquet"
    )


class Hexagon(NuBenchDetector):
    """`Detector` class for Hexagon."""

    geometry_table_path = os.path.join(
        NUBENCH_GEOMETRY_TABLE_DIR, "hexagon.parquet"
    )
