"""Locally-converted Prometheus datasets on the IceCube/Hexagon geometry.

These reuse NuBench's loading machinery but are not NuBench datasets: a single
``photons`` pulsemap serves every split, so there is no raw-train /
smeared-test distinction to bridge.
"""

from typing import Dict, List, Union

from graphnet.data.dataset import Dataset, EnsembleDataset
from graphnet.datasets.nubench_datasets import NuBenchDataset, NuBenchSpec
from graphnet.models.detector.nubench import Hexagon

# Event-level truth shared by both datasets.
_ICGEO_TRUTH = [
    "interaction",
    "initial_state_energy",
    "initial_state_type",
    "initial_state_zenith",
    "initial_state_azimuth",
    "initial_state_x",
    "initial_state_y",
    "initial_state_z",
]


class IceCubePrometheusDataset(NuBenchDataset):
    """Locally-converted Prometheus data on the IceCube/Hexagon geometry.

    These reuse :class:`~graphnet.datasets.nubench_datasets.NuBenchDataset`'s
    registry, selection, LMDB and labelling machinery, but they are *not*
    NuBench datasets: a single ``photons`` pulsemap serves train, val and test,
    so there is no raw-train / smeared-test distinction. A
    ``perturbation_dict`` on the data representation (if any) is applied
    uniformly to every split, and none of NuBench's presmeared-test handling
    applies.

    The files are not ERDA-hosted; the LMDB and selection parquet files must
    already exist under ``download_dir/<name>/`` (use ``backend="lmdb"``).

    Example::

        from graphnet.models.graphs import KNNGraph
        from graphnet.models.detector.nubench import Hexagon
        from graphnet.datasets import IceCubePrometheusDataset

        ds = IceCubePrometheusDataset(
            name="icecube86_icgeo",
            download_dir="/path/to/data",
            data_representation=KNNGraph(detector=Hexagon()),
            backend="lmdb",
        )
    """

    _registry: Dict[str, NuBenchSpec] = {
        "icecube86_icgeo": NuBenchSpec(
            erda_hash="LOCAL_NO_ERDA",
            detector_cls=Hexagon,
            experiment="IceCube/Hexagon all-sky (Prometheus, local)",
            comments=(
                "Locally-converted all-sky Prometheus generation on the "
                "IceCube hexagon geometry. Photons merged to pulses "
                "(3.2 ns window, charge = photon count), single `photons` "
                "pulsemap."
            ),
            event_truth=list(_ICGEO_TRUTH),
            pulsemap_per_split={
                "train": "photons",
                "val": "photons",
                "test": "photons",
            },
        ),
        "icecube86_icgeo_noise": NuBenchSpec(
            erda_hash="LOCAL_NO_ERDA",
            detector_cls=Hexagon,
            experiment="IceCube/Hexagon all-sky (Prometheus, local, +noise)",
            comments=(
                "Locally-converted all-sky Prometheus generation on the "
                "IceCube hexagon geometry with NuBench stochastic noise "
                "injected (ice 0.3 kHz) and merged into pulses (3.2 ns "
                "window); charge = signal+noise photon count, is_signal = "
                "signal fraction, single `photons` pulsemap."
            ),
            event_truth=list(_ICGEO_TRUTH),
            pulsemap_per_split={
                "train": "photons",
                "val": "photons",
                "test": "photons",
            },
        ),
    }

    _creator = "felixyu"
    _citation = "https://arxiv.org/abs/2511.13111"

    def _warn_if_missing_smear_perturbation(self) -> None:
        """Skip NuBench's smear warning.

        Every split shares the one ``photons`` pulsemap, so there is no raw
        train / smeared test gap for a perturbation to bridge.
        """
        return

    def _create_dataset(
        self,
        selection: Union[List[int], List[List[int]], List[float]],
    ) -> Union[EnsembleDataset, Dataset]:
        """Build a split from its single pulsemap.

        Every split uses the perturbing ``data_representation`` directly, so a
        configured perturbation is applied uniformly -- unlike NuBench, there
        is no already-smeared test pulsemap to leave unperturbed.
        """
        pmap = dict(self._spec.pulsemap_per_split)
        if self._custom_test_pulsemap is not None:
            pmap["test"] = self._custom_test_pulsemap
        if selection is self._test_selection:
            key = "test"
        elif selection is getattr(self, "_val_selection", None):
            key = "val"
        else:
            key = "train"
        self._dataset_args["pulsemaps"] = [pmap[key]]
        assert self._data_representation is not None
        self._dataset_args["data_representation"] = self._data_representation
        # Skip past NuBenchDataset._create_dataset (its presmeared-test twin
        # logic) to the generic dataset builder.
        return super(NuBenchDataset, self)._create_dataset(selection)
