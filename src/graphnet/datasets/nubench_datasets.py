"""Curated datasets from the NuBench benchmark suite (arXiv:2511.13111)."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Type, Union
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from graphnet.data import ERDAHostedDataset
from graphnet.data.dataset import Dataset, EnsembleDataset
from graphnet.models.data_representation import DataRepresentation
from graphnet.models.detector.nubench import (
    NuBenchDetector,
    Cluster,
    FlowerL,
    FlowerS,
    FlowerXL,
    Hexagon,
    Triangle,
)
from graphnet.training.labels import Direction, Track


def _read_file_fully(path: str) -> bytes:
    """Read all bytes of ``path`` via raw ``os.read`` to the stat'd size.

    Inside the GPU training process, higher-level reads (pandas/pyarrow, and
    even buffered ``open().read()``) of a small selection parquet on the
    network filesystem can return short -- yielding a truncated buffer with no
    ``PAR1`` footer ("Parquet magic bytes not found"). This loops on raw
    ``os.read`` until the file's fstat size is reached.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        chunks = []
        got = 0
        while got < size:
            chunk = os.read(fd, size - got)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _read_event_nos(path: str) -> List[int]:
    """Read the ``event_no`` column from a selection parquet robustly.

    Reads the whole file into memory (raw ``os.read`` to the stat size) and
    parses from a buffer, retrying on a short read that lacks the ``PAR1``
    footer. This guards against an occasional truncated read of a small
    selection file on a network filesystem; a hard, persistent failure (e.g. a
    dead Lustre OST for that file's stripe) still raises after the retries.
    """
    last_err = ""
    for attempt in range(5):
        data = _read_file_fully(path)
        if data[-4:] == b"PAR1":
            table = pq.read_table(pa.BufferReader(data), columns=["event_no"])
            return table.column("event_no").to_pylist()
        last_err = f"short read ({len(data)} bytes, no PAR1 footer)"
        if attempt < 4:
            time.sleep(0.5)
    raise OSError(f"Could not fully read {path}: {last_err}")


FEATURES_NUBENCH = [
    "sensor_pos_x",
    "sensor_pos_y",
    "sensor_pos_z",
    "t",
    "charge",
]

TRUTH_NUBENCH = [
    "interaction",
    "initial_state_energy",
    "initial_state_type",
    "initial_state_zenith",
    "initial_state_azimuth",
    "initial_state_x",
    "initial_state_y",
    "initial_state_z",
    "bjorken_x",
    "bjorken_y",
    "visible_inelasticity",
    "muon_azimuth",
    "muon_zenith",
]

_DEFAULT_SELECTIONS = {
    "train": "selections/train_selection.parquet",
    "test": "selections/test_selection.parquet",
}

_DEFAULT_PULSEMAPS = {
    "train": "merged_photons",
    "val": "merged_photons",
    "test": "pulses_no_noise",
}


@dataclass(frozen=True)
class NuBenchSpec:
    """Static configuration for a single NuBench dataset."""

    erda_hash: str
    detector_cls: Type[NuBenchDetector]
    experiment: str
    comments: str
    features: List[str] = field(default_factory=lambda: list(FEATURES_NUBENCH))
    event_truth: List[str] = field(default_factory=lambda: list(TRUTH_NUBENCH))
    db_relpath: str = "merged/merged.db"
    lmdb_relpath: str = "merged/merged.lmdb"
    # WIP: precomputed LMDBs are not yet hosted on ERDA. Once they are,
    # populate this with the share hash and ``prepare_data`` will fetch
    # them automatically, mirroring the SQLite flow.
    lmdb_erda_hash: Optional[str] = None
    selection_relpaths: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_SELECTIONS)
    )
    pulsemap_per_split: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_PULSEMAPS)
    )


class NuBenchDataset(ERDAHostedDataset):
    """Single entry point for every NuBench benchmark dataset.

    Pick a dataset by its registry name (see :meth:`available_datasets`)
    and pass a :class:`DataRepresentation` whose detector matches the
    dataset. The tarball is downloaded from ERDA on first use and
    extracted into ``{download_dir}/{name}/``.

    The NuBench convention is that train/val events live in the
    ``merged_photons`` pulsemap while test events live in
    ``pulses_no_noise``. This class builds each split against the
    correct pulsemap automatically.

    Example::

        from graphnet.models.graphs import KNNGraph
        from graphnet.models.detector.nubench import Hexagon
        from graphnet.datasets import NuBenchDataset

        ds = NuBenchDataset(
            name="hexagon_ice_le",
            download_dir="/path/to/nubench_data",
            data_representation=KNNGraph(detector=Hexagon()),
        )
    """

    _registry: Dict[str, NuBenchSpec] = {
        "cluster": NuBenchSpec(
            erda_hash="EBamFwOU2D",
            detector_cls=Cluster,
            experiment="Cluster (NuBench)",
            comments=(
                "NuBench neutrino events from the Cluster geometry "
                "(inspired by Baikal-GVD), simulated in water with "
                "energies in the 10 GeV - 100 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "flower_l": NuBenchSpec(
            erda_hash="EJylHQXkBr",
            detector_cls=FlowerL,
            experiment="Flower L (NuBench)",
            comments=(
                "NuBench neutrino events from the Flower L geometry "
                "(inspired by KM3NeT-ARCA), simulated in water with "
                "energies in the 10 GeV - 100 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "flower_s": NuBenchSpec(
            erda_hash="cUPqNKMRbQ",
            detector_cls=FlowerS,
            experiment="Flower S (NuBench)",
            comments=(
                "NuBench neutrino events from the Flower S geometry "
                "(inspired by KM3NeT-ORCA), simulated in water with "
                "energies in the 10 GeV - 1 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "flower_xl": NuBenchSpec(
            erda_hash="foVpx81yBz",
            detector_cls=FlowerXL,
            experiment="Flower XL (NuBench)",
            comments=(
                "NuBench neutrino events from the Flower XL geometry "
                "(inspired by TRIDENT), simulated in water with "
                "energies in the 10 GeV - 100 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "hexagon": NuBenchSpec(
            erda_hash="GTf1gIlBbZ",
            detector_cls=Hexagon,
            experiment="Hexagon (NuBench)",
            comments=(
                "NuBench neutrino events from the Hexagon geometry "
                "(inspired by IceCube), simulated in water with "
                "energies in the 10 GeV - 100 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "hexagon_ice_le": NuBenchSpec(
            erda_hash="b9VHSF9X64",
            detector_cls=Hexagon,
            experiment="IceCube Hexagon Ice LE (NuBench)",
            comments=(
                "NuBench neutrino events from the Hexagon geometry "
                "(inspired by IceCube), simulated in ice. Low-energy "
                "sample with energies in the 10 GeV - 1 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "triangle": NuBenchSpec(
            erda_hash="ER3B0TlPqR",
            detector_cls=Triangle,
            experiment="Triangle (NuBench)",
            comments=(
                "NuBench neutrino events from the Triangle geometry "
                "(inspired by P-ONE), simulated in water with "
                "energies in the 10 GeV - 100 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        # Locally-converted Prometheus sample on the IceCube/Hexagon geometry
        # (felixyu all_sky_prometheus_icgeo_gen). Same 5160-module geometry as
        # `hexagon`, so it reuses the Hexagon detector. Photons are merged into
        # pulses at conversion time (3.2 ns TTS window; charge = photon count,
        # t = mean arrival), with the NuBench >=4-pulse cut; a single `photons`
        # pulsemap serves all splits. Not ERDA-hosted: the LMDB and selection
        # files must already exist under download_dir/<name>/.
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
            event_truth=[
                "interaction",
                "initial_state_energy",
                "initial_state_type",
                "initial_state_zenith",
                "initial_state_azimuth",
                "initial_state_x",
                "initial_state_y",
                "initial_state_z",
            ],
            pulsemap_per_split={
                "train": "photons",
                "val": "photons",
                "test": "photons",
            },
        ),
        # Same as `icecube86_icgeo`, but with NuBench stochastic noise baked in:
        # per event, noise photons are sampled across a >=5 us trigger window
        # (ice IceCube rate 0.3 kHz, N_PMT=1 -> ~7.7 noise/event) and merged
        # with the signal photons. `charge` is the signal+noise photon count and
        # `is_signal` is the per-pulse signal fraction (0 pure-noise .. 1 pure-
        # signal). Pulse smearing is left as an on-the-fly augmentation. Same
        # `photons` pulsemap/truth/geometry; just a different LMDB under
        # download_dir/<name>/.
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
            event_truth=[
                "interaction",
                "initial_state_energy",
                "initial_state_type",
                "initial_state_zenith",
                "initial_state_azimuth",
                "initial_state_x",
                "initial_state_y",
                "initial_state_z",
            ],
            pulsemap_per_split={
                "train": "photons",
                "val": "photons",
                "test": "photons",
            },
        ),
    }

    _truth_table = "mc_truth"
    _available_backends = ["sqlite", "lmdb"]
    _creator = "NuBench Team"
    _citation = "https://arxiv.org/abs/2511.13111"
    _pulse_truth = None

    def __init__(
        self,
        name: str,
        download_dir: str,
        data_representation: DataRepresentation,
        train_selection: Optional[List[int]] = None,
        test_selection: Optional[List[int]] = None,
        backend: str = "sqlite",
        pre_computed_representation: Optional[str] = "GraphDefinition",
        disable_test_perturbation: bool = True,
        **kwargs: Any,
    ) -> None:
        """Construct a NuBench dataset by registry name.

        Args:
            name: Registry key of the NuBench dataset (see
                :meth:`available_datasets`).
            download_dir: Directory to download and extract the
                dataset into.
            data_representation: Data representation whose detector
                must match the one expected by the selected dataset.
            train_selection: Optional list of ``event_no`` to use for
                the train split, overriding the default selection file.
                Must be a subset of the default train selection.
            test_selection: Optional list of ``event_no`` to use for the
                test split. Must be a subset of either the default train
                selection or the default test selection; the matching
                pulsemap is used automatically -- a subset of the train
                selection is served from the train (``merged_photons``)
                pulsemap, a subset of the test selection from the
                ``pulses_no_noise`` pulsemap. A selection spanning both (or
                neither) raises ``ValueError``. The two pulsemaps are
                different processing stages of the chain and are not directly
                comparable.
            backend: ``"sqlite"`` (default, ERDA-hosted) or ``"lmdb"``.
                The on-disk layout under ``download_dir/<name>/`` is
                parallel for both backends: SQLite expects
                ``merged/merged.db`` while LMDB expects
                ``merged/merged.lmdb``. Parquet selection files live at
                ``selections/`` in either case. WIP: the LMDB backend
                is currently produced locally by the
                ``scripts/convert_sqlite_to_lmdb.py`` converter;
                precomputed LMDBs will be ERDA-hosted in the future,
                at which point ``prepare_data`` will download them
                automatically just like SQLite.
            pre_computed_representation: LMDB-only. Field name under
                which the precomputed ``DataRepresentation`` was stored
                at conversion time. Defaults to ``"GraphDefinition"``.
                Set to ``None`` to read raw tables instead.
            disable_test_perturbation: If True (default), the test split is
                built from a perturbation-free copy of
                ``data_representation`` so its already-smeared
                ``pulses_no_noise`` events are not perturbed a second time.
                Set to False to keep applying ``data_representation``'s
                ``perturbation_dict`` to the test split as well.
            **kwargs: Forwarded to :class:`ERDAHostedDataset`.
        """
        if name not in self._registry:
            raise ValueError(
                f"Unknown NuBench dataset {name!r}. "
                f"Available: {sorted(self._registry)}"
            )
        spec = self._registry[name]

        actual_detector = type(data_representation._detector)
        if not issubclass(actual_detector, spec.detector_cls):
            raise ValueError(
                f"NuBench dataset {name!r} requires a data representation "
                f"with detector {spec.detector_cls.__name__}, got "
                f"{actual_detector.__name__}."
            )

        self._name = name
        self._spec = spec
        self._custom_train_selection = train_selection
        self._custom_test_selection = test_selection
        # Pulsemap for a custom test partition, resolved in `_prepare_args` from
        # whether the selection is a subset of the train or the test selection.
        self._custom_test_pulsemap: Optional[str] = None
        self._experiment = spec.experiment
        self._comments = spec.comments
        self._features = spec.features
        self._event_truth = spec.event_truth
        self._file_hashes = {"sqlite": spec.erda_hash}
        if spec.lmdb_erda_hash is not None:
            # WIP: enables ERDA download for the LMDB backend once the
            # converted databases are published.
            self._file_hashes["lmdb"] = spec.lmdb_erda_hash
        self._pre_computed_representation = pre_computed_representation
        # Seed with the training pulsemap; `_create_dataset` swaps it per
        # split so train/val/test can use different pulsemaps.
        self._pulsemaps = [spec.pulsemap_per_split["train"]]

        # Test events live in the `pulses_no_noise` pulsemap, whose charge
        # and time already carry NuBench's pulse smearing (std 0.25 p.e. and
        # 1 ns; arXiv:2511.13111). By default a perturbation-free twin lets
        # `_create_dataset` build the test split without re-applying that
        # smearing, while train/val keep the configured perturbation that
        # emulates it on the raw `merged_photons` hits. With the flag off the
        # test split shares the perturbing representation. Set before
        # `super().__init__`, which builds the test split via `_create_dataset`
        # and so reads these attributes.
        self._disable_test_perturbation = disable_test_perturbation
        self._test_data_representation = (
            self._without_perturbation(data_representation)
            if disable_test_perturbation
            else data_representation
        )

        super().__init__(
            download_dir=download_dir,
            data_representation=data_representation,
            backend=backend,
            **kwargs,
        )

    @classmethod
    def available_datasets(cls) -> List[str]:
        """Return the list of registered NuBench dataset names."""
        return sorted(cls._registry)

    @property
    def dataset_dir(self) -> str:
        """Return the root directory of the extracted dataset."""
        return os.path.join(self._download_dir, self._name)

    def prepare_data(self) -> None:
        """Ensure dataset files are present.

        For ``sqlite`` (and, once published, ``lmdb`` with an ERDA
        hash) this triggers the ERDA download/extract when files are
        missing. For ``lmdb`` without a published hash — the current
        WIP state — nothing is downloaded: the LMDB must be produced
        locally by ``scripts/convert_sqlite_to_lmdb.py`` and a clear
        error is raised if it isn't there.
        """
        if self._files_present():
            return
        if self._backend == "lmdb" and "lmdb" not in self._file_hashes:
            # WIP: no ERDA hash yet for converted LMDBs. Tell the user
            # to run the local converter instead of trying to download.
            raise FileNotFoundError(
                f"NuBench dataset {self._name!r} (backend='lmdb'): "
                f"expected {self._spec.lmdb_relpath} and parquet selection "
                f"files under {self.dataset_dir}/selections/. Precomputed "
                "LMDBs are not yet ERDA-hosted; run "
                "scripts/convert_sqlite_to_lmdb.py first."
            )
        super().prepare_data()
        if not self._files_present():
            raise FileNotFoundError(
                f"NuBench dataset {self._name!r}: expected files not found "
                f"under {self.dataset_dir} after download+extract."
            )

    def _files_present(self) -> bool:
        """Check that the database and selection files exist on disk."""
        if self._backend == "lmdb":
            db_rel = self._spec.lmdb_relpath
        else:
            db_rel = self._spec.db_relpath
        required = [db_rel, *self._spec.selection_relpaths.values()]
        return all(
            os.path.exists(os.path.join(self.dataset_dir, rel))
            for rel in required
        )

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        if backend.lower() == "lmdb":
            db_path = os.path.join(self.dataset_dir, self._spec.lmdb_relpath)
        else:
            db_path = os.path.join(self.dataset_dir, self._spec.db_relpath)
        default_train_sel = _read_event_nos(
            os.path.join(
                self.dataset_dir, self._spec.selection_relpaths["train"]
            )
        )
        test_sel = _read_event_nos(
            os.path.join(
                self.dataset_dir, self._spec.selection_relpaths["test"]
            )
        )

        train_sel = default_train_sel
        if self._custom_train_selection is not None:
            train_sel = self._apply_custom_selection(
                self._custom_train_selection, default_train_sel, "train"
            )
        if self._custom_test_selection is not None:
            # A custom test partition may be a held-out slice of the train data
            # (events live in `merged_photons`) or a slice of the official test
            # set (events live in `pulses_no_noise`). The two default selections
            # are disjoint, so subset membership unambiguously identifies which,
            # and hence which pulsemap to serve (see `_create_dataset`). Raise if
            # the selection belongs to neither.
            custom = list(self._custom_test_selection)
            custom_set = set(custom)
            if custom_set.issubset(default_train_sel):
                self._custom_test_pulsemap = self._spec.pulsemap_per_split[
                    "train"
                ]
            elif custom_set.issubset(test_sel):
                self._custom_test_pulsemap = self._spec.pulsemap_per_split[
                    "test"
                ]
            else:
                orphans = sorted(
                    custom_set - set(default_train_sel) - set(test_sel)
                )
                raise ValueError(
                    f"Custom test selection is not a subset of either the "
                    f"train or the test selection: {len(orphans)} event_no(s) "
                    f"in neither (e.g. {orphans[:5]})."
                )
            test_sel = custom

        if (
            self._custom_train_selection is not None
            or self._custom_test_selection is not None
        ):
            # Custom selections can place the same event_no in both splits (a
            # custom test drawn from the train pool is the common case), so
            # enforce a disjoint train/test split. The default selections are
            # disjoint by construction, hence the guard.
            overlap = set(test_sel).intersection(train_sel)
            if overlap:
                raise ValueError(
                    f"Custom train and test selections overlap: "
                    f"{len(overlap)} shared event_no(s) (e.g. "
                    f"{sorted(overlap)[:5]}). Train and test must be disjoint."
                )

        dataset_args = {
            "path": db_path,
            "pulsemaps": self._pulsemaps,
            "features": features,
            "truth": truth,
            "truth_table": self._truth_table,
            "data_representation": self._data_representation,
            "labels": {
                "direction": Direction(
                    azimuth_key="initial_state_azimuth",
                    zenith_key="initial_state_zenith",
                ),
                "track": Track(
                    pid_key="initial_state_type",
                    interaction_key="interaction",
                ),
            },
        }
        if backend.lower() == "lmdb":
            dataset_args["pre_computed_representation"] = (
                self._pre_computed_representation
            )
        return dataset_args, train_sel, test_sel

    @staticmethod
    def _apply_custom_selection(
        custom: List[int], default: List[int], split: str
    ) -> List[int]:
        default_set = set(default)
        missing = [e for e in custom if e not in default_set]
        if missing:
            raise ValueError(
                f"Custom {split} selection is not a subset of the default "
                f"{split} selection: {len(missing)} event_no(s) missing "
                f"(e.g. {missing[:5]})."
            )
        return list(custom)

    @staticmethod
    def _without_perturbation(
        data_representation: DataRepresentation,
    ) -> DataRepresentation:
        """Return a representation that never perturbs its inputs.

        A representation with no ``perturbation_dict`` is returned unchanged;
        otherwise a deep copy with perturbation disabled is returned so the
        original (used for the train/val splits) keeps perturbing.
        """
        if not isinstance(
            getattr(data_representation, "_perturbation_dict", None), dict
        ):
            return data_representation
        unperturbed = deepcopy(data_representation)
        unperturbed._perturbation_dict = None
        return unperturbed

    def _create_dataset(
        self,
        selection: Union[List[int], List[List[int]], List[float]],
    ) -> Union[EnsembleDataset, Dataset]:
        """Select the correct pulsemap for this split, then delegate."""
        pmap = dict(self._spec.pulsemap_per_split)
        # A custom test partition is served from the pulsemap resolved in
        # `_prepare_args` (train `merged_photons` or test `pulses_no_noise`),
        # depending on which default selection it is a subset of.
        if self._custom_test_pulsemap is not None:
            pmap["test"] = self._custom_test_pulsemap
        if selection is self._test_selection:
            key = "test"
        elif selection is getattr(self, "_val_selection", None):
            key = "val"
        else:
            key = "train"
        self._dataset_args["pulsemaps"] = [pmap[key]]
        # The default test pulses (`pulses_no_noise`) are already smeared, so
        # build that split from the perturbation-free twin. A custom test
        # partition redirected to the train (`merged_photons`) pulsemap is raw
        # and must keep the perturbing original to emulate the smearing, like
        # train/val.
        test_is_presmeared = (
            key == "test"
            and pmap["test"] == self._spec.pulsemap_per_split["test"]
        )
        # Smearing the `pulses_no_noise` test pulsemap (default test set or a
        # custom subset of it) re-applies a perturbation on top of NuBench's
        # baked-in 0.25 p.e. / 1 ns smearing, i.e. double-smears the test
        # inputs. Warn when that combination is requested and actually perturbs.
        if (
            test_is_presmeared
            and not self._disable_test_perturbation
            and isinstance(
                getattr(self._data_representation, "_perturbation_dict", None),
                dict,
            )
        ):
            self.warning_once(
                "Perturbing the `pulses_no_noise` test pulsemap, whose charge "
                "and time already carry NuBench's 0.25 p.e. / 1 ns smearing: "
                "the test inputs will be smeared twice. Set "
                "`disable_test_perturbation=True` unless this is intended."
            )
        self._dataset_args["data_representation"] = (
            self._test_data_representation
            if test_is_presmeared
            else self._data_representation
        )
        return super()._create_dataset(selection)
