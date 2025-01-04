"""Snowstorm dataset module hosted on the IceCube Collaboration servers."""

from graphnet.data.constants import FEATURES, TRUTH
from typing import Dict, Any, Optional, List, Tuple, Union
from graphnet.data.curated_datamodule import IceCubeHostedDataset
from sklearn.model_selection import train_test_split
from glob import glob
from graphnet.data.utilities import query_database
import os
from graphnet.training.labels import Direction, Track
from graphnet.models.graphs import GraphDefinition


class SnowStormDataset(IceCubeHostedDataset):
    """IceCube SnowStorm simulation dataset.

    More information can be found at
    https://wiki.icecube.wisc.edu/index.php/SnowStorm_MC#File_Locations
    This is a IceCube Collaboration simulation dataset.
    Requires a username and password.
    """

    _experiment = "IceCube SnowStorm dataset"
    _creator = "Severin Magel"
    _comments = (
        "Contains ~X million track events."
        " Simulation produced by the IceCube Collaboration, "
        "https://wiki.icecube.wisc.edu/index.php/SnowStorm_MC#File_Locations"
    )
    _citation = None
    _available_backends = ["sqlite"]
    # Static Member Variables:
    _pulsemaps = ["SRTInIcePulses"]
    _truth_table = "truth"
    _pulse_truth = None
    _features = FEATURES.ICECUBE86
    _event_truth = TRUTH.ICECUBE86
    _tar_flags = "--strip-components=4"
    _data_root_dir = "/data/ana/graphnet/Snowstorm_l2"

    def __init__(
        self,
        sets: List[int],
        graph_definition: GraphDefinition,
        download_dir: str,
        truth: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        validation_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SnowStorm dataset."""
        self._zipped_files = [
            os.path.join(self._data_root_dir, f"{s}.tar.gz") for s in sets
        ]
        print(self._zipped_files)

        super().__init__(
            graph_definition=graph_definition,
            download_dir=download_dir,
            truth=truth,
            features=features,
            backend="sqlite",
            train_dataloader_kwargs=train_dataloader_kwargs,
            validation_dataloader_kwargs=validation_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
        )

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare arguments for dataset.

        Args:
            backend: backend of dataset. Either "parquet" or "sqlite".
            features: List of features from user to use as input.
            truth: List of event-level truth variables from user.

        Returns: Dataset arguments, train/val selection, test selection
        """
        assert backend == "sqlite"
        dataset_paths = glob(
            os.path.join(self.dataset_dir, "**/*.db"), recursive=True
        )
        print(dataset_paths)

        train_val = []
        test = []

        for path in dataset_paths:
            print(path)
            event_nos = query_database(
                database=path,
                query=f"SELECT event_no FROM {self._truth_table}",
            )
            train_val_temp, test_temp = train_test_split(
                event_nos["event_no"].tolist(),
                test_size=0.10,
                random_state=42,
                shuffle=True,
            )

            train_val.extend(train_val_temp)
            test.extend(test_temp)

        print(len(train_val))
        print(len(test))

        dataset_args = {
            "truth_table": self._truth_table,
            "pulsemaps": self._pulsemaps,
            "path": dataset_paths,
            "graph_definition": self._graph_definition,
            "features": features,
            "truth": truth,
            "labels": {
                "direction": Direction(
                    azimuth_key="initial_state_azimuth",
                    zenith_key="initial_state_zenith",
                ),
                "track": Track(
                    pid_key="initial_state_type", interaction_key="interaction"
                ),
            },
        }

        return dataset_args, train_val, test
