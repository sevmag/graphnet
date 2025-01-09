"""Snowstorm dataset module hosted on the IceCube Collaboration servers."""

import pandas as pd
import re
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from glob import glob
from sklearn.model_selection import train_test_split

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.curated_datamodule import IceCubeHostedDataset
from graphnet.data.utilities import query_database
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
    _citation = "arXiv:1909.01530"
    _available_backends = ["sqlite"]
    # Static Member Variables:
    _pulsemaps = ["SRTInIcePulses"]
    _truth_table = "truth"
    _pulse_truth = None
    _features = FEATURES.SNOWSTORM
    _event_truth = TRUTH.SNOWSTORM
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

        self._create_comment()

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

        # get event numbers from all datasets
        event_no = []

        # get set number
        pattern = rf"{re.escape(self.dataset_dir)}/(\d+)/.*"
        self._event_counts: dict[str, int] = {}
        self._event_counts = {}
        for path in dataset_paths:
            print(path)

            # Extract the ID
            match = re.search(pattern, path)
            assert match
            set_nb = match.group(1)
            print(set_nb)

            query_df = query_database(
                database=path,
                query=f"SELECT event_no FROM {self._truth_table}",
            )
            query_df["path"] = path
            event_no.append(query_df)

            # save event count for description
            if set_nb in self._event_counts:
                self._event_counts[set_nb] += query_df.shape[0]
            else:
                self._event_counts[set_nb] = query_df.shape[0]

        event_no = pd.concat(event_no, axis=0)

        print(self._event_counts)

        print(event_no)

        # split the non-unique event numbers into train/val and test
        train_val, test = train_test_split(
            event_no,
            test_size=0.10,
            random_state=42,
            shuffle=True,
        )

        print(train_val.shape, test.shape)

        train_val = train_val.groupby("path")
        test = test.groupby("path")

        # parse into right format for CuratedDataset
        train_val_selection = []
        test_selection = []
        for path in dataset_paths:
            train_val_selection.append(
                train_val["event_no"].get_group(path).tolist()
            )
            test_selection.append(test["event_no"].get_group(path).tolist())

        print(len(train_val_selection))
        print(len(test_selection))

        dataset_args = {
            "truth_table": self._truth_table,
            "pulsemaps": self._pulsemaps,
            "path": dataset_paths,
            "graph_definition": self._graph_definition,
            "features": features,
            "truth": truth,
        }

        return dataset_args, train_val_selection, test_selection

    @classmethod
    def _create_comment(cls) -> None:
        """Print the number of events in each set."""
        fixed_string = (
            " Simulation produced by the IceCube Collaboration, "
            + "https://wiki.icecube.wisc.edu/index.php/SnowStorm_MC#File_Locations"  # noqa: E501
        )
        tot = 0
        set_string = ""
        for k, v in cls._event_counts.items():
            set_string += f"\tSet {k} contains {v:10d} events\n"
            tot += v
        print(tot)
        cls._comments = (
            f"Contains ~{tot/1e6:.1f} million events:\n"
            + set_string
            + fixed_string
        )
