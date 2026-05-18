"""SQLite Extractor base for conversion from internal SQLite format."""

from abc import abstractmethod
from typing import List, Tuple

import pandas as pd
import polars as pl

from graphnet.data.extractors import Extractor


class SQLiteExtractor(Extractor):
    """Base class for extracting information from GraphNeT SQLite databases.

    Reads the requested event subset via ``polars.read_database_uri``, which
    delegates to the vectorized ``connectorx`` backend, then converts the
    result to pandas at the return boundary so downstream writers keep
    their existing pandas API.
    """

    def __init__(self, extractor_name: str):
        """Construct SQLiteExtractor.

        Args:
            extractor_name: Name of the `SQLiteExtractor` instance.
                Used to keep track of the provenance of different data,
                and to name tables to which this data is saved.
        """
        super().__init__(extractor_name=extractor_name)

    @abstractmethod
    def __call__(
        self, fileset: Tuple[str, List[int]]
    ) -> pd.DataFrame:  # type: ignore[override]
        """Extract information using a DB path and event subset.

        Args:
            fileset: Tuple of (sqlite db path, list of event numbers).
        """
        db_path, event_nos = fileset
        event_list = ",".join(map(str, event_nos))
        query = (
            f"SELECT * FROM {self._extractor_name} "
            f"WHERE event_no IN ({event_list})"
        )
        df = pl.read_database_uri(query=query, uri=f"sqlite://{db_path}")
        return df.to_pandas()
