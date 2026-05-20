"""LMDB-specific utility functions for use in `graphnet.data`."""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import lmdb
import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from graphnet.models.data_representation import DataRepresentation


def _resolve_serializer(
    serialization_method: str,
) -> Optional[Callable[[Any], bytes]]:
    """Resolve serialization callable from serialization method name.

    Args:
        serialization_method: Name of the serialization method
            (e.g., "pickle", "json", "msgpack", "dill").

    Returns:
        Serialization callable that takes an object and returns bytes, or
        None if the method is not supported or is custom.
    """
    if serialization_method == "pickle":
        import pickle

        return pickle.dumps
    if serialization_method == "json":
        import json

        return lambda obj: json.dumps(obj).encode("utf-8")
    if serialization_method == "msgpack":
        try:
            import msgpack  # type: ignore

            return lambda obj: msgpack.packb(obj, use_bin_type=True)
        except ImportError:
            raise ImportError("msgpack is not installed.")
    if serialization_method == "dill":
        try:
            import dill  # type: ignore

            return dill.dumps
        except ImportError:
            raise ImportError("dill is not installed.")
    # For "__custom__" or unknown methods, return None
    return None


def _resolve_deserializer(
    serialization_method: str,
) -> Optional[Callable[[bytes], Any]]:
    """Resolve deserialization callable from serialization method name.

    Args:
        serialization_method: Name of the serialization method
            (e.g., "pickle", "json", "msgpack", "dill").

    Returns:
        Deserialization callable that takes bytes and returns the deserialized
        object, or None if the method is not supported or is custom.
    """
    if serialization_method == "pickle":
        import pickle

        return pickle.loads
    if serialization_method == "json":
        import json

        return lambda data: json.loads(data.decode("utf-8"))
    if serialization_method == "msgpack":
        try:
            import msgpack  # type: ignore

            return lambda data: msgpack.unpackb(data, raw=False)
        except ImportError:
            raise ImportError("msgpack is not installed.")
    if serialization_method == "dill":
        try:
            import dill  # type: ignore

            return dill.loads
        except ImportError:
            raise ImportError("dill is not installed.")
    # For "__custom__" or unknown methods, return None
    return None


def get_serialization_method_name(lmdb_path: str) -> Optional[str]:
    """Retrieve the serialization method name for an LMDB database.

    Args:
        lmdb_path: Path to the LMDB database directory.

    Returns:
        The serialization method name
        (e.g., "pickle", "json", "msgpack", "dill")
        or None if the metadata is not found or the database cannot be opened.
    """
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=True)
        with env.begin(write=False) as txn:
            metadata_key = b"__meta_serialization__"
            metadata_value = txn.get(metadata_key)
            if metadata_value is not None:
                return metadata_value.decode("utf-8")
        env.close()
    except Exception:
        pass
    return None


def get_serialization_method(
    lmdb_path: str,
) -> Optional[Callable[[bytes], Any]]:
    """Retrieve the deserialization callable for an LMDB database.

    Args:
        lmdb_path: Path to the LMDB database directory.

    Returns:
        Deserialization callable that takes bytes and returns the deserialized
        object, or None if the metadata is not found, the database cannot be
        opened, or the serialization method is custom/unsupported.
    """
    method_name = get_serialization_method_name(lmdb_path)
    if method_name is not None:
        return _resolve_deserializer(method_name)
    return None


def query_database(
    database: str,
    index: int,
    deserialization: Optional[Callable[[bytes], Any]] = None,
) -> Any:
    """Retrieve and deserialize a record from an LMDB database.

    Args:
        database: Path to the LMDB database directory.
        index: The index (event_no) of the record to retrieve.
        deserialization: Optional deserialization callable. If not provided,
            the deserialization method will be fetched from the database
            metadata.

    Returns:
        The deserialized object stored at the given index.

    Raises:
        KeyError: If the index is not found in the database.
        ValueError: If deserialization is required but cannot be determined
            from the database metadata.
    """
    # Get deserialization method if not provided
    if deserialization is None:
        deserialization = get_serialization_method(database)
        if deserialization is None:
            raise ValueError(
                "Deserialization method could not be determined from "
                "database metadata. Please provide a deserialization "
                "callable."
            )

    # Open database and retrieve record
    env = lmdb.open(database, readonly=True, lock=False, subdir=True)
    try:
        with env.begin(write=False) as txn:
            # Convert index to key (same format as in LMDBWriter)
            key_bytes = str(index).encode("utf-8")
            value_bytes = txn.get(key_bytes)
            if value_bytes is None:
                raise KeyError(f"Index {index} not found in database.")
            # Deserialize and return
            result = deserialization(value_bytes)
        return result
    except (KeyError, ValueError):
        raise
    except Exception as e:
        # Re-raise other exceptions with context
        raise RuntimeError(f"Failed to query database at index {index}") from e
    finally:
        env.close()


def _get_data_representation_metadata_dict(
    lmdb_path: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve the raw data representation metadata dictionary from LMDB.

    This is a helper function for internal use.

    Args:
        lmdb_path: Path to the LMDB database directory.

    Returns:
        A dictionary mapping field names to data representation configs,
        or None if the metadata is not found or the database cannot be opened.
    """
    try:
        # First get the serialization method to deserialize the metadata
        serialization_method = get_serialization_method(lmdb_path)
        if serialization_method is None:
            return None

        env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=True)
        with env.begin(write=False) as txn:
            metadata_key = b"__meta_data_representations__"
            metadata_value = txn.get(metadata_key)
            if metadata_value is not None:
                # Deserialize the metadata
                return serialization_method(metadata_value)
        env.close()
    except Exception:
        pass
    return None


def get_data_representation_from_metadata(
    lmdb_path: str,
    field_name: str,
    trust: bool = False,
) -> Optional["DataRepresentation"]:
    """Retrieve a DataRepresentation instance from an LMDB database metadata.

    Args:
        lmdb_path: Path to the LMDB database directory.
        field_name: The field name of the data representation to retrieve
            (e.g., "GraphDefinition", "GraphDefinition_0", etc.).
        trust: Whether to trust the ModelConfig enough to `eval(...)`
            any lambda function expressions contained. Defaults to False.

    Returns:
        A DataRepresentation instance reconstructed from the stored config,
        or None if the metadata is not found, the field name doesn't exist,
        or the database cannot be opened.

    Raises:
        KeyError: If the field_name is not found in the metadata.
    """
    from graphnet.models.data_representation import DataRepresentation

    # Get the full metadata dictionary
    metadata = _get_data_representation_metadata_dict(lmdb_path)
    if metadata is None:
        return None

    # Extract the config for the specific field name
    if field_name not in metadata:
        raise KeyError(
            f"Field name '{field_name}' not found in data representation "
            f"metadata. Available field names: {list(metadata.keys())}"
        )

    model_config = metadata[field_name]

    # Use DataRepresentation.from_config to create the instance
    return DataRepresentation.from_config(model_config, trust=trust)


def get_all_indices(database: str) -> List[int]:
    """Retrieve all indices (event numbers) from an LMDB database.

    Args:
        database: Path to the LMDB database directory.

    Returns:
        A sorted list of all indices (event numbers) in the database,
        excluding the metadata entry.

    Raises:
        RuntimeError: If the database cannot be opened or accessed.
    """
    env = lmdb.open(database, readonly=True, lock=False, subdir=True)
    try:
        indices: List[int] = []
        metadata_key = b"__meta_serialization__"

        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key_bytes, _ in cursor:
                # Skip the metadata entries
                if (
                    key_bytes == metadata_key
                    or key_bytes == b"__meta_data_representations__"
                ):
                    continue
                # Convert key back to integer
                try:
                    key_str = key_bytes.decode("utf-8")
                    index = int(key_str)
                    indices.append(index)
                except (ValueError, UnicodeDecodeError):
                    # Skip keys that can't be decoded as integers
                    # (shouldn't happen in normal operation,
                    # but handle gracefully)
                    continue

        return sorted(indices)
    except Exception as e:
        raise RuntimeError("Failed to retrieve indices from database") from e
    finally:
        env.close()


_META_SERIALIZATION_KEY = b"__meta_serialization__"
_META_DATA_REPRESENTATIONS_KEY = b"__meta_data_representations__"


def assign_data_representation_field_names(
    data_representations: List["DataRepresentation"],
    existing_field_names: Optional[Iterable[str]] = None,
) -> Dict[str, "DataRepresentation"]:
    """Assign LMDB field names for a list of `DataRepresentation` instances.

    Mirrors the disambiguation policy used by `LMDBWriter` at write time:
    the first instance of a class is stored under its class name; subsequent
    instances get a numeric suffix (`<ClassName>_<n>`).

    Args:
        data_representations: Instances to assign field names to.
        existing_field_names: Field names already present in an LMDB. Used to
            seed disambiguation so retrospective additions pick the next free
            `<ClassName>_<n>` slot rather than clashing.

    Returns:
        Mapping from field name to `DataRepresentation` instance.
    """
    field_name_to_rep: Dict[str, "DataRepresentation"] = {}
    class_name_counts: Dict[str, int] = {}
    reserved: set = set(existing_field_names or [])

    def _next_free(class_name: str) -> str:
        if class_name in class_name_counts:
            class_name_counts[class_name] += 1
            return f"{class_name}_{class_name_counts[class_name]}"
        class_name_counts[class_name] = 0
        return class_name

    for data_rep in data_representations:
        base_class_name = data_rep.__class__.__name__

        if base_class_name in reserved or base_class_name in field_name_to_rep:
            # Base name unavailable -- start from _0 and walk forward.
            if base_class_name not in class_name_counts:
                class_name_counts[base_class_name] = -1
            key_name = _next_free(base_class_name)
            while key_name in reserved or key_name in field_name_to_rep:
                key_name = _next_free(base_class_name)
        else:
            key_name = _next_free(base_class_name)

        field_name_to_rep[key_name] = data_rep

    return field_name_to_rep


def compute_data_representations_dict(
    pulse_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    field_name_to_rep: Dict[str, "DataRepresentation"],
    truth_label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run each `DataRepresentation.forward(...)` for a single event.

    Args:
        pulse_df: Per-event pulse-level features as a DataFrame.
        truth_df: Per-event truth as a one-row DataFrame.
        field_name_to_rep: Mapping from LMDB field name to data
            representation, as produced by
            `assign_data_representation_field_names`.
        truth_label_names: Optional subset of truth columns to forward to the
            representation.

    Returns:
        Mapping from field name to the output of `data_rep.forward(...)`.

    Raises:
        ValueError: If `pulse_df` or `truth_df` is empty. An LMDB event
            without pulses or truth indicates upstream corruption.
    """
    if pulse_df.empty or truth_df.empty:
        raise ValueError(
            "Cannot compute data representation: pulse_df or truth_df is "
            "empty. Every event in the LMDB should carry both."
        )
    if len(truth_df) != 1:
        raise ValueError(
            f"Expected exactly one truth row per event, got {len(truth_df)}. "
            "Multiple truth rows would silently drop information when "
            "taking iloc[0]."
        )

    truth_row = truth_df.iloc[0].to_dict()
    if truth_label_names is not None:
        truth_row = {
            k: truth_row[k] for k in truth_label_names if k in truth_row
        }
    truth_dicts = [truth_row]

    output: Dict[str, Any] = {}
    for field_name, data_rep in field_name_to_rep.items():
        feature_names = list(data_rep._input_feature_names)
        x = pulse_df[feature_names].to_numpy()
        output[field_name] = data_rep.forward(
            input_features=x,
            input_feature_names=feature_names,
            truth_dicts=truth_dicts,
        )  # type: ignore[arg-type]

    return output


def build_event_value(
    per_event_tables: Dict[str, pd.DataFrame],
    data_representations: Optional[List["DataRepresentation"]] = None,
    pulsemap_extractor_name: Optional[str] = None,
    truth_extractor_name: Optional[str] = None,
    truth_label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the LMDB value dict for a single event from per-event tables.

    Always includes the raw extractor tables. If `data_representations` is
    provided, also computes them and stores their outputs under
    `"data_representations"` keyed by class name.
    """
    extractor_dict = {
        name: df.to_dict(orient="list")
        for name, df in per_event_tables.items()
    }
    if data_representations is None:
        return extractor_dict

    if pulsemap_extractor_name is None or truth_extractor_name is None:
        raise ValueError(
            "pulsemap_extractor_name and truth_extractor_name must be set "
            "when using data_representation."
        )

    if pulsemap_extractor_name not in per_event_tables:
        raise KeyError(
            f"Pulsemap extractor {pulsemap_extractor_name!r} not found in "
            f"per-event tables (available: {list(per_event_tables)})."
        )
    if truth_extractor_name not in per_event_tables:
        raise KeyError(
            f"Truth extractor {truth_extractor_name!r} not found in "
            f"per-event tables (available: {list(per_event_tables)})."
        )
    pulse_df = per_event_tables[pulsemap_extractor_name]
    truth_df = per_event_tables[truth_extractor_name]

    field_name_to_rep = assign_data_representation_field_names(
        data_representations
    )
    extractor_dict["data_representations"] = compute_data_representations_dict(
        pulse_df=pulse_df,
        truth_df=truth_df,
        field_name_to_rep=field_name_to_rep,
        truth_label_names=truth_label_names,
    )
    return extractor_dict


def build_data_representation_metadata(
    field_name_to_rep: Dict[str, "DataRepresentation"],
) -> Dict[str, Any]:
    """Build the metadata dict (`{field_name: config}`) for LMDB storage."""
    return {
        field_name: data_rep.config
        for field_name, data_rep in field_name_to_rep.items()
    }


def write_data_representation_metadata(
    txn: lmdb.Transaction,
    metadata_dict: Dict[str, Any],
    serializer: Callable[[Any], bytes],
) -> None:
    """Write the data-representation metadata dict into an LMDB transaction."""
    txn.put(
        _META_DATA_REPRESENTATIONS_KEY,
        serializer(metadata_dict),
        overwrite=True,
    )


def add_data_representations_to_lmdb(  # noqa: C901
    lmdb_path: str,
    data_representations: Union[
        "DataRepresentation", List["DataRepresentation"]
    ],
    pulsemap_extractor_name: str,
    truth_extractor_name: str,
    truth_label_names: Optional[List[str]] = None,
    event_nos: Optional[List[int]] = None,
    overwrite: bool = False,
    map_size_bytes: int = 8 * 1024 * 1024 * 1024,
    batch_size: int = 1000,
    num_workers: int = 1,
) -> Dict[str, "DataRepresentation"]:
    """Retroactively add precomputed data representations to an LMDB.

    Walks every event in `lmdb_path` (or only `event_nos` if given),
    recomputes the requested `DataRepresentation`(s) from the stored raw
    extractor tables, and writes the outputs back under
    `value["data_representations"][<field_name>]`. The
    `__meta_data_representations__` metadata is updated so that
    `LMDBDataset(pre_computed_representation=<field_name>)` and
    `get_data_representation_from_metadata(...)` keep working.

    Args:
        lmdb_path: Path to an existing LMDB directory.
        data_representations: One or more `DataRepresentation` instances to
            compute and store. Each must accept the stored pulse-feature
            columns as input.
        pulsemap_extractor_name: Name of the extractor (stored top-level key)
            providing per-event pulse features.
        truth_extractor_name: Name of the extractor providing event truth.
        truth_label_names: Optional subset of truth columns to pass to
            `data_rep.forward(...)`.
        event_nos: Optional subset of event numbers to process. If None
            (default), every event in the LMDB is processed. Useful when
            different event subsets use different `pulsemap_extractor_name`
            values -- call once per subset.
        overwrite: If False (default), refuse to clobber any existing
            representation whose field name would collide with a newly
            assigned one. If True, conflicting names are reused.
        map_size_bytes: LMDB map size for the read-write reopen.
        batch_size: Number of events per write transaction.
        num_workers: Number of worker processes for `data_rep.forward(...)`.
            1 (default) keeps everything in the main process. With >1,
            workers compute representations in parallel and the main
            process serializes writes back to the LMDB (LMDB allows only
            one writer per env).

    Returns:
        Mapping from the field names that were written to their corresponding
        `DataRepresentation` instances. Useful for the caller to know what
        name to pass as `pre_computed_representation` afterwards.
    """
    if isinstance(data_representations, list):
        new_reps = data_representations
    else:
        new_reps = [data_representations]
    if not new_reps:
        raise ValueError("data_representations must be non-empty.")

    serialization_method = get_serialization_method_name(lmdb_path)
    if serialization_method is None:
        raise ValueError(
            f"Could not determine serialization method for LMDB at "
            f"{lmdb_path!r}. The database may be missing metadata or "
            f"corrupted."
        )
    if serialization_method == "__custom__":
        raise ValueError(
            "LMDB was written with a custom serializer; retrospective "
            "writes are not supported because the original serializer "
            "callable is not recoverable from metadata."
        )

    serializer = _resolve_serializer(serialization_method)
    deserializer = _resolve_deserializer(serialization_method)
    if serializer is None or deserializer is None:
        raise ValueError(
            f"Serialization method {serialization_method!r} is not "
            f"supported for retrospective writes."
        )

    existing_metadata = _get_data_representation_metadata_dict(lmdb_path) or {}
    existing_field_names = list(existing_metadata.keys())

    if overwrite:
        field_name_to_rep = assign_data_representation_field_names(new_reps)
    else:
        field_name_to_rep = assign_data_representation_field_names(
            new_reps, existing_field_names=existing_field_names
        )
        collisions = [
            name for name in field_name_to_rep if name in existing_metadata
        ]
        if collisions:
            raise ValueError(
                f"Field name(s) {collisions} already present in LMDB "
                f"metadata. Pass overwrite=True to replace them."
            )

    env = lmdb.open(
        lmdb_path,
        map_size=map_size_bytes,
        subdir=True,
        readonly=False,
        lock=True,
        max_dbs=1,
    )
    try:
        indices = (
            list(event_nos)
            if event_nos is not None
            else get_all_indices(lmdb_path)
        )
        _process_events(
            env=env,
            indices=indices,
            lmdb_path=lmdb_path,
            serializer=serializer,
            deserializer=deserializer,
            field_name_to_rep=field_name_to_rep,
            pulsemap_extractor_name=pulsemap_extractor_name,
            truth_extractor_name=truth_extractor_name,
            truth_label_names=truth_label_names,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        merged_metadata = dict(existing_metadata)
        merged_metadata.update(
            build_data_representation_metadata(field_name_to_rep)
        )
        with env.begin(write=True) as meta_txn:
            write_data_representation_metadata(
                meta_txn, merged_metadata, serializer
            )

        env.sync()
    finally:
        env.close()

    return field_name_to_rep


def _read_event_value(
    env: lmdb.Environment,
    index: int,
    lmdb_path: str,
    deserializer: Callable[[bytes], Any],
    pulsemap_extractor_name: str,
    truth_extractor_name: str,
) -> Dict[str, Any]:
    """Read+deserialize a single event and validate required extractors."""
    key = str(index).encode("utf-8")
    with env.begin(write=False) as txn:
        value_bytes = txn.get(key)
    if value_bytes is None:
        raise KeyError(f"Event {index} not present in {lmdb_path!r}.")
    value = deserializer(value_bytes)

    if pulsemap_extractor_name not in value:
        raise KeyError(
            f"Event {index} is missing pulsemap extractor "
            f"{pulsemap_extractor_name!r}. Available top-level keys for "
            f"this event: {sorted(value.keys())}."
        )
    if truth_extractor_name not in value:
        raise KeyError(
            f"Event {index} is missing truth extractor "
            f"{truth_extractor_name!r}. Available top-level keys for "
            f"this event: {sorted(value.keys())}."
        )
    return value


def _flush_batch(
    env: lmdb.Environment,
    keys: List[bytes],
    values: List[bytes],
) -> None:
    """Write a batch of (key, value) pairs in a single write txn."""
    if not keys:
        return
    with env.begin(write=True) as txn:
        for k, v in zip(keys, values):
            txn.put(k, v, overwrite=True)


def _process_events(
    env: lmdb.Environment,
    indices: List[int],
    lmdb_path: str,
    serializer: Callable[[Any], bytes],
    deserializer: Callable[[bytes], Any],
    field_name_to_rep: Dict[str, "DataRepresentation"],
    pulsemap_extractor_name: str,
    truth_extractor_name: str,
    truth_label_names: Optional[List[str]],
    batch_size: int,
    num_workers: int,
) -> None:
    """Iterate `indices`, compute reps, and write merged values back."""
    if num_workers <= 1:
        _process_events_serial(
            env=env,
            indices=indices,
            lmdb_path=lmdb_path,
            serializer=serializer,
            deserializer=deserializer,
            field_name_to_rep=field_name_to_rep,
            pulsemap_extractor_name=pulsemap_extractor_name,
            truth_extractor_name=truth_extractor_name,
            truth_label_names=truth_label_names,
            batch_size=batch_size,
        )
    else:
        _process_events_parallel(
            env=env,
            indices=indices,
            lmdb_path=lmdb_path,
            serializer=serializer,
            deserializer=deserializer,
            field_name_to_rep=field_name_to_rep,
            pulsemap_extractor_name=pulsemap_extractor_name,
            truth_extractor_name=truth_extractor_name,
            truth_label_names=truth_label_names,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def _process_events_serial(
    env: lmdb.Environment,
    indices: List[int],
    lmdb_path: str,
    serializer: Callable[[Any], bytes],
    deserializer: Callable[[bytes], Any],
    field_name_to_rep: Dict[str, "DataRepresentation"],
    pulsemap_extractor_name: str,
    truth_extractor_name: str,
    truth_label_names: Optional[List[str]],
    batch_size: int,
) -> None:
    """Single-process loop: read, compute, batch-write."""
    batch_keys: List[bytes] = []
    batch_values: List[bytes] = []
    for index in tqdm(
        indices,
        desc=f"Adding {list(field_name_to_rep)} to LMDB",
        unit="event",
    ):
        value = _read_event_value(
            env=env,
            index=index,
            lmdb_path=lmdb_path,
            deserializer=deserializer,
            pulsemap_extractor_name=pulsemap_extractor_name,
            truth_extractor_name=truth_extractor_name,
        )
        try:
            rep_output = compute_data_representations_dict(
                pulse_df=pd.DataFrame(value[pulsemap_extractor_name]),
                truth_df=pd.DataFrame(value[truth_extractor_name]),
                field_name_to_rep=field_name_to_rep,
                truth_label_names=truth_label_names,
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to compute data representation for event "
                f"{index}: {e}"
            ) from e

        value.setdefault("data_representations", {}).update(rep_output)
        batch_keys.append(str(index).encode("utf-8"))
        batch_values.append(serializer(value))
        if len(batch_keys) >= batch_size:
            _flush_batch(env, batch_keys, batch_values)
            batch_keys = []
            batch_values = []
    _flush_batch(env, batch_keys, batch_values)


# Worker globals -- populated by `_worker_init` in each Pool worker.
_WORKER_FIELD_NAME_TO_REP: Dict[str, Any] = {}
_WORKER_TRUTH_LABELS: Optional[List[str]] = None


def _worker_init(
    field_names: List[str],
    rep_configs: List[Any],
    truth_label_names: Optional[List[str]],
) -> None:
    """Pool initializer: rebuild reps from their configs once per worker."""
    from graphnet.models.data_representation import DataRepresentation

    global _WORKER_FIELD_NAME_TO_REP, _WORKER_TRUTH_LABELS
    _WORKER_FIELD_NAME_TO_REP = {
        name: DataRepresentation.from_config(cfg, trust=True)
        for name, cfg in zip(field_names, rep_configs)
    }
    _WORKER_TRUTH_LABELS = truth_label_names


def _worker_compute(
    args: tuple,
) -> Tuple[int, bytes]:
    """Worker payload: run `compute_data_representations_dict` on one event.

    Returns the rep output as `pickle.dumps(...)` bytes rather than the
    raw dict of `torch_geometric.Data`. This bypasses torch's queue
    reducers, which would otherwise mmap each tensor to a shared-memory
    file and exhaust the host's mmap/FD/SHM budget at high
    `num_workers` * tensors-per-event.
    """
    import pickle

    index, pulse_dict, truth_dict = args
    try:
        rep_output = compute_data_representations_dict(
            pulse_df=pd.DataFrame(pulse_dict),
            truth_df=pd.DataFrame(truth_dict),
            field_name_to_rep=_WORKER_FIELD_NAME_TO_REP,
            truth_label_names=_WORKER_TRUTH_LABELS,
        )
    except ValueError as e:
        raise ValueError(
            f"Failed to compute data representation for event {index}: {e}"
        ) from e
    return index, pickle.dumps(rep_output)


def _process_events_parallel(
    env: lmdb.Environment,
    indices: List[int],
    lmdb_path: str,
    serializer: Callable[[Any], bytes],
    deserializer: Callable[[bytes], Any],
    field_name_to_rep: Dict[str, "DataRepresentation"],
    pulsemap_extractor_name: str,
    truth_extractor_name: str,
    truth_label_names: Optional[List[str]],
    batch_size: int,
    num_workers: int,
) -> None:
    """Compute reps in a Pool, write back from the main process."""
    import multiprocessing as mp

    field_names = list(field_name_to_rep.keys())
    rep_configs = [field_name_to_rep[name].config for name in field_names]

    # Per-event deserialized values; popped when the worker returns.
    pending: Dict[int, Dict[str, Any]] = {}

    def _work_iter() -> Iterator[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        for index in indices:
            value = _read_event_value(
                env=env,
                index=index,
                lmdb_path=lmdb_path,
                deserializer=deserializer,
                pulsemap_extractor_name=pulsemap_extractor_name,
                truth_extractor_name=truth_extractor_name,
            )
            pending[index] = value
            yield (
                index,
                value[pulsemap_extractor_name],
                value[truth_extractor_name],
            )

    batch_keys: List[bytes] = []
    batch_values: List[bytes] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(field_names, rep_configs, truth_label_names),
    ) as pool:
        import pickle

        results = pool.imap_unordered(
            _worker_compute, _work_iter(), chunksize=8
        )
        try:
            for index, rep_output_bytes in tqdm(
                results,
                total=len(indices),
                desc=f"Adding {list(field_name_to_rep)} to LMDB",
                unit="event",
            ):
                rep_output = pickle.loads(rep_output_bytes)
                value = pending.pop(index)
                value.setdefault("data_representations", {}).update(rep_output)
                batch_keys.append(str(index).encode("utf-8"))
                batch_values.append(serializer(value))
                if len(batch_keys) >= batch_size:
                    _flush_batch(env, batch_keys, batch_values)
                    batch_keys = []
                    batch_values = []
        except BaseException:
            # Any worker exception (or main-loop failure) -- kill all
            # in-flight workers immediately rather than letting them
            # drain. The exception itself is the one raised by the first
            # failing imap_unordered result, so re-raise after cleanup.
            pool.terminate()
            pool.join()
            raise
    _flush_batch(env, batch_keys, batch_values)
