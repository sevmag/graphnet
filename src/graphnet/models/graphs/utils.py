"""Utility functions for construction of graphs."""

from typing import List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
from graphnet.constants import DATA_DIR


def lex_sort(x: np.array, cluster_columns: List[int]) -> np.ndarray:
    """Sort numpy arrays according to columns on ´cluster_columns´.

    Note that `x` is sorted along the dimensions in `cluster_columns`
    backwards. I.e. `cluster_columns = [0,1,2]`
    means `x` is sorted along `[2,1,0]`.

    Args:
        x: array to be sorted.
        cluster_columns: Columns of `x` to be sorted along.

    Returns:
        A sorted version of `x`.
    """
    tmp_list = []
    for cluster_column in cluster_columns:
        tmp_list.append(x[:, cluster_column])
    return x[np.lexsort(tuple(tmp_list)), :]


def gather_cluster_sequence(
    x: np.ndarray, feature_idx: int, cluster_columns: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn `x` into rows of clusters with sequences along columns.

    Sequences along columns are added which correspond to
    gathered sequences of the feature in `x` specified by column index
    `feature_idx` associated with each column. Sequences are padded with NaN to
    be of same length. Dimension of clustered array is `[n_clusters, l +
    len(cluster_columns)]`,where l is the largest sequence length.

    **Example**:
    Suppose `x` represents a neutrino event and we have chosen to cluster on
    the PMT positions and that `feature_idx` correspond to pulse time.

    The resulting array will have dimensions `[n_pmts, m + 3]` where `m` is the
    maximum number of same-pmt pulses found in `x`, and `+3`for the three
    spatial directions  defining each cluster.

    Args:
        x:  Array for clustering
        feature_idx: Index of the feature in `x` to
        be gathered for each cluster.
        cluster_columns: Index in `x` from which to build clusters.

    Returns:
        array: Array with dimensions  `[n_clusters, l + len(cluster_columns)]`
        column_offset: Indices of the columns in `array` that defines clusters.
    """
    # sort pulses according to cluster columns
    x = lex_sort(x=x, cluster_columns=cluster_columns)

    # Calculate clusters and counts
    unique_sensors, counts = np.unique(
        x[:, cluster_columns], return_counts=True, axis=0
    )
    # sort DOMs and pulse-counts
    sensor_counts = counts.reshape(-1, 1)
    contingency_table = np.concatenate([unique_sensors, sensor_counts], axis=1)
    sensors_in_contingency_table = np.arange(0, unique_sensors.shape[1], 1)
    contingency_table = lex_sort(
        x=contingency_table, cluster_columns=sensors_in_contingency_table
    )
    unique_sensors = contingency_table[:, 0 : unique_sensors.shape[1]]
    count_part = contingency_table[:, unique_sensors.shape[1] :]
    flattened_counts = count_part.flatten()
    counts = flattened_counts.astype(int)

    # Pad unique sensor columns with NaN's up until the maximum number of
    # Same pmt-pulses. Each of padded columns represents a pulse.
    pad = np.empty((unique_sensors.shape[0], max(counts)))
    pad[:] = np.nan
    array = np.concatenate([unique_sensors, pad], axis=1)
    column_offset = unique_sensors.shape[1]

    # Construct indices for loop
    cumsum = np.zeros(len(np.cumsum(counts)) + 1)
    cumsum[0] = 0
    cumsum[1:] = np.cumsum(counts)
    cumsum = cumsum.astype(int)

    # Insert pulse attribute in place of NaN.
    for k in range(len(counts)):
        array[k, column_offset : (column_offset + counts[k])] = x[
            cumsum[k] : cumsum[k + 1], feature_idx
        ]
    return array, column_offset, counts


def identify_indices(
    feature_names: List[str], cluster_on: List[str]
) -> Tuple[List[int], List[int], List[str]]:
    """Identify indices for clustering and summarization."""
    features_for_summarization = []
    for feature in feature_names:
        if feature not in cluster_on:
            features_for_summarization.append(feature)
    cluster_indices = [feature_names.index(column) for column in cluster_on]
    summarization_indices = [
        feature_names.index(column) for column in features_for_summarization
    ]
    return cluster_indices, summarization_indices, features_for_summarization


def cluster_summarize_with_percentiles(
    x: np.ndarray,
    summarization_indices: List[int],
    cluster_indices: List[int],
    percentiles: List[int],
    add_counts: bool,
) -> np.ndarray:
    """Turn `x` into clusters with percentile summary.

    From variables specified by column indices `cluster_indices`, `x` is turned
    into clusters. Information in columns of `x` specified by indices
    `summarization_indices` with each cluster is summarized using percentiles.
    It is assumed `x` represents a single event.

    **Example use-case**:
    Suppose `x` contains raw pulses from a neutrino event where some DOMs have
    multiple measurements of Cherenkov radiation. If `cluster_indices` is set
    to the columns corresponding to the xyz-position of the DOMs, and the
    features specified in `summarization_indices` correspond to time, charge,
    then each row in the returned array will correspond to a DOM,
    and the time and charge for each DOM will be summarized by percentiles.
    Returned output array has dimensions
    `[n_clusters,
    len(percentiles)*len(summarization_indices) + len(cluster_indices)]`

    Args:
        x: Array to be clustered
        summarization_indices: List of column indices that defines features
                                that will be summarized with percentiles.
        cluster_indices: List of column indices on which the clusters
                        are constructed.
        percentiles: percentiles used to summarize `x`. E.g. [10,50,90].

    Returns:
        Percentile-summarized array
    """
    pct_dict = {}
    for feature_idx in summarization_indices:
        summarized_array, column_offset, counts = gather_cluster_sequence(
            x, feature_idx, cluster_indices
        )
        pct_dict[feature_idx] = np.nanpercentile(
            summarized_array[:, column_offset:], percentiles, axis=1
        ).T

    for i, key in enumerate(pct_dict.keys()):
        if i == 0:
            array = summarized_array[:, 0:column_offset]

        array = np.concatenate([array, pct_dict[key]], axis=1)

    if add_counts:
        array = np.concatenate(
            [array, np.log10(counts).reshape(-1, 1)], axis=1
        )

    return array


def ice_transparency(
    z_offset: Optional[float] = None,
    z_scaling: Optional[float] = None,
) -> Tuple[interp1d, interp1d]:
    """Return interpolation functions for optical properties of IceCube.

        NOTE: The resulting interpolation functions assumes that the
        Z-coordinate of pulse are scaled as `z = z/500`.
        Any deviation from this scaling method results in inaccurate results.

    Args:
        z_offset: Offset to be added to the depth of the DOM.
        z_scaling: Scaling factor to be applied to the depth of the DOM.

    Returns:
        f_scattering: Function that takes a normalized depth and returns the
        corresponding normalized scattering length.
        f_absorption: Function that takes a normalized depth and returns the
        corresponding normalized absorption length.
    """
    # Data from page 31 of https://arxiv.org/pdf/1301.5361.pdf
    df = pd.read_parquet(
        os.path.join(DATA_DIR, "ice_properties/ice_transparency.parquet"),
    )

    z_offset = z_offset or -1950.0
    z_scaling = z_scaling or 500.0

    df["z_norm"] = (df["depth"] + z_offset) / z_scaling
    df[["scattering_len_norm", "absorption_len_norm"]] = (
        RobustScaler().fit_transform(df[["scattering_len", "absorption_len"]])
    )

    f_scattering = interp1d(df["z_norm"], df["scattering_len_norm"])
    f_absorption = interp1d(df["z_norm"], df["absorption_len_norm"])
    return f_scattering, f_absorption


def weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the weighted median of an array of values.

    Args:
        values: Array-like object containing the values for which
            the weighted median is to be computed.
        weights: Array-like object containing the weights corresponding
            to each value in `values`.

    Returns:
        float: The computed weighted median of the input values.

    Raises:
        ValueError:
            If `values` and `weights` have different lengths or are empty.
    """
    # Convert input values and weights to numpy arrays
    values = np.array(values)
    weights = np.array(weights)

    # Get the indices that would sort the array
    sort_indices = np.argsort(values)

    # Sort values and weights according to the sorted indices
    values_sorted = values[sort_indices]
    weights_sorted = weights[sort_indices]

    # Compute the cumulative sum of the sorted weights
    cumsum = weights_sorted.cumsum()

    # Calculate the cutoff as half of the total weight sum
    cutoff = weights_sorted.sum() / 2.0

    # Return the smallest value for which the cumulative sum is greater
    # than or equal to the cutoff
    return values_sorted[cumsum >= cutoff][0]


def time_of_percentile(
    time_arr: np.array,
    value_arr: np.array,
    percentiles: list,
) -> list:
    """Get time values at specified percentiles of a cumulative sum.

    Using the time and value series provided in `time_arr` and `value_arr`,
    this function computes the cumulative sum of `value_arr` and identifies
    the time points at which the cumulative sum reaches or exceeds the
    specified percentiles. The result is useful for analyzing the distribution
    of cumulative values over time.

    **Example use-case**:
    Suppose `time_arr` and `value_arr` represent time and charge values.
    Then this would give timing information about how quick the charge gets
    deposited in a cluster E.g. DOM.

    Args:
        time_arr: Array of time values associated with measurements or events.
            Should be in ascending order.
        value_arr: Array of values corresponding to each time point,
            USED TO compute the cumulative sum.
        percentiles: List of percentile thresholds (ranging from 0 to 100)
            at which to evaluate the time values.

    Returns:
        List of time values at the specified percentiles.
    """
    ret = []
    assert (
        time_arr.shape == value_arr.shape
    ), f"Time array: {time_arr.shape}, Value array: {value_arr.shape}"
    cumulative = np.cumsum(value_arr)
    pct_values = np.percentile(
        cumulative,
        percentiles,
    ).tolist()
    for pct in pct_values:
        idx = np.where(cumulative - pct >= 0, cumulative, np.inf).argmin()
        ret.append(time_arr[idx])

    return ret


def cumulative_after_t(
    time_arr: np.array,
    value_arr: np.array,
    times: List[float],
) -> list:
    """Get cumulative value of `value_arr` at/after specified times.

    Using time points provided in `times`, this function calculates cumulative
    timeseries of `value_arr` at points occurring at time in `time_arr`.
    The cumulative sum is evaluated for each time in `times`, and the result
    is returned as a list of cumulative values.

    **Example use-case**:
    Suppose `time_arr` and `value_arr` represent the time and charge collected
    in a DOM. You may want to determine the deposited charge at
    different time thresholds to analyze how much charge has been accumulated
    up to these times and use this as a Cluster Feature.

    Args:
        time_arr: Array of time values associated with events.
            The first time value should be 0 so that every time value
            in `times` is just the time after the start.
        value_arr: Array of values to be cumulatively summed.
        times: List of time thresholds after which the cumulative sums
               are evaluated. All positive values.

    Returns:
        List of cumulative sums evaluated at the specified times.
    """
    assert (
        time_arr.shape == value_arr.shape
    ), f"Time array: {time_arr.shape}, Value array: {value_arr.shape}"
    ret = []
    for t in times:
        if t > time_arr.max():
            ret.append(np.cumsum(value_arr).max())
        else:
            idx = np.where(time_arr - t >= 0, time_arr, np.inf).argmin()
            ret.append(np.cumsum(value_arr)[idx])
    return ret
