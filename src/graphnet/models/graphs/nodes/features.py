"""Class(es) for getting Cluster features intended for the node definiton.

graphnet.models.graphs.nodes.Cluster
"""

from abc import abstractmethod
from typing import List
import numpy as np

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.graphs.utils import time_of_percentile, cumulative_after_t


class ClusterFeature(Model):
    """Base class for summarizing feature series per cluster.

    Enables feature summarization for clustered data
    (e.g., dom_x, dom_y, dom_z coordinates),
    such as calculating total charge per DOM, pulse count per event, etc.

    Supports flexible feature selection with options to:
    - Use all features
    - Selectively include/exclude time-based features
    - Specify custom feature subsets
    """

    def __init__(
        self,
        features: List[str],
        use_on_time: bool,
        use_all_features: bool,
    ) -> None:
        """Initialize ClusterFeature with feature selection parameters.

        Args:
            features (List[str]):
                List of specific features to use
                NOTE: Can use all features except for
                    the cluster features and time feature
            use_on_time (bool):
                Whether to use the time as a feature
                as well
            use_all_features (bool):
                Flag to use all available features
                NOTE: this will not automatically use the time feature
                    you need to set `use_on_time` to True as well to do that

        Raises:
            AssertionError: If both `use_all_features` is True
                and `features` is non-empty
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._use_all_features = use_all_features
        self._use_on_time = use_on_time
        if len(features) > 0:
            assert (
                self._use_all_features is False
            ), "CANNOT use non-empty `feature` and True `use_all_features`"
        self._features = features

    @final
    def _feature_selection(self, infeatures: list, time_label: str) -> None:
        """Select features for cluster summarization.

        Manages feature selection based on initialization parameters:
        - If `use_all_features` is True, selects all input features
            that are not time or cluster features
        - Adds the time feature if `use_on_time` is set to True

        Args:
            infeatures (list):
                List of all available input features
            time_label (str):
                Name of the time feature column
        """
        if self._use_all_features:
            self._features = infeatures[:]
            if self._use_on_time is False:
                self._features.remove(time_label)
        if self._use_on_time is True:
            if time_label not in self._features:
                self._features.append(time_label)

    @abstractmethod
    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> List[float]:
        """Calculate the value of the new cluster features.

        Args:
            time_arr (np.ndarray):
                time sequence per cluster
            value_arr(np.ndarray):
                feature value sequence per cluster
            t_ref: float,

        Returns:
            A list of values for the cluster feature for the current
            input feature.
        """
        raise NotImplementedError

    @abstractmethod
    def _cf_labels(self, infeature: str) -> List[str]:
        """Return the final feature labels for the clusters.

        Args:
            infeature (str):
                string label of input feature

        Returns:
            A list of cluster feature labels for
            the `infeature_label`
        """
        raise NotImplementedError

    @final
    def _label(self, infeature: str) -> list:
        if infeature in self._features:
            return self._cf_labels(infeature)
        else:
            return []

    @final
    def _get_feature(
        self,
        infeature: str,
        time_arr: np.ndarray,
        value_arr: int,
        t_ref: float,
    ) -> list:
        """Extract cluster features for a specific input feature."""
        if infeature in self._features:
            ret = self._calculate_feature(
                time_arr=time_arr,
                value_arr=value_arr,
                t_ref=t_ref,
            )
            return ret
        else:
            return []


class Percentiles(ClusterFeature):
    """Calculate percentiles for the specified features in the cluster.

    Attributes:
        percentiles (List[int]): List of percentiles to calculate.
    """

    def __init__(
        self,
        percentiles: List[int],
        features: List[str] = [],
        use_on_time: bool = False,
        use_all_features: bool = True,
    ) -> None:
        """Initialize the Percentiles instance.

        Args:
            percentiles (List[int]):
                Percentiles to compute
                E.g. [25, 50, 75]
            features (List[str]):
                List of specific features to include.
            use_on_time (bool):
                Whether to include time as a feature.
            use_all_features (bool):
                Whether to use all features
                NOTE: This does not automatically select the time
                    feature. To include time set the
                    flag `use_on_time` to True
        """
        super().__init__(
            features=features,
            use_on_time=use_on_time,
            use_all_features=use_all_features,
        )
        self._percentiles = percentiles

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_pct{pct}" for pct in self._percentiles]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        return np.percentile(value_arr, self._percentiles).tolist()


class Totals(ClusterFeature):
    """Calculate the total (sum) of the specified features in the cluster."""

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = True,
    ) -> None:
        """Initialize the Totals instance.

        Args:
            features (List[str]):
                List of specific features to include.
            use_all_features (bool):
                Whether to use all features (without from time).
                To include time set the flag `use_on_time` to True
        """
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_tot"]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        return [value_arr.sum()]


class CumulativeAfterT(ClusterFeature):
    """Calculate cumulative values after time thresholds.

    Attributes:
        times (List[float]):
            Time thresholds for calculating cumulative values.
    """

    def __init__(
        self,
        times: List[float],
        features: List[str] = [],
        use_all_features: bool = True,
    ) -> None:
        """Initialize CumulativeAfterT instance.

        Args:
            times (List[float]):
                Time points for cumulative calculations.
            features (List[str]):
                List of features to include (default: []).
            use_all_features (bool):
                Whether to use all features (without from time).
                To include time set the flag `use_on_time` to True
        """
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )
        self._times = times

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_after{t}" for t in self._times]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        # time of first pulse
        t0 = time_arr.min()
        return cumulative_after_t(
            time_arr=time_arr - t0,
            value_arr=value_arr,
            times=self._times,
        )


class TimeOfPercentiles(ClusterFeature):
    """Calculate the time corresponding to percentiles of input features.

    Attributes:
        percentiles (List[float]):
            Percentiles to compute times for.
    """

    def __init__(
        self,
        percentiles: List[float],
        features: List[str] = [],
        use_all_features: bool = True,
    ) -> None:
        """Initialize TimeOfPercentiles instance.

        Args:
            percentiles (List[float]):
                Percentiles for which to calculate times.
            features (List[str]):
                List of features to include (default: []).
            use_all_features (bool):
                Whether to use all features (without from time).
                To include time set the flag `use_on_time` to True
        """
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )
        self._percentiles = percentiles

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"t_pct{pct}{infeature}" for pct in self._percentiles]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        # time of first pulse
        t0 = time_arr.min()
        return time_of_percentile(
            time_arr=time_arr - t0,
            value_arr=value_arr,
            percentiles=self._percentiles,
        )


class Std(ClusterFeature):
    """Calculate the standard deviation as cluster feature."""

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = True,
        use_on_time: bool = False,
    ) -> None:
        """Initialize Std instance.

        Args:
            features (List[str]):
                List of features to include (default: []).
            use_all_features (bool):
                Whether to use all features (without from time).
                To include time set the flag `use_on_time` to True
            use_on_time (bool):
                Whether to include time-based features (default: False).
        """
        super().__init__(
            features=features,
            use_on_time=use_on_time,
            use_all_features=use_all_features,
        )

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_std"]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        return [value_arr.std()]


class Spread(ClusterFeature):
    """NEED TO IMPLEMENT STILL."""

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = False,
        use_on_time: bool = False,
    ) -> None:
        """Initialize Spread instance.

        Args:
            features (List[str]):
                List of features to include (default: []).
            use_all_features (bool):
                Whether to use all features (without from time).
                To include time set the flag `use_on_time` to True
            use_on_time (bool):
                Whether to include time-based features (default: False).
        """
        super().__init__(
            features=features,
            use_on_time=use_on_time,
            use_all_features=use_all_features,
        )

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_spread"]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        raise NotImplementedError


class MinimumValue(ClusterFeature):
    """Determine the minimum value for the specified features in thecluster."""

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = False,
    ) -> None:
        """Initialize MinimumValue instance.

        Args:
            features (List[str]):
                List of features to include (default: []).
            use_all_features (bool):
                Whether to use all features (without from time).
                To include time set the flag `use_on_time` to True
        """
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_min"]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        return [value_arr.min()]


class TimeOfFirstPulse(ClusterFeature):
    """Extract the time of the first pulse in the cluster."""

    def __init__(
        self,
    ) -> None:
        """Initialize TimeOfFirstPulse instance.

        Uses time-based features exclusively.
        """
        super().__init__(
            features=[],
            use_on_time=True,
            use_all_features=False,
        )

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return [f"{infeature}_first_pulse"]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        # time of first pulse
        t0 = time_arr.min()
        return [t0]


class AddCounts(ClusterFeature):
    """Count the number of occurrences (e.g., pulses) in the cluster."""

    def __init__(
        self,
    ) -> None:
        """Initialize AddCounts instance.

        Uses time-based features exclusively.
        """
        super().__init__(
            features=[],
            use_on_time=True,
            use_all_features=False,
        )

    def _cf_labels(self, infeature: str) -> list:
        """Return the final feature labels for the clusters."""
        return ["counts"]

    def _calculate_feature(
        self,
        time_arr: np.ndarray,
        value_arr: np.ndarray,
        t_ref: float,
    ) -> list:
        return [len(value_arr)]
