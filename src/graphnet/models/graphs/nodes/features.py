"""Class(es) for getting Cluster featuers 
intended for the node definiton:
graphnet.models.graphs.nodes.Cluster"""


from abc import abstractmethod
from typing import List
import numpy as np

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.graphs.utils import (
    time_of_percentile,
    cumulative_after_t
)

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
    ):
        """Initialize ClusterFeature with feature selection parameters.

        Args:
            features: List of specific features to use
                NOTE: Can use all features except for 
                    the cluster features and time feature
            use_on_time: Whether to use the time as a feature
                as well
            use_all_features: Flag to use all available features
                NOTE: this will not automatically use the time feature
                    you need to set `use_on_time` to True as well to do that

        Raises:
            AssertionError: If both `use_all_features` is True and `features` is non-empty
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._use_all_features = use_all_features
        self._use_on_time = use_on_time
        if len(features)>0:
            assert self._use_all_features==False, \
            "CANNOT use non empty `feature` list and True `use_all_features`"
        self._features = features

    @final
    def _feature_selection(
        self,
        infeatures:list,
        time_label:str
    ) -> None:
        """Select features for cluster summarization.

        Manages feature selection based on initialization parameters:
        - If `use_all_features` is True, selects all input features
            that are not time or cluster features
        - Adds the time feature if `use_on_time` is set to True

        Args:
            infeatures: List of all available input features
            time_label: Name of the time feature column
        """
        if self._use_all_features:
            self._features = infeatures[:]
            if self._use_on_time == False:
                self._features.remove(time_label)
        if self._use_on_time==True:
            if time_label not in self._features:
                self._features.append(time_label)

    @abstractmethod
    def _calculate_feature(
        self,
        **kwargs
    )->List[float]:
        """Calculate the value of the new cluster features.

        Args:
            kwargs: anything passed in `_get_feature` can be used.
                NOTE: adjust `_get_feature` if more inputs are needed
                NOTE: keep the **kwargs argument always to prevent
                    `TypeError` 'got an unexpected keyword argument'

        Returns:
            A list of values for the cluster feature for the current
            input feature.
        """
        raise NotImplementedError

    abstractmethod
    def _cf_labels(self,infeature_label:str)->List[str]:
        """Return the final feature labels for the
        clusters for the input feature `infeature_label`
        
        Args:
            infeature: string label of input feature
        
        Returns:
            A list of cluster feature labels for 
            the `infeature_label`
        """
        raise NotImplementedError

    @final
    def _label(self,infeature):
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
        t0:float,
        t_ref:float,
    ):
        """Extract cluster features for a specific input feature."""
        if infeature in self._features:
            ret = self._calculate_feature(
                time_arr=time_arr,
                value_arr=value_arr,
                t0=t0,
                t_ref=t_ref,
            )
            return ret
        else:
            return []

class Percentiles(ClusterFeature):

    def __init__(
        self,
        percentiles: List[int],
        features: List[str] = [],
        used_on_time: bool = False,
        use_all_features: bool = True,
    ):
        super().__init__(
            features=features,
            use_on_time=used_on_time,
            use_all_features=use_all_features
        )
        self._percentiles = percentiles

    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_pct{pct}" for pct in self._percentiles]
    
    def _calculate_feature(
        self,
        value_arr,
        **kwargs
    )->list:
        return np.percentile(
            value_arr,
            self._percentiles
        ).tolist()

class Totals(ClusterFeature):

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = True,
    ):
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )

    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_tot"]
    
    def _calculate_feature(
        self,
        value_arr,
        **kwargs
    )->list:
        return [value_arr.sum()]


class CumulativeAfterT(ClusterFeature):

    def __init__(
        self,
        times: List[float],
        features: List[str] = [],
        use_all_features: bool = True,
    ):
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )
        self._times = times
    
    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_after{t}" for t in self._times]
    
    def _calculate_feature(
        self,
        time_arr: np.array,
        value_arr: np.array,
        t0,
        **kwargs
    )->list:
        return cumulative_after_t(
            time_arr=time_arr-t0,
            value_arr=value_arr,
            times=self._times,
        )

class TimeOfPercentiles(ClusterFeature):
    def __init__(
        self,
        percentiles: List[float],
        features: List[str] = [],
        use_all_features: bool = True,
    ):
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )
        self._percentiles = percentiles
    
    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"t_pct{pct}{infeature}" for pct in self._percentiles]
    
    def _calculate_feature(
        self,
        time_arr,
        value_arr,
        t0,
        **kwargs
    )->list:
        return time_of_percentile(
            time_arr=time_arr-t0,
            value_arr=value_arr,
            percentiles=self._percentiles
        )

class Std(ClusterFeature):

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = True,
        use_on_time: bool = False,
    ):
        super().__init__(
            features=features,
            use_on_time=use_on_time,
            use_all_features=use_all_features,
        )

    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_std"]
    
    def _calculate_feature(
        self,
        value_arr,
        **kwargs
    )->list:
        return [value_arr.std()]

class Spread(ClusterFeature):

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = False,
        use_on_time: bool = False,
    ):
        super().__init__(
            features=features,
            use_on_time=use_on_time,
            use_all_features=use_all_features,
        )

    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_spread"]
    
    def _calculate_feature(
        self,
        **kwargs
    )->list:
        return [np.nan]
    
class MinimumValue(ClusterFeature):

    def __init__(
        self,
        features: List[str] = [],
        use_all_features: bool = False,
    ):
        super().__init__(
            features=features,
            use_on_time=False,
            use_all_features=use_all_features,
        )

    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_min"]
    
    def _calculate_feature(
        self,
        value_arr,
        **kwargs
    )->list:
        return [value_arr.min()]

class TimeOfFirstPulse(ClusterFeature):
    
    def __init__(
        self,
    ):
        super().__init__(
            features=[],
            use_on_time=True,
            use_all_features=False ,
        )
    
    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"{infeature}_first_pulse"]
    
    def _calculate_feature(
        self,
        t0,
        **kwargs
    )->list:
        return [t0]

class AddCounts(ClusterFeature):
    def __init__(
        self,
    ):
        super().__init__(
            features=[],
            use_on_time=True,
            use_all_features=False,
        )
    
    def _cf_labels(self,infeature:str)->list:
        """Return the Cluster feature labels for the
        input feature `infeature_label`"""
        return [f"counts"]
    
    def _calculate_feature(
        self,
        value_arr,
        **kwargs
    )->list:
        return [len(value_arr)]