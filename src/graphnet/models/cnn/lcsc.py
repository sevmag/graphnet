"""Module for the Lightning CNN signal classifier (LCSC).

All credits go to Alexander Harnisch (https://github.com/AlexHarn)
"""

from .cnn import CNN
import torch
from torch_geometric.data import Data
from typing import List, Union, Tuple, Type, Optional


class LCSC(CNN):
    """Lightning CNN Signal Classifier (LCSC).

    All credits go to Alexander Harnisch (
    https://github.com/AlexHarn)

    Intended to be used with the IceCube 86 image containing
    only the Main Array image.
    """

    def __init__(
        self,
        num_input_features: int,
        out_put_dim: int = 2,
        input_norm: bool = True,
        num_conv_layers: int = 8,
        conv_filters: List[int] = [50, 50, 50, 50, 50, 50, 50, 10],
        kernel_size: Union[int, List[Union[int, List[int]]]] = 3,
        padding: Union[str, int, List[Union[str, int]]] = "Same",
        pooling_type: List[Union[None, str]] = [
            None,
            "Avg",
            None,
            "Avg",
            None,
            "Avg",
            None,
            "Avg",
        ],
        pooling_kernel_size: List[Union[None, int, List[int]]] = [
            None,
            [1, 1, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
        ],
        pooling_stride: Union[int, List[Union[None, int, List[int]]]] = [
            None,
            [1, 1, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
            None,
            [2, 2, 2],
        ],
        num_fc_neurons: int = 50,
        norm_list: bool = True,
        norm_type: str = "Batch",
        image_size: Tuple[int, int, int] = (10, 10, 60),
    ) -> None:
        """Initialize the Lightning CNN signal classifier (LCSC).

        Args:
            num_input_features: Number of input features.
            out_put_dim: Number of output dimensions of final MLP.
                Defaults to 2.
            input_norm: Whether to apply normalization to the input.
                Defaults to True.
            num_conv_layers: Number of convolutional layers.
                Defaults to 8.
            conv_filters: List of number of convolutional
                filters to use in hidden layers.
                Defaults to [50, 50, 50, 50, 50, 50, 50, 50, 10].
                NOTE needs to have the length of `num_conv_layers`.
            kernel_size:
                Size of the convolutional kernels.
                Options are:
                    int: single integer for all dimensions
                        and all layers,
                        e.g. 3 would equal [3, 3, 3] for each layer.
                    list: list of integers specifying the kernel size,
                        for each layer for all dimensions equally,
                        e.g. [3, 5, 6] would equal [[3,3,3], [5,5,5], [6,6,6]].
                        NOTE: needs to have the length of `num_conv_layers`.
                    If a list of lists is provided, each list will be used
                        for the corresponding layer as kernel size.
                NOTE: If a list if passed it needs to have the length
                     of `num_conv_layers`.
            padding: Padding for the
                convolutional layers.
                Options are:
                    'Same' for same convolutional padding,
                    int: single integer for all dimensions and all layers,
                        e.g. 1 would equal [1, 1, 1].
                    list: list of integers specifying the padding for each
                        dimension, for each layer equally,
                        e.g. [1, 2, 3].
                        NOTE: If a list is passed it needs to have the length
                            of `num_conv_layers`.
                Defaults to 'Same'.
            pooling_type: List of pooling types
                for layers.
                Options are
                    None  : No pooling is used,
                    'Avg'   : Average pooling is used,
                    'Max'   : Max pooling is used
                    Defaults to [
                        None, 'Avg',
                        None, 'Avg',
                        None, 'Avg',
                        None, 'Avg'
                    ].
                    NOTE: the length of the list must be equal to
                        `num_conv_layers`.
            pooling_kernel_size:
                List of pooling kernel sizes for each layer.
                If an integer is provided, it will be used for all layers.
                In case of a list the options for its elements are:
                    list: list of integers for each dimension, e.g. [1, 1, 2].
                    int: single integer for all dimensions,
                        e.g. 2 would equal [2, 2, 2].
                    If None, no pooling is applied.
                NOTE: If a list is passed it needs to have the length
                    of `num_conv_layers`.
                Defaults to [
                    None, [1, 1, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2]
                ].
            pooling_stride:
                List of pooling strides for each layer.
                If an integer is provided, it will be used for all layers.
                In case of a list the options for its elements are:
                    list: list of integers for each dimension, e.g. [1, 1, 2].
                    int: single integer for all dimensions,
                        e.g. 2 would equal [2, 2, 2].
                    If None, no pooling is applied.
                NOTE: If a list is passed it needs to have the length
                    of `num_conv_layers`.
                Defaults to [
                    None, [1, 1, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2],
                    None, [2, 2, 2]
                ].
            num_fc_neurons: Number of neurons in the
                fully connected layers.
                Defaults to 50.
            norm_list: Whether to apply normalization
                for each convolutional layer.
                If a boolean is provided, it will be used for all layers.
                Defaults to True.
                NOTE: If a list is passed it needs to have the length
                    of `num_conv_layers`.
            norm_type: Type of normalization to use.
                Options are 'Batch' or 'Instance'.
                Defaults to 'Batch'.
            image_size: Size of the input image
                in the format (height, width, depth).
                NOTE: Only needs to be changed if the input image is not
                    the standard IceCube 86 image size.
        """
        super().__init__(nb_inputs=num_input_features, nb_outputs=out_put_dim)

        (self.conv_filters, self.kernel_size, self.padding) = (
            self._parse_conv_arguments(
                kernel_size,
                num_conv_layers,
                padding,
                conv_filters,
            )
        )

        (self.pooling_kernel_size, self.pooling_type, self.pooling_stride) = (
            self._parse_pooling_arguments(
                pooling_type,
                pooling_kernel_size,
                pooling_stride,
                num_conv_layers,
            )
        )

        (self.norm_list, self.norm_class, self.input_normal) = (
            self._parse_norm_arguments(
                norm_list,
                norm_type,
                num_conv_layers,
                input_norm,
                num_input_features,
            )
        )

        self.num_conv_layers = num_conv_layers
        self.num_input_features = num_input_features
        self.image_size = image_size
        self.num_fc_neurons = num_fc_neurons
        self.out_put_dim = out_put_dim
        self.input_norm = input_norm

        self._init_cnn_layers()

    def _calc_output_dimension(
        self,
        dimensions: List[int],
        out_channels: int,
        kernel_size: Union[None, int, List[int]],
        padding: Union[str, int, List[int]] = 0,
        stride: Union[None, int, List[int]] = 1,
    ) -> List[int]:
        """Calculate the output dimension after a CNN layers.

        Works for Conv3D, MaxPool3D and AvgPool3D layers.

        Args:
            dimensions (Tuple[int]): Current dimensions of the input tensor.
                (C,H,W,D) where C is the number of channels,
                H is the height, W is the width and D is the depth.
            out_channels (int): Number of output channels.
            kernel_size (Union[int,List[int]]): Size of the kernel.
                If an integer is provided, it will be used for all dimensions.
            padding (Union[int,List[int]]): Padding size.
                If an integer is provided, it will be used for all dimensions.
                If 'Same', the padding will be calculated to keep the
                output size the same as the input size.
                Defaults to 0.
            stride (Union[int,List[int]]): Stride size.
                If an integer is provided, it will be used for all dimensions.
                Defaults to 1.

        Returns:
            Tuple[int]: New dimensions after the layer.

        NOTE: For the pooling layers, set out_channels equal to the
            input channels. Since they do not change the number of channels.
        """
        krnl_sz: int
        if isinstance(padding, str):
            if not padding.lower() == "same":
                raise ValueError(
                    f"`padding` must be 'Same' or an integer, not {padding}!"
                )
            dimensions[0] = out_channels
        else:
            for i in range(1, 4):
                if isinstance(kernel_size, list):
                    krnl_sz = kernel_size[i - 1]
                else:
                    assert isinstance(kernel_size, int)
                    krnl_sz = kernel_size
                if isinstance(padding, list):
                    pad = padding[i - 1]
                else:
                    pad = padding
                if isinstance(stride, list):
                    strd = stride[i - 1]
                else:
                    assert isinstance(stride, int)
                    strd = stride
                dimensions[i] = (dimensions[i] + 2 * pad - krnl_sz) // strd + 1

        return dimensions

    @staticmethod
    def _parse_conv_arguments(
        kernel_size: Union[int, List[Union[int, List[int]]]],
        num_conv_layers: int,
        padding: Union[str, int, List[Union[str, int]]],
        conv_filters: List[int],
    ) -> Tuple[List[int], List[Union[int, List[int]]], List[Union[str, int]]]:
        """Parse convolution layer arguments."""
        # Check input parameters
        ret_conv_filters: List[int]
        ret_kernel_size: List[Union[int, List[int]]]
        ret_padding: List[Union[str, int]]
        if isinstance(conv_filters, int):
            ret_conv_filters = [conv_filters for _ in range(num_conv_layers)]
        else:
            if not isinstance(conv_filters, list):
                raise TypeError(
                    (
                        f"`conv_filters` must be a "
                        f"list or an integer, not {type(conv_filters)}!"
                    )
                )
            if len(conv_filters) != num_conv_layers:
                raise ValueError(
                    f"`conv_filters` must have {num_conv_layers} "
                    f"elements, not {len(conv_filters)}!"
                )
            ret_conv_filters = conv_filters

        if isinstance(kernel_size, int):
            ret_kernel_size = [
                [kernel_size, kernel_size, kernel_size]
                for _ in range(num_conv_layers)
            ]
        else:
            if not isinstance(kernel_size, list):
                raise TypeError(
                    (
                        "`kernel_size` must be a list or an "
                        f"integer, not {type(kernel_size)}!"
                    )
                )
            if len(kernel_size) != num_conv_layers:
                raise ValueError(
                    (
                        f"`kernel_size` must have {num_conv_layers} "
                        f"elements, not {len(kernel_size)}!"
                    )
                )
            if not all(isinstance(k, (int, list)) for k in kernel_size):
                raise TypeError(
                    (
                        "`kernel_size` must be a list of integers or "
                        f"an integer, not {[type(k) for k in kernel_size]}!"
                    )
                )
            ret_kernel_size = kernel_size

        if isinstance(padding, int):
            ret_padding = [padding for _ in range(num_conv_layers)]
        elif isinstance(padding, str):
            if padding.lower() == "same":
                ret_padding = ["same" for i in range(num_conv_layers)]
            else:
                raise ValueError(
                    (
                        "`padding` must be 'Same' or an integer, "
                        f"not {padding}!"
                    )
                )
        else:
            if not isinstance(padding, list):
                raise TypeError(
                    (
                        f"`padding` must be a list or "
                        f"an integer, not {type(padding)}!"
                    )
                )
            if len(padding) != num_conv_layers:
                raise ValueError(
                    f"`padding` must have {num_conv_layers} "
                    f"elements, not {len(padding)}!"
                )
            ret_padding = padding

        return ret_conv_filters, ret_kernel_size, ret_padding

    @staticmethod
    def _parse_pooling_arguments(
        pooling_type: List[Union[None, str]],
        pooling_kernel_size: List[Union[None, int, List[int]]],
        pooling_stride: Union[int, List[Union[None, int, List[int]]]],
        num_conv_layers: int,
    ) -> Tuple[
        List[Union[None, int, List[int]]],
        List[Union[None, str]],
        List[Union[None, int, List[int]]],
    ]:
        """Parse pooling layer arguments."""
        if isinstance(pooling_kernel_size, int):
            pooling_kernel_size = [
                pooling_kernel_size for _ in range(num_conv_layers)
            ]
        else:
            if not isinstance(pooling_kernel_size, list):
                raise TypeError(
                    (
                        "`pooling_kernel_size` must be a list or "
                        f"an integer, not {type(pooling_kernel_size)}!"
                    )
                )
            if len(pooling_kernel_size) != num_conv_layers:
                raise ValueError(
                    (
                        f"`pooling_kernel_size` must have "
                        f"{num_conv_layers} elements, not "
                        f"{len(pooling_kernel_size)}!"
                    )
                )

        if isinstance(pooling_stride, int):
            pooling_stride = [pooling_stride for i in range(num_conv_layers)]
        else:
            if not isinstance(pooling_stride, list):
                raise TypeError(
                    (
                        "`pooling_stride` must be a list or an integer, "
                        f"not {type(pooling_stride)}!"
                    )
                )
            if len(pooling_stride) != num_conv_layers:
                raise ValueError(
                    (
                        f"`pooling_stride` must have {num_conv_layers} "
                        f"elements, not {len(pooling_stride)}!"
                    )
                )

        if pooling_type is not None:
            if not isinstance(pooling_type, list):
                raise TypeError(
                    (
                        "`pooling_type` must be a list or None, "
                        f"not {type(pooling_type)}!"
                    )
                )
            if not len(pooling_type) == num_conv_layers:
                raise ValueError(
                    (
                        f"`pooling_type` must have {num_conv_layers} "
                        f"elements, not {len(pooling_type)}!"
                    )
                )
            if not all(t in ["Max", "Avg", None] for t in pooling_type):
                raise ValueError(
                    (
                        f"`pooling_type` must be one of ['Max', 'Avg', None], "
                        f"got: {pooling_type}!"
                    )
                )

        return pooling_kernel_size, pooling_type, pooling_stride

    @staticmethod
    def _parse_norm_arguments(
        norm_list: bool,
        norm_type: str,
        num_conv_layers: int,
        input_norm: bool,
        num_input_features: int,
    ) -> Tuple[List[bool], Type[torch.nn.Module], Optional[torch.nn.Module]]:
        """Parse normalization layer arguments."""
        ret_norm_list: List[bool]
        ret_norm_class: Type[torch.nn.Module]
        ret_input_normal: Optional[torch.nn.Module]

        if isinstance(norm_list, bool):
            ret_norm_list = [norm_list for _ in range(num_conv_layers)]
        else:
            if not isinstance(norm_list, list):
                raise TypeError(
                    (
                        "`norm_list` must be a list or a boolean, "
                        f"not {type(norm_list)}!"
                    )
                )

            if len(norm_list) != num_conv_layers:
                raise ValueError(
                    (
                        f"`norm_list` must have {num_conv_layers} "
                        f"elements, not {len(norm_list)}!"
                    )
                )
            ret_norm_list = norm_list

        if norm_type.lower() == "instance":
            ret_norm_class = torch.nn.InstanceNorm3d
            if input_norm:
                ret_input_normal = torch.nn.InstanceNorm3d(num_input_features)
            else:
                ret_input_normal = None
        elif norm_type.lower() == "batch":
            ret_norm_class = torch.nn.BatchNorm3d
            if input_norm:
                # No momentum or learnable parameters for input normalization,
                # just use the average
                ret_input_normal = torch.nn.BatchNorm3d(
                    num_input_features, momentum=None, affine=False
                )
            else:
                ret_input_normal = None
        else:
            raise ValueError(
                (
                    "`norm_type` has to be 'instance' or "
                    f"'batch, not '{norm_type}'!"
                )
            )
        return ret_norm_list, ret_norm_class, ret_input_normal

    def _init_cnn_layers(self) -> None:
        """Initialize all layers."""
        # Initialize layers
        self.conv = torch.nn.ModuleList()
        self.pool = torch.nn.ModuleList()
        self.normal = torch.nn.ModuleList()

        dimensions: List[int] = [
            self.num_input_features,
            *self.image_size,
        ]  # (nb_features per pixel, height, width, depth)
        for i in range(self.num_conv_layers):
            self.conv.append(
                torch.nn.Conv3d(
                    dimensions[0],
                    self.conv_filters[i],
                    kernel_size=self.kernel_size[i],
                    padding=self.padding[i],
                )
            )
            dimensions = self._calc_output_dimension(
                dimensions,
                self.conv_filters[i],
                self.kernel_size[i],
                self.padding[i],
            )
            if self.pooling_type[i] is None or self.pooling_type[i] == "None":
                self.pool.append(None)
            elif self.pooling_type[i] == "Avg":
                self.pool.append(
                    torch.nn.AvgPool3d(
                        kernel_size=self.pooling_kernel_size[i],
                        stride=self.pooling_stride[i],
                    )
                )
                dimensions = self._calc_output_dimension(
                    dimensions,
                    out_channels=dimensions[
                        0
                    ],  # same out channels as input channels for pooling
                    kernel_size=self.pooling_kernel_size[i],
                    stride=self.pooling_stride[i],
                )
            elif self.pooling_type[i] == "Max":
                self.pool.append(
                    torch.nn.MaxPool3d(
                        kernel_size=self.pooling_kernel_size[i],
                        stride=self.pooling_stride[i],
                    )
                )
                dimensions = self._calc_output_dimension(
                    dimensions,
                    out_channels=dimensions[
                        0
                    ],  # same out channels as input channels for pooling
                    kernel_size=self.pooling_kernel_size[i],
                    stride=self.pooling_stride[i],
                )
            else:
                raise ValueError(
                    "Pooling type must be 'None', 'Avg' or 'Max'!"
                )
            if self.norm_list[i]:
                self.normal.append(self.norm_class(dimensions[0]))
            else:
                self.normal.append(None)

        latent_dim = (
            dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3]
        )

        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(latent_dim, self.num_fc_neurons)
        self.fc2 = torch.nn.Linear(self.num_fc_neurons, self.out_put_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the LCSC."""
        assert len(data.x) == 1, "Only Main Array image is supported for LCSC"
        x = data.x[0]
        if self.input_normal:
            x = self.input_normal(x)
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            if self.pool[i] is not None:
                x = self.pool[i](x)
            x = torch.nn.functional.elu(x)
            if self.normal[i] is not None:
                x = self.normal[i](x)

        x = self.flatten(x)
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return x
