# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Sequence, Tuple, Union
from holidays import NO
from matplotlib.pyplot import flag

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock

from .transformer import Transformer_Unit
from .maxvit3d import MaxViT

__all__ = ["DynUNet", "DynUnet", "Dynunet"]


class DynUNetSkipLayer(nn.Module):
    """
    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the Torchscript restrictions on
    looping over lists of layers and accumulating lists of output tensors which must be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: Optional[List[torch.Tensor]]

    def __init__(self, index, downsample, upsample, next_layer, heads=None, super_head=None):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x):
        downout = self.downsample(x)
        nextout = self.next_layer(downout)
        upout = self.upsample(nextout, downout)
        if self.super_head is not None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)
        return upout

class ClassLayer(nn.Module):
    def __init__(self, in_cnn_c,in_trans_c,outc,dropout=0.5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_cnn_c+in_trans_c,outc)
        )
        self.flatten = nn.Flatten()

    def forward(self, x_cnn,x_trans):
        x1 = self.avgpool(x_cnn)
        x2 = self.avgpool(x_trans)
        x = torch.cat((x1,x2),dim=1)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.mlp1(x)
        return x
  
 
class BottleneckLayer(nn.Module):
    def __init__(self, conv_block,spatial_dims,inch,outch,kernel_size,strides,norm_name,act_name,dim,dropout):
        super().__init__()
        self.conv_block = conv_block
        self.spatial_dims = spatial_dims
        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm_name = norm_name
        self.act_name = act_name
        self.dim = dim
        self.dropout = dropout
        
        self.conv_seg = nn.Sequential(
            self.conv_block(
                self.spatial_dims,
                self.inch,
                self.outch,
                self.kernel_size,
                self.strides,
                self.norm_name,
                self.act_name,
                dropout=self.dropout,
                ),
            self.conv_block(
                self.spatial_dims,
                self.outch,
                self.dim,
                self.kernel_size,
                [1,1,1],
                self.norm_name,
                self.act_name,
                dropout=self.dropout,
                ),
            )
        self.conv_de = nn.Sequential(
            self.conv_block(
                self.spatial_dims,
                self.dim,
                self.outch,
                self.kernel_size,
                [1,1,1],
                self.norm_name,
                self.act_name,
                dropout=self.dropout,
                ),
            self.conv_block(
                self.spatial_dims,
                self.outch,
                self.outch,
                self.kernel_size,
                [1,1,1],
                self.norm_name,
                self.act_name,
                dropout=self.dropout,
                )
            )
        
        self.trans =  MaxViT(dim=self.dim,window_size = (10,10,6),dropout=0.2)

    def forward(self, x):
        self.X1 = x
        x = self.conv_seg(x)
        x = self.trans(x)
        self.X2 = x
        x = self.conv_de(x)
        return x



class DynUNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        trans_dim: int = 512,
        class_num: int = 2,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock  #首次带有步长的两个卷积块
        self.trans_bias = trans_bias
        self.trans_dim = trans_dim
        self.class_num = class_num
        if filters is not None:
            self.filters = filters
            self.check_filters()
        else:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()  # conv block 首次无步长
        self.downsamples = self.get_downsamples()  # conv block一次， 下采样 不包含最后一次下采样
        self.bottleneck = self.get_bottleneck()   # 瓶颈层 原始： 包含最后一次下采样 
        self.upsamples = self.get_upsamples()   # conv block （UnetUpBlock）一次， 上采样，包含第一上采样， 反卷积，拼接，基础卷积（不能选带残差的）
        self.output_block = self.get_output_block(0)  # 输出卷积
        self.output_block_jdm = self.get_output_block(0)  # 输出卷积
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * self.deep_supr_num
        if self.deep_supervision:
            self.deep_supervision_heads = self.get_deep_supervision_heads()  # 多个输出卷积  无上采样等其他尺寸操作
            self.check_deep_supr_num()

        self.apply(self.initialize_weights)
        self.check_kernel_stride()
        self.cls = ClassLayer(self.filters[-2],self.trans_dim,class_num)

        def create_skips(index, downsamples, upsamples, bottleneck, superheads=None):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """
            # print(bottleneck)
            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck

            if superheads is None:
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck)
                return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

            super_head_flag = False
            if index == 0:  # don't associate a supervision head with self.input_block
                rest_heads = superheads
            else:
                if len(superheads) > 0:
                    super_head_flag = True
                    rest_heads = superheads[1:]
                else:
                    rest_heads = nn.ModuleList()

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=rest_heads)
            if super_head_flag:
                return DynUNetSkipLayer(
                    index,
                    downsample=downsamples[0],
                    upsample=upsamples[0],
                    next_layer=next_layer,
                    heads=self.heads,
                    super_head=superheads[0],
                )

            return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

        if not self.deep_supervision:
            self.skip_layers = create_skips(
                0, [self.input_block] + list(self.downsamples), self.upsamples[::-1], self.bottleneck
            )
        else:
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1],
                self.bottleneck,
                superheads=self.deep_supervision_heads,
            )

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise ValueError("deep_supr_num should be larger than 0.")

    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]
   
    def get_net_layer(self):
        last_layer = self.skip_layers
        flag = True
        while flag:
            try:
                getattr(last_layer.next_layer,'next_layer')
                last_layer = getattr(last_layer,'next_layer')
            except:
                flag = False
        return last_layer
   
   
    def forward(self, x, flag=False):
        out = self.skip_layers(x)
        self.downout = None
        out_ctl = self.output_block(out)
        out_jdm = self.output_block(out)
        if self.training and self.deep_supervision:
            out_ctl_all = [out_ctl]
            out_jdm_all = [out_jdm]
            for feature_map in self.heads:
                out_ctl_all.append(interpolate(feature_map, out_ctl.shape[2:]))
                out_jdm_all.append(interpolate(feature_map, out_jdm.shape[2:]))
            return torch.stack(out_ctl_all, dim=1),torch.stack(out_jdm_all, dim=1), self.cls(self.bottleneck.X1,self.bottleneck.X2)
        if flag:
            return out_ctl, out_jdm
        return out_ctl, out_jdm, self.cls(self.bottleneck.X1,self.bottleneck.X2)
  

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self):
        return  BottleneckLayer(
            self.conv_block,
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dim = self.trans_dim,
            dropout=self.dropout)
    


    def get_output_block(self, idx: int):
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size, trans_bias=self.trans_bias
        )

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


DynUnet = Dynunet = DynUNet


# net = DynUNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=2,
#     kernel_size= [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
#     strides= [[1,1,1], [2,2,1], [2,2,2], [2,2,2], [2,2,2]],
#     upsample_kernel_size= [[2,2,1], [2,2,2], [2,2,2], [2,2,2]],
#     filters= [16, 32, 64, 128, 256],
#     deep_supervision= False,
#     deep_supr_num= 1,
#     res_block=  False,
#     trans_bias= False)

# x = torch.rand((1,1,32,32,32))
# y = net(x)
# # print(net)
# print(y.shape)
