# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch import nn
from torch.autograd import Function
import pdb

import dreamplace.ops.pin_utilization.pin_utilization_cpp as pin_utilization_cpp
import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.pin_utilization.pin_utilization_cuda as pin_utilization_cuda


class PinUtilization(nn.Module):
    def __init__(
        self,
        node_size_x,
        node_size_y,
        pin_weights,
        flat_node2pin_start_map,
        fp_info,
        num_movable_nodes,
        num_filler_nodes,
        num_bins_x,
        num_bins_y,
        unit_pin_capacity,
        pin_stretch_ratio,
    ):
        super(PinUtilization, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.fp_info = fp_info
        self.num_nodes = len(node_size_x)
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes
        self.num_physical_nodes = self.num_nodes - num_filler_nodes
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y

        self.unit_pin_capacity = unit_pin_capacity
        self.pin_stretch_ratio = pin_stretch_ratio

        # for each physical node, we use the pin counts as the weights
        if pin_weights is not None:
            self.pin_weights = pin_weights
        elif flat_node2pin_start_map is not None:
            self.pin_weights = (
                flat_node2pin_start_map[1 : self.num_physical_nodes + 1]
                - flat_node2pin_start_map[: self.num_physical_nodes]
            ).to(self.node_size_x.dtype)
        else:
            assert "either pin_weights or flat_node2pin_start_map is required"

        self.reset()

    def reset(self):
        self.bin_size_x = (
            self.fp_info.routing_grid_xh - self.fp_info.routing_grid_xl
        ) / self.num_bins_x
        self.bin_size_y = (
            self.fp_info.routing_grid_yh - self.fp_info.routing_grid_yl
        ) / self.num_bins_y

        # to make the pin density map smooth, we stretch each pin to a ratio of the pin utilization bin
        self.half_node_size_stretch_x = 0.5 * self.node_size_x[
            : self.num_physical_nodes
        ].clamp(min=self.bin_size_x * self.pin_stretch_ratio)
        self.half_node_size_stretch_y = 0.5 * self.node_size_y[
            : self.num_physical_nodes
        ].clamp(min=self.bin_size_y * self.pin_stretch_ratio)

    def forward(self, pos):
        self.bin_size_x = (
            self.fp_info.routing_grid_xh - self.fp_info.routing_grid_xl
        ) / self.num_bins_x
        self.bin_size_y = (
            self.fp_info.routing_grid_yh - self.fp_info.routing_grid_yl
        ) / self.num_bins_y

        if pos.is_cuda:
            func = pin_utilization_cuda.forward
        else:
            func = pin_utilization_cpp.forward
        output = func(
            pos,
            self.node_size_x,
            self.node_size_y,
            self.half_node_size_stretch_x,
            self.half_node_size_stretch_y,
            self.pin_weights,
            self.fp_info.xl,
            self.fp_info.yl,
            self.fp_info.xh,
            self.fp_info.yh,
            self.bin_size_x,
            self.bin_size_y,
            self.num_physical_nodes,
            self.num_bins_x,
            self.num_bins_y,
        )

        # convert demand to utilization in each bin
        output.mul_(1 / (self.bin_size_x * self.bin_size_y * self.unit_pin_capacity))

        return output
