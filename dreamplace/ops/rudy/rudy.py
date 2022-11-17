# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

##
# @file   rudy.py
# @author Jake Gu
# @date   Dec 2019
# @brief  Compute Rudy map
#
import math
import torch
from torch import nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import pdb

import dreamplace.ops.rudy.rudy_cpp as rudy_cpp
import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.rudy.rudy_cuda as rudy_cuda


class Rudy(nn.Module):
    def __init__(
        self,
        netpin_start,
        flat_netpin,
        net_weights,
        fp_info,
        num_bins_x,
        num_bins_y,
        unit_horizontal_capacity,
        unit_vertical_capacity,
        initial_horizontal_utilization_map=None,
        initial_vertical_utilization_map=None,
    ):
        super(Rudy, self).__init__()
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.fp_info = fp_info
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y

        # initialize parameters
        self.unit_horizontal_capacity = unit_horizontal_capacity
        self.unit_vertical_capacity = unit_vertical_capacity

        self.initial_horizontal_utilization_map = initial_horizontal_utilization_map
        self.initial_vertical_utilization_map = initial_vertical_utilization_map

    def forward(self, pin_pos):
        self.bin_size_x = (
            self.fp_info.routing_grid_xh - self.fp_info.routing_grid_xl
        ) / self.num_bins_x
        self.bin_size_y = (
            self.fp_info.routing_grid_yh - self.fp_info.routing_grid_yl
        ) / self.num_bins_y

        horizontal_utilization_map = torch.zeros(
            (self.num_bins_x, self.num_bins_y),
            dtype=pin_pos.dtype,
            device=pin_pos.device,
        )
        vertical_utilization_map = torch.zeros_like(horizontal_utilization_map)
        if pin_pos.is_cuda:
            func = rudy_cuda.nets_forward
        else:
            func = rudy_cpp.nets_forward
        func(
            pin_pos,
            self.netpin_start,
            self.flat_netpin,
            self.net_weights,
            self.bin_size_x,
            self.bin_size_y,
            self.fp_info.xl,
            self.fp_info.yl,
            self.fp_info.xh,
            self.fp_info.yh,
            self.num_bins_x,
            self.num_bins_y,
            horizontal_utilization_map,
            vertical_utilization_map,
        )

        # convert demand to utilization in each bin
        bin_area = self.bin_size_x * self.bin_size_y
        horizontal_utilization_map.mul_(1 / (bin_area * self.unit_horizontal_capacity))
        vertical_utilization_map.mul_(1 / (bin_area * self.unit_vertical_capacity))

        if self.initial_horizontal_utilization_map is not None:
            horizontal_utilization_map.add_(self.initial_horizontal_utilization_map)
        if self.initial_vertical_utilization_map is not None:
            vertical_utilization_map.add_(self.initial_vertical_utilization_map)

        # infinity norm
        route_utilization_map = torch.max(
            horizontal_utilization_map.abs_(), vertical_utilization_map.abs_()
        )

        return route_utilization_map
