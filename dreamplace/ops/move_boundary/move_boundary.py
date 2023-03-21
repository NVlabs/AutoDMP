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

##
# @file   move_boundary.py
# @author Yibo Lin
# @date   Jun 2018
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.move_boundary.move_boundary_cpp as move_boundary_cpp
import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.move_boundary.move_boundary_cuda as move_boundary_cuda


class MoveBoundaryFunction(Function):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """

    @staticmethod
    def forward(
        pos,
        node_size_x,
        node_size_y,
        xl,
        yl,
        xh,
        yh,
        num_movable_nodes,
        num_filler_nodes,
    ):
        if pos.is_cuda:
            func = move_boundary_cuda.forward
        else:
            func = move_boundary_cpp.forward
        output = func(
            pos.view(pos.numel()),
            node_size_x,
            node_size_y,
            xl,
            yl,
            xh,
            yh,
            num_movable_nodes,
            num_filler_nodes,
        )
        return output


class MoveBoundary(object):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """

    def __init__(
        self, node_size_x, node_size_y, fp_info, num_movable_nodes, num_filler_nodes
    ):
        super(MoveBoundary, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.fp_info = fp_info
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes

    def forward(self, pos):
        return MoveBoundaryFunction.forward(
            pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            xl=self.fp_info.xl,
            yl=self.fp_info.yl,
            xh=self.fp_info.xh,
            yh=self.fp_info.yh,
            num_movable_nodes=self.num_movable_nodes,
            num_filler_nodes=self.num_filler_nodes,
        )

    def __call__(self, pos):
        return self.forward(pos)
