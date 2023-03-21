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
# @file   macro_legalize.py
# @author Yibo Lin
# @date   Nov 2019
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.macro_legalize.macro_legalize_cpp as macro_legalize_cpp


class MacroLegalizeFunction(Function):
    """Legalize movable macros without considering standard cells"""

    @staticmethod
    def forward(
        init_pos,
        pos,
        node_size_x,
        node_size_y,
        flat_region_boxes,
        flat_region_boxes_start,
        node2fence_region_map,
        node_weights,
        xl,
        yl,
        xh,
        yh,
        site_width,
        row_height,
        num_bins_x,
        num_bins_y,
        num_movable_nodes,
        num_terminal_NIs,
        num_filler_nodes,
    ):
        if pos.is_cuda:
            output = macro_legalize_cpp.forward(
                init_pos.view(init_pos.numel()).cpu(),
                pos.view(pos.numel()).cpu(),
                node_size_x.cpu(),
                node_size_y.cpu(),
                node_weights.cpu(),
                flat_region_boxes.cpu(),
                flat_region_boxes_start.cpu(),
                node2fence_region_map.cpu(),
                xl,
                yl,
                xh,
                yh,
                site_width,
                row_height,
                num_bins_x,
                num_bins_y,
                num_movable_nodes,
                num_terminal_NIs,
                num_filler_nodes,
            ).cuda()
        else:
            output = macro_legalize_cpp.forward(
                init_pos.view(init_pos.numel()),
                pos.view(pos.numel()),
                node_size_x,
                node_size_y,
                node_weights,
                flat_region_boxes,
                flat_region_boxes_start,
                node2fence_region_map,
                xl,
                yl,
                xh,
                yh,
                site_width,
                row_height,
                num_bins_x,
                num_bins_y,
                num_movable_nodes,
                num_terminal_NIs,
                num_filler_nodes,
            )
        return output


class MacroLegalize(object):
    """Legalize movable macros without considering standard cells"""

    def __init__(
        self,
        node_size_x,
        node_size_y,
        node_weights,
        flat_region_boxes,
        flat_region_boxes_start,
        node2fence_region_map,
        fp_info,
        num_bins_x,
        num_bins_y,
        num_movable_nodes,
        num_terminal_NIs,
        num_filler_nodes,
    ):
        super(MacroLegalize, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.node_weights = (
            node_weights  # node_weights of cells when computing displacement
        )
        self.flat_region_boxes = flat_region_boxes
        self.flat_region_boxes_start = flat_region_boxes_start
        self.node2fence_region_map = node2fence_region_map
        self.fp_info = fp_info
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminal_NIs = num_terminal_NIs
        self.num_filler_nodes = num_filler_nodes

    def __call__(self, init_pos, pos):
        """
        @param init_pos the reference position for displacement minization
        @param pos current roughly legal position
        """
        return MacroLegalizeFunction.forward(
            init_pos,
            pos,
            node_size_x=self.node_size_x,
            node_size_y=self.node_size_y,
            node_weights=self.node_weights,
            flat_region_boxes=self.flat_region_boxes,
            flat_region_boxes_start=self.flat_region_boxes_start,
            node2fence_region_map=self.node2fence_region_map,
            xl=self.fp_info.xl,
            yl=self.fp_info.yl,
            xh=self.fp_info.xh,
            yh=self.fp_info.yh,
            site_width=self.fp_info.site_width,
            row_height=self.fp_info.row_height,
            num_bins_x=self.num_bins_x,
            num_bins_y=self.num_bins_y,
            num_movable_nodes=self.num_movable_nodes,
            num_terminal_NIs=self.num_terminal_NIs,
            num_filler_nodes=self.num_filler_nodes,
        )
