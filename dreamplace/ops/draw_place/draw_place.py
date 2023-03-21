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
# @file   draw_place.py
# @author Yibo Lin
# @date   Jan 2019
# @brief  Plot placement to an image
#

import os
import sys
import torch
from torch.autograd import Function

import dreamplace.ops.draw_place.draw_place_cpp as draw_place_cpp
import dreamplace.ops.draw_place.PlaceDrawer as PlaceDrawer


class DrawPlaceFunction(Function):
    @staticmethod
    def forward(
        pos,
        node_size_x,
        node_size_y,
        pin_offset_x,
        pin_offset_y,
        pin2node_map,
        xl,
        yl,
        xh,
        yh,
        site_width,
        row_height,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_filler_nodes,
        filename,
        show_fillers=False,
    ):
        ret = draw_place_cpp.forward(
            pos,
            node_size_x.cpu(),
            node_size_y.cpu(),
            pin_offset_x.cpu(),
            pin_offset_y.cpu(),
            pin2node_map.cpu(),
            xl,
            yl,
            xh,
            yh,
            site_width,
            row_height,
            bin_size_x,
            bin_size_y,
            num_movable_nodes,
            num_filler_nodes,
            filename,
            show_fillers,
        )
        # if C/C++ API failed, try with python implementation
        if not filename.endswith(".gds") and not ret:
            ret = PlaceDrawer.PlaceDrawer.forward(
                pos,
                node_size_x.cpu(),
                node_size_y.cpu(),
                pin_offset_x.cpu(),
                pin_offset_y.cpu(),
                pin2node_map.cpu(),
                xl,
                yl,
                xh,
                yh,
                site_width,
                row_height,
                bin_size_x,
                bin_size_y,
                num_movable_nodes,
                num_filler_nodes,
                filename,
            )
        return ret


class DrawPlace(object):
    """
    @brief Draw placement
    """

    def __init__(
        self,
        node_size_x,
        node_size_y,
        pin_offset_x,
        pin_offset_y,
        pin2node_map,
        fp_info,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_filler_nodes,
    ):
        """
        @brief initialization
        """
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.pin2node_map = pin2node_map
        self.fp_info = fp_info
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_filler_nodes = num_filler_nodes

    def forward(self, pos, filename):
        """
        @param pos cell locations, array of x locations and then y locations
        @param filename suffix specifies the format
        """
        return DrawPlaceFunction.forward(
            pos,
            self.node_size_x,
            self.node_size_y,
            self.pin_offset_x,
            self.pin_offset_y,
            self.pin2node_map,
            self.fp_info.xl,
            self.fp_info.yl,
            self.fp_info.xh,
            self.fp_info.yh,
            self.fp_info.site_width,
            self.fp_info.row_height,
            self.bin_size_x,
            self.bin_size_y,
            self.num_movable_nodes,
            self.num_filler_nodes,
            filename,
        )

    def __call__(self, pos, filename):
        """
        @brief top API
        @param pos cell locations, array of x locations and then y locations
        @param filename suffix specifies the format
        """
        return self.forward(pos, filename)
