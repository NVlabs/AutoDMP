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
# @file   global_swap.py
# @author Yibo Lin
# @date   Jan 2019
# @brief  detailed placement using global swap
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.global_swap.global_swap_cpp as global_swap_cpp
import dreamplace.ops.global_swap.global_swap_concurrent_cpp as global_swap_concurrent_cpp
import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.global_swap.global_swap_cuda as global_swap_cuda

import pdb

import logging

logger = logging.getLogger(__name__)


class GlobalSwapFunction(Function):
    """Detailed placement with global swap"""

    @staticmethod
    def forward(
        pos,
        node_size_x,
        node_size_y,
        flat_region_boxes,
        flat_region_boxes_start,
        node2fence_region_map,
        flat_net2pin_map,
        flat_net2pin_start_map,
        pin2net_map,
        flat_node2pin_map,
        flat_node2pin_start_map,
        pin2node_map,
        pin_offset_x,
        pin_offset_y,
        net_mask,
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
        batch_size,
        max_iters,
        algorithm,
    ):
        if pos.is_cuda:
            func = global_swap_cuda.global_swap
        else:
            if algorithm == "concurrent":
                func = global_swap_concurrent_cpp.global_swap
            else:
                func = global_swap_cpp.global_swap
        output = func(
            pos.view(pos.numel()),
            node_size_x,
            node_size_y,
            flat_region_boxes,
            flat_region_boxes_start,
            node2fence_region_map,
            flat_net2pin_map,
            flat_net2pin_start_map,
            pin2net_map,
            flat_node2pin_map,
            flat_node2pin_start_map,
            pin2node_map,
            pin_offset_x,
            pin_offset_y,
            net_mask,
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
            batch_size,
            max_iters,
        )
        return output


class GlobalSwap(object):
    """Detailed placement with global swap"""

    def __init__(
        self,
        node_size_x,
        node_size_y,
        flat_region_boxes,
        flat_region_boxes_start,
        node2fence_region_map,
        flat_net2pin_map,
        flat_net2pin_start_map,
        pin2net_map,
        flat_node2pin_map,
        flat_node2pin_start_map,
        pin2node_map,
        pin_offset_x,
        pin_offset_y,
        net_mask,
        fp_info,
        num_bins_x,
        num_bins_y,
        num_movable_nodes,
        num_terminal_NIs,
        num_filler_nodes,
        batch_size=32,
        max_iters=10,
        algorithm="concurrent",
    ):
        super(GlobalSwap, self).__init__()
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.flat_region_boxes = flat_region_boxes
        self.flat_region_boxes_start = flat_region_boxes_start
        self.node2fence_region_map = node2fence_region_map
        self.flat_net2pin_map = flat_net2pin_map
        self.flat_net2pin_start_map = flat_net2pin_start_map
        self.pin2net_map = pin2net_map
        self.flat_node2pin_map = flat_node2pin_map
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.pin2node_map = pin2node_map
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.net_mask = net_mask
        self.fp_info = fp_info
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminal_NIs = num_terminal_NIs
        self.num_filler_nodes = num_filler_nodes
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.algorithm = algorithm

    def __call__(self, pos, scale_factor=1.0):
        """the coordinate system may need to be scaled"""
        with torch.no_grad():
            # scale to integer system
            if scale_factor != 1.0:
                inv_scale_factor = 1.0 / scale_factor
                logger.info(
                    "scale coodindate system by %g for refinement" % (inv_scale_factor)
                )
                pos.mul_(inv_scale_factor).round_()
                self.node_size_x.mul_(inv_scale_factor).round_()
                self.node_size_y.mul_(inv_scale_factor).round_()
                self.flat_region_boxes.mul_(inv_scale_factor).round_()
                self.pin_offset_x.mul_(inv_scale_factor)
                self.pin_offset_y.mul_(inv_scale_factor)
                xl = round(self.fp_info.xl * inv_scale_factor)
                yl = round(self.fp_info.yl * inv_scale_factor)
                xh = round(self.fp_info.xh * inv_scale_factor)
                yh = round(self.fp_info.yh * inv_scale_factor)
                site_width = round(self.fp_info.site_width * inv_scale_factor)
                row_height = round(self.fp_info.row_height * inv_scale_factor)
            else:
                xl = self.fp_info.xl
                yl = self.fp_info.yl
                xh = self.fp_info.xh
                yh = self.fp_info.yh
                site_width = self.fp_info.site_width
                row_height = self.fp_info.row_height

            out = GlobalSwapFunction.forward(
                pos,
                node_size_x=self.node_size_x,
                node_size_y=self.node_size_y,
                flat_region_boxes=self.flat_region_boxes,
                flat_region_boxes_start=self.flat_region_boxes_start,
                node2fence_region_map=self.node2fence_region_map,
                flat_net2pin_map=self.flat_net2pin_map,
                flat_net2pin_start_map=self.flat_net2pin_start_map,
                pin2net_map=self.pin2net_map,
                flat_node2pin_map=self.flat_node2pin_map,
                flat_node2pin_start_map=self.flat_node2pin_start_map,
                pin2node_map=self.pin2node_map,
                pin_offset_x=self.pin_offset_x,
                pin_offset_y=self.pin_offset_y,
                net_mask=self.net_mask,
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                site_width=site_width,
                row_height=row_height,
                num_bins_x=self.num_bins_x,
                num_bins_y=self.num_bins_y,
                num_movable_nodes=self.num_movable_nodes,
                num_terminal_NIs=self.num_terminal_NIs,
                num_filler_nodes=self.num_filler_nodes,
                batch_size=self.batch_size,
                max_iters=self.max_iters,
                algorithm=self.algorithm,
            )

            # scale back
            if scale_factor != 1.0:
                logger.info("scale back by %g" % (scale_factor))
                pos.mul_(scale_factor)
                self.node_size_x.mul_(scale_factor)
                self.node_size_y.mul_(scale_factor)
                self.flat_region_boxes.mul_(scale_factor)
                self.pin_offset_x.mul_(scale_factor)
                self.pin_offset_y.mul_(scale_factor)
                out.mul_(scale_factor)

            return out
