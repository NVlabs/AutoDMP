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
# @file   EvalMetrics.py
# @author Yibo Lin
# @date   Sep 2018
# @brief  Evaluation metrics
#
# 8/4/2022 Mark Ren  add weight_hpwl report

import time
import torch
import pdb


class EvalMetrics(object):
    """
    @brief evaluation metrics at one step
    """

    def __init__(self, iteration=None, detailed_step=None):
        """
        @brief initialization
        @param iteration optimization step
        """
        self.iteration = iteration
        self.detailed_step = detailed_step
        self.objective = None
        self.wirelength = None
        self.density = None
        self.density_weight = None
        self.hpwl = None
        self.rsmt_wl = None
        self.overflow = None
        self.goverflow = None
        self.route_utilization = None
        self.pin_utilization = None
        self.max_density = None
        self.gmax_density = None
        self.gamma = None
        self.eval_time = None
        self.weight_hpwl = None
        self.macro_overlap = None
        self.macro_overlap_weight = None

    def __str__(self):
        """
        @brief convert to string
        """
        content = ""
        if self.iteration is not None:
            content = "iteration %4d" % (self.iteration)
        if self.detailed_step is not None:
            content += ", (%4d, %2d, %2d)" % (
                self.detailed_step[0],
                self.detailed_step[1],
                self.detailed_step[2],
            )
        if self.objective is not None:
            content += ", Obj %.6E" % (self.objective)
        if self.wirelength is not None:
            content += ", WL %.3E" % (self.wirelength)
        if self.density is not None:
            if self.density.numel() == 1:
                content += ", Density %.3E" % (self.density)
            else:
                content += ", Density [%s]" % ", ".join(
                    ["%.3E" % i for i in self.density]
                )
        if self.density_weight is not None:
            if self.density_weight.numel() == 1:
                content += ", DensityWeight %.6E" % (self.density_weight)
            else:
                content += ", DensityWeight [%s]" % ", ".join(
                    ["%.3E" % i for i in self.density_weight]
                )
        if self.hpwl is not None:
            content += ", HPWL %.6E" % (self.hpwl)
        if self.weight_hpwl is not None:
            content += ", weight HPWL %.6E" % (self.weight_hpwl)
        if self.rsmt_wl is not None:
            content += ", RSMT %.3E" % (self.rsmt_wl)
        if self.overflow is not None:
            if self.overflow.numel() == 1:
                content += ", Overflow %.6E" % (self.overflow)
            else:
                content += ", Overflow [%s]" % ", ".join(
                    ["%.3E" % i for i in self.overflow]
                )
        if self.goverflow is not None:
            content += ", Global Overflow %.6E" % (self.goverflow)
        if self.max_density is not None:
            if self.max_density.numel() == 1:
                content += ", MaxDensity %.3E" % (self.max_density)
            else:
                content += ", MaxDensity [%s]" % ", ".join(
                    ["%.3E" % i for i in self.max_density]
                )
        if self.route_utilization is not None:
            content += ", RouteOverflow %.6E" % (self.route_utilization)
        if self.pin_utilization is not None:
            content += ", PinOverflow %.6E" % (self.pin_utilization)
        if self.macro_overlap is not None:
            content += ", MacroOverlap %.3E" % (self.macro_overlap)
        if self.macro_overlap_weight is not None:
            content += ", MacroOverlapWeight %.6E" % (self.macro_overlap_weight)
        if self.gamma is not None:
            content += ", gamma %.6E" % (self.gamma)
        if self.eval_time is not None:
            content += ", time %.3fms" % (self.eval_time * 1000)

        return content

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def evaluate(self, placedb, ops, var, data_collections=None):
        """
        @brief evaluate metrics
        @param placedb placement database
        @param ops a list of ops
        @param var variables
        @param data_collections placement data collections
        """
        tt = time.time()
        with torch.no_grad():
            if "objective" in ops:
                self.objective = ops["objective"](var).data
            if "wirelength" in ops:
                self.wirelength = ops["wirelength"](var).data
            if "density" in ops:
                self.density = ops["density"](var).data
            if "hpwl" in ops:
                self.hpwl = ops["hpwl"](var).data
            if "weight_hpwl" in ops:
                self.weight_hpwl = ops["weight_hpwl"](var).data
            if "rsmt_wl" in ops:
                self.rsmt_wl = ops["rsmt_wl"](var).data
            if "overflow" in ops:
                overflow, max_density = ops["overflow"](var)
                if overflow.numel() == 1:
                    self.overflow = overflow.data / placedb.total_movable_node_area
                    self.max_density = max_density.data
                else:
                    self.overflow = (
                        overflow.data
                        / data_collections.total_movable_node_area_fence_region
                    )
                    self.max_density = max_density.data
            if "goverflow" in ops:
                overflow, max_density = ops["goverflow"](var)
                self.goverflow = overflow.data / placedb.total_movable_node_area
                self.gmax_density = max_density.data
            if "route_utilization" in ops:
                route_utilization_map = ops["route_utilization"](var)
                route_utilization_map_sum = route_utilization_map.sum()
                self.route_utilization = (
                    route_utilization_map.sub_(1).clamp_(min=0).sum()
                    / route_utilization_map_sum
                )
            if "pin_utilization" in ops:
                pin_utilization_map = ops["pin_utilization"](var)
                pin_utilization_map_sum = pin_utilization_map.sum()
                self.pin_utilization = (
                    pin_utilization_map.sub_(1).clamp_(min=0).sum()
                    / pin_utilization_map_sum
                )
            if "macro_overlap" in ops:
                self.macro_overlap = ops["macro_overlap"](var).data
        self.eval_time = time.time() - tt
