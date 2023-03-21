# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from torch import nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import logging
import os

opj = os.path.join

import dreamplace.ops.rudy.rudy_cpp as rudy_cpp
import dreamplace.configure as configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.rudy.rudy_cuda as rudy_cuda

matplotlib.use("Agg")


class RudyWithMacros(nn.Module):
    def __init__(
        self,
        netpin_start,
        flat_netpin,
        net_weights,
        fp_info,
        num_bins_x,
        num_bins_y,
        node_size_x,
        node_size_y,
        num_movable_nodes,
        movable_macro_mask,
        num_terminals,
        fixed_macro_mask,
        params,
    ):
        super(RudyWithMacros, self).__init__()
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.fp_info = fp_info
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.num_movable_nodes = num_movable_nodes
        self.movable_macro_mask = movable_macro_mask
        self.num_terminals = num_terminals
        self.fixed_macro_mask = fixed_macro_mask
        self.params = params
        self.macro_indexes = torch.cat(
            (
                torch.where(self.movable_macro_mask)[0],
                self.num_movable_nodes + torch.where(self.fixed_macro_mask)[0],
            )
        )

    @torch.no_grad()
    def forward(self, pos, pin_pos):
        self.bin_size_x = (
            self.fp_info.routing_grid_xh - self.fp_info.routing_grid_xl
        ) / self.num_bins_x
        self.bin_size_y = (
            self.fp_info.routing_grid_yh - self.fp_info.routing_grid_yl
        ) / self.num_bins_y

        num_bins = self.num_bins_x * self.num_bins_y
        bin_capa_H = self.fp_info.routing_H / num_bins
        bin_capa_V = self.fp_info.routing_V / num_bins

        horizontal_bin_capacity = torch.full(
            (self.num_bins_x, self.num_bins_y),
            fill_value=bin_capa_H,
            dtype=pin_pos.dtype,
            device=pin_pos.device,
        )
        vertical_bin_capacity = torch.full_like(
            horizontal_bin_capacity, fill_value=bin_capa_V
        )

        horizontal_utilization_map = torch.zeros_like(horizontal_bin_capacity)
        vertical_utilization_map = torch.zeros_like(horizontal_utilization_map)

        if pos.is_cuda:
            func_nets = rudy_cuda.nets_forward
            func_macros = rudy_cuda.macros_forward
        else:
            func_nets = rudy_cpp.nets_forward
            func_macros = rudy_cpp.macros_forward

        func_nets(
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

        if self.macro_indexes.size(0) > 0:
            num_nodes = pos.numel() // 2
            macro_pos_x = pos[self.macro_indexes].contiguous()
            macro_pos_y = pos[num_nodes + self.macro_indexes].contiguous()
            macro_size_x = self.node_size_x[self.macro_indexes].contiguous()
            macro_size_y = self.node_size_y[self.macro_indexes].contiguous()
            macro_area = macro_size_x * macro_size_y
            # torch.save({"px": macro_pos_x, "py": macro_pos_y, "sx": macro_size_x, "sy": macro_size_y}, 'macros.pt')

            unit_macro_util_H = self.fp_info.macro_util_H / macro_area
            unit_macro_util_V = self.fp_info.macro_util_V / macro_area

            horizontal_macro_demand = torch.zeros_like(horizontal_bin_capacity)
            vertical_macro_demand = torch.zeros_like(vertical_bin_capacity)

            func_macros(
                macro_pos_x,
                macro_pos_y,
                macro_size_x,
                macro_size_y,
                unit_macro_util_H,
                unit_macro_util_V,
                self.bin_size_x,
                self.bin_size_y,
                self.fp_info.xl,
                self.fp_info.yl,
                self.fp_info.xh,
                self.fp_info.yh,
                self.num_bins_x,
                self.num_bins_y,
                horizontal_macro_demand,
                vertical_macro_demand,
            )
            
            horizontal_bin_capacity -= horizontal_macro_demand
            vertical_bin_capacity -= vertical_macro_demand

        # convert demand to utilization in each bin
        horizontal_utilization_map.div_(horizontal_bin_capacity)
        vertical_utilization_map.div_(vertical_bin_capacity)

        # Gaussian filter
        hsigma = (1.0 / 16.0) * (self.fp_info.xh - self.fp_info.xl) / self.bin_size_x
        vsigma = (1.0 / 16.0) * (self.fp_info.yh - self.fp_info.yl) / self.bin_size_y
        gaussian_filter = T.GaussianBlur(kernel_size=(3, 3), sigma=min(hsigma, vsigma))
        horizontal_utilization_map = gaussian_filter(
            horizontal_utilization_map.unsqueeze_(0)
        ).squeeze_(0)
        vertical_utilization_map = gaussian_filter(
            vertical_utilization_map.unsqueeze_(0)
        ).squeeze_(0)

        if self.params.plot_flag:
            path = "%s/%s" % (self.params.result_dir, self.params.design_name())
            logging.info("writing congestion maps to %s" % (path))
            plot(
                self.bin_size_x,
                self.bin_size_y,
                horizontal_utilization_map.clone().cpu().numpy(),
                opj(path, f"{self.params.design_name()}.hcong"),
                "2D",
                0.0,
                2.0,
            )

            plot(
                self.bin_size_x,
                self.bin_size_y,
                vertical_utilization_map.clone().cpu().numpy(),
                opj(path, f"{self.params.design_name()}.vcong"),
                "2D",
                0.0,
                2.0,
            )

        # infinity norm
        route_utilization_map = torch.max(
            horizontal_utilization_map.abs_(), vertical_utilization_map.abs_()
        )

        return route_utilization_map


def plot(sx, sy, map, name, dim="2D", vmin=0.0, vmax=1.0, hist=False):
    nx, ny = map.shape
    aspect_ratio = ny * sy / (nx * sx)
    fig = plt.figure()
    # heatmap
    if dim == "2D":
        plt.imshow(
            np.transpose(map),
            origin="lower",
            extent=(0, map.shape[1], 0, map.shape[0]),
            cmap=cm.jet,
            aspect=aspect_ratio,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar()
        plt.show()
        plt.savefig(name + ".2d.png", dpi=800)
        plt.close()
    elif dim == "3D":
        ax = fig.gca(projection="3d")
        x = np.arange(map.shape[0])
        y = np.arange(map.shape[1])
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, map, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("density")
        plt.savefig(name + ".3d.png")
        plt.close()
    if hist:
        # histogram
        plt.hist(map.flatten(), density=True, bins=50)
        plt.show()
        plt.savefig(name + ".2d.hist.png")
        plt.close()
