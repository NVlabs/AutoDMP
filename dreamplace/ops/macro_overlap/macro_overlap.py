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

from dataclasses import dataclass
import torch
from torch import nn

# "virtual" boundary nodes
@dataclass
class BoundaryNode:
    size_x: ...
    size_y: ...
    pos_x: ...  # center position
    pos_y: ...  # center position


def create_boundary_nodes(fp_info, num_x, num_y, padding=2):
    sq_x = (fp_info.xh - fp_info.xl) / num_x
    sq_y = (fp_info.yh - fp_info.yl) / num_y
    build_node = lambda i, j: BoundaryNode(
        sq_x, sq_y, fp_info.xl + sq_x / 2 + j * sq_x, fp_info.yl + sq_y / 2 + i * sq_y
    )
    bndry_nodes = []
    for i in range(-padding, num_y + padding):
        for j in range(-padding, num_x + padding):
            if i <= 0 or i >= num_y - 1 or j <= 0 or j >= num_x - 1:
                bndry_nodes.append(build_node(i, j))
    return bndry_nodes, sq_x * sq_y * len(bndry_nodes)


class MacroOverlap(nn.Module):
    def __init__(
        self,
        fp_info,
        node_size_x,
        node_size_y,
        num_movable_nodes,
        movable_macro_mask,
        boundary_only=False,
    ):
        super(MacroOverlap, self).__init__()
        self.fp_info = fp_info
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.num_movable_nodes = num_movable_nodes
        self.movable_macro_mask = movable_macro_mask
        self.boundary_only = boundary_only

        self.num_movable_macros = None
        self.bndry_nodes = None
        self.bndry_nodes_area = None
        self.num_bndry_nodes = None
        self.num_blocks = None
        self.node_area = None
        self.delta_size_x = None
        self.delta_size_y = None
        self.sum_area = None

        self._setup_matrices()

    def _setup_matrices(self):
        # movable macros
        self.macro_size_x = self.node_size_x[: self.num_movable_nodes][
            self.movable_macro_mask
        ]
        self.macro_size_y = self.node_size_y[: self.num_movable_nodes][
            self.movable_macro_mask
        ]

        self.num_movable_macros = self.macro_size_x.shape[0]
        if self.num_movable_macros == 0:
            return

        # pad movable macros
        self.macro_size_x += 2 * self.fp_info.macro_padding_x
        self.macro_size_y += 2 * self.fp_info.macro_padding_y

        # add the boundary nodes
        if self.fp_info.bndry_padding_x > 0 and self.fp_info.bndry_padding_y > 0:
            num_x = int(
                (self.fp_info.xh - self.fp_info.xl) / self.fp_info.bndry_padding_x
            )
            num_y = int(
                (self.fp_info.yh - self.fp_info.yl) / self.fp_info.bndry_padding_y
            )
            self.bndry_nodes, self.bndry_nodes_area = create_boundary_nodes(
                self.fp_info, num_x, num_y
            )
        else:
            self.bndry_nodes = []
            self.bndry_nodes_area = 1

        self.num_bndry_nodes = len(self.bndry_nodes)
        bndry_size_x = torch.tensor(
            [n.size_x for n in self.bndry_nodes],
            dtype=self.macro_size_x.dtype,
            device=self.macro_size_x.device,
        )
        bndry_size_y = torch.tensor(
            [n.size_y for n in self.bndry_nodes],
            dtype=self.macro_size_y.dtype,
            device=self.macro_size_y.device,
        )
        self.macro_size_x = torch.cat((self.macro_size_x, bndry_size_x))
        self.macro_size_y = torch.cat((self.macro_size_y, bndry_size_y))

        self.bndry_pos_x = torch.tensor(
            [n.pos_x for n in self.bndry_nodes],
            dtype=self.macro_size_x.dtype,
            device=self.macro_size_x.device,
        )
        self.bndry_pos_y = torch.tensor(
            [n.pos_y for n in self.bndry_nodes],
            dtype=self.macro_size_y.dtype,
            device=self.macro_size_y.device,
        )

        self.num_blocks = self.macro_size_x.shape[0]
        self.macro_area = self.macro_size_x * self.macro_size_y
        self.total_macro_area = torch.sum(self.macro_area)

        # delta wij = (wi + wj) / 2
        self.delta_size_x = (
            torch.transpose(
                self.macro_size_x.expand(self.num_blocks, self.num_blocks),
                0,
                1,
            )
            + self.macro_size_x
        ).div_(2.0)

        # delta hij = (hi + hj) / 2
        self.delta_size_y = (
            torch.transpose(
                self.macro_size_y.expand(self.num_blocks, self.num_blocks),
                0,
                1,
            )
            + self.macro_size_y
        ).div_(2.0)

        # Ai + Aj
        self.sum_area = (
            torch.transpose(
                self.macro_area.expand(self.num_blocks, self.num_blocks),
                0,
                1,
            )
            + self.macro_area
        )

    def forward(self, pos):
        if self.num_movable_macros == 0:
            return torch.zeros(1, dtype=pos[0].dtype, device=pos[0].device)

        num_nodes = pos.numel() // 2
        # center positions of macros
        x = (
            pos[: self.num_movable_nodes][self.movable_macro_mask]
            + self.macro_size_x[: self.num_movable_macros] / 2
        )
        y = (
            pos[num_nodes : num_nodes + self.num_movable_nodes][self.movable_macro_mask]
            + self.macro_size_y[: self.num_movable_macros] / 2
        )

        # add the boundary nodes
        x = torch.cat((x, self.bndry_pos_x))
        y = torch.cat((y, self.bndry_pos_y))

        # delta xij = |xi - xj|
        delta_x = torch.cdist(x.view(-1, 1), x.view(-1, 1), p=1.0)
        # delta yij = |yi - yj|
        delta_y = torch.cdist(y.view(-1, 1), y.view(-1, 1), p=1.0)

        # bell shape x overlap
        Px = torch.where(
            delta_x <= self.delta_size_x,
            torch.where(
                delta_x <= self.delta_size_x / 2,
                1 - 2 * ((delta_x / self.delta_size_x) ** 2),
                2 * (((delta_x - self.delta_size_x) / self.delta_size_x) ** 2),
            ),
            torch.tensor(0.0, dtype=pos[0].dtype, device=pos[0].device),
        )
        # bell shape y overlap
        Py = torch.where(
            delta_y <= self.delta_size_y,
            torch.where(
                delta_y <= self.delta_size_y / 2,
                1 - 2 * ((delta_y / self.delta_size_y) ** 2),
                2 * (((delta_y - self.delta_size_y) / self.delta_size_y) ** 2),
            ),
            torch.tensor(0.0, dtype=pos[0].dtype, device=pos[0].device),
        )

        overlaps = self.sum_area * Px * Py
        if self.boundary_only:
            overlap_loss = (
                torch.sum(overlaps[: -self.num_bndry_nodes, -self.num_bndry_nodes :])
                / self.bndry_nodes_area
            ) ** 2
        else:
            overlap_loss = (
                torch.sum(torch.triu(overlaps, diagonal=1)) / self.total_macro_area
            ) ** 2

        return overlap_loss
