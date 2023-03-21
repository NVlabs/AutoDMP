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

import math
import torch
from torch import nn


class MacroRefinement(nn.Module):
    def __init__(
        self,
        node_orient,
        node_size_x,
        node_size_y,
        pin_offset_x,
        pin_offset_y,
        flat_node2pin_map,
        flat_node2pin_start_map,
        pin2net_map,
        movable_macro_mask,
        hpwl_op,
        hpwl_op_net_mask,
    ):
        super(MacroRefinement, self).__init__()

        self.node_orient = node_orient
        self.node_size_x = node_size_x
        self.node_size_y = node_size_y
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.flat_node2pin_map = flat_node2pin_map
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.pin2net_map = pin2net_map
        self.hpwl_op = hpwl_op
        self.hpwl_op_net_mask = hpwl_op_net_mask
        self.original_net_mask = hpwl_op_net_mask.clone()
        self.best_hpwl = None

        #! assume movable nodes are first in tensor
        self.movable_macro_mask = movable_macro_mask
        self.macros_indexes = torch.where(self.movable_macro_mask)[0]
        self.num_macros = len(self.macros_indexes)
        if self.num_macros == 0:
            return

        # macro pins
        self.macro_pins = []
        for i in self.macros_indexes:
            start = self.flat_node2pin_start_map[i]
            end = self.flat_node2pin_start_map[i + 1]
            self.macro_pins.append(self.flat_node2pin_map[start:end].long())

        # macro orientations
        self.macro_orient = [
            o.decode("ascii")
            for o in self.node_orient[self.macros_indexes.detach().cpu()]
        ]
        # search sequence: vflip -> hflip -> vflip
        self.flip_sequence = [self.vflip, self.hflip, self.vflip]
        self.orient_map = {
            "N": ["N", "FN", "S", "FS"],
            "FN": ["FN", "N", "FS", "S"],
            "S": ["S", "FS", "N", "FN"],
            "FS": ["FS", "S", "FN", "N"],
        }

        # macro nets
        self.macro_net_mask = torch.full_like(self.original_net_mask, 0)
        macro_net_indexes = self.pin2net_map[torch.cat(self.macro_pins)]
        self.macro_net_mask.index_fill_(0, macro_net_indexes.long(), 1)

    def vflip(self, macro):
        self.pin_offset_x[self.macro_pins[macro]] = (
            self.macro_size_x[macro] - self.pin_offset_x[self.macro_pins[macro]]
        )

    def hflip(self, macro):
        self.pin_offset_y[self.macro_pins[macro]] = (
            self.macro_size_y[macro] - self.pin_offset_y[self.macro_pins[macro]]
        )

    def find_best_flip(self, macro, pos):
        best_flip = 0
        for idx, op in enumerate(self.flip_sequence, 1):
            op(macro)
            if self.evaluate(pos):
                best_flip = idx
        # set macro to best flip
        if best_flip == 0:
            self.hflip(macro)
        elif best_flip == 1:
            self.hflip(macro)
            self.vflip(macro)
        elif best_flip == 2:
            self.vflip(macro)
        # update orientation
        if best_flip != 0:
            self.compute_orient(macro, best_flip)

    def evaluate(self, pos):
        hpwl = self.hpwl_op(pos)
        if self.best_hpwl > hpwl:
            self.best_hpwl = hpwl
            return True
        return False

    def compute_orient(self, macro, flip):
        orig_orient = self.macro_orient[macro]
        self.macro_orient[macro] = self.orient_map[orig_orient][flip]

    @torch.no_grad()
    def forward(self, pos, improvement=0.001):
        """
        Greedy approach
        -- flip each macro one after another
        -- keep flip if HPWL reduces
        """
        if self.num_macros == 0:
            return zip([], [])
        else:
            max_iter = int(100 / math.sqrt(self.num_macros))

        # macro sizes
        self.macro_size_x = self.node_size_x[self.macros_indexes]
        self.macro_size_y = self.node_size_y[self.macros_indexes]

        if self.best_hpwl is None:
            self.hpwl_op_net_mask.copy_(self.macro_net_mask)
            self.best_hpwl = self.hpwl_op(pos)

        cur_best_hpwl = self.best_hpwl
        iteration = 0
        while iteration <= max_iter:
            iteration += 1
            for macro_idx in range(self.num_macros):
                self.find_best_flip(macro_idx, pos)
            if (cur_best_hpwl - self.best_hpwl) / cur_best_hpwl < improvement:
                break
            cur_best_hpwl = self.best_hpwl

        # reset net mask
        self.hpwl_op_net_mask.copy_(self.original_net_mask)

        return zip(self.macros_indexes, self.macro_orient)
