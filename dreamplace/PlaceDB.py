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
# @file   PlaceDB.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  placement database
#

import sys
import os
import re
import math
import time
import numpy as np
import torch
import pickle
import logging
import dreamplace.Params as Params
import dreamplace
import dreamplace.ops.place_io.place_io as place_io
import dreamplace.ops.fence_region.fence_region as fence_region
import pdb

datatypes = {"float32": np.float32, "float64": np.float64}


class PlaceDB(object):
    """
    @brief placement database
    """

    def __init__(self):
        """
        initialization
        To avoid the usage of list, I flatten everything.
        """
        self.rawdb = None  # raw placement database, a C++ object
        self.pydb = None  # raw placement database, a Python object

        self.place_info = {}  # placement information
        self.num_physical_nodes = 0  # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.num_terminals = 0  # number of terminals, essentially fixed macros
        self.num_terminal_NIs = (
            0  # number of terminal_NIs that can be overlapped, essentially IO pins
        )
        self.node_name2id_map = {}  # node name to id map, cell name
        self.node_names = None  # 1D array, cell name
        self.node_x = None  # 1D array, cell position x
        self.node_y = None  # 1D array, cell position y
        self.node_orient = None  # 1D array, cell orientation
        self.node_size_x = None  # 1D array, cell width
        self.node_size_y = None  # 1D array, cell height

        self.node2orig_node_map = None  # some fixed cells may have non-rectangular shapes; we flatten them and create new nodes
        # this map maps the current multiple node ids into the original one

        self.pin_direct = None  # 1D array, pin direction IO
        self.pin_offset_x = None  # 1D array, pin offset x to its node
        self.pin_offset_y = None  # 1D array, pin offset y to its node

        self.net_name2id_map = {}  # net name to id map
        self.net_names = None  # net name
        self.net_weights = None  # weights for each net

        self.net2pin_map = []  # array of 1D array, each row stores pin id
        self.flat_net2pin_map = None  # flatten version of net2pin_map
        self.flat_net2pin_start_map = (
            None  # starting index of each net in flat_net2pin_map
        )

        self.node2pin_map = None  # array of 1D array, contains pin id of each node
        self.flat_node2pin_map = None  # flatten version of node2pin_map
        self.flat_node2pin_start_map = (
            None  # starting index of each node in flat_node2pin_map
        )

        self.pin2node_map = None  # 1D array, contain parent node id of each pin
        self.pin2net_map = []  # 1D array, contain parent net id of each pin

        self.rows = None  # NumRows x 4 array, stores xl, yl, xh, yh of each row

        self.regions = None  # array of 1D array, placement regions like FENCE and GUIDE
        self.flat_region_boxes = None  # flat version of regions
        self.flat_region_boxes_start = (
            None  # start indices of regions, length of num regions + 1
        )
        self.node2fence_region_map = (
            None  # map cell to a region, maximum integer if no fence region
        )

        self.xl = 0
        self.yl = 0
        self.xh = 0
        self.yh = 0

        self.row_height = None
        self.site_width = None

        self.bin_size_x = None
        self.bin_size_y = None
        self.num_bins_x = None
        self.num_bins_y = None

        self.num_movable_pins = None

        self.total_movable_node_area = None  # total movable cell area
        self.total_fixed_node_area = None  # total fixed cell area
        self.total_space_area = None  # total placeable space area excluding fixed cells

        # enable filler cells
        # the Idea from e-place and RePlace
        self.total_filler_node_area = None
        self.num_filler_nodes = 0

        self.routing_grid_xl = 0
        self.routing_grid_yl = 0
        self.routing_grid_xh = 0
        self.routing_grid_yh = 0
        self.num_routing_grids_x = 1
        self.num_routing_grids_y = 1
        self.num_routing_layers = None
        self.unit_horizontal_capacity = (
            None  # per unit distance, projected to one layer
        )
        self.unit_vertical_capacity = None  # per unit distance, projected to one layer
        self.unit_horizontal_capacities = None  # per unit distance, layer by layer
        self.unit_vertical_capacities = None  # per unit distance, layer by layer
        self.initial_horizontal_demand_map = None  # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer
        self.initial_vertical_demand_map = None  # routing demand map from fixed cells, indexed by (grid x, grid y), projected to one layer

        self.dtype = None

    def unscale_pl(self, shift_factor, scale_factor):
        """
        @brief unscale placement solution only
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        unscale_factor = 1.0 / scale_factor
        node_x = self.node_x * unscale_factor + shift_factor[0]
        node_y = self.node_y * unscale_factor + shift_factor[1]
        return node_x, node_y

    def scale(self, shift_factor, scale_factor):
        """
        @brief shift and scale coordinates
        @param shift_factor shift factor to make the origin of the layout to (0, 0)
        @param scale_factor scale factor
        """
        logging.info(
            "shift coordinate system by (%g, %g), scale coordinate system by %g"
            % (shift_factor[0], shift_factor[1], scale_factor)
        )

        # node positions
        self.node_x -= shift_factor[0]
        self.node_x *= scale_factor
        self.node_y -= shift_factor[1]
        self.node_y *= scale_factor

        # node sizes
        self.node_size_x *= scale_factor
        self.node_size_y *= scale_factor

        # pin offsets
        self.pin_offset_x *= scale_factor
        self.pin_offset_y *= scale_factor

        # floorplan
        self.xl -= shift_factor[0]
        self.xl *= scale_factor
        self.yl -= shift_factor[1]
        self.yl *= scale_factor
        self.xh -= shift_factor[0]
        self.xh *= scale_factor
        self.yh -= shift_factor[1]
        self.yh *= scale_factor
        self.row_height *= scale_factor
        self.site_width *= scale_factor

        # routing
        self.routing_grid_xl -= shift_factor[0]
        self.routing_grid_xl *= scale_factor
        self.routing_grid_yl -= shift_factor[1]
        self.routing_grid_yl *= scale_factor
        self.routing_grid_xh -= shift_factor[0]
        self.routing_grid_xh *= scale_factor
        self.routing_grid_yh -= shift_factor[1]
        self.routing_grid_yh *= scale_factor
        self.routing_V *= scale_factor
        self.routing_H *= scale_factor
        self.macro_util_V *= scale_factor
        self.macro_util_H *= scale_factor

        # shift factor for rectangle
        box_shift_factor = np.array(
            [shift_factor, shift_factor], dtype=self.rows.dtype
        ).reshape(1, -1)

        # placement rows
        self.rows -= box_shift_factor
        self.rows *= scale_factor
        self.total_space_area *= scale_factor * scale_factor

        # regions
        if len(self.flat_region_boxes) > 0:
            self.flat_region_boxes -= box_shift_factor
            self.flat_region_boxes *= scale_factor
        for i in range(len(self.regions)):
            # may have performance issue
            self.regions[i] -= box_shift_factor
            self.regions[i] *= scale_factor

    def sort(self):
        """
        @brief Sort net by degree.
        Sort pin array such that pins belonging to the same net is abutting each other
        """
        logging.info("sort nets by degree and pins by net")

        # sort nets by degree
        net_degrees = np.array([len(pins) for pins in self.net2pin_map])
        net_order = (
            net_degrees.argsort()
        )  # indexed by new net_id, content is old net_id
        self.net_names = self.net_names[net_order]
        self.net2pin_map = self.net2pin_map[net_order]
        for net_id, net_name in enumerate(self.net_names):
            self.net_name2id_map[net_name] = net_id
        for new_net_id in range(len(net_order)):
            for pin_id in self.net2pin_map[new_net_id]:
                self.pin2net_map[pin_id] = new_net_id
        ## check
        # for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id

        # sort pins such that pins belonging to the same net is abutting each other
        pin_order = (
            self.pin2net_map.argsort()
        )  # indexed new pin_id, content is old pin_id
        self.pin2net_map = self.pin2net_map[pin_order]
        self.pin2node_map = self.pin2node_map[pin_order]
        self.pin_direct = self.pin_direct[pin_order]
        self.pin_offset_x = self.pin_offset_x[pin_order]
        self.pin_offset_y = self.pin_offset_y[pin_order]
        old2new_pin_id_map = np.zeros(len(pin_order), dtype=np.int32)
        for new_pin_id in range(len(pin_order)):
            old2new_pin_id_map[pin_order[new_pin_id]] = new_pin_id
        for i in range(len(self.net2pin_map)):
            for j in range(len(self.net2pin_map[i])):
                self.net2pin_map[i][j] = old2new_pin_id_map[self.net2pin_map[i][j]]
        for i in range(len(self.node2pin_map)):
            for j in range(len(self.node2pin_map[i])):
                self.node2pin_map[i][j] = old2new_pin_id_map[self.node2pin_map[i][j]]
        ## check
        # for net_id in range(len(self.net2pin_map)):
        #    for j in range(len(self.net2pin_map[net_id])):
        #        assert self.pin2net_map[self.net2pin_map[net_id][j]] == net_id
        # for node_id in range(len(self.node2pin_map)):
        #    for j in range(len(self.node2pin_map[node_id])):
        #        assert self.pin2node_map[self.node2pin_map[node_id][j]] == node_id

    @property
    def num_movable_nodes(self):
        """
        @return number of movable nodes
        """
        return self.num_physical_nodes - self.num_terminals - self.num_terminal_NIs

    @property
    def num_nodes(self):
        """
        @return number of movable nodes, terminals, terminal_NIs, and fillers
        """
        return self.num_physical_nodes + self.num_filler_nodes

    @property
    def num_nets(self):
        """
        @return number of nets
        """
        return len(self.net2pin_map)

    @property
    def num_pins(self):
        """
        @return number of pins
        """
        return len(self.pin2net_map)

    @property
    def width(self):
        """
        @return width of layout
        """
        return self.xh - self.xl

    @property
    def height(self):
        """
        @return height of layout
        """
        return self.yh - self.yl

    @property
    def area(self):
        """
        @return area of layout
        """
        return self.width * self.height

    def bin_xl(self, id_x):
        """
        @param id_x horizontal index
        @return bin xl
        """
        return self.xl + id_x * self.bin_size_x

    def bin_xh(self, id_x):
        """
        @param id_x horizontal index
        @return bin xh
        """
        return min(self.bin_xl(id_x) + self.bin_size_x, self.xh)

    def bin_yl(self, id_y):
        """
        @param id_y vertical index
        @return bin yl
        """
        return self.yl + id_y * self.bin_size_y

    def bin_yh(self, id_y):
        """
        @param id_y vertical index
        @return bin yh
        """
        return min(self.bin_yl(id_y) + self.bin_size_y, self.yh)

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return number of bins
        """
        return int(np.ceil((h - l) / bin_size))

    def bin_centers(self, l, h, bin_size):
        """
        @brief compute bin centers
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return array of bin centers
        """
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins):
            bin_l = l + id_x * bin_size
            bin_h = min(bin_l + bin_size, h)
            centers[id_x] = (bin_l + bin_h) / 2
        return centers

    @property
    def routing_grid_size_x(self):
        return (self.routing_grid_xh - self.routing_grid_xl) / self.num_routing_grids_x

    @property
    def routing_grid_size_y(self):
        return (self.routing_grid_yh - self.routing_grid_yl) / self.num_routing_grids_y

    def net_hpwl(self, x, y, net_id):
        """
        @brief compute HPWL of a net
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of a net
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes] + self.pin_offset_x[pins]) - np.amin(
            x[nodes] + self.pin_offset_x[pins]
        )
        hpwl_y = np.amax(y[nodes] + self.pin_offset_y[pins]) - np.amin(
            y[nodes] + self.pin_offset_y[pins]
        )

        return (hpwl_x + hpwl_y) * self.net_weights[net_id]

    def hpwl(self, x, y):
        """
        @brief compute total HPWL
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl

    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        """
        @brief compute overlap between two boxes
        @return overlap area between two rectangles
        """
        return max(min(xh1, xh2) - max(xl1, xl2), 0.0) * max(
            min(yh1, yh2) - max(yl1, yl2), 0.0
        )

    def density_map(self, x, y):
        """
        @brief this density map evaluates the overlap between cell and bins
        @param x horizontal cell locations
        @param y vertical cell locations
        @return density map
        """
        bin_index_xl = np.maximum(np.floor(x / self.bin_size_x).astype(np.int32), 0)
        bin_index_xh = np.minimum(
            np.ceil((x + self.node_size_x) / self.bin_size_x).astype(np.int32),
            self.num_bins_x - 1,
        )
        bin_index_yl = np.maximum(np.floor(y / self.bin_size_y).astype(np.int32), 0)
        bin_index_yh = np.minimum(
            np.ceil((y + self.node_size_y) / self.bin_size_y).astype(np.int32),
            self.num_bins_y - 1,
        )

        density_map = np.zeros([self.num_bins_x, self.num_bins_y])

        for node_id in range(self.num_physical_nodes):
            for ix in range(bin_index_xl[node_id], bin_index_xh[node_id] + 1):
                for iy in range(bin_index_yl[node_id], bin_index_yh[node_id] + 1):
                    density_map[ix, iy] += self.overlap(
                        self.bin_xl(ix),
                        self.bin_yl(iy),
                        self.bin_xh(ix),
                        self.bin_yh(iy),
                        x[node_id],
                        y[node_id],
                        x[node_id] + self.node_size_x[node_id],
                        y[node_id] + self.node_size_y[node_id],
                    )

        for ix in range(self.num_bins_x):
            for iy in range(self.num_bins_y):
                density_map[ix, iy] /= (self.bin_xh(ix) - self.bin_xl(ix)) * (
                    self.bin_yh(iy) - self.bin_yl(iy)
                )

        return density_map

    def density_overflow(self, x, y, target_density):
        """
        @brief if density of a bin is larger than target_density, consider as overflow bin
        @param x horizontal cell locations
        @param y vertical cell locations
        @param target_density target density
        @return density overflow cost
        """
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map - target_density, 0.0)))

    def print_node(self, node_id):
        """
        @brief print node information
        @param node_id cell index
        """
        logging.debug(
            "node %s(%d), size (%g, %g), pos (%g, %g)"
            % (
                self.node_names[node_id],
                node_id,
                self.node_size_x[node_id],
                self.node_size_y[node_id],
                self.node_x[node_id],
                self.node_y[node_id],
            )
        )
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (
                self.node_names[self.pin2node_map[pin_id]],
                self.net_names[self.pin2net_map[pin_id]],
                pin_id,
            )
        logging.debug(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index
        """
        logging.debug("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (
                self.node_names[self.pin2node_map[pin_id]],
                self.net_names[self.pin2net_map[pin_id]],
                pin_id,
            )
        logging.debug(pins)

    def print_row(self, row_id):
        """
        @brief print row information
        @param row_id row index
        """
        logging.debug("row %d %s" % (row_id, self.rows[row_id]))

    # def flatten_nested_map(self, net2pin_map):
    #    """
    #    @brief flatten an array of array to two arrays like CSV format
    #    @param net2pin_map array of array
    #    @return a pair of (elements, cumulative column indices of the beginning element of each row)
    #    """
    #    # flat netpin map, length of #pins
    #    flat_net2pin_map = np.zeros(len(self.pin2net_map), dtype=np.int32)
    #    # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
    #    flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
    #    count = 0
    #    for i in range(len(net2pin_map)):
    #        flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
    #        flat_net2pin_start_map[i] = count
    #        count += len(net2pin_map[i])
    #    assert flat_net2pin_map[-1] != 0
    #    flat_net2pin_start_map[len(net2pin_map)] = len(self.pin2net_map)

    #    return flat_net2pin_map, flat_net2pin_start_map

    def read(self, params):
        """
        @brief read using c++
        @param params parameters
        """
        self.dtype = datatypes[params.dtype]
        self.rawdb = place_io.PlaceIOFunction.read(params)
        self.initialize_from_rawdb(params)

    def initialize_from_rawdb(self, params):
        """
        @brief initialize data members from raw database
        @param params parameters
        """
        if self.pydb is None:
            self.pydb = place_io.PlaceIOFunction.pydb(self.rawdb)
        pydb = self.pydb

        self.num_physical_nodes = pydb.num_nodes
        self.num_terminals = pydb.num_terminals
        self.num_terminal_NIs = pydb.num_terminal_NIs
        self.node_name2id_map = pydb.node_name2id_map
        self.node_names = np.array(pydb.node_names, dtype=np.string_)
        # If the placer directly takes a global placement solution,
        # the cell positions may still be floating point numbers.
        # It is not good to use the place_io OP to round the positions.
        # Currently we only support BOOKSHELF format.
        use_read_pl_flag = False
        if (not params.global_place_flag) and os.path.exists(params.aux_input):
            filename = None
            with open(params.aux_input, "r") as f:
                for line in f:
                    line = line.strip()
                    if ".pl" in line:
                        tokens = line.split()
                        for token in tokens:
                            if token.endswith(".pl"):
                                filename = token
                                break
            filename = os.path.join(os.path.dirname(params.aux_input), filename)
            if filename is not None and os.path.exists(filename):
                self.node_x = np.zeros(self.num_physical_nodes, dtype=self.dtype)
                self.node_y = np.zeros(self.num_physical_nodes, dtype=self.dtype)
                self.node_orient = np.zeros(self.num_physical_nodes, dtype=np.string_)
                self.read_pl(params, filename)
                use_read_pl_flag = True
        if not use_read_pl_flag:
            self.node_x = np.array(pydb.node_x, dtype=self.dtype)
            self.node_y = np.array(pydb.node_y, dtype=self.dtype)
            self.node_orient = np.array(pydb.node_orient, dtype=np.string_)
        self.node_size_x = np.array(pydb.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(pydb.node_size_y, dtype=self.dtype)
        self.node2orig_node_map = np.array(pydb.node2orig_node_map, dtype=np.int32)
        self.pin_direct = np.array(pydb.pin_direct, dtype=np.string_)
        self.pin_offset_x = np.array(pydb.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(pydb.pin_offset_y, dtype=self.dtype)
        self.net_name2id_map = pydb.net_name2id_map
        self.net_names = np.array(pydb.net_names, dtype=np.string_)
        self.net2pin_map = pydb.net2pin_map
        self.flat_net2pin_map = np.array(pydb.flat_net2pin_map, dtype=np.int32)
        self.flat_net2pin_start_map = np.array(
            pydb.flat_net2pin_start_map, dtype=np.int32
        )
        self.net_weights = np.array(pydb.net_weights, dtype=self.dtype)
        self.node2pin_map = pydb.node2pin_map
        self.flat_node2pin_map = np.array(pydb.flat_node2pin_map, dtype=np.int32)
        self.flat_node2pin_start_map = np.array(
            pydb.flat_node2pin_start_map, dtype=np.int32
        )
        self.pin2node_map = np.array(pydb.pin2node_map, dtype=np.int32)
        self.pin2net_map = np.array(pydb.pin2net_map, dtype=np.int32)
        self.rows = np.array(pydb.rows, dtype=self.dtype)
        self.regions = pydb.regions
        for i in range(len(self.regions)):
            self.regions[i] = np.array(self.regions[i], dtype=self.dtype)
        self.flat_region_boxes = np.array(pydb.flat_region_boxes, dtype=self.dtype)
        self.flat_region_boxes_start = np.array(
            pydb.flat_region_boxes_start, dtype=np.int32
        )
        self.node2fence_region_map = np.array(
            pydb.node2fence_region_map, dtype=np.int32
        )
        # print(self.flat_region_boxes, self.flat_region_boxes_start, self.node2fence_region_map)
        # print(self.flat_region_boxes.shape, self.flat_region_boxes_start.shape, self.node2fence_region_map.shape)
        #### nonfence region is set to INT_MAX, we set it to #regions??? not compatible with other APIs
        # self.node2fence_region_map = np.minimum(self.node2fence_region_map, len(self.regions))
        self.xl = float(pydb.xl)
        self.yl = float(pydb.yl)
        self.xh = float(pydb.xh)
        self.yh = float(pydb.yh)
        self.row_height = float(pydb.row_height)
        self.site_width = float(pydb.site_width)
        self.num_movable_pins = pydb.num_movable_pins
        self.total_space_area = float(pydb.total_space_area)

        self.routing_grid_xl = float(pydb.routing_grid_xl)
        self.routing_grid_yl = float(pydb.routing_grid_yl)
        self.routing_grid_xh = float(pydb.routing_grid_xh)
        self.routing_grid_yh = float(pydb.routing_grid_yh)
        if pydb.num_routing_grids_x:
            self.num_routing_grids_x = pydb.num_routing_grids_x
            self.num_routing_grids_y = pydb.num_routing_grids_y
            self.num_routing_layers = len(pydb.unit_horizontal_capacities)
            self.unit_horizontal_capacity = np.array(
                pydb.unit_horizontal_capacities, dtype=self.dtype
            ).sum()
            self.unit_vertical_capacity = np.array(
                pydb.unit_vertical_capacities, dtype=self.dtype
            ).sum()
            self.unit_horizontal_capacities = np.array(
                pydb.unit_horizontal_capacities, dtype=self.dtype
            )
            self.unit_vertical_capacities = np.array(
                pydb.unit_vertical_capacities, dtype=self.dtype
            )
            self.initial_horizontal_demand_map = (
                np.array(pydb.initial_horizontal_demand_map, dtype=self.dtype)
                .reshape((-1, self.num_routing_grids_x, self.num_routing_grids_y))
                .sum(axis=0)
            )
            self.initial_vertical_demand_map = (
                np.array(pydb.initial_vertical_demand_map, dtype=self.dtype)
                .reshape((-1, self.num_routing_grids_x, self.num_routing_grids_y))
                .sum(axis=0)
            )
        else:
            self.num_routing_grids_x = params.route_num_bins_x
            self.num_routing_grids_y = params.route_num_bins_y
            self.num_routing_layers = 1
            self.unit_horizontal_capacity = params.unit_horizontal_capacity
            self.unit_vertical_capacity = params.unit_vertical_capacity

        # convert node2pin_map to array of array
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i], dtype=np.int32)
        self.node2pin_map = np.array(self.node2pin_map, dtype=object)

        # convert net2pin_map to array of array
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i], dtype=np.int32)
        self.net2pin_map = np.array(self.net2pin_map, dtype=object)

    def __call__(self, params):
        """
        @brief top API to read placement files
        @param params parameters
        """
        tt = time.time()

        self.read(params)
        self.initialize(params)

        logging.info("reading benchmark takes %g seconds" % (time.time() - tt))

    def calc_num_filler_for_fence_region(
        self, region_id, node2fence_region_map, target_density
    ):
        """
        @description: calculate number of fillers for each fence region
        @param fence_regions{type}
        @return:
        """
        num_regions = len(self.regions)
        node2fence_region_map = node2fence_region_map[self.movable_slice]
        if region_id < len(self.regions):
            fence_region_mask = node2fence_region_map == region_id
        else:
            fence_region_mask = node2fence_region_map >= len(self.regions)

        num_movable_nodes = self.num_movable_nodes

        movable_node_size_x = self.node_size_x[:num_movable_nodes][fence_region_mask]
        # movable_node_size_y = self.node_size_y[:num_movable_nodes][fence_region_mask]

        lower_bound = np.percentile(movable_node_size_x, 5)
        upper_bound = np.percentile(movable_node_size_x, 95)
        filler_size_x = np.mean(
            movable_node_size_x[
                (movable_node_size_x >= lower_bound)
                & (movable_node_size_x <= upper_bound)
            ]
        )
        filler_size_y = self.row_height

        area = (self.xh - self.xl) * (self.yh - self.yl)

        total_movable_node_area = np.sum(
            self.node_size_x[:num_movable_nodes][fence_region_mask]
            * self.node_size_y[:num_movable_nodes][fence_region_mask]
        )

        if region_id < num_regions:
            ## placeable area is not just fention region area. Macros can have overlap with fence region. But we approximate by this method temporarily
            region = self.regions[region_id]
            placeable_area = np.sum(
                (region[:, 2] - region[:, 0]) * (region[:, 3] - region[:, 1])
            )
        else:
            ### invalid area outside the region, excluding macros? ignore overlap between fence region and macro
            fence_regions = np.concatenate(self.regions, 0).astype(np.float32)
            fence_regions_size_x = fence_regions[:, 2] - fence_regions[:, 0]
            fence_regions_size_y = fence_regions[:, 3] - fence_regions[:, 1]
            fence_region_area = np.sum(fence_regions_size_x * fence_regions_size_y)

            placeable_area = (
                max(self.total_space_area, self.area - self.total_fixed_node_area)
                - fence_region_area
            )

        ### recompute target density based on the region utilization
        utilization = min(total_movable_node_area / placeable_area, 1.0)
        if target_density < utilization:
            ### add a few fillers to avoid divergence
            target_density_fence_region = min(1, utilization + 0.01)
        else:
            target_density_fence_region = target_density

        target_density_fence_region = max(0.35, target_density_fence_region)

        total_filler_node_area = max(
            placeable_area * target_density_fence_region - total_movable_node_area, 0.0
        )

        num_filler = int(
            round(total_filler_node_area / (filler_size_x * filler_size_y))
        )
        logging.info(
            "Region:%2d movable_node_area =%10.1f, placeable_area =%10.1f, utilization =%.3f, filler_node_area =%10.1f, #fillers =%8d, filler sizes =%2.4gx%g\n"
            % (
                region_id,
                total_movable_node_area,
                placeable_area,
                utilization,
                total_filler_node_area,
                num_filler,
                filler_size_x,
                filler_size_y,
            )
        )

        return (
            num_filler,
            target_density_fence_region,
            filler_size_x,
            filler_size_y,
            total_movable_node_area,
            np.sum(fence_region_mask.astype(np.float32)),
        )

    def crop_to_site(self, v, axis: str = "x", mode: str = "close"):
        ops = {"close": np.round, "up": np.ceil, "down": np.floor}
        op = ops[mode]
        if axis == "x":
            return self.site_width * op(v / self.site_width)
        elif axis == "y":
            return self.row_height * op(v / self.row_height)

    def update_macros(self, params, area_threshold=10, height_threshold=2):
        # set large cells as macros
        node_areas = self.node_size_x * self.node_size_y
        mean_area = node_areas[self.movable_slice].mean() * area_threshold
        row_height = self.node_size_y[self.movable_slice].min() * height_threshold

        # movable macros
        self.movable_macro_mask = (node_areas[self.movable_slice] > mean_area) & (
            self.node_size_y[self.movable_slice] > row_height
        )
        self.movable_macro_idx = np.where(self.movable_macro_mask)[0]
        self.num_movable_macros = self.movable_macro_idx.shape[0]
        # fixed macros
        self.fixed_macro_mask = (node_areas[self.fixed_slice] > mean_area) & (
            self.node_size_y[self.fixed_slice] > row_height
        )
        self.fixed_macro_idx = (
            self.num_movable_nodes + np.where(self.fixed_macro_mask)[0]
        )
        self.num_fixed_macros = self.fixed_macro_idx.shape[0]

        # setup macro padding for overlap loss
        self.macro_padding_x = params.macro_padding_x
        self.macro_padding_y = params.macro_padding_y
        self.bndry_padding_x = params.bndry_padding_x
        self.bndry_padding_y = params.bndry_padding_y

        # make sure the macros & halo sizes are multiples of site
        params.macro_halo_x = self.crop_to_site(params.macro_halo_x, "x")
        params.macro_halo_y = self.crop_to_site(params.macro_halo_y, "y")
        self.node_size_x[self.movable_macro_idx] = self.crop_to_site(
            self.node_size_x[self.movable_macro_idx], "x"
        )
        self.node_size_y[self.movable_macro_idx] = self.crop_to_site(
            self.node_size_y[self.movable_macro_idx], "y"
        )

        # add halo around macros
        if params.macro_halo_x >= 0 and params.macro_halo_y >= 0:
            # increase macro sizes
            self.node_size_x[self.movable_macro_idx] += 2 * params.macro_halo_x
            self.node_size_y[self.movable_macro_idx] += 2 * params.macro_halo_y
            # self.node_size_x[self.fixed_macro_idx] += 2 * params.macro_halo_x
            # self.node_size_y[self.fixed_macro_idx] += 2 * params.macro_halo_y

            # shift macro positions
            self.node_x[self.movable_macro_idx] -= params.macro_halo_x
            self.node_y[self.movable_macro_idx] -= params.macro_halo_y
            # self.node_x[self.fixed_macro_idx] -= params.macro_halo_x
            # self.node_y[self.fixed_macro_idx] -= params.macro_halo_y

            # shift macro pins
            self.movable_macro_pins = np.isin(self.pin2node_map, self.movable_macro_idx)
            self.pin_offset_x[self.movable_macro_pins] += params.macro_halo_x
            self.pin_offset_y[self.movable_macro_pins] += params.macro_halo_y
            # self.fixed_macro_pins = np.isin(self.pin2node_map, self.fixed_macro_idx)
            # self.pin_offset_x[self.fixed_macro_pins] += params.macro_halo_x
            # self.pin_offset_y[self.fixed_macro_pins] += params.macro_halo_y

    def set_net_weights(self):
        # with open("risa_weights.pkl", 'rb') as f:
        #     weights_dict = pickle.load(f)
        weights_dict = {
            1: 1.0000,
            2: 1.0000,
            3: 1.0000,
            4: 1.0828,
            5: 1.1536,
            6: 1.2206,
            7: 1.2823,
            8: 1.3385,
            9: 1.3991,
            10: 1.4493,
            11: 1.6899,
            12: 1.6899,
            13: 1.6899,
            14: 1.6899,
            15: 1.6899,
            16: 1.8924,
            17: 1.8924,
            18: 1.8924,
            19: 1.8924,
            20: 1.8924,
            21: 2.0743,
            22: 2.0743,
            23: 2.0743,
            24: 2.0743,
            25: 2.0743,
            26: 2.2334,
            27: 2.2334,
            28: 2.2334,
            29: 2.2334,
            30: 2.2334,
            31: 2.3892,
            32: 2.3892,
            33: 2.3892,
            34: 2.3892,
            35: 2.3892,
            36: 2.5356,
            37: 2.5356,
            38: 2.5356,
            39: 2.5356,
            40: 2.5356,
            41: 2.6625,
            42: 2.6625,
            43: 2.6625,
            44: 2.6625,
            45: 2.6625,
        }
        num_pins_in_net = np.ediff1d(self.flat_net2pin_start_map)
        weights = np.full(num_pins_in_net.shape, 2.7933)
        for k in weights_dict:
            weights[num_pins_in_net == k] = weights_dict[k]
        self.net_weights *= weights

    def pin_density_inflation(self, pin_density, pin_accessibility=2.0):
        # pin_accessibility: virtually increases # pins due to cell blockages/pin shapes
        if 0 < pin_density < 1:
            num_tracks_height = 5  # assume 6.5-track high-density library ~ 5 M1/M2 tracks for signal routing
            num_pins_in_cell = np.ediff1d(self.flat_node2pin_start_map)
            inflated_widths = self.crop_to_site(
                np.minimum(
                    2.5 * self.node_size_x,
                    num_pins_in_cell
                    * pin_accessibility
                    * self.row_height
                    / (num_tracks_height**2 * pin_density),
                ),
                "x",
            )
            # inflate standard cells only
            self.node_size_x[self.movable_slice][~self.movable_macro_mask] = np.maximum(
                self.node_size_x[self.movable_slice][~self.movable_macro_mask],
                inflated_widths[self.movable_slice][~self.movable_macro_mask],
            )

    def set_routing_info(self, route_file):
        self.routing_V = (
            10 * self.area / (100 * self.site_width)
        )  # default: 10 layers and pitch is 100x site width
        self.routing_H = 10 * self.area / (100 * self.site_width)
        self.macro_util_V = np.zeros(
            self.num_movable_macros + self.num_fixed_macros, dtype=self.dtype
        )
        self.macro_util_H = np.zeros(
            self.num_movable_macros + self.num_fixed_macros, dtype=self.dtype
        )
        self.macros_routing = {}

        if os.path.isfile(route_file):
            with open(route_file, "r") as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) == 2:
                        self.routing_V = float(line[0])
                        self.routing_H = float(line[1])
                    elif len(line) == 3:
                        self.macros_routing[line[0]] = [float(line[1]), float(line[2])]

        if self.macros_routing:
            movable_macros_indexes = np.where(self.movable_macro_mask)[0]
            fixed_macro_indexes = (
                self.num_movable_nodes + np.where(self.fixed_macro_mask)[0]
            )
            macros_indexes = np.concatenate(
                (movable_macros_indexes, fixed_macro_indexes)
            )
            for name, util in self.macros_routing.items():
                idx = np.where(macros_indexes == self.node_name2id_map[name])[0]
                self.macro_util_V[idx], self.macro_util_H[idx] = util

        self.routing_grid_xl = self.xl
        self.routing_grid_yl = self.yl
        self.routing_grid_xh = self.xh
        self.routing_grid_yh = self.yh

    def initialize(self, params):
        """
        @brief initialize data members after reading
        @param params parameters
        """

        # setup utility slices
        self.movable_slice = slice(0, self.num_movable_nodes)
        self.fixed_slice = slice(
            self.num_movable_nodes, self.num_movable_nodes + self.num_terminals
        )
        self.io_slice = slice(
            self.num_movable_nodes + self.num_terminals,
            self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs,
        )

        # set macros
        self.update_macros(params)

        # set net weights for improved HPWL % RSMT correlation
        if params.risa_weights == 1:
            self.set_net_weights()

        # pin density inflation
        self.pin_density_inflation(params.pin_density)

        # routing information for congestion estimation
        if params.route_info_input == "default":
            aux_name = params.aux_input.rsplit(".", 1)[0]
            params.route_info_input = f"{aux_name}.route_info"
        self.set_routing_info(params.route_info_input)

        # shift and scale
        # adjust shift_factor and scale_factor if not set
        params.shift_factor[0] = self.xl
        params.shift_factor[1] = self.yl
        logging.info(
            "set shift_factor = (%g, %g), as original row bbox = (%g, %g, %g, %g)"
            % (
                params.shift_factor[0],
                params.shift_factor[1],
                self.xl,
                self.yl,
                self.xh,
                self.yh,
            )
        )

        if params.scale_factor == 0.0 or self.site_width != 1.0:
            params.scale_factor = 1.0 / self.site_width
        if self.row_height % self.site_width != 0:
            logging.warn(
                "row_height is not divisible by site_width, might create issues during legalization"
            )
        if not self.site_width.is_integer() or not self.row_height.is_integer():
            logging.warn(
                "site_width or row_height is not an integer, might create issues during legalization"
            )
        logging.info(
            "set scale_factor = %g, as site_width = %g"
            % (params.scale_factor, self.site_width)
        )
        self.scale(params.shift_factor, params.scale_factor)

        params.macro_halo_x *= params.scale_factor
        params.macro_halo_y *= params.scale_factor
        self.macro_padding_x *= params.scale_factor
        self.macro_padding_y *= params.scale_factor
        self.bndry_padding_x *= params.scale_factor
        self.bndry_padding_y *= params.scale_factor

        content = """
================================= Benchmark Statistics =================================
#nodes = %d, #terminals = %d, # terminal_NIs = %d, #movable = %d, #nets = %d
die area = (%g, %g, %g, %g) %g
row height = %g, site width = %g
""" % (
            self.num_physical_nodes,
            self.num_terminals,
            self.num_terminal_NIs,
            self.num_movable_nodes,
            len(self.net_names),
            self.xl,
            self.yl,
            self.xh,
            self.yh,
            self.area,
            self.row_height,
            self.site_width,
        )

        # set number of bins
        # derive bin dimensions by keeping the aspect ratio
        # this bin setting is not for global placement, only for other steps
        # global placement has its bin settings defined in global_place_stages
        aspect_ratio = (self.yh - self.yl) / (self.xh - self.xl)
        num_bins_x = int(
            math.pow(
                2,
                max(
                    np.ceil(
                        math.log2(math.sqrt(self.num_movable_nodes / aspect_ratio))
                    ),
                    0,
                ),
            )
        )
        num_bins_y = int(
            math.pow(
                2,
                max(
                    np.ceil(
                        math.log2(math.sqrt(self.num_movable_nodes * aspect_ratio))
                    ),
                    0,
                ),
            )
        )
        self.num_bins_x = max(params.num_bins_x, num_bins_x)
        self.num_bins_y = max(params.num_bins_y, num_bins_y)
        # set bin size
        self.bin_size_x = (self.xh - self.xl) / self.num_bins_x
        self.bin_size_y = (self.yh - self.yl) / self.num_bins_y

        content += "num_bins = %dx%d, bin sizes = %gx%g\n" % (
            self.num_bins_x,
            self.num_bins_y,
            self.bin_size_x / self.row_height,
            self.bin_size_y / self.row_height,
        )

        # set num_movable_pins
        if self.num_movable_pins is None:
            self.num_movable_pins = 0
            for node_id in self.pin2node_map:
                if node_id < self.num_movable_nodes:
                    self.num_movable_pins += 1
        content += "#pins = %d, #movable_pins = %d\n" % (
            self.num_pins,
            self.num_movable_pins,
        )

        # set total cell area
        self.total_movable_node_area = float(
            np.sum(
                self.node_size_x[self.movable_slice]
                * self.node_size_y[self.movable_slice]
            )
        )
        # total fixed node area should exclude the area outside the layout and the area of terminal_NIs
        self.total_fixed_node_area = float(
            np.sum(
                np.maximum(
                    np.minimum(
                        self.node_x[self.fixed_slice]
                        + self.node_size_x[self.fixed_slice],
                        self.xh,
                    )
                    - np.maximum(
                        self.node_x[self.fixed_slice],
                        self.xl,
                    ),
                    0.0,
                )
                * np.maximum(
                    np.minimum(
                        self.node_y[self.fixed_slice]
                        + self.node_size_y[self.fixed_slice],
                        self.yh,
                    )
                    - np.maximum(
                        self.node_y[self.fixed_slice],
                        self.yl,
                    ),
                    0.0,
                )
            )
        )
        content += (
            "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n"
            % (
                self.total_movable_node_area,
                self.total_fixed_node_area,
                self.total_space_area,
            )
        )

        # set movable macro area
        self.total_movable_macro_area = np.sum(
            (self.node_size_x * self.node_size_y)[self.movable_slice][
                self.movable_macro_mask
            ]
        )

        # check movable macros, adjust area to treat movable macros as fixed macros
        if self.num_movable_macros > 0:
            logging.info(
                "detect movable macros %d, area %g, reduce those area from movable_area",
                self.num_movable_macros,
                self.total_movable_macro_area,
            )
            self.total_movable_node_area -= self.total_movable_macro_area
            self.total_fixed_node_area += self.total_movable_macro_area
            self.total_space_area -= self.total_movable_macro_area
            content += (
                "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n"
                % (
                    self.total_movable_node_area,
                    self.total_fixed_node_area,
                    self.total_space_area,
                )
            )

        target_density = min(self.total_movable_node_area / self.total_space_area, 1.0)
        if target_density > params.target_density:
            logging.warn(
                "target_density %g is smaller than utilization %g, ignored"
                % (params.target_density, target_density)
            )
            params.target_density = target_density
        content += "utilization = %g, target_density = %g\n" % (
            self.total_movable_node_area / self.total_space_area,
            params.target_density,
        )

        # calculate fence region virtual macro
        if len(self.regions) > 0:
            virtual_macro_for_fence_region = [
                fence_region.slice_non_fence_region(
                    region,
                    self.xl,
                    self.yl,
                    self.xh,
                    self.yh,
                    merge=True,
                    plot=False,
                    figname=f"vmacro_{region_id}_merged.png",
                    device="cpu",
                    macro_pos_x=self.node_x[self.fixed_slice],
                    macro_pos_y=self.node_y[self.fixed_slice],
                    macro_size_x=self.node_size_x[self.fixed_slice],
                    macro_size_y=self.node_size_y[self.fixed_slice],
                )
                .cpu()
                .numpy()
                for region_id, region in enumerate(self.regions)
            ]
            virtual_macro_for_non_fence_region = np.concatenate(self.regions, 0)
            self.virtual_macro_fence_region = virtual_macro_for_fence_region + [
                virtual_macro_for_non_fence_region
            ]

        # insert filler nodes
        if len(self.regions) > 0:
            ### calculate fillers if there is fence region
            self.filler_size_x_fence_region = []
            self.filler_size_y_fence_region = []
            self.num_filler_nodes = 0
            self.num_filler_nodes_fence_region = []
            self.num_movable_nodes_fence_region = []
            self.total_movable_node_area_fence_region = []
            self.target_density_fence_region = []
            self.filler_start_map = None
            filler_node_size_x_list = []
            filler_node_size_y_list = []
            self.total_filler_node_area = 0
            for i in range(len(self.regions) + 1):
                (
                    num_filler_i,
                    target_density_i,
                    filler_size_x_i,
                    filler_size_y_i,
                    total_movable_node_area_i,
                    num_movable_nodes_i,
                ) = self.calc_num_filler_for_fence_region(
                    i, self.node2fence_region_map, params.target_density
                )
                self.num_movable_nodes_fence_region.append(num_movable_nodes_i)
                self.num_filler_nodes_fence_region.append(num_filler_i)
                self.total_movable_node_area_fence_region.append(
                    total_movable_node_area_i
                )
                self.target_density_fence_region.append(target_density_i)
                self.filler_size_x_fence_region.append(filler_size_x_i)
                self.filler_size_y_fence_region.append(filler_size_y_i)
                self.num_filler_nodes += num_filler_i
                filler_node_size_x_list.append(
                    np.full(
                        num_filler_i,
                        fill_value=filler_size_x_i,
                        dtype=self.node_size_x.dtype,
                    )
                )
                filler_node_size_y_list.append(
                    np.full(
                        num_filler_i,
                        fill_value=filler_size_y_i,
                        dtype=self.node_size_y.dtype,
                    )
                )
                filler_node_area_i = num_filler_i * (filler_size_x_i * filler_size_y_i)
                self.total_filler_node_area += filler_node_area_i
                content += (
                    "Region: %2d filler_node_area = %10.2f, #fillers = %8d, filler sizes = %2.4gx%g\n"
                    % (
                        i,
                        filler_node_area_i,
                        num_filler_i,
                        filler_size_x_i,
                        filler_size_y_i,
                    )
                )

            self.total_movable_node_area_fence_region = np.array(
                self.total_movable_node_area_fence_region
            )
            self.num_movable_nodes_fence_region = np.array(
                self.num_movable_nodes_fence_region
            )

        if params.enable_fillers:
            # the way to compute this is still tricky; we need to consider place_io together on how to
            # summarize the area of fixed cells, which may overlap with each other.
            if len(self.regions) > 0:
                self.filler_start_map = np.cumsum(
                    [0] + self.num_filler_nodes_fence_region
                )
                self.num_filler_nodes_fence_region = np.array(
                    self.num_filler_nodes_fence_region
                )
                self.node_size_x = np.concatenate(
                    [self.node_size_x] + filler_node_size_x_list
                )
                self.node_size_y = np.concatenate(
                    [self.node_size_y] + filler_node_size_y_list
                )
                content += (
                    "total_filler_node_area = %10.2f, #fillers = %8d, average filler sizes = %2.4gx%g\n"
                    % (
                        self.total_filler_node_area,
                        self.num_filler_nodes,
                        self.total_filler_node_area
                        / self.num_filler_nodes
                        / self.row_height,
                        self.row_height,
                    )
                )
            else:
                node_size_order = np.argsort(self.node_size_x[self.movable_slice])
                filler_size_x = np.mean(
                    self.node_size_x[
                        node_size_order[
                            int(self.num_movable_nodes * 0.05) : int(
                                self.num_movable_nodes * 0.95
                            )
                        ]
                    ]
                )
                filler_size_y = self.row_height
                placeable_area = max(
                    self.area - self.total_fixed_node_area, self.total_space_area
                )
                content += "use placeable_area = %g to compute fillers\n" % (
                    placeable_area
                )
                self.total_filler_node_area = max(
                    placeable_area * params.target_density
                    - self.total_movable_node_area,
                    0.0,
                )
                self.num_filler_nodes = int(
                    round(self.total_filler_node_area / (filler_size_x * filler_size_y))
                )
                self.node_size_x = np.concatenate(
                    [
                        self.node_size_x,
                        np.full(
                            self.num_filler_nodes,
                            fill_value=filler_size_x,
                            dtype=self.node_size_x.dtype,
                        ),
                    ]
                )
                self.node_size_y = np.concatenate(
                    [
                        self.node_size_y,
                        np.full(
                            self.num_filler_nodes,
                            fill_value=filler_size_y,
                            dtype=self.node_size_y.dtype,
                        ),
                    ]
                )
                content += (
                    "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n"
                    % (
                        self.total_filler_node_area,
                        self.num_filler_nodes,
                        filler_size_x,
                        filler_size_y,
                    )
                )
        else:
            self.total_filler_node_area = 0
            self.num_filler_nodes = 0
            filler_size_x, filler_size_y = 0, 0
            if len(self.regions) > 0:
                self.filler_start_map = np.zeros(len(self.regions) + 2, dtype=np.int32)
                self.num_filler_nodes_fence_region = np.zeros(
                    len(self.num_filler_nodes_fence_region)
                )

            content += (
                "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n"
                % (
                    self.total_filler_node_area,
                    self.num_filler_nodes,
                    filler_size_x,
                    filler_size_y,
                )
            )

        if params.routability_opt_flag:
            content += "================================== routing information =================================\n"
            content += "routing grids (%d, %d)\n" % (
                self.num_routing_grids_x,
                self.num_routing_grids_y,
            )
            content += "routing grid sizes (%g, %g)\n" % (
                self.routing_grid_size_x,
                self.routing_grid_size_y,
            )
            content += "routing capacity H/V (%g, %g) per tile\n" % (
                self.unit_horizontal_capacity * self.routing_grid_size_y,
                self.unit_vertical_capacity * self.routing_grid_size_x,
            )
        content += "========================================================================================"

        logging.info(content)

        # setup utility slices
        self.filler_slice = slice(
            self.num_nodes - self.num_filler_nodes, self.num_nodes
        )
        self.all_slice = slice(0, self.num_nodes)

    def write(self, params, filename, sol_file_format=None):
        """
        @brief write placement solution
        @param filename output file name
        @param sol_file_format solution file format, DEF|DEFSIMPLE|BOOKSHELF|BOOKSHELFALL
        """
        tt = time.time()
        logging.info("writing to %s" % (filename))
        if sol_file_format is None:
            if filename.endswith(".def"):
                sol_file_format = place_io.SolutionFileFormat.DEF
            else:
                sol_file_format = place_io.SolutionFileFormat.BOOKSHELF

        # unscale locations
        node_x, node_y = self.unscale_pl(params.shift_factor, params.scale_factor)

        # Global placement may have floating point positions.
        # Currently only support BOOKSHELF format.
        # This is mainly for debug.
        if (
            not params.legalize_flag
            and not params.detailed_place_flag
            and sol_file_format == place_io.SolutionFileFormat.BOOKSHELF
        ):
            self.write_pl(params, filename, node_x, node_y)
        else:
            place_io.PlaceIOFunction.write(
                self.rawdb, filename, sol_file_format, node_x, node_y
            )
        logging.info(
            "write %s takes %.3f seconds" % (str(sol_file_format), time.time() - tt)
        )

    def read_pl(self, params, pl_file):
        """
        @brief read .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("reading %s" % (pl_file))
        count = 0
        with open(pl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA"):
                    continue
                # node positions
                pos = re.search(
                    r"([^\s]+)\s+(\d+(\.\d*)?|\.\d+)\s+(\d+(\.\d*)?|\.\d+)\s*:\s*(\w+)",
                    line,
                )
                if pos:
                    node_id = self.node_name2id_map[pos.group(1)]
                    self.node_x[node_id] = float(pos.group(2))
                    self.node_y[node_id] = float(pos.group(4))
                    self.node_orient[node_id] = pos.group(6)
        logging.info("read_pl takes %.3f seconds" % (time.time() - tt))

    def write_pl(self, params, pl_file, node_x, node_y):
        """
        @brief write .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("writing to %s" % (pl_file))
        content = "UCLA pl 1.0\n"
        str_node_names = np.array(self.node_names).astype(np.str)
        str_node_orient = np.array(self.node_orient).astype(np.str)
        for i in range(self.num_movable_nodes):
            content += "\n%s %g %g : %s" % (
                str_node_names[i],
                node_x[i],
                node_y[i],
                str_node_orient[i],
            )
        # use the original fixed cells, because they are expanded if they contain shapes
        fixed_node_indices = list(self.rawdb.fixedNodeIndices())
        for i, node_id in enumerate(fixed_node_indices):
            content += "\n%s %g %g : %s /FIXED" % (
                str(self.rawdb.nodeName(node_id)),
                float(self.rawdb.node(node_id).xl()),
                float(self.rawdb.node(node_id).yl()),
                "N",  # still hard-coded
            )
        for i in range(
            self.num_movable_nodes + self.num_terminals,
            self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs,
        ):
            content += "\n%s %g %g : %s /FIXED_NI" % (
                str_node_names[i],
                node_x[i],
                node_y[i],
                str_node_orient[i],
            )
        with open(pl_file, "w") as f:
            f.write(content)
        logging.info("write_pl takes %.3f seconds" % (time.time() - tt))

    def write_nets(self, params, net_file):
        """
        @brief write .net file
        @param params parameters
        @param net_file .net file
        """
        tt = time.time()
        logging.info("writing to %s" % (net_file))
        content = "UCLA nets 1.0\n"
        content += "\nNumNets : %d" % (len(self.net2pin_map))
        content += "\nNumPins : %d" % (len(self.pin2net_map))
        content += "\n"

        for net_id in range(len(self.net2pin_map)):
            pins = self.net2pin_map[net_id]
            content += "\nNetDegree : %d %s" % (len(pins), self.net_names[net_id])
            for pin_id in pins:
                # (self.pin_offset_x[pin_id] - self.node_x[pin_id]/2) / params.scale_factor
                content += "\n\t%s %s : %d %d" % (
                    self.node_names[self.pin2node_map[pin_id]],
                    self.pin_direct[pin_id],
                    self.pin_offset_x[pin_id] / params.scale_factor,
                    self.pin_offset_y[pin_id] / params.scale_factor,
                )

        with open(net_file, "w") as f:
            f.write(content)
        logging.info("write_nets takes %.3f seconds" % (time.time() - tt))

    def apply(self, params, node_x, node_y):
        """
        @brief apply placement solution and update database
        """
        # assign solution
        self.node_x[self.movable_slice] = node_x[self.movable_slice]
        self.node_y[self.movable_slice] = node_y[self.movable_slice]

        # unscale locations
        node_x, node_y = self.unscale_pl(params.shift_factor, params.scale_factor)

        # update raw database
        place_io.PlaceIOFunction.apply(self.rawdb, node_x, node_y)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")

    params = Params.Params()
    params.load(sys.argv[sys.argv[1]])
    logging.info("parameters = %s" % (params))

    db = PlaceDB()
    db(params)

    db.print_node(1)
    db.print_net(1)
    db.print_row(1)
