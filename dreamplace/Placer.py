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
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
import os
import sys
import time
import torch
import random
import numpy as np
import logging

# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import dreamplace.configure as configure
import dreamplace.Params as Params
import dreamplace.PlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace


# set up logging
logging.root.name = "DREAMPlace"
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-7s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("DREAMPlace.log", mode="w"),
        # logging.StreamHandler(sys.stdout),
    ],
)


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PlacementEngine:
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    def __init__(self, params):
        # load parameters
        self.params = Params.Params()
        self.update_params(params)
        self.params.printWelcome()

        if self.params.evaluate_pl == 1:
            self.params.global_place_flag = 1
            self.params.global_place_stages[0]["iteration"] = 0
            self.params.random_center_init_flag = 0
            self.params.enable_fillers = 0
            self.params.detailed_place_flag = 0
            self.params.legalize_flag = 0
            self.params.routability_opt_flag = 0
            self.params.macro_halo_x = 0
            self.params.macro_halo_y = 0
            self.params.plot_flag = 1
            self.params.gp_noise_ratio = 0.0
            self.params.pin_density = -1
            logging.critical("running in evaluation mode")

        # seed for reproducibility
        seed_all(self.params.random_seed)

        # control multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)
        torch.set_num_threads(self.params.num_threads)

        assert (not self.params.gpu) or configure.compile_configurations[
            "CUDA_FOUND"
        ] == "TRUE", "CANNOT enable GPU without CUDA compiled"

        self.placedb = None
        self.placer = None

        self.rsmt = float("inf")
        self.hpwl = float("inf")
        self.congestion = float("inf")
        self.density = float("inf")
        self.metrics = None

    def setup_rawdb(self):
        # read cpp database
        tt = time.time()
        if self.placedb is None:
            self.placedb = PlaceDB.PlaceDB()
            self.placedb.read(self.params)
        logging.info("setting up raw database takes %.2f seconds" % (time.time() - tt))

    def setup_placedb(self):
        # setup python placement database
        tt = time.time()
        self.setup_rawdb()
        self.placedb.initialize_from_rawdb(self.params)
        self.placedb.initialize(self.params)
        logging.info(
            "setting up placement database takes %.2f seconds" % (time.time() - tt)
        )

    def update_params(self, new_params):
        self.params.update(new_params)
        logging.info("parameters = %s" % (self.params))

    def place(self):
        # solve placement
        tt = time.time()
        self.placer = NonLinearPlace.NonLinearPlace(self.params, self.placedb)
        logging.info(
            "non-linear placement initialization takes %.2f seconds"
            % (time.time() - tt)
        )
        self.rsmt, self.hpwl, self.metrics = self.placer(self.params, self.placedb)
        logging.info("non-linear placement takes %.2f seconds" % (time.time() - tt))

    def external_detailed_placer(self):
        # call external detailed placement
        # TODO: support more external placers, currently only support
        # 1. NTUplace3/NTUplace4h with Bookshelf format
        # 2. NTUplace_4dr with LEF/DEF format
        logging.info(
            "Use external detailed placement engine %s"
            % (self.params.detailed_place_engine)
        )
        if self.params.solution_file_suffix() == "pl" and any(
            dp_engine in self.params.detailed_place_engine
            for dp_engine in ["ntuplace3", "ntuplace4h"]
        ):
            dp_out_file = self.gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if (
                self.params.target_density < 1.0
                and not self.params.routability_opt_flag
            ):
                target_density_cmd = " -util %f" % (self.params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                self.params.detailed_place_engine,
                self.params.aux_input,
                self.gp_out_file,
                target_density_cmd,
                dp_out_file,
                self.params.detailed_place_command,
            )
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info(
                "External detailed placement takes %.2f seconds" % (time.time() - tt)
            )

            if self.params.plot_flag:
                # read solution and evaluate
                self.placedb.read_pl(self.params, dp_out_file + ".ntup.pl")
                iteration = len(self.metrics)
                pos = self.placer.init_pos
                pos[0 : self.placedb.num_physical_nodes] = self.placedb.node_x
                pos[
                    self.placedb.num_nodes : self.placedb.num_nodes
                    + self.placedb.num_physical_nodes
                ] = self.placedb.node_y
                hpwl, density_overflow, max_density = self.placer.validate(
                    self.placedb, pos, iteration
                )
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density)
                )
                self.placer.plot(self.params, self.placedb, iteration, pos)
        elif "ntuplace_4dr" in self.params.detailed_place_engine:
            dp_out_file = self.gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (self.params.detailed_place_engine)
            for lef in self.params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (self.gp_out_file)
            if self.params.verilog_input:
                cmd += " -verilog %s" % (self.params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(self.params.verilog_input))
                benchmark_dir
            )
            cmd += " -noglobal %s ; " % (self.params.detailed_place_command)
            # cmd += " %s ; " % (self.params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file
            )
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info(
                "External detailed placement takes %.2f seconds" % (time.time() - tt)
            )
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )

    def get_congestion(self):
        pos = self.placer.data_collections.pos[0]
        # use new congestion estimation
        if self.params.route_info_input != "":
            (
                congestion_map,
                max_overflow,
                total_overflow,
            ) = self.placer.op_collections.get_congestion_map_op(
                pos,
                True,
            )
            logging.info(f"Overflow max/total: {max_overflow}/{total_overflow}")
        else:
            congestion_map = self.placer.op_collections.get_congestion_map_op(pos)

        congestion, _ = torch.topk(
            congestion_map.flatten(), k=int(0.1 * congestion_map.numel())
        )
        self.congestion = float(congestion.mean())
        logging.info(f"Congestion score {self.congestion}")

    def save_placement(self):
        # write placement solution
        self.path = "%s/%s" % (self.params.result_dir, self.params.design_name())
        if not os.path.exists(self.path):
            os.system("mkdir -p %s" % (self.path))
        self.gp_out_file = os.path.join(
            self.path,
            "%s.gp.%s"
            % (self.params.design_name(), self.params.solution_file_suffix()),
        )
        self.placedb.write(self.params, self.gp_out_file)

    def run(self):
        # run entire placement flow
        tt = time.time()
        self.setup_placedb()
        self.place()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ]
        # ) as p:
        #     self.place()
        # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

        if self.rsmt != float("inf"):
            self.save_placement()
            self.density = float(self.params.target_density)

            if self.params.detailed_place_engine and os.path.exists(
                self.params.detailed_place_engine
            ):
                self.external_detailed_placer()
            elif self.params.detailed_place_engine:
                logging.warning(
                    "External detailed placement engine %s or aux file NOT found"
                    % (self.params.detailed_place_engine)
                )

            if self.params.get_congestion_map and hasattr(
                self.placer.op_collections, "get_congestion_map_op"
            ):
                tt2 = time.time()
                self.get_congestion()
                logging.info(
                    "congestion extraction takes %.3f seconds" % (time.time() - tt2)
                )

            logging.info("placement takes %.3f seconds" % (time.time() - tt))
        else:
            logging.warning("placement failed")

        final_ppa = {
            "rsmt": self.rsmt,
            "hpwl": self.hpwl,
            "congestion": self.congestion,
            "density": self.density,
        }
        niter = len(self.metrics["objective"])
        other_metrics = {
            "iteration": niter,
            "objective": -1 if niter == 0 else self.metrics["objective"][-1],
            "overflow": -1 if niter == 0 else self.metrics["overflow"][-1],
            "max_density": -1 if niter == 0 else self.metrics["density"][-1],
        }
        final_ppa.update(other_metrics)
        logging.info(f"Final PPA: {final_ppa}")

        return final_ppa


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """

    if len(sys.argv) == 1 or "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        params = Params.Params()
        params.printWelcome()
        params.printHelp()
        exit()

    engine = PlacementEngine(sys.argv[1:])
    ppa = engine.run()
