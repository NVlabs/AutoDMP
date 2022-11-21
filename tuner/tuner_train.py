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

import os
import argparse
import time
import torch
import shutil
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import MOBOHB as MOBOHB

opj = os.path.join

# disable heavy logging from C++
os.environ["DREAMPLACE_DISABLE_PRINT"] = "1"

from tuner_worker import AutoDMPWorker
from tuner_utils import parse_dictionary, str2bool, dp_to_def
from tuner_analyze import get_candidates, plot_pareto


parser = argparse.ArgumentParser(
    description="Optimization of AutoDMP parameters",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--multiobj",
    type=str2bool,
    help="Enable Multi-Objective BOHB",
    default=False,
)
parser.add_argument(
    "--cfgSearchFile",
    help="Search-space config file",
    default="",
)
parser.add_argument(
    "--min_points_in_model",
    type=int,
    help="Number of observations to start building a KDE",
    default=32,
)
parser.add_argument(
    "--min_budget",
    type=int,
    help="Minimum budget used during the optimization.",
    default=1,
)
parser.add_argument(
    "--max_budget",
    type=int,
    help="Maximum budget used during the optimization.",
    default=1,
)
parser.add_argument(
    "--n_iterations",
    type=int,
    help="Number of iterations performed by the optimizer",
    default=200,
)
parser.add_argument(
    "--n_workers",
    type=int,
    help="Number of parallel workers used by the optimizer",
    default=8,
)
parser.add_argument(
    "--n_samples", type=int, help="Number of samples per iteration", default=64
)
parser.add_argument(
    "--congestion_ratio", type=float, help="Congestion cost ratio", default=0.5
)
parser.add_argument(
    "--density_ratio", type=float, help="Density cost ratio", default=0.5
)
parser.add_argument(
    "--num_pareto", type=int, help="Number of Pateto points to propose", default=5
)
parser.add_argument("--log_dir", help="Result log dir", default="logs_tuner")
parser.add_argument("--run_id", help="Run id for communication", default="0")
parser.add_argument(
    "--run_args", nargs="*", help="Args for AutoDMP", action=parse_dictionary
)
parser.add_argument(
    "--worker", help="Flag to turn this into a worker process", action="store_true"
)
parser.add_argument("--worker_id", type=int, help="Worker id", default=0)
args = parser.parse_args()


# Worker
if args.worker:
    time.sleep(5)  # artificial delay to make sure the nameserver is already running
    print(f"Starting worker process number {args.worker_id}")
    print(f"Worker args: {args}")

    if args.run_args["gpu"] == "1":
        # alternate the gpu_id of workers
        gpu_id = args.worker_id % torch.cuda.device_count()
        args.run_args["gpu_id"] = gpu_id
        print(f"Assigning worker {args.worker_id} to GPU {gpu_id}")

    w = AutoDMPWorker(
        nameserver="127.0.0.1",
        run_id=args.run_id,
        log_dir=args.log_dir,
        congestion_ratio=args.congestion_ratio,
        density_ratio=args.density_ratio,
        default_config=args.run_args,
        multiobj=args.multiobj,
    )
    w.run(background=False)
    exit(0)


# Master
print("Starting master process")
print(f"Master args: {args}")

# Create Log directory
os.makedirs(args.log_dir, exist_ok=True)
result_logger = hpres.json_result_logger(directory=args.log_dir, overwrite=True)

# Start a nameserver
NS = hpns.NameServer(run_id=args.run_id, host="127.0.0.1", port=None)
NS.start()

# Run an optimizer
if args.multiobj:
    motpe_params = {
        "init_method": "random",
        "num_initial_samples": 10,
        "num_candidates": 24,
        "gamma": 0.10,
    }
    bohb = MOBOHB(
        configspace=AutoDMPWorker.get_configspace(args.cfgSearchFile),
        parameters=motpe_params,
        run_id=args.run_id,
        min_points_in_model=args.min_points_in_model,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        num_samples=args.n_samples,
        result_logger=result_logger,
    )
else:
    bohb = BOHB(
        configspace=AutoDMPWorker.get_configspace(args.cfgSearchFile),
        run_id=args.run_id,
        min_points_in_model=args.min_points_in_model,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        num_samples=args.n_samples,
        result_logger=result_logger,
    )
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

# Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Analysis
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print("A total of %i unique configurations where sampled." % len(id2config.keys()))
print("A total of %i runs where executed." % len(res.get_all_runs()))
all_runs = res.get_all_runs()
print(
    "The run took %.1f seconds to complete."
    % (all_runs[-1].time_stamps["finished"] - all_runs[0].time_stamps["started"])
)

# Propose Pareto points
dp_dir, netlist = os.path.split(args.run_args["aux_input"])
netlist = netlist.replace(".aux", "")
result = hpres.logged_results_to_HBS_result(args.log_dir)
candidates, df = get_candidates(result, num=args.num_pareto)
print("Pareto candidates are:")
print(candidates.to_markdown())

# Save Pareto configs and generate DEFs
print("Generating DEF files")
dest = opj(args.log_dir, "best_cfgs")
os.makedirs(dest, exist_ok=True)
df.to_pickle(opj(dest, f"{netlist}.dataframe.pkl"))
for _, row in candidates.iterrows():
    cfg_id = "run-" + "_".join([s for s in re.findall(r"\b\d+\b", row["ID"])])
    print(f"Generating DEF for candidate {cfg_id}")
    src_cfg, dest_cfg = opj(args.log_dir, cfg_id), opj(dest, cfg_id)
    shutil.copytree(src_cfg, dest_cfg, dirs_exist_ok=True)
    # generate DEF
    def_file = opj(dp_dir, f"{netlist}.ref.def")
    macro_file = opj(dp_dir, f"{netlist}.macros")
    pl_file = opj(src_cfg, netlist, f"{netlist}.gp.pl")
    new_def_file = opj(dest_cfg, f"{netlist}.AutoDMP.def")
    dp_to_def(def_file, pl_file, macro_file, new_def_file)

# Plot Pareto curve
plot_pareto(result, args.num_pareto, opj(dest, f"{netlist}.pareto.png"))
