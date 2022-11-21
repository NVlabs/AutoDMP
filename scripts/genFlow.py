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
import subprocess
import glob
import time
from pathlib import Path

opj = os.path.join


def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-autodmp_dir", help="Path to AutoDMP directory", type=str)
    parser.add_argument("-cadence_dir", help="Path to Cadence directory", type=str)
    parser.add_argument(
        "-base_ppa", help="Name for getting Base PPA parameters", type=str
    )
    parser.add_argument(
        "--reuse_params",
        help="Path to reuse_params json file",
        type=str,
        default="\\'\\'",
    )
    parser.add_argument(
        "-unopt_netlist",
        help="0: Use optimized netlist in AutoDMP, 1: Use unoptimized netlist in AutoDMP",
        type=int,
        choices=[0, 1],
    )
    parser.add_argument(
        "-macros_only",
        help="0: Full placement, 1: Macro placement only. Only keep macros from AutoDMP placement",
        type=int,
        choices=[0, 1],
    )
    parser.add_argument(
        "-bo_method", help="0: Single-Obj, 1: Multi-Obj", type=int, choices=[0, 1]
    )
    parser.add_argument("-bo_space", help="BO config space", type=str)
    parser.add_argument(
        "--so_d_weight", help="Single-Obj density weight", type=float, default=0.1
    )
    parser.add_argument(
        "--so_c_weight", help="Single-Obj congestion weight", type=float, default=0.1
    )
    parser.add_argument(
        "--gpu",
        help="0: No GPU present, 1: GPU present",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--num_workers", help="Number of parallel workers", type=int, default=8
    )
    parser.add_argument(
        "--iterations", help="Number of MOBO iterations", type=int, default=1000
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_config_args()

    if args.bo_method == 1:
        exp_name = f"{args.base_ppa}_multi_obj_{'unopt_netlist' if args.unopt_netlist else 'opt_netlist'}_{'macro_pl' if args.macros_only else 'full_pl'}"
    else:
        exp_name = f"{args.base_ppa}_single_obj_{'unopt_netlist' if args.unopt_netlist else 'opt_netlist'}_{'macro_pl' if args.macros_only else 'full_pl'}_d_{args.so_d_weight}_c_{args.so_c_weight}"

    run_dir = f"{'/'.join(args.cadence_dir.split('/')[:-1])}/{exp_name}"
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # copy Cadence files and Tcl flow
    subprocess.call(f"cp {args.cadence_dir}/*.{{tcl,sh}} {run_dir}/", shell=True)
    subprocess.call(f"cp -r {args.cadence_dir}/syn_handoff {run_dir}/", shell=True)
    subprocess.call(
        f"cp -r {opj(args.autodmp_dir, 'scripts/*')} {run_dir}/", shell=True
    )

    # run pre-AutoDMP
    os.system(f"chmod 777 {run_dir}/run_flow.sh")
    os.system(f"./run_flow.sh PreDP {args.unopt_netlist}")

    # run AutoDMP Bayesian optimization
    gpu = args.gpu
    multiobj = args.bo_method
    cfg = args.bo_space
    aux = glob.glob(f"{run_dir}/dpCollaterals/*.aux")[0]
    base_ppa = args.base_ppa
    reuse_params = args.reuse_params
    iterations = args.iterations
    workers = args.num_workers
    d_ratio = args.so_d_weight
    c_ratio = args.so_c_weight
    m_points = 32

    cmd = [
        "#!/bin/bash -x",
        f"cd {run_dir}",
        f"./run_flow.sh Tuner {gpu} {multiobj} {cfg} {aux} {base_ppa} {reuse_params} {iterations} {workers} {d_ratio} {c_ratio} {m_points} {opj(args.autodmp_dir, 'tuner')}",
        f"echo 'complete' > {run_dir}/training.complete",
    ]

    with open("slurm_run.sh", "w") as f:
        f.writelines("\n".join(cmd))

    os.system(f"chmod 777 -R {run_dir}")
    subprocess.call(f"./slurm.sh", shell=True)

    while not os.path.exists(f"{run_dir}/training.complete"):
        print("Waiting")
        time.sleep(120)
    print("AutoDMP completed")

    # run post-AutoDMP BO candidates
    rootdir = Path(os.getcwd())
    candidates = [d for d in rootdir.glob("tunerLog/*/best_cfgs/*") if d.is_dir()]
    commands = []
    for candidate in candidates:
        run_dir_candidate = opj(
            "/".join(run_dir.split("/")[:-1]), exp_name + "_pareto_" + candidate.name
        )
        subprocess.call(f"mkdir -p {run_dir_candidate}", shell=True)
        subprocess.call(
            f"cp {args.cadence_dir}/*.{{tcl,sh}} {run_dir_candidate}/", shell=True
        )
        subprocess.call(
            f"cp -r {args.cadence_dir}/syn_handoff {run_dir_candidate}/", shell=True
        )
        subprocess.call(
            f"cp -r {opj(args.autodmp_dir, 'scripts/*')} {run_dir_candidate}/",
            shell=True,
        )
        subprocess.call(
            f"cp {opj(candidate, '*.def')} {run_dir_candidate}/", shell=True
        )
        subprocess.call(f"cp -r {run_dir}/enc {run_dir_candidate}/", shell=True)
        commands.append(
            f"cd {run_dir_candidate}; ./run_flow.sh PostDP {args.unopt_netlist} {args.macros_only}; mv {run_dir_candidate} {run_dir}/"
        )
        os.system(f"chmod 777 -R {run_dir_candidate}")

    procs = [subprocess.Popen(cmd, shell=True) for cmd in commands]
    for p in procs:
        p.wait()
