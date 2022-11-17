import os
import shutil
import argparse
import subprocess
import glob
from pathlib import Path

opj = os.path.join


def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cadence_dir", help="Path to Cadence directory")
    parser.add_argument("-dreamplace_dir", help="Path to DREAMPlace directory")
    parser.add_argument("-unopt_netlist", help="Use unoptimized netlist in DREAMPlace")
    parser.add_argument("-macros_only", help="Only keep macros from DREAMPlace placement")
    parser.add_argument("-bo_method", help="Single-Obj vs. Multi-Obj")
    parser.add_argument("-bo_space", help="BO config space")
    parser.add_argument("--so_d_weight", help="Single-Obj density weight", default=0.1)
    parser.add_argument("--so_c_weight", help="Single-Obj congestion weight", default=0.1)
    args = parser.parse_args()
    return args

args = parse_config_args()

run_dir = f"{args.cadence_dir}_{args.unopt_netlist}_{args.macros_only}"
os.makedirs(run_dir, exist_ok=True)
os.chdir(run_dir)

# copy Cadence files and Tcl flow
shutil.copytree(args.cadence_dir, run_dir)
shutil.copytree(opj(args.dreamplace_dir, "tuner/cadence"), run_dir)

# run pre-DREAMPlace
subprocess.call(f"sh run_flow.sh PreDP {args.unoptNetlist}", shell=True)

# run DREAMPlace Bayesian Optimization
gpu = 1
multiobj = int(args.bo_method)
cfg = args.bo_space
aux = glob.glob(f"{run_dir}/dpCollaterals/*.aux")[0]
base_ppa = ...
reuse_params = ...
iterations = 1000
workers = 8
d_ratio = args.so_d_weight
c_ratio = args.so_c_weight
m_points = 32

cmd = f"{gpu} {multiobj} {cfg} {aux} {base_ppa} {reuse_params} {iterations} {workers} {d_ratio} {c_ratio} {m_points}"
subprocess.call(f"sh run_flow.sh Tuner {cmd}", shell=True)

# run post-DREAMplace BO candidates
rootdir = Path(os.getcwd())
candidates = [d for d in rootdir.glob('tunerLog/best_cfgs/*') if d.is_dir()]
commands = []
for candidate in candidates:
    run_dir_candidate = opj(args.cadence_dir, "..", candidate.name)
    shutil.copytree(args.cadence_dir, run_dir_candidate)
    shutil.copytree(candidate, run_dir_candidate)
    commands.append(f"os.chdir({run_dir_candidate}); sh run_flow.sh PostDP {args.unoptNetlist} {args.macros_only};")
 
procs = [subprocess.Popen(cmd) for cmd in commands ]
for p in procs:
   p.wait()
    
#subprocess.call(f"sh run_flow.sh PostDP {args.unoptNetlist} {args.macros_only}", shell=True)



