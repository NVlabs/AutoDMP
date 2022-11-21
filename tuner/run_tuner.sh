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

# example use: sh run_tuner.sh 0 0 \"\" ariane.aux ariane \"\" 10 4 0.1 0.1 32 tunerLog

#!/bin/bash -x

# export PYTHONPATH=/AutoDMP

gpu=${1}
multiobj=${2}
cfg=${3}
aux=${4}
base_ppa=${5}
reuse_params=${6}
iterations=${7}
workers=${8}
d_ratio=${9}
c_ratio=${10}
m_points=${11}
script_dir=${12}
log_dir=${13}
auxbase=$(basename $aux .aux)
# script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Parameters:" $@
printf "# Parameters: %s\n" "${#@}"

# kill previous processes
# ps -fA | grep tuner_train | awk '{print $2}' | xargs kill -9 $1

# launch master process
python $script_dir/tuner_train.py --multiobj $multiobj --cfgSearchFile $cfg --n_workers $workers --n_iterations $iterations --min_points_in_model $m_points --log_dir $log_dir/$auxbase --run_args aux_input=$aux &

# launch worker processes
for i in $(seq $workers); do
    python $script_dir/tuner_train.py --multiobj $multiobj --log_dir $log_dir/$auxbase --worker --worker_id $i --run_args aux_input=$aux gpu=$gpu base_ppa=$base_ppa reuse_params=$reuse_params --density_ratio $d_ratio --congestion_ratio $c_ratio &
done
