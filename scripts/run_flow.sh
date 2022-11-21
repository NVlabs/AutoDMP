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

#!/bin/bash

function fn_runInvs {
    # export PATH=$PATH:path to innovus:path to genus
    # export CDS_LIC_FILE=""
    innovus -64 -files $1 -log $2
}

function fn_PreDP {
    mkdir -p log >/dev/null 2>&1
    cp run_invs_AutoDMP.tcl tmp.tcl
    sed -i "s/set var(AutoDMP,prePlace) [0-1]/set var(AutoDMP,prePlace) 1/g" tmp.tcl
    sed -i "s/set var(AutoDMP,unoptNetlist) [0-1]/set var(AutoDMP,unoptNetlist) ${1}/g" tmp.tcl
    fn_runInvs "tmp.tcl" "log/preDP_$1.log"
}

function fn_PostDP {
    mkdir -p log >/dev/null 2>&1
    cp run_invs_AutoDMP.tcl tmp.tcl
    sed -i "s/set var(AutoDMP,prePlace) [0-1]/set var(AutoDMP,prePlace) 0/g" tmp.tcl
    sed -i "s/set var(AutoDMP,unoptNetlist) [0-1]/set var(AutoDMP,unoptNetlist) ${1}/g" tmp.tcl
    sed -i "s/set var(AutoDMP,macrosOnly) [0-1]/set var(AutoDMP,macrosOnly) ${2}/g" tmp.tcl
    fn_runInvs "tmp.tcl" "log/postDP_$1_$2.log"
}

function fn_Tuner {
    mkdir -p tunerLog >/dev/null 2>&1
    # gpu multiobj cfg aux base_ppa reuse_params iterations workers d_ratio c_ratio m_points
    sh ../run_tuner.sh "$@" tunerLog 2>&1 | tee tunerLog/log.txt
}

# Run command
if [[ $1 = @(runInvs|PreDP|PostDP|Tuner) ]]; then
    until fn_$1 ${@:2}; do
        echo "Command crashed with exit code $?. Respawning..." >&2
        sleep 1
    done
else
    echo "Wrong command."
fi
