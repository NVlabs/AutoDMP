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

proc ppa_cost { wl hcong vcong density {wns 0} {tns 0} {pwr 0}} {
    return [expr 1e-6 * $wl + ($hcong + $vcong) + $density / 100]
}

proc load_candidate { candidate } {
    defIn -components ${candidate}
    dbWireCleanup; checkPlace; clearDrc;
    refine_macro_place
    # refinePlace -eco true
    redirect -variable ret { puts [reportDensityMap] }
    regexp "> 0.750 = (\[0-9\]+.\[0-9\]+) %" $ret _ density
    redirect -variable ret { puts [earlyGlobalRoute] }
    # report_power; timeDesign -preCTS;
    regexp "Total length: (\[0-9\]+)um" $ret _ wl
    regexp "Total half perimeter of net bounding box: (\[0-9\]+)um" $ret _ hpwl
    regexp "Overflow after Early Global Route (\[0-9\]+.\[0-9\]+)% H \\+ (\[0-9\]+.\[0-9\]+)% V" $ret _ hcong vcong
    set ppa [list $wl $hpwl $hcong $vcong $density]
    puts "Candidate ${candidate} PPA: ${ppa}"
    return $ppa
}

proc evaluate_paretos { } {
    set best_candidate ""
    set best_cost 2147483647
    puts "###### INFO: Evaluating AutoDMP Paretos ######"
    set candidates [open "|find best_cfgs -name \*.AutoDMP.def -print" r]
    while { [gets $candidates c] >= 0 } {
        lassign [load_candidate $c] wl hpwl hcong vcong density
        set cost [ppa_cost $wl $hcong $vcong $density]
        if { $cost < $best_cost } {
            set best_cost $cost
            set best_candidate $c
        }
    }
    return $best_candidate
}