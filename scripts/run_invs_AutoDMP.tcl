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

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
setMultiCpuUsage -localCpu 16

# design settings
source lib_setup.tcl
source design_setup.tcl

set handoff_dir "./syn_handoff"
set netlist ${handoff_dir}/${DESIGN}.v
set sdc ${handoff_dir}/${DESIGN}.sdc

# utilities
source ../../../../util/extract_report.tcl
source AutoDMP_utils.tcl

# report settings
set encDir enc
set dpDir dpCollaterals

if {![file exists $encDir/]} {
    exec mkdir $encDir/
}
if {![file exists $dpDir/]} {
    exec mkdir $dpDir/
}

# flow settings
set var(AutoDMP,prePlace) 0; # run pre- or post-AutoDMP flow
set var(AutoDMP,unoptNetlist) 0; # use unoptimized or optimized netlist
set var(AutoDMP,macrosOnly) 0; # use macros only vs. +std cells AutoDMP placement

set technology "NanGate45"
set designName "${DESIGN}_${technology}"

# ------------------------------------------------------------------------------
#  Pre-AutoDMP flow
# ------------------------------------------------------------------------------
if { ${var(AutoDMP,prePlace)} == 1 } {
    puts "###### INFO: Running Pre-AutoDMP flow ######"
    puts "###### Configuration: "
    parray var

    # create corners
    source mmmc_setup.tcl

    # default settings
    set init_pwr_net VDD
    set init_gnd_net VSS

    # default settings
    set init_verilog "$netlist"
    set init_design_netlisttype "Verilog"
    set init_design_settop 1
    set init_top_cell "$DESIGN"
    set init_lef_file "$lefs"

    # MCMM setup
    init_design -setup {WC_VIEW} -hold {BC_VIEW}
    set_power_analysis_mode -leakage_power_view WC_VIEW -dynamic_power_view WC_VIEW

    set_interactive_constraint_modes {CON}
    setAnalysisMode -reset
    setAnalysisMode -analysisType onChipVariation -cppr both

    clearGlobalNets
    globalNetConnect VDD -type pgpin -pin VDD -inst * -override
    globalNetConnect VSS -type pgpin -pin VSS -inst * -override
    globalNetConnect VDD -type tiehi -inst * -override
    globalNetConnect VSS -type tielo -inst * -override

    setOptMode -powerEffort low -leakageToDynamicRatio 0.5
    setGenerateViaMode -auto true
    generateVias

    # basic path groups
    createBasicPathGroups -expanded

    # initialize report
    echo "PD Stage, Core Area (um^2), Std Cell Area (um^2), Macro Area (um^2), Total Power (mW), Wirelength (um), WS (ns), TNS (ns), Congestion (H), Congestion (V)" > ${designName}_DETAILS.rpt
    set rpt_post_synth [extract_report postSynth]
    echo "$rpt_post_synth" >> ${designName}_DETAILS.rpt

    # setup floorplan
    defIn ${floorplan_def}
    if {[dbget top.terms.pStatus -v -e fixed] != "" } {
        source ../../../../util/place_pin.tcl
    }
    addHaloToBlock -allMacro $HALO_WIDTH $HALO_WIDTH $HALO_WIDTH $HALO_WIDTH

    # setup P&R options
    setPlaceMode -place_detail_legalization_inst_gap 1
    setFillerMode -fitGap true
    setDesignMode -topRoutingLayer $TOP_ROUTING_LAYER
    setDesignMode -bottomRoutingLayer 2

    if { ${var(AutoDMP,unoptNetlist)} == 1 } {
        # extract AutoDMP files
        source generate_bookshelf.tcl
        saveDesign -tcon -verilog -def ${encDir}/${designName}_preDP.enc
    } else {
        # concurrent macro placement
        place_design -concurrent_macros
        refine_macro_place
        dbset [dbget top.insts.cell.subClass block -p2].pStatus fixed

        # standard cell placement
        setPlaceMode -place_opt_run_global_place full
        place_opt_design

        # extract AutoDMP files
        source generate_bookshelf.tcl
        saveDesign -tcon -verilog -def ${encDir}/${designName}_preDP.enc
    }

    exit
}

# ------------------------------------------------------------------------------
#  Post-AutoDMP Flow
# ------------------------------------------------------------------------------
if { ${var(AutoDMP,prePlace)} == 0 } {
    puts "###### INFO: Running Post-AutoDMP flow ######"
    puts "###### Configuration: "
    parray var

    # load DB
    source ${encDir}/${designName}_preDP.enc

    # update constraint mode
    update_constraint_mode -name CON -sdc_files ${sdc}

    # load AutoDMP placement
    set best_candidate [evaluate_paretos]
    defIn -components ${best_candidate}
    refine_macro_place
    dbset [dbget top.insts.cell.baseClass block -p2].pStatus fixed
    # refinePlace -eco true

    if { ${var(AutoDMP,macrosOnly)} == 1 } {
        dbset [dbget top.insts.cell.subClass core -p2].pStatus unplaced; # unplace std cells
    }

    # PDN
    source ../../../../../Enablements/${technology}/util/pdn_config.tcl
    source pdn_flow.tcl

    # place-opt
    if { ${var(AutoDMP,macrosOnly)} == 1 } {
        setPlaceMode -place_opt_run_global_place full
        place_opt_design
    } else {
        # some cells might be unplaced due to P/G grid
        dbset [dbget top.insts.pStatus unplaced -p].pStatus placed

        if { ${var(AutoDMP,unoptNetlist)} == 1 } {
            # setDesignMode -flowEffort extreme
            place_opt_design -incremental
            # place_opt_design -incremental; # -incremental_timing
            # setDesignMode -flowEffort standard
        } else {
            place_opt_design -incremental
        }
    }
    saveDesign -tcon -verilog -def ${encDir}/${designName}_preCTS.enc
    set rpt_pre_cts [extract_report preCTS]
    echo "Post-AutoDMP $rpt_pre_cts" >> ${designName}_DETAILS.rpt

    # CTS
    set_ccopt_property post_conditioning_enable_routing_eco 1
    set_ccopt_property -cts_def_lock_clock_sinks_after_routing true
    setOptMode -unfixClkInstForOpt false

    create_ccopt_clock_tree_spec
    ccopt_design

    set_interactive_constraint_modes [all_constraint_modes -active]
    set_propagated_clock [all_clocks]
    set_clock_propagation propagated

    saveDesign -tcon -verilog -def $encDir/${designName}_cts.enc
    set rpt_post_cts [extract_report postCTS]
    echo "$rpt_post_cts" >> ${designName}_DETAILS.rpt

    # routing
    setNanoRouteMode -drouteVerboseViolationSummary 1
    setNanoRouteMode -routeWithSiDriven true
    setNanoRouteMode -routeWithTimingDriven true
    setNanoRouteMode -routeUseAutoVia true

    ##Recommended by lib owners
    # Prevent router modifying M1 pins shapes
    setNanoRouteMode -routeWithViaInPin "1:1"
    setNanoRouteMode -routeWithViaOnlyForStandardCellPin "1:1"

    ## limit VIAs to ongrid only for VIA1 (S1)
    setNanoRouteMode -drouteOnGridOnly "via 1:1"
    setNanoRouteMode -drouteAutoStop false
    setNanoRouteMode -drouteExpAdvancedMarFix true
    setNanoRouteMode -routeExpAdvancedTechnology true

    #SM suggestion for solving long extraction runtime during GR
    setNanoRouteMode -grouteExpWithTimingDriven false

    routeDesign
    saveDesign -tcon -verilog -def ${encDir}/${designName}_route.enc
    defOut -netlist -floorplan -routing ${designName}_route.def

    set rpt_post_route [extract_report postRoute]
    echo "$rpt_post_route" >> ${designName}_DETAILS.rpt

    ### Post-route Opt ###
    optDesign -postRoute
    set rpt_post_route [extract_report postRouteOpt]
    echo "$rpt_post_route" >> ${designName}_DETAILS.rpt

    ### Run DRC and LVS ###
    verify_connectivity -error 0 -geom_connect -no_antenna
    verify_drc -limit 0

    summaryReport -noHtml -outfile summaryReport/post_route.sum
    saveDesign -tcon -verilog -def ${encDir}/${designName}.enc
    defOut -netlist -floorplan -routing ${designName}.def

    exit
}
