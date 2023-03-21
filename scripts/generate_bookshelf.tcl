# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

####################################################
# Preprocessing
####################################################
if {![info exists SITE] || ![info exists TOP_ROUTING_LAYER] || ![info exists HALO_WIDTH]} {
    error {ERROR: please set variables SITE, TOP_ROUTING_LAYER, HALO_WIDTH}
}

# unfix cells
dbSet top.insts.pStatus placed

# remove buffers
deleteBufferTree

####################################################
# Query Objects
####################################################
set nets  [dbGet -u -e top.nets {.numTerms > 1 && .isPwrOrGnd == 0}]
set insts [dbGet -u -e top.insts]
set fixed_insts [dbGet -u -e $insts.pStatus fixed -p]
set terms [dbGet -u -e top.terms]
set pins  [dbGet -u -e $nets.instTerms]
set blkgs {}

set num_insts [llength $insts]
set num_nets  [llength $nets]
set num_terms [llength $terms]
set num_pins  [llength $pins]
set num_blkgs [llength $blkgs]
set num_fixed_insts [llength $fixed_insts]
set num_nodes [expr $num_insts + $num_terms + $num_blkgs]
set num_terminals [expr $num_fixed_insts + $num_terms + $num_blkgs]

set DB_UNITS [dbGet head.dbUnits]
set DB_DIGITS 0

####################################################
# Helpers
####################################################
proc ladd {l} {::tcl::mathop::+ {*}$l}

proc Rescale { x } {
    set digits $::DB_DIGITS
    set db_units $::DB_UNITS
    set value [expr double(round($db_units * $x * 1e$digits))/1e$digits]
    if { $digits == 0 } {
        return [expr round($value)]
    } else {
        return $value
    }
}

set ORIENTS [dict create "R0" "N" "R180" "S" "R90" "W" "R270" "E" "MY" "FN" "MX" "FS" "MX90" "FW" "MY90" "FE"]
proc Orient {o} {
    return [dict get $::ORIENTS $o]
}

####################################################
# Bookshelf
####################################################
set design [dbGet top.name]

#########################
# Aux
#########################
set fp_aux [open ${design}.aux w]
puts $fp_aux "RowBasedPlacement : ${design}.nodes ${design}.nets ${design}.wts ${design}.pl ${design}.scl"
close $fp_aux

#########################
# Nodes and Placement
#########################
set fp_nodes [open ${design}.nodes w]
puts $fp_nodes "UCLA nodes 1.0\n"
puts $fp_nodes "NumNodes : $num_nodes"
puts $fp_nodes "NumTerminals : $num_terminals\n"

set fp_pl [open ${design}.pl w]
puts $fp_pl "UCLA pl 1.0\n"

# write IOs
set pin_size_x [Rescale 0.0]
set pin_size_y [Rescale 0.0]
foreach p $terms {
    set name [dbGet $p.defName]
    set x [Rescale [dbGet $p.pt_x]]
    set y [Rescale [dbGet $p.pt_y]]
    puts $fp_nodes "$name $pin_size_x $pin_size_y terminal_NI"
    puts $fp_pl "$name $x $y : N /FIXED_NI"
}

# write cells
set MACRO_HALO [Rescale $HALO_WIDTH]
foreach e $insts {
    set name [dbGet $e.defName]
    set width [Rescale [dbGet $e.cell.size_x]]
    set height [Rescale [dbGet $e.cell.size_y]]
    set x [Rescale [dbGet $e.pt_x]]
    set y [Rescale [dbGet $e.pt_y]]
    set orient "N"
    set status [dbGet $e.pStatus]
    if { [dbGet $e.cell.baseClass] == "block" } {
        set width [expr $width + 2 * $MACRO_HALO]
        set height [expr $height + 2 * $MACRO_HALO]
        set x [expr $x - $MACRO_HALO]
        set y [expr $y - $MACRO_HALO]
        set orient [Orient [dbGet $e.orient]]
    }
    if { $status == "fixed" } {
        puts $fp_nodes "$name $width $height terminal"
        puts $fp_pl "$name $x $y : $orient /FIXED"
    } else {
        puts $fp_nodes "$name $width $height"
        puts $fp_pl "$name $x $y : $orient"
    }
}
close $fp_nodes
close $fp_pl

# save macros
set macros [dbGet [dbGet -p2 $insts.cell.baseClass "block"].defName]
set fp [open ${design}.macros w]
puts $fp "$MACRO_HALO"
puts $fp "$macros"
close $fp

#########################
# Nets and Weights
#########################
set fp_nets [open ${design}.nets w]
set fp_wts  [open ${design}.wts w]

puts $fp_nets "UCLA nets 1.0\n"
puts $fp_nets "NumNets : $num_nets"
puts $fp_nets "NumPins : $num_pins\n"

puts $fp_wts "UCLA wts 1.0\n"

foreach n $nets {
    set net_name [dbGet $n.defName]
    set degree [dbGet $n.numTerms]
    set net_pins [dbGet $n.allTerms]
    puts $fp_nets "NetDegree : $degree $net_name"
    foreach p $net_pins {
        if { [dbGet $p.objType] == "term" } {
            set node_name [dbGet $p.defName]
            set dir [dbGet $p.inOutDir]
            if { $dir == "input" } { set dir "O" } else { set dir "I" }
            set xoffset 0
            set yoffset 0
        } else {
            set node_name [dbGet $p.inst.defName]
            set dir [dbGet $p.cellTerm.direction]
            if { $dir == "input" } { set dir "I" } else { set dir "O" }
            set pin_name [dbGet $p.defName]
            set cell_pin [dbGet $p.inst.cell.terms {.name == $pin_name}]
            set xoffset [Rescale [dbGet $cell_pin.pt_x]]
            set yoffset [Rescale [dbGet $cell_pin.pt_y]]
            set width [Rescale [dbGet $p.inst.cell.size_x]]
            set height [Rescale [dbGet $p.inst.cell.size_y]]
            set xoffset [expr int($xoffset - $width/2)]
            set yoffset [expr int($yoffset - $height/2)]
        }
        puts $fp_nets "  $node_name $dir : $xoffset $yoffset"
    }
}
close $fp_nets
close $fp_wts

#########################
# SCL
#########################
set rows [dbGet -u -e -p2 top.fPlan.rows.site.name $SITE]
if { [llength [lsort -unique [dbGet $rows.numX]]] > 1 || [llength [lsort -unique -real [dbGet $rows.box_llx]]] > 1} {
    error {ERROR: There seems to be an issue with the placement rows}
}
set num_rows [llength $rows]

# sort rows by y coordinate
set rows_index [lsort -indices -real [dbGet $rows.box_lly]]
set rows [lmap i $rows_index {lindex $rows $i}]

set fp_scl [open ${design}.scl w]
puts $fp_scl "UCLA scl 1.0\n"
puts $fp_scl "NumRows : $num_rows\n"

foreach r $rows {
    puts $fp_scl "CoreRow Horizontal"
    puts $fp_scl "  Coordinate   : [Rescale [dbGet $r.box_lly]]"
    puts $fp_scl "  Height       : [Rescale [expr [dbGet $r.site.size_y] * [dbGet $r.numY]]]"
    puts $fp_scl "  Sitewidth    : [Rescale [dbGet $r.site.size_x]]"
    puts $fp_scl "  Sitespacing  : [Rescale [dbGet $r.site.size_x]]"
    puts $fp_scl "  Siteorient   : [string equal [dbGet $r.orient] "R0"]"
    puts $fp_scl "  Sitesymmetry : 1"
    puts $fp_scl "  SubrowOrigin : [Rescale [dbGet $r.box_llx]]  NumSites : [dbGet $r.numX]"
    puts $fp_scl "End"
}
close $fp_scl

#########################
# Routing information
#########################
set top_route $TOP_ROUTING_LAYER
set bot_route 2
set layers [dbGet [dbGet -e -p head.layers.type routing] {.num >= $bot_route && .num <= $top_route}]

proc MetalLength { lyr w h } {
    if {[dbGet $lyr.direction] == "Vertical"} {
        set dir "y"
        set pitch [dbGet $lyr.pitchX]
        set ml [expr floor($w / $pitch) * $h]
    } else {
        set dir "x"
        set pitch [dbGet $lyr.pitchY]
        set ml [expr floor($h / $pitch) * $w]
    }
    return [list $dir $ml]
}

set vlength {}
set hlength {}
set chip_width  [dbGet top.fPlan.box_sizex]
set chip_height [dbGet top.fPlan.box_sizey]
foreach lyr $layers {
    lassign [MetalLength $lyr $chip_width $chip_height] dir ml
    if { $dir == "y" } { lappend vlength $ml } else { lappend hlength $ml }
}
set vlength [Rescale [ladd $vlength]]
set hlength [Rescale [ladd $hlength]]

set fp_route [open ${design}.route_info w]
puts $fp_route "$vlength $hlength"
foreach m $macros {
    set ptr [dbGetInstByName $m]
    set width [dbGet $ptr.box_sizex]
    set height [dbGet $ptr.box_sizey]
    set ratio_h_w [expr $height / $width]
    set ml_v {}; set ml_h {};
    foreach lyr $layers {
        set obs_area [ladd [dbGet [dbGet -p2 $ptr.cell.allObstructions.layer.num [dbGet $lyr.num]].shapes.rect_area]]
        if { $obs_area == 0 } { continue }
        set obs_h [expr sqrt($obs_area * $ratio_h_w)]
        set obs_w [expr $obs_area / $obs_h]
        lassign [MetalLength $lyr $obs_w $obs_h] dir ml
        if { $dir == "y" } { lappend ml_v $ml } else { lappend ml_h $ml }
    }
    set obs_area_v [Rescale [ladd $ml_v]]
    set obs_area_h [Rescale [ladd $ml_h]]
    puts $fp_route "$m $obs_area_v $obs_area_h"
}
close $fp_route

#########################
# Post-processing
#########################
defOut -floorplan -unplaced -netlist ${design}.ref.def

# process names causing Bookshelf issues in AutoDMP
exec /bin/sed -i {s/\\\[/\[/g;s/\\\]/\]/g;s/\\\//\//g} ${design}.nodes
exec /bin/sed -i {s/\\\[/\[/g;s/\\\]/\]/g;s/\\\//\//g} ${design}.nets
exec /bin/sed -i {s/\\\[/\[/g;s/\\\]/\]/g;s/\\\//\//g} ${design}.pl
exec /bin/sed -i {s/\\\[/\[/g;s/\\\]/\]/g;s/\\\//\//g} ${design}.wts
exec /bin/sed -i {s/\\\[/\[/g;s/\\\]/\]/g;s/\\\//\//g} ${design}.macros
exec /bin/sed -i {s/\\\[/\[/g;s/\\\]/\]/g;s/\\\//\//g} ${design}.route_info
exec /bin/sed -i -r {s/([0-9]+\s)_/\1AutoDMP_/g} ${design}.nets

eval exec mv [glob ${design}.*] ${dpDir}/
