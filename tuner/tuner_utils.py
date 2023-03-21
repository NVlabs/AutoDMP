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

import argparse
import re
import os


# ----------------------------------------------------------------------------
# Parse dictionary from command line
class parse_dictionary(argparse.Action):
    # parse dictionary given in command line
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            if "=" in value:
                key, value = value.split("=")
            else:
                key = value
                value = None
            getattr(namespace, self.dest)[key] = value


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# ----------------------------------------------------------------------------
# Convert Bookshelf to DEF
def dp_to_def(def_file, pl_file, macro_file, target_filename=""):
    macro_halo = 0
    macros = []
    if macro_file != "":
        with open(macro_file, "r") as f:
            content = f.read().splitlines()
            macro_halo = int(content[0])
            macros = content[1].split()

    pl_dict = {}
    with open(pl_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip("\n").split(" ")
            if len(line) >= 4:
                # x, y, orientation
                pl_dict[line[0]] = [int(line[1]), int(line[2]), line[4]]

    def_lines = []
    with open(def_file, "r") as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if "COMPONENTS" in line:
                start = True
            if "END COMPONENTS" in line:
                start = False
            if start and line.startswith("-"):
                split = line.split(" ")
                instance = (
                    split[1].replace("\[", "[").replace("\]", "]").replace("\/", "/")
                )
                if instance in pl_dict and "FIXED" not in split:
                    x, y, o = pl_dict[instance]
                    # shift back x, y
                    if instance in macros:
                        x += macro_halo
                        y += macro_halo
                    old_pos = line[line.find("(") : line.find(")") + 1]
                    line = line.replace(old_pos, "( %d %d )" % (x, y))
                    line = re.sub(
                        r"(.*\) )(N|S|W|E|FN|FS|FW|FE)($| )", r"\1%s\3" % o, line
                    )
            def_lines.append(line)

    if target_filename == "":
        dir_name, _ = os.path.split(pl_file)
        _, filename = os.path.split(def_file)
        target_filename = os.path.join(dir_name, "%s.DP.def" % filename.split(".")[0])

    with open(target_filename, "w") as f:
        f.writelines(def_lines)
