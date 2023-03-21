# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# @file   Params.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  User parameters
#

import os
import sys
import json
import math
import logging
from collections import OrderedDict
import pdb


class Params:
    """
    @brief Parameter class
    """

    def __init__(self):
        """
        @brief initialization
        """
        filename = os.path.join(os.path.dirname(__file__), "params.json")
        self.__dict__ = {}
        params_dict = {}
        with open(filename, "r") as f:
            params_dict = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in params_dict.items():
            if "default" in value:
                self.__dict__[key] = value["default"]
            else:
                self.__dict__[key] = None
        self.__dict__["params_dict"] = params_dict

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """
========================================================
                       DREAMPlace
            Yibo Lin (http://yibolin.com)
   David Z. Pan (http://users.ece.utexas.edu/~dpan)
========================================================
"""
        logging.info(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        logging.info(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table
        """
        key_length = len("JSON Parameter")
        key_length_map = []
        default_length = len("Default")
        default_length_map = []
        description_length = len("Description")
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3:  # python 2
                flag = isinstance(value["default"], unicode)
            else:  # python 3
                flag = isinstance(value["default"], str)
            if flag and not value["default"] and "required" in value:
                return value["required"]
            else:
                return value["default"]

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value["description"]))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
            "JSON Parameter",
            " " * (key_length - len("JSON Parameter") + 1),
            "Default",
            " " * (default_length - len("Default") + 1),
            "Description",
            " " * (description_length - len("Description") + 1),
        )
        content += "| %s | %s | %s |\n" % (
            "-" * (key_length + 1),
            "-" * (default_length + 1),
            "-" * (description_length + 1),
        )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                key,
                " " * (key_length - key_length_map[count] + 1),
                str(getDefaultColumn(key, value)),
                " " * (default_length - default_length_map[count] + 1),
                value["description"],
                " " * (description_length - description_length_map[count] + 1),
            )
            count += 1
        return content

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != "params_dict":
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load from json
        """
        for key, value in data.items():
            self.__dict__[key] = value

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, "w") as f:
            json.dump(self.toJson(), f)

    def load(self, args):
        """
        @brief load parameters
        """
        if len(args) == 1 and args[0].endswith(".json"):
            with open(args[0], "r") as f:
                self.fromJson(json.load(f))
        else:
            self.fromCmdLine(args)

    def __str__(self):
        """
        @brief string
        """
        return str(self.toJson())

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def __eq__(self, other):
        """
        @brief test equality
        """
        if isinstance(other, Params):
            ignore = {"params_dict"}
            return all(
                (self.__dict__[key] == other.__dict__[key]) | (key in ignore)
                for key in self.__dict__
            )
        return False

    def design_name(self):
        """
        @brief speculate the design name for dumping out intermediate solutions
        """
        if self.aux_input:
            design_name = (
                os.path.basename(self.aux_input).replace(".aux", "").replace(".AUX", "")
            )
        elif self.verilog_input:
            design_name = (
                os.path.basename(self.verilog_input).replace(".v", "").replace(".V", "")
            )
        elif self.def_input:
            design_name = (
                os.path.basename(self.def_input).replace(".def", "").replace(".DEF", "")
            )
        return design_name

    def solution_file_suffix(self):
        """
        @brief speculate placement solution file suffix
        """
        if self.def_input is not None and os.path.exists(self.def_input):  # LEF/DEF
            return "def"
        else:  # Bookshelf
            return "pl"

    def fromCmdLine(self, args):
        """
        @brief load from command line
        """
        for key, value in (
            (k.lstrip("--"), v) for k, v in (arg.split("=") for arg in args)
        ):
            self.__dict__[key] = value

    def update(self, params):
        """
        @brief update parameters
        """
        if isinstance(params, dict):
            self.fromJson(params)
        elif isinstance(params, str):
            self.load([params])
        elif isinstance(params, list):
            self.load(params)
        elif isinstance(params, Params):
            self.fromJson(params.__dict__)
