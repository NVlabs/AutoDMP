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

from dataclasses import make_dataclass, fields, asdict
from operator import itemgetter
import os
import copy
import json
from pathlib import Path
import logging
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json as CS_JSON

from hpbandster.core.worker import Worker
from dreamplace.Placer import PlacementEngine

from tuner.tuner_configs import (
    DREAMPLACE_BASE_CONFIG,
    DREAMPLACE_BASE_PPA,
    DREAMPLACE_BAD_RATIO,
    DREAMPLACE_BEST_CFG,
)

opj = os.path.join

# Wrap DREAMPlace config in dataclass
def update_cfg(self, cfg):
    my_fields = [f.name for f in fields(self)]
    for p, v in cfg.items():
        if p in my_fields:
            setattr(self, p, type(getattr(self, p))(v))
        elif "GP_" in p:
            p = p.replace("GP_", "")
            gp = self.global_place_stages[0]
            if p in gp:
                gp[p] = type(gp[p])(v)


DREAMPlaceConfig = make_dataclass(
    "DREAMPlaceConfig", DREAMPLACE_BASE_CONFIG, namespace={"update_cfg": update_cfg}
)


class DREAMPlaceWorker(Worker):
    def __init__(
        self,
        log_dir,
        *args,
        default_config,
        congestion_ratio,
        density_ratio,
        multiobj=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        self.congestion_ratio = congestion_ratio
        self.density_ratio = density_ratio
        self.multiobj = multiobj

        # update default with best parameters
        path_reuse = Path(default_config["reuse_params"])
        if path_reuse.suffix == ".json" and path_reuse.is_file():
            with path_reuse.open() as f:
                best_params = json.load(f)
        else:
            best_params = DREAMPLACE_BEST_CFG.get(default_config["reuse_params"], {})
        print("Reusing best parameters:", best_params)
        self.default_config = {**best_params, **default_config}

        # setup PPA reference
        path_ppa = Path(default_config["base_ppa"])
        if path_ppa.suffix == ".json" and path_ppa.is_file():
            with path_ppa.open() as f:
                self.base_ppa = json.load(f)
        else:
            self.base_ppa = DREAMPLACE_BASE_PPA[default_config["base_ppa"]]
        print("PPA reference:", self.base_ppa)
        self.bad_run = {
            k: float(v * DREAMPLACE_BAD_RATIO) for k, v in self.base_ppa.items()
        }

        self._setup_placer()

    def _create_params(self, config):
        params = DREAMPlaceConfig(**DREAMPLACE_BASE_CONFIG)
        params.update_cfg(self.default_config)
        params.update_cfg(config)
        return asdict(params)

    def _setup_placer(self):
        params = self._create_params({})
        self.placer = PlacementEngine(params)
        self.placer.setup_rawdb()

    def _update_logger(self, working_directory, suffix=""):
        # change logger
        log = logging.getLogger()
        filehandler = logging.FileHandler(
            opj(working_directory, f"DREAMPlace{suffix}.log"), "w"
        )
        formatter = logging.Formatter("[%(levelname)-7s] %(name)s - %(message)s")
        filehandler.setFormatter(formatter)
        log = logging.getLogger()
        for hdlr in log.handlers[:]:  # remove existing file handler
            if isinstance(hdlr, logging.FileHandler):
                log.removeHandler(hdlr)
        log.addHandler(filehandler)
        log.setLevel(logging.DEBUG)

    def compute(self, config_id, config, budget, working_directory, **kwargs):
        config_identifier = "run-" + "_".join([str(x) for x in config_id])

        working_directory = opj(self.log_dir, config_identifier)
        os.makedirs(working_directory, exist_ok=True)

        self._update_logger(working_directory)

        config["result_dir"] = working_directory
        params = self._create_params(config)

        self.placer.update_params(params)

        config_filename = opj(working_directory, "parameters.json")
        self.placer.params.dump(config_filename)

        result = self.placer.run()

        if float("inf") in result.values():
            ppa = self.bad_run
        else:
            ppa = result

        rsmt_norm = ppa["rsmt"] / self.base_ppa["rsmt"]
        congestion_norm = ppa["congestion"] / self.base_ppa["congestion"]
        density_norm = ppa["density"] / self.base_ppa["density"]
        cost = (
            rsmt_norm
            + self.congestion_ratio * congestion_norm
            + self.density_ratio * density_norm
        )
        result.update({"cost": float(cost)})
        logging.info(f"BOHB Cost: {cost}")

        if self.multiobj:
            return {
                "loss": (rsmt_norm, congestion_norm, density_norm),
                "info": result,
            }
        else:
            return {
                "loss": float(cost),
                "info": result,
            }

    @staticmethod
    def get_configspace(config_file: str, seed=None):
        # read JSON config if provided
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                cs = CS_JSON.read(f.read())
                cs.seed(seed)
            return cs

        # otherwise, setup default config space
        cs = CS.ConfigurationSpace(seed)

        init_x = CSH.UniformFloatHyperparameter(
            "init_loc_perc_x", lower=0.2, upper=0.8, default_value=0.5
        )
        init_y = CSH.UniformFloatHyperparameter(
            "init_loc_perc_y", lower=0.2, upper=0.8, default_value=0.5
        )
        td = CSH.UniformFloatHyperparameter(
            "target_density", lower=0.50, upper=0.70, default_value=0.50
        )
        dw = CSH.UniformFloatHyperparameter(
            "density_weight", lower=1e-6, upper=1e-0, default_value=8e-3, log=True
        )
        halox = CSH.UniformIntegerHyperparameter(
            "macro_halo_x",
            lower=4000,
            upper=8000,
            default_value=5000,
            log=True,
        )
        haloy = CSH.UniformIntegerHyperparameter(
            "macro_halo_y",
            lower=4000,
            upper=8000,
            default_value=5000,
            log=True,
        )
        ovflow = CSH.UniformFloatHyperparameter(
            "stop_overflow", lower=0.06, upper=0.10, default_value=0.07
        )
        gamma = CSH.UniformFloatHyperparameter(
            "gamma", lower=0.10, upper=0.50, default_value=0.1318231577
        )
        lr = CSH.UniformFloatHyperparameter(
            "GP_learning_rate", lower=1e-4, upper=1e-2, default_value=2.5e-4
        )
        lr_decay = CSH.UniformFloatHyperparameter(
            "GP_learning_rate_decay", lower=0.99, upper=1.0, default_value=1.0
        )
        optimizer = CSH.CategoricalHyperparameter(
            "GP_optimizer", ["adam", "nesterov"], default_value="nesterov"
        )
        nbinx = CSH.CategoricalHyperparameter(
            "GP_num_bins_x", [256, 512, 1024, 2048], default_value=512
        )
        nbiny = CSH.CategoricalHyperparameter(
            "GP_num_bins_y", [256, 512, 1024, 2048], default_value=512
        )
        wl = CSH.CategoricalHyperparameter(
            "GP_wirelength",
            ["weighted_average", "logsumexp"],
            default_value="weighted_average",
        )
        llambda = CSH.UniformIntegerHyperparameter(
            "GP_Llambda_density_weight_iteration",
            lower=1,
            upper=3,
            default_value=1,
        )
        lsub = CSH.UniformIntegerHyperparameter(
            "GP_Lsub_iteration",
            lower=1,
            upper=3,
            default_value=1,
        )
        replace_wl = CSH.UniformIntegerHyperparameter(
            "RePlAce_ref_hpwl", lower=150000, upper=550000, default_value=350000
        )
        replace_low = CSH.UniformFloatHyperparameter(
            "RePlAce_LOWER_PCOF", lower=0.90, upper=0.99, default_value=0.95
        )
        replace_up = CSH.UniformFloatHyperparameter(
            "RePlAce_UPPER_PCOF", lower=1.02, upper=1.15, default_value=1.05
        )
        pd = CSH.UniformFloatHyperparameter(
            "pin_density", lower=0.1, upper=0.5, default_value=0.5
        )

        hyperparameters = {
            "all": [
                init_x,
                init_y,
                td,
                dw,
                halox,
                haloy,
                ovflow,
                gamma,
                lr,
                lr_decay,
                optimizer,
                nbinx,
                nbiny,
                wl,
                # llambda,
                # lsub,
                replace_wl,
                replace_low,
                replace_up,
                # pd,
            ],
            "refine": [init_x, init_y, td, dw, halox, haloy],
            "refine_fixed_macros": [init_x, init_y, td, dw],
            "optimizer": [gamma, lr, lr_decay, optimizer, nbinx, nbiny, wl],
            "RePlace": [replace_wl, replace_low, replace_up],
        }

        modes = ["all"]
        cs.add_hyperparameters(list(set(itemgetter(*modes)(hyperparameters))))

        return cs
