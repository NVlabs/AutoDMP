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

import traceback
import csv

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import logging

from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.optimizers.config_generators.mobohb_utils import (
    TPESampler,
    GammaFunction,
    default_weights,
)


class MOBOHB(base_config_generator):
    def __init__(
        self,
        configspace,
        parameters,
        history_dir,
        run_id,
        init=True,
        min_points_in_model=None,
        top_n_percent=10,
        num_samples=24,
        random_fraction=1 / 6,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        **kwargs
    ):
        """
        Fits for each given budget a kernel density estimator on the best N percent of the
        evaluated configurations on this budget.


        Parameters:
        -----------
        configspace: ConfigSpace
            Configuration space object
        top_n_percent: int
            Determines the percentile of configurations that will be used as training data
            for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
            for training.
        min_points_in_model: int
            minimum number of datapoints needed to fit a model
        num_samples: int
            number of samples drawn to optimize EI via sampling
        random_fraction: float
            fraction of random configurations returned
        bandwidth_factor: float
            widens the bandwidth for contiuous parameters for proposed points to optimize EI
        min_bandwidth: float
            to keep diversity, even when all (good) samples have the same value for one of the parameters,
            a minimum bandwidth (Default: 1e-3) is used instead of zero.

        """
        super().__init__(**kwargs)
        self.top_n_percent = top_n_percent
        self.configspace = configspace
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.parameters = parameters
        self.history_dir = history_dir
        self.run_id = run_id
        self.random_state = np.random.RandomState(int(self.run_id))
        self.mobohb_with_init = init

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = (
                2 * len(self.configspace.get_hyperparameters()) + 1
            )

        if self.min_points_in_model < len(self.configspace.get_hyperparameters()) + 1:
            self.logger.warning(
                "Invalid min_points_in_model value. Setting it to %i"
                % (len(self.configspace.get_hyperparameters()) + 1)
            )
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        hps = self.configspace.get_hyperparameters()

        self.configs = dict()
        self.losses = dict()
        self.good_config_rankings = dict()
        self.history = list()
        self.split_cache = dict()

        self.init = True

    def largest_budget_with_model(self):
        if len(self.history) == 0:
            return -float("inf")
        budgets = [h["budget"] for h in self.history]
        return max(budgets)

    def get_config(self, budget):
        """
        Function to sample a new configuration

        This function is called inside Hyperband to query a new configuration


        Parameters:
        -----------
        budget: float
            the budget for which this configuration is scheduled

        returns: config
            should return a valid configuration

        """

        self.logger.debug("start sampling a new configuration.")

        sample = None
        info_dict = {}

        budget = int(np.ceil(budget))

        if len(self.history) > 0:
            budgets = [h["budget"] for h in self.history]
            budget_type, budget_counts = np.unique(budgets, return_counts=True)
        else:
            budget_type = []
            budget_counts = 0

        if budget != 25:
            self.init = False

        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if self.mobohb_with_init:
            if np.random.rand() < self.random_fraction or self.init:
                sample = self.configspace.sample_configuration()
                info_dict["model_based_pick"] = False
        else:
            if (
                np.random.rand() < self.random_fraction
                or budget not in self.configs
                or not np.any(budget_counts > self.min_points_in_model)
            ):
                sample = self.configspace.sample_configuration()
                info_dict["model_based_pick"] = False

        if sample is None:
            try:

                # sample from largest budget
                idx_map = budget_counts > self.min_points_in_model
                largest_budget = max(budget_type[idx_map])
                history_largest_budget = [
                    h for h in self.history if h["budget"] == largest_budget
                ]

                self.split_cache = {}
                sample = {}
                for hp in self.configspace.get_hyperparameters():
                    sampler = TPESampler(
                        hp,
                        history_largest_budget,
                        self.random_state,
                        n_ei_candidates=self.parameters["num_candidates"],
                        gamma_func=GammaFunction(self.parameters["gamma"]),
                        weights_func=default_weights,
                        split_cache=self.split_cache,
                    )
                    sample[hp.name] = sampler.sample()
                    self.split_cache = sampler.split_cache
                    info_dict["model_based_pick"] = True
            except:
                self.logger.warning(
                    "Sampling based optimization with %i samples failed\n %s \nUsing random configuration"
                    % (self.num_samples, traceback.format_exc())
                )
                sample = self.configspace.sample_configuration()
                info_dict["model_based_pick"] = False

        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace, configuration=sample
            ).get_dictionary()
        except Exception as e:
            self.logger.warning(
                "Error (%s) converting configuration: %s -> "
                "using random configuration!",
                e,
                sample,
            )
            sample = self.configspace.sample_configuration().get_dictionary()
            info_dict["model_based_pick"] = False
        self.logger.debug("done sampling a new configuration.")
        return sample, info_dict

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array

    def new_result(self, job, update_model=True):
        """
        function to register finished runs

        Every time a run has finished, this function should be called
        to register it with the result logger. If overwritten, make
        sure to call this method from the base class to ensure proper
        logging.


        Parameters:
        -----------
        job: hpbandster.distributed.dispatcher.Job object
            contains all the info about the run
        """

        super().new_result(job)

        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result["loss"]

        budget = job.kwargs["budget"]
        budget = int(np.ceil(budget))
        if budget > 50:
            budget = 50

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)

        # record = {'Trial': len(self.history), 'Config': conf, 'Error': job.result['info']['acc_err'], 'norm_params': job.result['info']['norm_params'],
        #          'Params': job.result['info']['params'],
        #          'f': loss,
        #          'budget': budget}

        # print(job.result)
        record = {
            "Trial": len(self.history),
            "Config": conf,
            "Error": job.result["loss"][0],
            "norm_params": job.result["loss"][1],
            "f": loss,
            "budget": budget,
        }

        self.history.append(record)
        self.write()

    def write(self):
        self.currently_writing = True
        keys = self.history[0].keys()
        self.currently_writing = False
