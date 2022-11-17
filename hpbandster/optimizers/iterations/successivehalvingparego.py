import numpy as np
from hpbandster.core.base_iteration import BaseIteration

class SuccessiveHalvingParEGO(BaseIteration):

    def parEG0_scalarization(self, cost):
        w = np.random.random_sample(2)
        w /= np.sum(w)

        w_f = w * cost
        max_k = np.max(w_f)
        rho_sum_wf = self.rho * np.sum(w_f)
        return max_k + rho_sum_wf

    def _advance_to_next_stage(self, config_ids, losses):
        """
			SuccessiveHalving MOBOHB simply continues the best based on the current multi-objective loss.
		"""
        losses = self.parEG0_scalarization(losses)
        ranks = np.argsort(np.argsort(losses))
        return (ranks < self.num_configs[self.stage])