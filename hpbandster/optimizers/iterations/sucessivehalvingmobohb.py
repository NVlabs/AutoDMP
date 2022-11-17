from hpbandster.core.base_iteration import BaseIteration
import numpy as np
import sys
from pygmo import hypervolume

eps = sys.float_info.epsilon

def nondominated_sort(points):
    points = points.copy()
    ranks = np.zeros(len(points))
    r = 0
    c = len(points)
    while c > 0:
        extended = np.tile(points, (points.shape[0], 1, 1))
        dominance = np.sum(np.logical_and(
            np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
            np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)
        points[dominance == 0] = 1e9  # mark as used
        ranks[dominance == 0] = r
        r += 1
        c -= np.sum(dominance == 0)
    return ranks

class SuccessiveHalvingMOBOHB(BaseIteration):

    def _advance_to_next_stage(self, config_ids, losses):
        """
			SuccessiveHalving MOBOHB simply continues the best based on the current multi-objective loss.
		"""
        rank = nondominated_sort(losses)
        indices = np.array(range(len(losses)))
        keep_indices = np.array([], dtype=int)

        # nondominance rank-based selection
        i = 0
        while len(keep_indices) + sum(rank == i) <= self.num_configs[self.stage]:
            keep_indices = np.append(keep_indices, indices[rank == i])
            i += 1
        keep_indices = np.append(keep_indices, indices[rank == i])

        # hypervolume contribution-based selection
        #ys_r = losses[rank == i]
        #indices_r = indices[rank == i]
        #worst_point = np.max(losses, axis=0)
        #reference_point = np.maximum(
        #    np.maximum(
        #        1.1 * worst_point,  # case: value > 0
        #        0.9 * worst_point  # case: value < 0
        #    ),
        #    np.full(len(worst_point), eps)  # case: value = 0
        #)

        #S = []
        #contributions = []
        #for j in range(len(ys_r)):
        #    contributions.append(hypervolume([ys_r[j]]).compute(reference_point))
        #while len(keep_indices) + 1 <= self.num_configs[self.stage]:
        #    hv_S = 0
        #    if len(S) > 0:
        #        hv_S = hypervolume(S).compute(reference_point)
        #    index = np.argmax(contributions)
        #    contributions[index] = -1e9  # mark as already selected
        #    for j in range(len(contributions)):
        #        if j == index:
        #            continue
        #        p_q = np.max([ys_r[index], ys_r[j]], axis=0)
        #        contributions[j] = contributions[j] - (hypervolume(S + [p_q]).compute(reference_point) - hv_S)
        #    S = S + [ys_r[index]]
        #    keep_indices = np.append(keep_indices, indices_r[index])

        return_stat = np.zeros((len(losses))).astype(bool)
        return_stat[keep_indices] = True
        return return_stat

        # ranks = np.argsort(np.argsort(losses))
        # return (ranks < self.num_configs[self.stage])
