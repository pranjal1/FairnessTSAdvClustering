from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

from tslearn.neighbors import KNeighborsTimeSeries


class ClusteringKMedoid:
    def __init__(
        self, mode, K: int, X: np.ndarray, S: np.ndarray, min_dist_init: float = 0.0
    ) -> None:
        self.mode = mode
        assert self.mode in ["adversarial", "normal"]
        self.K = K
        self.X = X
        self.S = S

        self.min_dist_init = min_dist_init

        self.data_dim = len(self.X.shape)

        self.num_samples = self.X.shape[0]
        self.get_all_data_distn()
        if self.mode == "adversarial":
            self.cost_fn = self.calculate_fairness_utility
        else:
            self.cost_fn = self.calculate_cost

    def get_all_data_distn(self):
        _all_data_distn = defaultdict(int)
        for x in self.S:
            _all_data_distn[x] += 1

        self.all_data_distn = {
            k: v / sum(_all_data_distn.values()) for k, v in _all_data_distn.items()
        }
        print(self.all_data_distn)

    def get_cluster_assignment(self, centers):
        assert len(centers) == self.K
        centers = self.X[centers]
        X = np.repeat(self.X, self.K, axis=0)
        if self.data_dim == 3:
            C = np.tile(centers, (self.num_samples, 1, 1))
        elif self.data_dim == 2:
            C = np.tile(centers, (self.num_samples, 1))
        assert X.shape == C.shape
        if self.data_dim == 3:
            euc = np.sum((X - C) ** 2, axis=(1, 2)) ** 0.5
        elif self.data_dim == 2:
            euc = np.sum((X - C) ** 2, axis=1) ** 0.5
        euc_reshaped = euc.reshape(-1, self.K)
        cluster_assignment = np.argmin(euc_reshaped, axis=-1)
        cost = np.sum(np.min(euc_reshaped, axis=-1))
        return cluster_assignment, cost

    def calculate_cost(self, centers):
        cluster_assignment, cost = self.get_cluster_assignment(centers)
        return cluster_assignment, cost

    def calculate_fairness_utility(self, centers):
        cluster_assignment, _ = self.get_cluster_assignment(centers)
        _cluster_wise_distn = defaultdict(lambda: defaultdict(int))
        for c, sm in zip(cluster_assignment, self.S):
            _cluster_wise_distn[c][sm] += 1

        cluster_wise_distn = {}
        for ko, vo in _cluster_wise_distn.items():
            _tmp_dct = {}
            for ki, vi in vo.items():
                _tmp_dct[ki] = vi / sum(vo.values())
            cluster_wise_distn[ko] = _tmp_dct

        cluster_wise_rho = {}
        for cluster_num, distn in cluster_wise_distn.items():
            _tmp_dct = {}
            for sens_group, ratio in distn.items():
                _tmp_dct[sens_group] = [
                    self.all_data_distn[sens_group] / ratio,
                    ratio / self.all_data_distn[sens_group],
                ]
            cluster_wise_rho[cluster_num] = _tmp_dct

        all_rhos = []
        for ko, vo in cluster_wise_rho.items():
            for ki, vi in vo.items():
                all_rhos.extend(vi)

        return cluster_assignment, min(all_rhos)

    def get_init_centers(self):
        """return random points as initial centers"""
        index = np.random.choice(self.num_samples, self.K, replace=False)
        # bad_init = True

        # while bad_init:
        #     bad_init = False
        #     index = np.random.choice(self.num_samples, self.K, replace=False)
        #     init_centers = self.X[index]
        #     for i_idx, i in enumerate(init_centers):
        #         for j_idx, j in enumerate(init_centers):
        #             if i_idx != j_idx:
        #                 dist = np.linalg.norm(i - j)
        #                 if dist < self.min_dist_init:
        #                     print("bad init", i, j)
        #                     bad_init = True
        return index

    def get_centers(self, max_iter=1000):
        init_ids = self.get_init_centers()
        centers = init_ids
        # print("Initial centers are ", init_ids)
        members, tot_utility = self.cost_fn(init_ids)
        # print("Initial utility is:", tot_utility)

        cc, SWAPED = 0, True
        while True:
            # print("Iter", cc)
            SWAPED = False
            for i in range(self.num_samples):
                if i not in centers:
                    for j in range(len(centers)):
                        centers_ = deepcopy(centers)
                        centers_[j] = i
                        members_, tot_utility_ = self.cost_fn(centers_)
                        # print(centers_, tot_utility_)
                        if tot_utility_ < tot_utility:
                            members, tot_utility = members_, tot_utility_
                            centers = centers_
                            SWAPED = True
                            # print("Change centers to ", centers)
            if cc > max_iter:
                # print("End Searching by reaching maximum iteration", max_iter)
                break
            if not SWAPED:
                # print("End Searching by no swaps")
                break
            cc += 1

        # print("Final utility is ", tot_utility)
        return self.X[centers], centers, members, tot_utility
