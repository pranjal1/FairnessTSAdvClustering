import csv
import math
import copy
from typing import List
from collections import defaultdict
from itertools import permutations

import numpy as np
from numpy import linalg
from sklearn import datasets
from sklearn_extra.cluster import KMedoids

from .cluster import ClusteringKMedoid


def get_all_data_distn(S):
    _all_data_distn = defaultdict(int)
    for x in S:
        _all_data_distn[x] += 1

    all_data_distn = {
        k: v / sum(_all_data_distn.values()) for k, v in _all_data_distn.items()
    }
    return all_data_distn


def calculate_fairness_utility(centers, S, cluster_assignment):
    all_data_distn = get_all_data_distn(S)
    _cluster_wise_distn = defaultdict(lambda: defaultdict(int))
    for c, sm in zip(cluster_assignment, S):
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
                all_data_distn[sens_group] / ratio,
                ratio / all_data_distn[sens_group],
            ]
        cluster_wise_rho[cluster_num] = _tmp_dct

    all_rhos = []
    for ko, vo in cluster_wise_rho.items():
        for ki, vi in vo.items():
            all_rhos.extend(vi)

    return min(all_rhos)


class FairnessAttack:
    def __init__(
        self,
        K: int,
        X: np.ndarray,
        S: np.ndarray,
        samples_per_iteration: int,
        min_dist_init: float = 0.5,
    ) -> None:
        self.K = K
        self.X = X
        self.S = S
        self.min_dist_init = min_dist_init

        self.data_dim = len(self.X.shape)
        self.sample_size = len(self.X)
        self.samples_per_iteration = samples_per_iteration

    def run_kmedoids(self, n_clusters, X):
        num_samples = X.shape[0]
        KM = KMedoids(n_clusters=n_clusters).fit(X)
        centers = KM.cluster_centers_
        sorted_centers = np.array(sorted(centers, key=lambda x: np.sum(x)))

        # assign data pts to the sorted centers
        X = np.repeat(X, n_clusters, axis=0)
        if self.data_dim == 3:
            C = np.tile(centers, (num_samples, 1, 1))
        elif self.data_dim == 2:
            C = np.tile(centers, (num_samples, 1))
        assert X.shape == C.shape
        if self.data_dim == 3:
            euc = np.sum((X - C) ** 2, axis=(1, 2)) ** 0.5
        elif self.data_dim == 2:
            euc = np.sum((X - C) ** 2, axis=1) ** 0.5
        euc_reshaped = euc.reshape(-1, self.K)
        cluster_assignment = np.argmin(euc_reshaped, axis=-1)
        return (cluster_assignment, sorted_centers)

    def get_initial_clusters(self):
        print("Performing clustering pre-attack...")
        KM = KMedoids(self.K).fit(self.X)
        self.initial_labels, self.initial_centers = KM.labels_, KM.cluster_centers_
        self.pre_attack_utility = calculate_fairness_utility(
            self.initial_centers, self.S, self.initial_labels
        )

    def get_adversarial_cluster_centers(self):
        print("Obtaining adversarial cluster centers...")
        adv_clustering = ClusteringKMedoid(
            "adversarial", self.K, self.X, self.S, self.min_dist_init
        )
        (
            self.adv_centers,
            self.adv_centers_idx,
            self.adv_members,
            self.adv_utility,
        ) = adv_clustering.get_centers()

    def get_final_clusters(self):
        print("Performing clustering post-attack...")
        KM = KMedoids(self.K).fit(self.X_augmented)
        self.final_labels, self.final_centers = KM.labels_, KM.cluster_centers_
        self.post_attack_utility = calculate_fairness_utility(
            self.final_centers, self.S, self.final_labels
        )

    def perform_attack(self):
        self.adv_centers = np.array(sorted(self.adv_centers, key=lambda x: np.sum(x)))
        # self.S = np.reshape(self.S, (len(self.S), 1))

        self.X_augmented = copy.deepcopy(self.X)

        _, self.new_centers = self.run_kmedoids(self.K, self.X_augmented)

        V_k = [0] * self.K
        ep_k = [self.samples_per_iteration] * self.K
        mask_idx = []

        print("new centers", self.new_centers)
        print("adv centers", self.adv_centers)

        perms = []
        for p in permutations(range(self.K)):
            perms.append(list(p))

        cent_adv_perms = [self.adv_centers[[i]] for i in perms]

        while np.linalg.norm(self.new_centers - self.adv_centers) != 0:
            if np.linalg.norm(self.new_centers - self.adv_centers[::-1]) == 0:
                break
            print(len(self.X_augmented))
            ######## This code segment checks to see if we have arrived at similar centers since they might be shuffled (clustering is never consistent in labeling) #########################
            if self.K >= 2:
                done = False
                c_perms = [self.new_centers[i] for i in perms]
                for c in c_perms:
                    for co in cent_adv_perms:
                        if np.linalg.norm(c - co) == 0:
                            done = True
                            break
                if done:
                    break
            #################################
            # print(len(self.X_augmented))
            # print(np.unique(self.X_augmented[400:], return_counts=True))
            for k_i in range(self.K):
                # print("os", k_i, self.new_centers[k_i], self.adv_centers[k_i])
                if np.linalg.norm(self.new_centers[k_i] - self.adv_centers[k_i]) != 0:
                    # print("is", k_i, self.new_centers[k_i], self.adv_centers[k_i])
                    V = datasets.make_blobs(
                        n_samples=ep_k[k_i],
                        cluster_std=[0.0],
                        random_state=170,
                        centers=[self.adv_centers[k_i]],
                    )[
                        0
                    ]  # RS = 42000
                    self.X_augmented = np.vstack((self.X_augmented, V))
                    for i in range(self.sample_size, self.sample_size + ep_k[k_i]):
                        mask_idx.append(i)
                    self.sample_size += ep_k[k_i]
                    V_k[k_i] += ep_k[k_i]
                _, self.new_centers = self.run_kmedoids(self.K, self.X_augmented)
                # print(self.new_centers)
                if np.linalg.norm(self.new_centers - self.adv_centers) == 0:
                    break

    def main(self):
        self.get_initial_clusters()
        self.get_adversarial_cluster_centers()
        self.perform_attack()
        self.get_final_clusters()

        print("Pre-attack cluster centers: ")
        print(self.initial_centers)
        print("Pre-attack utility: ")
        print(self.pre_attack_utility)
        # print("Adversarial cluster centers: ")
        # print(self.adv_centers)
        print("Adversarial centers utility: ")
        print(self.adv_utility)
        # print("Post attack cluster centers: ")
        # print(self.final_centers)
        print("Post attack utility")
        print(self.post_attack_utility)
        print("Added samples: ")
        print(len(self.X_augmented) - len(self.X))
        print("Ratio: (|X_aug| - |X|)/|X|")
        print((len(self.X_augmented) - len(self.X)) / len(self.X))
