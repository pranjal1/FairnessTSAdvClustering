import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class ADLDataLoader:
    def __init__(
        self,
        folder_path,
        window_length=3,
        dgp_vae_ratio=0.4,
        dataset_seed=0,
        filter_persons=False,
    ):
        self.folder_path = folder_path
        self.dataset_seed = dataset_seed
        self.window_length = window_length
        self.filter_persons = filter_persons
        self.sampling_freq = 32

        self.dgp_vae_ratio = dgp_vae_ratio

        self.slicer_val = self.window_length * self.sampling_freq

        try:
            display()
            self.printer = display
        except:
            self.printer = print
        np.random.seed(self.dataset_seed)
        random.seed(self.dataset_seed)

    def load_data(self, print_distn=False):
        self.data_samples = []
        self.tgt_labels = []
        self.genders = []
        self.person_identifiers = []
        naming_pattern = (
            r".+\/(?P<activity>.+)\/.+\-(?P<gender>.)(?P<identifier>\d+(_\d+)?).txt"
        )
        for fp in tqdm(Path(self.folder_path).glob("*/*.txt")):
            pattern_test = re.match(naming_pattern, fp.as_posix())
            if not pattern_test:
                raise Exception(
                    "Folder specified does not have data in specified format"
                )
            data = np.loadtxt(fp.as_posix())
            gender = pattern_test.groupdict().get("gender")
            activity = pattern_test.groupdict().get("activity")
            if activity.endswith("_MODEL"):
                activity = activity[:-6]
            identifier = pattern_test.groupdict().get("identifier")
            identifier = identifier.split("_")[0]

            num_iters = data.shape[0] // self.slicer_val
            # last slice is discarded
            # non overlapping slices
            for slice_idx in range(num_iters):
                sliced_data = data[
                    (slice_idx) * self.slicer_val : (slice_idx + 1) * self.slicer_val
                ]
                self.data_samples.append(sliced_data)
                self.tgt_labels.append(activity)
                self.genders.append(gender)
                self.person_identifiers.append(identifier)

        # filter persons with less than some count, for now its 100
        if self.filter_persons:
            person_identity, person_count = np.unique(
                self.person_identifiers, return_counts=True
            )
            print("Original Distribution: ", person_identity, person_count)
            valid_person_identity = person_identity[person_count > 100]
            valid_ids = [
                i
                for i in range(len(self.person_identifiers))
                if self.person_identifiers[i] in valid_person_identity
            ]
            print("Original size = ", len(self.person_identifiers))
            print("New size = ", len(valid_ids))

            (
                self.data_samples,
                self.tgt_labels,
                self.genders,
                self.person_identifiers,
            ) = list(
                map(
                    lambda x: [d for i, d in enumerate(x) if i in valid_ids],
                    [
                        self.data_samples,
                        self.tgt_labels,
                        self.genders,
                        self.person_identifiers,
                    ],
                )
            )
            person_identity, person_count = np.unique(
                self.person_identifiers, return_counts=True
            )
            print("New Distribution: ", person_identity, person_count)

        assert (
            len(self.data_samples)
            == len(self.tgt_labels)
            == len(self.genders)
            == len(self.person_identifiers)
        )

        self.data_samples_max = np.max(np.concatenate(self.data_samples), axis=0)
        self.data_samples_min = np.min(np.concatenate(self.data_samples), axis=0)
        self.data_samples = list(
            map(
                lambda x: np.float32(
                    (x - self.data_samples_min)
                    / (self.data_samples_max - self.data_samples_min)
                ),
                self.data_samples,
            )
        )

        self.unique_tgt_labels = np.unique(self.tgt_labels)
        self.unique_genders = np.unique(self.genders)
        self.unique_person_identifiers = np.unique(self.person_identifiers)

        self.le_tgt = preprocessing.LabelEncoder()
        self.le_tgt.fit(self.tgt_labels)
        self.tgt_labels = np.float32(self.le_tgt.transform(self.tgt_labels))

        self.le_gender = preprocessing.LabelEncoder()
        self.le_gender.fit(self.genders)
        self.genders = np.float32(self.le_gender.transform(self.genders))

        self.le_person = preprocessing.LabelEncoder()
        self.le_person.fit(self.person_identifiers)
        self.person_identifiers = np.float32(
            self.le_person.transform(self.person_identifiers)
        )

        self.data_samples = np.array(self.data_samples)
        self.tgt_labels = np.array(self.tgt_labels)
        self.genders = np.array(self.genders)
        self.person_identifiers = np.array(self.person_identifiers)

        if not print_distn:
            return

        df_distributions = pd.DataFrame(
            {
                "tgt_labels": self.tgt_labels,
                "genders": self.genders,
                "person_identifiers": self.person_identifiers,
            }
        )
        df_distributions["counts"] = 1

        print("Details:")
        print("-" * 20)
        print("Window Length = ", self.window_length)
        print("Sampling Frequency = ", self.sampling_freq)
        print("Each sample contains ", self.slicer_val, " points")
        print("Target Classes distributions ")
        self.printer(
            df_distributions[["tgt_labels", "counts"]].groupby("tgt_labels").count()
        )
        print("Genders distribuitions ")
        self.printer(
            df_distributions[["genders", "tgt_labels", "counts"]]
            .groupby(["genders", "tgt_labels"])
            .count()
        )
        print("Person Identifiers ")
        self.printer(
            df_distributions[["person_identifiers", "tgt_labels", "counts"]]
            .groupby(["person_identifiers", "tgt_labels"])
            .count()
        )
