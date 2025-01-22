import os

import numpy as np
import torch
from fibad.data_sets.data_set_registry import fibad_data_set
from torch.utils.data.sampler import SubsetRandomSampler


@fibad_data_set
class KbmodStamps:
    """TODO: what is the actual shape of the data that we're going to want to import?
    my initial thoughts is that we'll have a single numpy array that we stitch together
    from the two datasets (adding a column with a classification based on which set
    they come from). We should also have the option to select which stamp type we are using
    (mean, median, sum, and var weighted). So we can just have all those stored as individual rows.

    We could have an "active columns" variable, with the indices of the columns we want to grab
    (corresponding to which coadd type we want to use), which could reflect in the `shape` function.
    """

    def __init__(self, config, split: str):
        coadd_type_to_column = {
            "median": 0,
            "mean": 1,
            "sum": 2,
            "var_weighted": 3,
        }

        cols = []

        for c in ["mean"]:
            cols.append(coadd_type_to_column[c])

        self.active_columns = np.array(cols)

        data_dir = config["general"]["data_dir"]
        true_positive_file_name = config["kbmod_ml"]["true_positive_file_name"]
        false_positive_file_name = config["kbmod_ml"]["false_positive_file_name"]

        true_data_path = os.path.join(data_dir, true_positive_file_name)
        false_data_path = os.path.join(data_dir, false_positive_file_name)

        if not os.path.isfile(true_data_path):
            raise ValueError(f"Could not find {true_positive_file_name} in provided {data_dir}")
        if not os.path.isfile(false_data_path):
            raise ValueError(f"could not find {false_positive_file_name} in provided {data_dir}")

        true_positive_samples = np.load(true_data_path)
        false_positive_samples = np.load(false_data_path)

        self._labels = np.concatenate(
            [
                np.ones(len(true_positive_samples), dtype=np.int8),
                np.zeros(len(false_positive_samples), dtype=np.int8),
            ]
        )
        self._data = np.concatenate([true_positive_samples[:, :3, :, :], false_positive_samples])

        if split != "test":
            num_train = len(self)
            indices = list(range(num_train))
            split_idx = 0
            if config["data_set"]["validate_size"]:
                split_idx = int(np.floor(config["data_set"]["validate_size"] * num_train))

            random_seed = None
            if config["data_set"]["seed"]:
                random_seed = config["data_set"]["seed"]
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[split_idx:], indices[:split_idx]

            # These samplers are used by PyTorch's DataLoader to split the dataset
            # into training and validation sets.
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.validation_sampler = SubsetRandomSampler(valid_idx)

    def shape(self):
        """data shape, including currently enabled columns"""
        cols = len(self.active_columns)
        width, height = self._data[0][0].shape

        return (cols, width, height)

    def __getitem__(self, idx):
        row = self._data[idx][self.active_columns]
        label = self._labels[idx]

        return torch.tensor(row), torch.tensor(label, dtype=torch.int8)

    def __len__(self):
        return len(self._data)
