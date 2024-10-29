import numpy as np
import os
import torch

from fibad.data_sets.data_set_registry import fibad_data_set
@fibad_data_set
class KbmodStamps():
    """TODO: what is the actual shape of the data that we're going to want to import?
    my initial thoughts is that we'll have a single numpy array that we stitch together
    from the two datasets (adding a column with a classification based on which set
    they come from). We should also have the option to select which stamp type we are using
    (mean, median, sum, and var weighted). So we can just have all those stored as individual rows.
    
    We could have an "active columns" variable, with the indices of the columns we want to grab
    (corresponding to which coadd type we want to use), which could reflect in the `shape` function.
    """
    def __init__(self, config):
        coadd_type_to_column = {
            "median": 0,
            "mean": 1,
            "sum": 2,
            "var_weighted": 3,
        }

        cols = []
        # for c in config["data_loader"]["coadd_types"]:
        for c in ["mean"]:
            cols.append(coadd_type_to_column[c])

        self.active_columns = np.array(cols)

        true_data_path = os.path.join(config["general"]["data_dir"], "true_train_stamps.npy")
        false_data_path = os.path.join(config["general"]["data_dir"], "false_train_stamps.npy")

        if not os.path.isfile(true_data_path):
            raise ValueError("could not find 'true_train_stamps.py' in provided 'data_dir'")
        if not os.path.isfile(false_data_path):
            raise ValueError("could not find 'false_train_stamps.py' in provided 'data_dir'")

        true_train = np.load(true_data_path)
        false_train = np.load(false_data_path)

        self._labels = np.concatenate([np.ones(len(true_train), dtype=int), np.zeros(len(false_train), dtype=int)])
        self._data = np.concatenate([true_train[:,:3,:,:], false_train])

    def shape(self):
        """data shape, including currently enabled columns"""
        cols = len(self.active_columns)
        width, height = self._data[0][0].shape

        return (cols, width, height)

    def __getitem__(self, idx):
        row = self._data[idx][self.active_columns]
        label = self._labels[idx]

        return torch.Tensor(row), torch.Tensor(label)
    
    def __len__(self):
        return len(self._data)