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
        for c in ["median", "sum"]:
            cols.append(coadd_type_to_column[c])

        self.active_columns = np.array(cols)

        data_path = os.path.join(config["general"]["data_dir"], "stamps.npy")
        labels_path = os.path.join(config["general"]["data_dir"], "labels.npy")

        if not os.path.isfile(data_path):
            raise ValueError("could not find 'stamps.py' in provided 'data_dir'")
        self._data = np.load(data_path)

        if not os.path.isfile(data_path):
            raise ValueError("could not find 'labels.npy' in provided 'data_dir'")
        self._labels = np.load(labels_path)

    def shape(self):
        """data shape, including currently enabled columns"""
        cols = len(self.active_columns)
        width, height = self._data[0][0].shape

        return (cols, width, height)

    def __getitem__(self, idx):
        row = self._data[idx][self.active_columns]
        label = self._labels[idx]

        return torch.Tensor(row), label
    
    def __len__(self):
        return len(self._data)