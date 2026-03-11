import os

import numpy as np
from hyrax.data_sets.data_set_registry import HyraxDataset
from torch.utils.data import Dataset


class KbmodStamps(HyraxDataset, Dataset):
    """TODO: what is the actual shape of the data that we're going to want to import?
    my initial thoughts is that we'll have a single numpy array that we stitch together
    from the two datasets (adding a column with a classification based on which set
    they come from). We should also have the option to select which stamp type we are using
    (mean, median, sum, and var weighted). So we can just have all those stored as individual rows.

    We could have an "active columns" variable, with the indices of the columns we want to grab
    (corresponding to which coadd type we want to use), which could reflect in the `shape` function.
    """

    def __init__(self, config, data_location=None):
        """Initialize the KbmodStamps dataset.

        Takes the `kbmod_ml` elements of the config dict to find the
        true positive and false positive .npy files, which are expected
        to be numpy arrays of shape (N_samples, 4, width, height), where
        the 4 corresponds to the 4 coadd types (median, mean, sum, var_weighted).
        Also does stamp normalization and label generation.
        """
        super().__init__(config)
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
        self._data = np.concatenate([true_positive_samples, false_positive_samples])

        self.normalize_stamps()

        metadata_table = self._read_metadata()
        super().__init__(config, metadata_table)

    def ids(self):
        """Return the ids of the data set"""
        return np.arange(len(self._data))

    def shape(self):
        """data shape, including currently enabled columns"""
        cols = len(self.active_columns)
        width, height = self._data[0][0].shape

        return (cols, width, height)

    def get_classification(self, idx):
        return self._labels[idx]

    def get_stamps(self, idx):
        row = self._data[idx][self.active_columns]
        return row

    def _read_metadata(self):
        """This is a pretend implementation so we don't use the path passed, which you might use
        to find your .csv/.fits/.tsv catalog file and call astropy's Table.read().

        We simply construct a table from our mock data"""
        from astropy.table import Table

        global ras, decs, filenames
        return Table(
            {
                "object_id": self.ids(),
                "classification": self._labels,
            }
        )

    def __len__(self):
        return len(self._data)

    def normalize_stamps(self):
        """Normalize each stamp."""
        normed_stamps = []
        sigmaG_coeff = 0.7413
        for stamp in self._data:
            row = []
            for col in self.active_columns:
                stamp = np.copy(stamp[:, col : col + 1, :, :])
                mean_pixel = np.nanmean(stamp)
                stamp[~np.isfinite(stamp)] = mean_pixel if np.isfinite(mean_pixel) else 0.0
                per25, per50, per75 = np.percentile(stamp, [25, 50, 75])
                sigmaG = sigmaG_coeff * (per75 - per25)
                stamp[stamp < (per50 - 2 * sigmaG)] = per50 - 2 * sigmaG
                stamp -= np.min(stamp)
                stamp /= np.sum(stamp)
                norm_mean_pixel = np.nanmean(stamp)
                stamp[~np.isfinite(stamp)] = norm_mean_pixel if np.isfinite(norm_mean_pixel) else 0.0
                row.append(stamp)
            normed_stamps.append(row)
        normed_stamps = np.array(normed_stamps)
        self._data = normed_stamps
