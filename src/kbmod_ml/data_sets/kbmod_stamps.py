import logging
import os
import numpy as np
from hyrax.datasets.dataset_registry import HyraxDataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class KbmodStamps(HyraxDataset, Dataset):

    def __init__(self, config, data_location=None):
        super().__init__(config)

        # data_dir = config["general"]["data_dir"]
        true_positive_file_name = config["kbmod_ml"]["true_positive_file_name"]
        false_positive_file_name = config["kbmod_ml"]["false_positive_file_name"]

        true_data_path = os.path.join(data_location, true_positive_file_name)
        false_data_path = os.path.join(data_location, false_positive_file_name)

        if not os.path.isfile(true_data_path):
            raise ValueError(f"Could not find {true_positive_file_name} in provided {data_location}")
        if not os.path.isfile(false_data_path):
            raise ValueError(f"could not find {false_positive_file_name} in provided {data_location}")

        true_positive_samples = np.load(true_data_path)
        logger.warning(f"Loaded {len(true_positive_samples)} true positive samples from {true_data_path}")
        false_positive_samples = np.load(false_data_path)
        logger.warning(f"Loaded {len(false_positive_samples)} false positive samples from {false_data_path}")

        n_tp = len(true_positive_samples)
        n_fp = len(false_positive_samples)

        raw_data = np.concatenate([true_positive_samples, false_positive_samples[:, 0:3]])
        raw_labels = np.concatenate([
            np.ones(n_tp, dtype=np.int64),
            np.zeros(n_fp, dtype=np.int64),
        ])

        self._data = raw_data.astype(np.float32)
        # self._data = self._normalize(raw_data)
        self._labels = raw_labels

        seed = config["data_set"]["seed"] if config["data_set"]["seed"] else 42
        # self._arrange_for_hyrax_splits(n_tp, n_fp, seed)

        self.augment = config["kbmod_ml"]["augment"]

        metadata_table = self._read_metadata()
        super().__init__(config, metadata_table)

    def _normalize(self, stamps):
        out = stamps.astype(np.float32)
        flat = out.reshape(len(out), -1)
        mu = flat.mean(axis=1, keepdims=True)
        sig = flat.std(axis=1, keepdims=True)
        sig[sig == 0] = 1.0
        return ((flat - mu) / sig).reshape(out.shape)

    def _arrange_for_hyrax_splits(self, n_tp, n_fp, seed):
        """Rearrange data so Hyrax's legacy create_splits produces Hurum's exact splits.

        Hurum splits TP and FP independently using default_rng(seed), then
        concatenates per split. Hyrax shuffles the combined array using
        np.random.seed(seed) and takes contiguous blocks. This method places
        Hurum's split data at the positions Hyrax will assign to each split.
        """
        total = n_tp + n_fp
        test_frac = 0.15
        val_frac = 0.15

        # Hurum's splits: TP and FP split independently
        rng = np.random.default_rng(seed)

        def hurum_split(n):
            idx = rng.permutation(n)
            n_test = int(n * test_frac)
            n_val = int(n * val_frac)
            return {
                "test": idx[:n_test],
                "val": idx[n_test:n_test + n_val],
                "train": idx[n_val + n_test:],
            }

        tp_splits = hurum_split(n_tp)
        fp_splits = hurum_split(n_fp)

        # Build Hurum's ordered data per split: concat(tp[split], fp[split])
        orig_data = self._data.copy()
        orig_labels = self._labels.copy()

        tp_data = orig_data[:n_tp]
        fp_data = orig_data[n_tp:]
        tp_labels = orig_labels[:n_tp]
        fp_labels = orig_labels[n_tp:]

        hurum_test_data = np.concatenate([tp_data[tp_splits["test"]], fp_data[fp_splits["test"]]])
        hurum_test_labels = np.concatenate([tp_labels[tp_splits["test"]], fp_labels[fp_splits["test"]]])

        # Interleave TP and FP in training set so sequential batches see both classes
        tp_train_data = tp_data[tp_splits["train"]]
        tp_train_labels = tp_labels[tp_splits["train"]]
        fp_train_data = fp_data[fp_splits["train"]]
        fp_train_labels = fp_labels[fp_splits["train"]]

        n_tp_train = len(tp_train_data)
        n_fp_train = len(fp_train_data)
        ratio = n_fp_train / n_tp_train

        interleaved_data = np.empty((n_tp_train + n_fp_train,) + tp_train_data.shape[1:],
                                     dtype=tp_train_data.dtype)
        interleaved_labels = np.empty(n_tp_train + n_fp_train, dtype=tp_train_labels.dtype)

        tp_idx = 0
        fp_idx = 0
        out_idx = 0
        fp_per_tp = ratio
        fp_debt = 0.0

        for _ in range(n_tp_train + n_fp_train):
            if tp_idx < n_tp_train and (fp_idx >= n_fp_train or fp_debt <= 0):
                interleaved_data[out_idx] = tp_train_data[tp_idx]
                interleaved_labels[out_idx] = tp_train_labels[tp_idx]
                tp_idx += 1
                fp_debt += fp_per_tp
            else:
                interleaved_data[out_idx] = fp_train_data[fp_idx]
                interleaved_labels[out_idx] = fp_train_labels[fp_idx]
                fp_idx += 1
                fp_debt -= 1.0
            out_idx += 1

        hurum_train_data = interleaved_data
        hurum_train_labels = interleaved_labels

        hurum_val_data = np.concatenate([tp_data[tp_splits["val"]], fp_data[fp_splits["val"]]])
        hurum_val_labels = np.concatenate([tp_labels[tp_splits["val"]], fp_labels[fp_splits["val"]]])

        # Hyrax's legacy split indices (must match create_splits in pytorch_ignite.py)
        hyrax_indices = list(range(total))
        np.random.seed(seed)
        np.random.shuffle(hyrax_indices)

        num_test = int(np.round(total * test_frac))
        num_train = int(np.round(total * (1.0 - test_frac - val_frac)))

        hyrax_test_positions = hyrax_indices[:num_test]
        hyrax_train_positions = hyrax_indices[num_test:num_test + num_train]
        hyrax_val_positions = hyrax_indices[num_test + num_train:]

        # Place Hurum's data at Hyrax's positions
        new_data = np.empty_like(self._data)
        new_labels = np.empty_like(self._labels)

        for i, pos in enumerate(hyrax_test_positions):
            new_data[pos] = hurum_test_data[i]
            new_labels[pos] = hurum_test_labels[i]

        for i, pos in enumerate(hyrax_train_positions):
            new_data[pos] = hurum_train_data[i]
            new_labels[pos] = hurum_train_labels[i]

        for i, pos in enumerate(hyrax_val_positions):
            new_data[pos] = hurum_val_data[i]
            new_labels[pos] = hurum_val_labels[i]

        self._data = new_data
        self._labels = new_labels

    def ids(self):
        return np.arange(len(self._data))

    def shape(self):
        width, height = self._data[0][0].shape
        return (3, width, height)

    def get_object_id(self, idx):
        # Return a 0-padded string of the index with the number of digits needed
        # for the largest index, e.g. "000123" for index 123 if there are less
        # than 1 million samples
        num_digits = len(str(len(self._data) - 1))
        return f"{idx:0{num_digits}d}"


    def get_classification(self, idx):
        return self._labels[idx]

    def get_stamps(self, idx):
        x = self._data[idx]
        if self.augment:
            # reproduce this code using numpy instead of torch
            x = np.rot90(x, k=np.random.randint(0, 4), axes=(1, 2))
            if np.random.rand() > 0.5: x = np.flip(x, axis=2)
            if np.random.rand() > 0.5: x = np.flip(x, axis=1)
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05
        return x

    def get_normalized_stamps(self, idx):
        x = self.get_stamps(idx)
        flat = x.reshape(len(x), -1)
        mu = flat.mean(axis=1, keepdims=True)
        sig = flat.std(axis=1, keepdims=True)
        sig[sig == 0] = 1.0
        return ((flat - mu) / sig).reshape(x.shape)

    def _read_metadata(self):
        from astropy.table import Table
        return Table({
            "object_id": self.ids(),
            "classification": self._labels,
        })

    def __len__(self):
        return len(self._data)
