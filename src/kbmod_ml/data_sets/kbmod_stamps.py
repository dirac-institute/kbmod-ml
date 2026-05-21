import logging
import os
import numpy as np
from hyrax.datasets.dataset_registry import HyraxDataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class KbmodStamps(HyraxDataset, Dataset):
    """KBMOD stamp dataset matching the notebook's training pipeline.

    - Per-stamp z-score normalization is applied once at load time
      (mean=0, std=1 across all 3x21x21 = 1323 pixels of each stamp).
    - Augmentation (rot90, flips, Gaussian noise) is applied per-call
      inside get_stamps when self.augment is True. It is done here
      and NOT in the model, to avoid double augmentation.
    - The TP/FP class balance is handled via get_sampler(), which Hyrax's
      dist_data_loader will call if present. Use_weighted_sampler=True
      in the train data request config, False for validate/test.
    """

    def __init__(self, config, data_location=None):
        super().__init__(config)

        true_positive_file_name = config["kbmod_ml"]["true_positive_file_name"]
        false_positive_file_name = config["kbmod_ml"]["false_positive_file_name"]

        true_data_path = os.path.join(data_location, true_positive_file_name)
        false_data_path = os.path.join(data_location, false_positive_file_name)

        if not os.path.isfile(true_data_path):
            raise ValueError(f"Could not find {true_positive_file_name} in provided {data_location}")
        if not os.path.isfile(false_data_path):
            raise ValueError(f"Could not find {false_positive_file_name} in provided {data_location}")

        true_positive_samples = np.load(true_data_path)
        logger.warning(f"Loaded {len(true_positive_samples)} true positive samples from {true_data_path}")
        false_positive_samples = np.load(false_data_path)
        logger.warning(f"Loaded {len(false_positive_samples)} false positive samples from {false_data_path}")

        # Both should be (N, 3, 21, 21). If FP has extra channels, take the
        # first 3 to match TP. If FP has fewer than 3, fail loudly.
        if true_positive_samples.ndim != 4 or true_positive_samples.shape[1:] != (3, 21, 21):
            raise ValueError(
                f"Expected TP shape (N, 3, 21, 21), got {true_positive_samples.shape}"
            )
        if false_positive_samples.ndim != 4 or false_positive_samples.shape[2:] != (21, 21):
            raise ValueError(
                f"Expected FP shape (N, C>=3, 21, 21), got {false_positive_samples.shape}"
            )
        if false_positive_samples.shape[1] < 3:
            raise ValueError(
                f"FP file has only {false_positive_samples.shape[1]} channels, need >= 3"
            )
        if false_positive_samples.shape[1] > 3:
            logger.warning(
                f"FP file has {false_positive_samples.shape[1]} channels; "
                f"taking first 3 to match TP."
            )
            false_positive_samples = false_positive_samples[:, 0:3]

        n_tp = len(true_positive_samples)
        n_fp = len(false_positive_samples)

        raw_data = np.concatenate([true_positive_samples, false_positive_samples])
        raw_labels = np.concatenate([
            np.ones(n_tp, dtype=np.int64),
            np.zeros(n_fp, dtype=np.int64),
        ])

        # Per-stamp z-score normalization
        self._data = self._normalize(raw_data)
        self._labels = raw_labels

        seed = 42
        if "data_set" in config and config["data_set"].get("seed"):
            seed = config["data_set"]["seed"]

        # Whether to apply stochastic augmentation in get_stamps.
        self.augment = config["kbmod_ml"]["augment"]

        # Whether to return a WeightedRandomSampler from get_sampler().
        self.use_weighted_sampler = config["kbmod_ml"].get("use_weighted_sampler", False)


        if config["kbmod_ml"].get("arrange_for_hyrax_splits", False):
            self._arrange_for_hyrax_splits(n_tp, n_fp, seed)

        metadata_table = self._read_metadata()
        super().__init__(config, metadata_table)

    @staticmethod
    def _normalize(stamps):
        out = stamps.astype(np.float32)
        flat = out.reshape(len(out), -1)
        mu = flat.mean(axis=1, keepdims=True)
        sig = flat.std(axis=1, keepdims=True)
        sig[sig == 0] = 1.0
        return ((flat - mu) / sig).reshape(out.shape)

    def sampler(self, indexes):
        """WeightedRandomSampler for class-balanced training batches.
        """
        if not self.use_weighted_sampler:
            return None

        from torch.utils.data import WeightedRandomSampler
        import torch

        labels = self._labels[indexes]
        n_tp = (labels == 1).sum()
        n_fp = (labels == 0).sum()

        if n_tp == 0 or n_fp == 0:
            logger.warning(
                f"get_sampler: split has n_tp={n_tp}, n_fp={n_fp} — "
                "skipping WeightedRandomSampler, falling back to default."
            )
            return None

        weights = np.where(labels == 1, 1.0 / n_tp, 1.0 / n_fp)
        logger.info(
            f"WeightedRandomSampler: {n_tp} TP, {n_fp} FP, "
            f"ratio {n_fp/n_tp:.1f}:1 over {len(indexes)} samples"
        )
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float32),
            num_samples=len(weights),
            replacement=True,
        )

    def _arrange_for_hyrax_splits(self, n_tp, n_fp, seed):
        """Rearrange data so Hyrax's create_splits. Split TP and FP independently 
        using default_rng(seed),then concatenates per split. Hyrax shuffles the
        combined array using np.random.seed(seed) and takes contiguous blocks. The train portion
        is additionally interleaved by class so sequential batches see both classes 
        in proportion (a deterministic substitute for WeightedRandomSampler).
        """
        total = n_tp + n_fp
        test_frac = 0.15
        val_frac = 0.15

        rng = np.random.default_rng(seed)

        def notebook_split(n):
            idx = rng.permutation(n)
            n_test = int(n * test_frac)
            n_val = int(n * val_frac)
            return {
                "test": idx[:n_test],
                "val": idx[n_test:n_test + n_val],
                "train": idx[n_val + n_test:],
            }

        tp_splits = notebook_split(n_tp)
        fp_splits = notebook_split(n_fp)

        orig_data = self._data.copy()
        orig_labels = self._labels.copy()

        tp_data = orig_data[:n_tp]
        fp_data = orig_data[n_tp:]
        tp_labels = orig_labels[:n_tp]
        fp_labels = orig_labels[n_tp:]

        notebook_test_data = np.concatenate(
            [tp_data[tp_splits["test"]], fp_data[fp_splits["test"]]]
        )
        notebook_test_labels = np.concatenate(
            [tp_labels[tp_splits["test"]], fp_labels[fp_splits["test"]]]
        )

  
        tp_train_data = tp_data[tp_splits["train"]]
        tp_train_labels = tp_labels[tp_splits["train"]]
        fp_train_data = fp_data[fp_splits["train"]]
        fp_train_labels = fp_labels[fp_splits["train"]]

        n_tp_train = len(tp_train_data)
        n_fp_train = len(fp_train_data)
        fp_per_tp = n_fp_train / n_tp_train

        interleaved_data = np.empty(
            (n_tp_train + n_fp_train,) + tp_train_data.shape[1:],
            dtype=tp_train_data.dtype,
        )
        interleaved_labels = np.empty(
            n_tp_train + n_fp_train, dtype=tp_train_labels.dtype
        )

        tp_idx = 0
        fp_idx = 0
        out_idx = 0
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

        notebook_train_data = interleaved_data
        notebook_train_labels = interleaved_labels

        notebook_val_data = np.concatenate(
            [tp_data[tp_splits["val"]], fp_data[fp_splits["val"]]]
        )
        notebook_val_labels = np.concatenate(
            [tp_labels[tp_splits["val"]], fp_labels[fp_splits["val"]]]
        )

  
        hyrax_indices = list(range(total))
        np.random.seed(seed)
        np.random.shuffle(hyrax_indices)

        num_test = int(np.round(total * test_frac))
        num_train = int(np.round(total * (1.0 - test_frac - val_frac)))

        hyrax_test_positions = hyrax_indices[:num_test]
        hyrax_train_positions = hyrax_indices[num_test:num_test + num_train]
        hyrax_val_positions = hyrax_indices[num_test + num_train:]

        new_data = np.empty_like(self._data)
        new_labels = np.empty_like(self._labels)

        for i, pos in enumerate(hyrax_test_positions):
            new_data[pos] = notebook_test_data[i]
            new_labels[pos] = notebook_test_labels[i]
        for i, pos in enumerate(hyrax_train_positions):
            new_data[pos] = notebook_train_data[i]
            new_labels[pos] = notebook_train_labels[i]
        for i, pos in enumerate(hyrax_val_positions):
            new_data[pos] = notebook_val_data[i]
            new_labels[pos] = notebook_val_labels[i]

        self._data = new_data
        self._labels = new_labels

    def ids(self):
        return np.arange(len(self._data))

    def shape(self):
        _, h, w = self._data[0].shape
        return (3, h, w)

    def get_object_id(self, idx):
        num_digits = len(str(len(self._data) - 1))
        return f"{idx:0{num_digits}d}"

    def get_classification(self, idx):
        return self._labels[idx]

    def get_stamps(self, idx):
        """Return one stamp, optionally augmented.
        """
        x = self._data[idx]
        if self.augment:
            x = np.rot90(x, k=np.random.randint(0, 4), axes=(1, 2)).copy()
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.05
        return x

    def _read_metadata(self):
        from astropy.table import Table
        return Table({
            "object_id": self.ids(),
            "classification": self._labels,
        })

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self.get_stamps(idx), self.get_classification(idx)
