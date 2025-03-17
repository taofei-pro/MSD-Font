import bisect
import numpy as np
import albumentations
import torch
from torch.utils.data import Dataset, ConcatDataset


class Txt2ImgIterableBaseDataset(Dataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.size = size

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        raise NotImplementedError()


class ConcatDatasetWithIndex(ConcatDataset):
    """
    A ConcatDataset that provides the index of the original dataset for each sample
    """
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx
