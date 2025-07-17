import itertools
import h5py
import numpy as np
from torch.utils.data import Dataset, Sampler

def iterate_once(iterable):
    """Iterate through the iterable once, returning a shuffled version."""
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    """Iterate through the indices infinitely, yielding shuffled batches."""
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks."""
    args = [iter(iterable)] * n
    return zip(*args)

class BaseDataSets(Dataset):
    """Base dataset class for loading medical imaging data."""

    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            if num is not None:
                self.sample_list = self.sample_list[:num]
            print("Train total {} samples".format(len(self.sample_list)))
        elif self.split in ("val", "test"):
            with open(self._base_dir + f"/{self.split}.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print(f"{self.split.capitalize()} total {len(self.sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if self.transform is not None:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample["casename"] = case
        return sample

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch for (primary_batch, secondary_batch) in
            zip(grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
