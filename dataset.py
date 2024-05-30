from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import itertools
from torch.utils.data.sampler import Sampler
from torchvision import datasets, models, transforms
import math
from copy import deepcopy
import random
from utils.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)  # spiltext() a.npy to a and .npy
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')  # show npy num

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, npy):
        img_nd = npy
        # print(img_nd.shape)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
            # print(img_nd.shape)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # print(img_trans.shape)
        # if img_trans.max() > 1:
        #     img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + '/' + idx + '.*')
        img_file = glob(self.imgs_dir + '/' + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0])
        img = np.load(img_file[0])

        # assert img.size == mask.size, \
        assert img.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        mask = mask.astype("float32")
        img = img.astype("float32")

        return img, mask


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
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
