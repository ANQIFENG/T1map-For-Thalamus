#!/usr/bin/env python

import os
import torch
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")

config_split = {'0': {'train_idxs': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'val_idxs': [3, 4], 'test_idxs': [0, 1, 2]},
                '1': {'train_idxs': [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'val_idxs': [6, 7], 'test_idxs': [3, 4, 5]},
                '2': {'train_idxs': [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'val_idxs': [9, 10], 'test_idxs': [6, 7, 8]},
                '3': {'train_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'val_idxs': [12, 13], 'test_idxs': [9, 10, 11]},
                '4': {'train_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23], 'val_idxs': [15, 16], 'test_idxs': [12, 13, 14]},
                '5': {'train_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 21, 22, 23], 'val_idxs': [18, 19], 'test_idxs': [15, 16, 17]},
                '6': {'train_idxs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23], 'val_idxs': [21, 22], 'test_idxs': [18, 19, 20]},
                '7': {'train_idxs': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'val_idxs': [0, 1], 'test_idxs': [21, 22, 23]}}


# Read nifti data
def load_data(data_path):
    data = nib.load(data_path).get_fdata().astype(np.float32)
    data = data.transpose((3, 0, 1, 2))  # transpose h*w*l*c to c*h*w*l
    return data


# Read nifti labels
def load_label(label_path):
    label = nib.load(label_path).get_fdata().astype(np.int32)
    return label


# Min-Max normalization function
def min_max_normalization(data):
    for i in range(data.shape[0]):
        modality = data[i]
        min_val = modality.min()
        max_val = modality.max()
        data[i] = (modality - min_val) / (max_val - min_val)
    return data


class ThalamusDataset(Dataset):

    def __init__(self, data_dir, label_dir, split, division, augment=False):
        super(ThalamusDataset, self).__init__()
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.split = split
        self.division = division
        self.data_file_list = sorted(list(os.listdir(self.data_dir)))
        self.label_file_list = sorted(list(os.listdir(self.label_dir)))
        self.augment = augment
        self.crop_size = (96, 96, 96)

        if self.augment:
            self.transformations = tio.Compose([
                tio.RandomFlip(axes=('LR'), flip_probability=0.4),
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=2),
                tio.RandomGamma(log_gamma=(-0.3, 0.3)),
                tio.CropOrPad(self.crop_size)
            ])
        else:
            self.transformations = tio.Compose([
                tio.CropOrPad(self.crop_size),
            ])

    def __len__(self):
        return len(config_split[self.split][self.division + "_idxs"])

    def __getitem__(self, idx):

        data_fn = [self.data_file_list[index] for index in config_split[self.split][self.division + "_idxs"]][idx]
        label_fn = [self.label_file_list[index] for index in config_split[self.split][self.division + "_idxs"]][idx]

        assert data_fn.split('_')[0] == label_fn.split('_')[0]
        data_path = os.path.join(self.data_dir, data_fn)
        label_path = os.path.join(self.label_dir, label_fn)

        data_np = load_data(data_path)
        label_np = load_label(label_path)

        subject = tio.Subject(image=tio.ScalarImage(tensor=data_np), label=tio.LabelMap(tensor=label_np[np.newaxis]))
        transformed = self.transformations(subject)

        data_np = transformed.image.data.detach().numpy()
        label_np = transformed.label.data.detach().numpy()

        normalized_data_np = min_max_normalization(data_np)

        data_tensor = torch.tensor(normalized_data_np, dtype=torch.float32)
        label_tensor = torch.tensor(label_np, dtype=torch.int32)

        if self.division == 'test':
            return data_tensor, label_tensor, data_fn, label_fn
        else:
            return data_tensor, label_tensor


def ThalamusDataloader(data_dir, label_dir, batch_size, split, division, augment=False, shuffle=False, num_workers=2):
    dataset = ThalamusDataset(data_dir=data_dir, label_dir=label_dir, split=split, division=division, augment=augment)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
