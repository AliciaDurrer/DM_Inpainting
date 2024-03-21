""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
from skimage import exposure
import argparse
import pandas as pd

import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel


class BRATSDDPMTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.train == False:
            print(self.root_dir)
            brats_2023 = pd.read_csv(
                os.path.join(self.root_dir, "ids/test_data.tsv"), delim_whitespace=True
            )
            # print('brats_2023 val', brats_2023)
        return brats_2023

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        out = []

        diseased_img = np.asarray(
            nibabel.load(self.dataset["voided"].iloc[index]).get_fdata()
        )
        mask_img = np.asarray(
            nibabel.load(self.dataset["mask"].iloc[index]).get_fdata()
        )

        mask_pad = np.pad(mask_img, ((8, 8), (8, 8), (50, 51)))
        mask_tensor = torch.tensor(mask_pad)
        mask_prep = mask_tensor[::2, ::2, ::2]
        mask_prep = mask_prep.unsqueeze(0)
        mask_prep = mask_prep.float()

        mask_prep = (
            2
            * (
                (mask_prep - torch.min(mask_prep))
                / (torch.max(mask_prep) - torch.min(mask_prep))
            )
            - 1
        )

        diseased_pad = np.pad(diseased_img, ((8, 8), (8, 8), (50, 51)))
        diseased_tensor = torch.tensor(diseased_pad)
        diseased = diseased_tensor[::2, ::2, ::2]
        diseased = diseased.unsqueeze(0)
        diseased = diseased.float()

        diseased_normalized = (
            2
            * (
                (diseased - torch.min(diseased))
                / (torch.max(diseased) - torch.min(diseased))
            )
            - 1
        )

        img_out = torch.stack((diseased_normalized, mask_prep)).squeeze(1)

        path_voided = self.dataset["voided"].iloc[index]

        return img_out, path_voided
