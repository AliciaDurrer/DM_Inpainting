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


class BRATSDDPMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True):
        """
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        """

        self.root_dir = root_dir
        self.train = train
        # print('self.root', self.root_dir)
        self.dataset = self.get_dataset()

    # print('got dataset')

    def get_dataset(self):
        if self.train:
            brats_2023 = pd.read_csv(
                os.path.join(self.root_dir, "ids/train_ddpm.tsv"), delim_whitespace=True
            )
        #    print('brats_2023 train', brats_2023)
        else:
            brats_2023 = pd.read_csv(
                os.path.join(self.root_dir, "ids/val_ddpm.tsv"), delim_whitespace=True
            )
            # print('brats_2023 val', brats_2023)
        return brats_2023

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        out = []
        #  print('index', index)
        # print(self.dataset)
        # print('self.dataset[brats_2023].iloc[index]', self.dataset['healthy'].iloc[index])
        # print('self.dataset[brats_2023].iloc[index]', self.dataset['diseased'].iloc[index])

        healthy_img = np.asarray(
            nibabel.load(self.dataset["healthy"].iloc[index]).get_fdata()
        )
        diseased_img = np.asarray(
            nibabel.load(self.dataset["diseased"].iloc[index]).get_fdata()
        )
        mask_img = np.asarray(
            nibabel.load(self.dataset["mask"].iloc[index]).get_fdata()
        )

        healthy_pad = np.pad(healthy_img, ((16, 16), (16, 16), (16, 16)))
        healthy_tensor = torch.tensor(healthy_pad)
        healthy = healthy_tensor[::2, ::2, ::2]
        healthy = healthy.unsqueeze(0)
        healthy = healthy.float()

        healthy_normalized = (
            2
            * (
                (healthy - torch.min(healthy))
                / (torch.max(healthy) - torch.min(healthy))
            )
            - 1
        )

        diseased_pad = np.pad(diseased_img, ((16, 16), (16, 16), (16, 16)))
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

        mask_pad = np.pad(mask_img, ((16, 16), (16, 16), (16, 16)))
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

        img_out = torch.stack(
            (diseased_normalized, mask_prep, healthy_normalized)
        ).squeeze(1)

        return {"data": img_out}
