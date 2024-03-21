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


class BRATSDataset(torch.utils.data.Dataset):
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
        print("self.root", self.root_dir)

        self.dataset = self.get_dataset()

    def get_dataset(self):
        if self.train:
            df = pd.read_csv(os.path.join(self.root_dir, "ids/train.tsv"))
            brats_2023 = df[~df["images"].isna()]
            print("brats 2023 training", brats_2023)
        else:
            df = pd.read_csv(os.path.join(self.root_dir, "ids/val.tsv"))
            brats_2023 = df[~df["images"].isna()]
            print("brats 2023 validation", brats_2023)
        return brats_2023

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        out = []

        nib_img = nibabel.load(self.dataset["images"].iloc[index])
        # print('(self.dataset[images].iloc[index])', (self.dataset['images'].iloc[index]))

        # img_type = (self.dataset['images'].iloc[index]).split('/')[-1].split('_')[-1]

        out.append(torch.tensor(nib_img.get_fdata()))

        out = torch.stack(out)

        image_np = np.asarray(out)
        image_pad = np.pad(image_np, ((0, 0), (16, 16), (16, 16), (16, 16)))
        image_tensor = torch.tensor(image_pad)
        image = image_tensor[:, ::2, ::2, ::2]
        image = image.float()

        image_normalized = (
            2 * ((image - torch.min(image)) / (torch.max(image) - torch.min(image))) - 1
        )

        # print('image normalized', img_type, image_normalized.shape, torch.min(image_normalized), torch.max(image_normalized))
        # breakpoint()

        return {"data": image_normalized}
