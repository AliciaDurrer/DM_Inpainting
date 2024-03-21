import os
import os.path

import nibabel
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(
        self, directory, test_flag=True, normalize=None, mode="test", img_size=256
    ):
        """
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        """
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        if test_flag:
            self.seqtypes = ["voided", "mask"]
        else:
            self.seqtypes = ["diseased", "mask", "healthy"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have a datadir
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split("-")[-1].split(".")[0]
                    if seqtype in self.seqtypes:
                        datapoint[seqtype] = os.path.join(root, f)
                        self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]

        out_single = []

        if self.test_flag:
            for seqtype in self.seqtypes:
                nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                    np.float32
                )
                t1_numpy_pad = np.pad(nib_img, ((8, 8), (8, 8), (50, 51)))
                t1_normalized = (
                    2
                    * (
                        (t1_numpy_pad - np.min(t1_numpy_pad))
                        / (np.max(t1_numpy_pad) - np.min(t1_numpy_pad))
                    )
                    - 1
                )
                path = filedict[seqtype]
                img_preprocessed = torch.tensor(t1_normalized)
                out_single.append(img_preprocessed)
            out_single = torch.stack(out_single)
            image = out_single

            if self.img_size == 128:
                image = image[:, ::2, ::2, ::2]

            return (image, path)

        else:
            for seqtype in self.seqtypes:
                nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                    np.float32
                )
                t1_numpy_pad = np.pad(nib_img, ((16, 16), (16, 16), (16, 16)))
                t1_normalized = (
                    2
                    * (
                        (t1_numpy_pad - np.min(t1_numpy_pad))
                        / (np.max(t1_numpy_pad) - np.min(t1_numpy_pad))
                    )
                    - 1
                )
                img_preprocessed = torch.tensor(t1_normalized)
                out_single.append(img_preprocessed)
            out_single = torch.stack(out_single)
            image = out_single[:2, ...]
            label = out_single[2, ...]

            label = label.unsqueeze(0)

            if self.img_size == 128:
                image = image[:, ::2, ::2, ::2]
                label = label[:, ::2, ::2, ::2]


            return (image, label)

    def __len__(self):
        return len(self.database)
