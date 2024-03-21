import os
import os.path

import nibabel
import numpy as np
import torch
import torch.nn


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        """
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ["voided", "mask"]
        else:
            self.seqtypes = ["diseased", "mask", "healthy"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files = sorted(files)
                datapoint = dict()
                for f in files:
                    seqtype = f.split("-")[-1].split(".")[0]
                    datapoint[seqtype] = os.path.join(root, f)
                assert (
                    set(datapoint.keys()) == self.seqtypes_set
                ), f"datapoint is incomplete, keys are {datapoint.keys()}"
                self.database.append(datapoint)
                print("datapoint", datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]

        out_single = []

        if self.test_flag:
            for seqtype in self.seqtypes:
                if seqtype == "voided":
                    nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                        np.float32
                    )
                    path = filedict[seqtype]
                    t1_numpy_pad = np.pad(nib_img, ((8, 8), (8, 8), (50, 51)))
                    t1_clipped = np.clip(
                        t1_numpy_pad,
                        np.quantile(t1_numpy_pad, 0.001),
                        np.quantile(t1_numpy_pad, 0.999),
                    )
                    t1_normalized = (t1_clipped - np.min(t1_clipped)) / (
                        np.max(t1_clipped) - np.min(t1_clipped)
                    )
                    img_preprocessed = torch.tensor(t1_normalized)
                elif seqtype == "mask":
                    nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                        np.float32
                    )
                    path = filedict[seqtype]
                    mask_numpy_pad = np.pad(nib_img, ((8, 8), (8, 8), (50, 51)))
                    img_preprocessed = torch.tensor(mask_numpy_pad)

                else:
                    print("unknown seqtype")

                out_single.append(img_preprocessed)

            out_single = torch.stack(out_single)

            image = out_single[0:2, ...]
            path = filedict[seqtype]

            return (image, path)

        else:
            image = out[0:2, ...]
            label = out[2, ...][None, ...]

            return (image, label)

    def __len__(self):
        return len(self.database)
