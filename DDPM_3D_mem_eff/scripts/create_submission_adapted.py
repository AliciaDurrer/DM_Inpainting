import os

import nibabel as nib
import numpy as np
import torch as th
from PIL import ImageFilter
from scipy.ndimage import gaussian_filter1d


def adapt(input_data=None, samples_dir=None, adapted_samples_dir=None):
    if os.path.exists(samples_dir + "/log.txt"):
        os.remove(samples_dir + "/log.txt")

    if os.path.exists(samples_dir + "/progress.csv"):
        os.remove(samples_dir + "/progress.csv")

    for root, dirs, files in os.walk(samples_dir):
        for file_id in files:
            folder_name = file_id.split("-")[2] + "-" + file_id.split("-")[3]

            nifti_file = nib.load(str(root) + "/" + str(file_id))

            org_nifti = nib.load(
                str(input_data)
                + "/BraTS-GLI-"
                + str(folder_name)
                + "/BraTS-GLI-"
                + str(folder_name)
                + "-t1n-voided.nii.gz"
            )

            target_header = org_nifti.header

            np_org = np.asarray(org_nifti.dataobj)
            np_org_clipped = np.percentile(np_org, [0.5, 99.5])

            np_redef = np.asarray(nifti_file.dataobj)

            # normalize between ...

            clipped_image = np.clip(np_redef, 0, 1)
            start = np.min(np_org_clipped)
            end = np.max(np_org_clipped)
            width = end - start
            norm_img = (clipped_image - clipped_image.min()) / (
                clipped_image.max() - clipped_image.min()
            ) * width + start

            # reshape to [240, 240, 155]

            cropped_image = norm_img[8:-8, 8:-8, 50:-51]

            if not os.path.exists(adapted_samples_dir):
                os.makedirs(adapted_samples_dir)

            nib.save(
                nib.Nifti1Image(cropped_image, None, target_header),
                str(adapted_samples_dir) + "/" + str(file_id),
            )

        break
