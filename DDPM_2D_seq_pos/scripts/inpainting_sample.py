"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""


# Adapted for BraTS challenge 2023 pipeline

import argparse
import os
import nibabel as nib
import sys
import random

sys.path.append(".")
import numpy as np
import time
import math
from tqdm import tqdm
import torch as th
import torch.distributed as dist
import create_submission_adapted
import eval_sam
from datetime import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
    today = datetime.now()
    logger.log("SAMPLING " + str(today))
    logger.log("args: " + str(args))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    data = iter(datal)
    print("data", data)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dir_path = str(args.data_dir)
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, path)):
            count += 1

    for d in range(0, count):
        print("sampling file ", str(d + 1), " of ", str(count), "...")

        batch, path, slicedict = next(data)

        c = th.randn_like(batch[:, :1, ...])

        slice_ID = path[0].split("/")[-2]

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        diseased_3D = np.asarray(batch[0, 0, ...])
        mask_3D = np.asarray(batch[0, 1, ...])

        volume_3D = []

        prev_sam_slice = np.zeros((1, 1, 224, 224))

        for i in tqdm(range(mask_3D.shape[-1])):
            if np.sum(mask_3D[:, :, i]) != 0:
                # print('ID ', str(slice_ID), ' mask slice ', str(i), ' not empty. sampling ...')

                prev_sam_slice_th = (
                    th.from_numpy(prev_sam_slice).unsqueeze(0).unsqueeze(1)
                )

                img_to_sample = th.cat(
                    (batch[..., i], prev_sam_slice_th, c[..., i]), dim=1
                )
                current_slice_nr = i

                p_s = (
                    path[0].split("/")[-1].split(".")[0].split("-")[2]
                    + "-"
                    + path[0].split("/")[-1].split(".")[0].split("-")[3]
                )

                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known
                    if not args.use_ddim
                    else diffusion.ddim_sample_loop_known
                )

                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    img_to_sample,
                    current_slice_nr,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

                end.record()
                th.cuda.synchronize()

                s = sample.clone().detach()

                sam = np.asarray(s[0, 0, ...].detach().cpu())

                prev_sam_slice = np.copy(sam)
                slice_for_vol = np.copy(sam)

            else:
                slice_for_vol = np.copy(diseased_3D[..., i])
                prev_sam_slice = np.copy(slice_for_vol)

                p_s = (
                    path[0].split("/")[-1].split(".")[0].split("-")[2]
                    + "-"
                    + path[0].split("/")[-1].split(".")[0].split("-")[3]
                )

            volume_3D.append(th.from_numpy(slice_for_vol).unsqueeze(2))
            vol_stack = th.cat(volume_3D, dim=2)
            vol_3D = np.asarray(vol_stack)
            nib.save(
                nib.Nifti1Image(vol_3D, None),
                (
                    str(args.log_dir)
                    + "/BraTS-GLI-"
                    + str(p_s)
                    + "-t1n-inference.nii.gz"
                ),
            )

    create_submission_adapted.adapt(
        input_data=args.data_dir,
        samples_dir=args.log_dir,
        adapted_samples_dir=args.adapted_samples,
    )
    # eval_sam.eval_adapted_samples(dataset_path_eval=args.data_dir, solutionFilePaths_gt=args.gt_dir,
    #                              resultsFolder_dir=args.adapted_samples)


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        gt_dir="",
        adapted_samples="",
        subbatch=16,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
