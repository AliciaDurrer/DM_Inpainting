"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import sys

import nibabel as nib
from prettytable import PrettyTable

sys.path.append(".")
import pathlib
import time
import warnings
from datetime import datetime

import create_submission_adapted
import nibabel as nib
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    args = create_argparser().parse_args()
    seed = args.seed

    dist_util.setup_dist(devices=args.devices)
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)
    today = datetime.now()
    logger.log("SAMPLING START " + str(today))
    logger.log("args: " + str(args))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print(args.data_dir)
    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,  # how many volumes loaded at once, currently written for one volume only
        shuffle=False,
    )
    data = iter(datal)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(
        dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
    )  # allow for 2 devices
    model.eval()

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")
        # model.convert_to_fp16()

    dir_path = str(args.data_dir)
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, path)):
            count += 1

    for d in range(0, count):
        print("sampling file ", str(d + 1), " of ", str(count), "...")

        batch, path = next(data)

        p_s = (
            path[0].split("/")[-1].split(".")[0].split("-")[2]
            + "-"
            + path[0].split("/")[-1].split(".")[0].split("-")[3]
        )

        if not os.path.isfile(
            str(args.output_dir) + "/BraTS-GLI-" + str(p_s) + "-t1n-inference.nii.gz"
        ):
            logger.log("file ", str(p_s), " does not exist yet, starting sampling...")
            logger.log("    sampling start ", datetime.now())

            c = th.randn_like(batch[:, :1, ...])

            img = th.cat((batch, c), dim=1)

            p_s = (
                path[0].split("/")[-1].split(".")[0].split("-")[2]
                + "-"
                + path[0].split("/")[-1].split(".")[0].split("-")[3]
            )

            model_kwargs = {}

            sample_fn = (
                diffusion.p_sample_loop_known
                if not args.use_ddim
                else diffusion.ddim_sample_loop_known
            )

            sample, x_noisy, org = sample_fn(
                model,
                (img.shape[0], 3, args.image_size, args.image_size, args.image_size),
                img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            sa = sample.clone().detach()
            sa = sa.squeeze(1).squeeze(0)
            sam = np.asarray(sa.detach().cpu())

            nib.save(
                nib.Nifti1Image(sam, None),
                (
                    str(args.output_dir)
                    + "/BraTS-GLI-"
                    + str(p_s)
                    + "-t1n-inference.nii.gz"
                ),
            )

            logger.log("    sampling end ", datetime.now())

        else:
            logger.log("file ", str(p_s), " exists already, skipping it")

    create_submission_adapted.adapt(
        input_data=args.data_dir,
        samples_dir=args.output_dir,
        adapted_samples_dir=args.adapted_samples_dir,
    )

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        output_dir="",
        adapted_samples_dir="adapted_results",
        data_mode="validation",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        mode="segmentation",
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,  # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
    )
    defaults.update(
        {k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults}
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
