"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""


# BraTS 2023 inpainting challenge pipeline

import argparse
import math
import os
import random
import sys

import nibabel as nib
from prettytable import PrettyTable

sys.path.append(".")
import time
from datetime import datetime

import create_submission_adapted
import eval_sam
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
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)
    today = datetime.now()
    logger.log("SAMPLING START " + str(today))
    logger.log("args: " + str(args))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

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

        p_s = (
            path[0].split("/")[-1].split(".")[0].split("-")[2]
            + "-"
            + path[0].split("/")[-1].split(".")[0].split("-")[3]
        )

        if not os.path.isfile(
            str(args.log_dir) + "/BraTS-GLI-" + str(p_s) + "-t1n-inference.nii.gz"
        ):
            print("file does not exist yet, starting sampling...")
            generated_3D = np.asarray(batch[:, 0, :, :, :].squeeze())

            batch_size_vol = args.subbatch  # how many slices per volume
            nr_batches = len(slicedict) / batch_size_vol

            nr_batches = math.ceil(nr_batches)

            for b in range(0, nr_batches):
                out_batch = []
                s_sub = []

                if len(slicedict) > b * batch_size_vol + batch_size_vol:
                    for s in slicedict[
                        b * batch_size_vol : (b * batch_size_vol + batch_size_vol)
                    ]:
                        out_batch.append(batch[..., s])

                        s_sub.append(s)

                    out_batch = th.stack(out_batch)

                    out_batch = out_batch.squeeze(1)

                    out_batch = out_batch.squeeze(4)

                    c = th.randn_like(out_batch[:, :1, ...])

                    out_batch = th.cat((out_batch, c), dim=1)

                    model_kwargs = {}

                    sample_fn = (
                        diffusion.p_sample_loop_known
                        if not args.use_ddim
                        else diffusion.ddim_sample_loop_known
                    )

                    sample, x_noisy, org = sample_fn(
                        model,
                        (out_batch.shape[0], 3, args.image_size, args.image_size),
                        out_batch,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )

                    sa = sample.clone().detach()
                    sa = sa.squeeze(1)
                    sam = np.asarray(sa.detach().cpu())

                    su = 0
                    for s_sub_sub in s_sub:
                        generated_3D[..., s_sub_sub] = sam[su, :, :]
                        su += 1

                    nib.save(
                        nib.Nifti1Image(generated_3D, None),
                        (
                            str(args.log_dir)
                            + "/BraTS-GLI-"
                            + str(p_s)
                            + "-t1n-inference.nii.gz"
                        ),
                    )

                else:
                    for s in slicedict[b * batch_size_vol :]:
                        out_batch.append(batch[..., s])

                        s_sub.append(s)

                    out_batch = th.stack(out_batch)

                    out_batch = out_batch.squeeze(1)

                    out_batch = out_batch.squeeze(4)

                    c = th.randn_like(out_batch[:, :1, ...])

                    out_batch = th.cat((out_batch, c), dim=1)

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
                        (out_batch.shape[0], 3, args.image_size, args.image_size),
                        out_batch,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )

                    sa = sample.clone().detach()
                    sa = sa.squeeze(1)
                    sam = np.asarray(sa.detach().cpu())

                    su = 0
                    for s_sub_sub in s_sub:
                        generated_3D[..., s_sub_sub] = sam[su, :, :]
                        su += 1

                    nib.save(
                        nib.Nifti1Image(generated_3D, None),
                        (
                            str(args.log_dir)
                            + "/BraTS-GLI-"
                            + str(p_s)
                            + "-t1n-inference.nii.gz"
                        ),
                    )
        else:
            print("file exists already, skipping it")

    create_submission_adapted.adapt(
        input_data=args.data_dir,
        samples_dir=args.log_dir,
        adapted_samples_dir=args.adapted_samples,
    )


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        # gt_dir="",
        adapted_samples="",
        subbatch=16,
        clip_denoised=True,
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
