"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import nibabel as nib
import sys
import random

# sys.path.append("..")
sys.path.append(".")
import numpy as np
import time
import torch as th
import torch.distributed as dist
import nibabel as nib
import pathlib
import warnings
from datetime import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


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
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("[INFO] creating model and diffusion...")
    logger.log("[ARGS] ", args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("[INFO] load model from: {}".format(args.model_path))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(
        dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev()
    )  # allow for 2 devices

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    ds = BRATSVolumes(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,  # how many volumes loaded at once, currently written for one volume only
        shuffle=False,
    )
    data = iter(datal)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

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

        logger.log("sampling file ", str(p_s))
        logger.log("    sampling start ", datetime.now())

        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"reseeded (in for loop) to {seed}")

        seed += 1

        batch = batch.to(dist_util.dev())

        input_masked = batch[0, 0, :, :, :]
        input_masked = input_masked.unsqueeze(0).unsqueeze(1)

        mask = batch[0, 1, :, :, :]
        mask = mask.unsqueeze(0).unsqueeze(1)

        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(input_masked)
        input_masked_dwt = th.cat([LLL / 3.0, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        # print('input_masked_dwt', input_masked_dwt.shape)

        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(mask)
        mask_dwt = th.cat([LLL / 3.0, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        # print('mask_dwt', mask_dwt.shape)

        img = th.randn(
            args.batch_size,
            8,
            args.image_size // 2,
            args.image_size // 2,
            args.image_size // 2,
        ).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model=model,
            shape=img.shape,
            noise=img,
            input_masked=input_masked_dwt,
            mask=mask_dwt,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        B, _, H, W, D = sample.size()

        #  print('sample before', sample.shape)

        sample = idwt(
            sample[:, 0, :, :, :].view(B, 1, H, W, D) * 3.0,
            sample[:, 1, :, :, :].view(B, 1, H, W, D),
            sample[:, 2, :, :, :].view(B, 1, H, W, D),
            sample[:, 3, :, :, :].view(B, 1, H, W, D),
            sample[:, 4, :, :, :].view(B, 1, H, W, D),
            sample[:, 5, :, :, :].view(B, 1, H, W, D),
            sample[:, 6, :, :, :].view(B, 1, H, W, D),
            sample[:, 7, :, :, :].view(B, 1, H, W, D),
        )

        sample = (sample + 1) / 2.0

        # print('sample idwt', sample.shape, th.min(sample), th.max(sample))

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        if len(batch.shape) == 5:
            batch = batch.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        # print('sample squeezed', sample.shape)
        # print('batch squeezed', batch.shape)

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # for i in range(sample.shape[0]):
        output_name = os.path.join(args.output_dir, f"sample_{p_s}.nii.gz")
        img = nib.Nifti1Image(sample.detach().cpu().numpy()[0, :, :, :], np.eye(4))
        nib.save(img=img, filename=output_name)
        # output_name_masked = os.path.join(args.output_dir, f'input_{p_s}.nii.gz')
        # img_masked = nib.Nifti1Image(input_masked.detach().cpu().numpy()[0,0,:, :, :], np.eye(4))
        # nib.save(img=img_masked, filename=output_name_masked)
        # output_name_mask = os.path.join(args.output_dir, f'mask_{p_s}.nii.gz')
        # img_mask = nib.Nifti1Image(mask.detach().cpu().numpy()[0,0,:, :, :], np.eye(4))
        # nib.save(img=img_mask, filename=output_name_mask)
        print(f"saved to {output_name}")


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode="validation",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir="./results",
        mode="default",
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
