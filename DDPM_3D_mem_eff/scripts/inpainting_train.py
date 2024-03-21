"""
Train a diffusion model on images.
"""
import sys
import argparse
import torch as th

# import torch.utils.tensorboard
import random

sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.output_dir)
    today = datetime.now()
    logger.log("TRAINING " + str(today))
    logger.log("args: " + str(args))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=1000
    )

    logger.log("creating data loader...")
    ds = BRATSDataset(args.data_dir, test_flag=False)
    print("ds", ds)
    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    #  data = iter(datal)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # dataset=args.dataset,
        # summary_writer=summary_writer,
        # mode='segmentation',
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        output_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="brats3d",
        use_tensorboard=True,
        tensorboard_path="",  # set path to existing logdir for resuming
        devices=[0],
        dims=3,  # 2 for 2d images, 3 for 3d volumes
        learn_sigma=False,
        num_groups=29,
        channel_mult="1,3,4,4,4,4,4",
        in_channels=3,  # 4 MRI sequences + out_channels  (+ #dimensions if concat_coords==True)
        out_channels=1,  # out_channels = number  of classes
        bottleneck_attention=False,
        num_workers=0,
        resample_2d=False,
        #  mode='segmentation',
        renormalize=True,
        additive_skips=True,
        decoder_device_thresh=15,
        half_res_crop=False,  # only relevant for image_size=128: crop to half resolution instead of subsampling, if image_size=128
        concat_coords=False,  # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
