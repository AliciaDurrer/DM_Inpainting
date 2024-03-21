import os
import sys

import numpy as np
import torch

base_path = "/home/user/LDM_3D/"

sys.path.insert(
    0,
    os.path.abspath(base_path),
)
from re import I

sys.path.append("../")
import create_submission_adapted
import evaluation.pytorch_ssim
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
from ddpm import GaussianDiffusion, Trainer, Unet3D
from hydra import compose, initialize, initialize_config_dir, initialize_config_module
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from train.get_dataset import get_dataset
from vq_gan_3d.model.vqgan import VQGAN


def cycle(dl):
    while True:
        for data in dl:
            yield data


USE_DATASET = "BRATS23"

if USE_DATASET == "BRATS23":
    DDPM_CHECKPOINT = base_path + "/models/model-ddpm.pt"
    VQGAN_CHECKPOINT = base_path + "/models/model-vqgan.ckpt"

    with initialize(config_path="../LDM_3D/config"):
        cfg = compose(
            config_name="base_cfg.yaml",
            overrides=[
                "model=ddpm",
                "dataset=brats_ddpm_test",
                f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
                "model.diffusion_img_size=32",
                "model.diffusion_depth_size=32",
                "model.diffusion_num_channels=16",
                "model.dim_mults=[1,2,4,8]",
                "model.batch_size=2",
                "model.gpus=1",
            ],
        )


################################################# VQGAN only ####################################################################

test_ds = get_dataset(cfg)

vqgan = VQGAN.load_from_checkpoint(VQGAN_CHECKPOINT)
vqgan = vqgan.cuda()
vqgan.eval()

if test_ds:
    ds = test_ds
else:
    assert folder is not None, "Provide a folder path to the dataset"
    ds = Dataset(folder, image_size, channels=channels, num_frames=num_frames)
dl = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True)
dl = cycle(dl)

output_folder = base_path + "/results"
data_dir = "/home/user/testing/"
dir_path = str(data_dir)
count = 0
for path in os.listdir(dir_path):
    if os.path.isdir(os.path.join(dir_path, path)):
        count += 1

i = 0
for i in range(0, count):
    print("sampling file ", str(i + 1), " of ", str(count), "...")

    data, path = next(dl)

    data = torch.tensor(data).cuda()

    filename_new = str(path).split("/")[-1]
    foldername_new = filename_new.split("-")[2] + "-" + filename_new.split("-")[3]

    input_data = data[:, :2, :, :, :].unsqueeze(1)

    input_data_np = np.asarray(input_data.detach().cpu())

    ################################################# VQGAN & DDPM ####################################################################

    test_ds = get_dataset(cfg)

    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=test_ds,
        train_batch_size=1,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        # logger=cfg.model.logger
    )

    trainer.load(DDPM_CHECKPOINT, map_location="cuda:0")
    sample = trainer.ema_model.sample(input_data, batch_size=1)
    sample = np.asarray(sample.detach().cpu())
    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

    nib.save(
        nib.Nifti1Image(sample[0][0], None),
        str(output_folder)
        + "/BraTS-GLI-"
        + str(foldername_new)
        + "-t1n-inference.nii.gz",
    )

    i += 1

create_submission_adapted.adapt(
    str(data_dir), str(output_folder), "results_sam_postprocessed"
)
