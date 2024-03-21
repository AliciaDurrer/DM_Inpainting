# Medical Diffusion adapted for Inpainting

Extended and modified the version of the latent diffusion model by Khader et al. (https://github.com/FirasGit/medicaldiffusion/tree/master) for the task of inpainting.

# Setup

Create a virtual environment 
```
conda create -n medicaldiffusion python=3.8
``` 
and activate it with 
```
conda activate medicaldiffusion
```
Subsequently, download and install the required libraries by running 
```
pip install -r requirements.txt
```

# Training

First train the VQ-GAN (adjust as needed):

```
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=brats model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='t1n' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[4,4,4] model.num_workers=32 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=30000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1
```

Then train the DDPM (adjust as needed):
```
python train/train_ddpm.py model=ddpm dataset=brats_ddpm model.results_folder_postfix='t1n' model.vqgan_ckpt=/home/user/BRATS/t1n/lightning_logs/version_0/checkpoints/epoch-step-train/recon_loss.ckpt model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=16 model.dim_mults=[1,2,4,8] model.batch_size=16 model.gpus=1
```

# Sampling

Create a "models" folder.
Create a list of the testing files, adapt "data_dir", by using:
```
python3 create_datalist_testing.py

```
Adapt "root_dir" in config/dataset/brats_ddpm_test.yaml.
Adapt "results_folder" in config/model/ddpm.yaml.
Adapt GPU, "sys.path.insert", "output_folder" and "data_dir" in sampling.py. Run:

```
python3 sampling.py

```

The final samples will be saved in "results_sam_postprocessed".

