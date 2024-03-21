# general settings
GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='sample';            # train vs sample
DATA_MODE="test"        # train, test, validation data
MODEL='ours_unet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

# settings for sampling/inference
ITERATIONS=0;      # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;  # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="dir";     # tensorboard dir to be set for the evaluation

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "Ours (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "Ours (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE_TRAIN=256;
  IMAGE_SIZE_SAMPLE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "Ours (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "Ours (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE_TRAIN=256;
  IMAGE_SIZE_SAMPLE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
else
  echo "MODEL TYPE NOT FOUND";
fi

COMMON="
--dataset=brats3d
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=24
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
--use_wgupdown=False
"
TRAIN="
--data_dir=/home/user/data/training/
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE_TRAIN}
--use_fp16=False
--lr=1e-5
--save_interval=5000
--num_workers=12
--devices=${GPU}
"
SAMPLE="
--data_dir=/home/user/data/testing/
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE_SAMPLE}
--use_fp16=False
--model_path=./model/model.pt
--devices=${GPU}
--output_dir=./sampling_output/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train.py $TRAIN $COMMON;
else
  python scripts/generation_sample.py $SAMPLE $COMMON;
  python scripts/postprocess_samples.py $SAMPLE;
fi
