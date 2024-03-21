# general settings
GPU=1;            # set to 01 for using gpu 0 and 1
SEED=0;           # randomness seed for sampling
CHANNELS=29;      # number of model channels
MODE='test';     # train vs sample
DATA_MODE="test" # train, test, validation data
MODEL='fullres'; #'patchddm', 'fullres', 'halfres'

# settings for sampling/inference
ITERATIONS=001;  # training iteration (as amultiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;# number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="MonXX_HH-MM-SS_HOSTNAME"; # tensorboard dir to be set for the evaluation


# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'patchddm' ]]; then
  echo "PATCHDDM";
  IN_CHANNELS=8;
  CONCAT_COORDS=True;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=256;
  HALF_RES_CROP_TRAIN=True;
elif [[ $MODEL == 'fullres' ]]; then
  echo "FULLRES";
  IN_CHANNELS=3;
  CONCAT_COORDS=False;
  IMAGE_SIZE_TRAIN=256;
  IMAGE_SIZE_SAMPLE=256;
  HALF_RES_CROP_TRAIN=False;
elif [[ $MODEL == 'halfres' ]]; then
  echo "HALFRES";
  IN_CHANNELS=5;
  CONCAT_COORDS=False;
  IMAGE_SIZE_TRAIN=128;
  IMAGE_SIZE_SAMPLE=128;
  HALF_RES_CROP_TRAIN=False;
else
  echo "MODEL TYPE NOT FOUND";
fi

COMMON="
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=1,3,4,4,4,4,4
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=1
--num_groups=${CHANNELS}
--in_channels=${IN_CHANNELS}
--concat_coords=${CONCAT_COORDS}
--out_channels=1
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=True
--decoder_device_thresh=15
"
TRAIN="
--data_dir=/home/user/data
--image_size=${IMAGE_SIZE_TRAIN}
--half_res_crop=${HALF_RES_CROP_TRAIN}
--use_fp16=True
--lr=1e-5
--save_interval=5000
--num_workers=8
--devices=${GPU}
"
SAMPLE="
--data_dir=/home/user/data
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE_SAMPLE}
--half_res_crop=False
--use_fp16=False
--model_path=./model/model.pt
--devices=${GPU}
--output_dir=./output_dir
--num_samples=40
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
"
if [[ $MODE == 'train' ]]; then
  python scripts/inpainting_train.py $TRAIN $COMMON;
else
  python scripts/inpainting_sample.py $SAMPLE $COMMON;
fi
