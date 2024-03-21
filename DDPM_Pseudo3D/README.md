# Inpainting of Healthy Brain Tissue using 2.5D Diffusion Models

baseline model extended and modified using work by Zhu et al. (https://link.springer.com/chapter/10.1007/978-3-031-43999-5_56).

Set the flags as follows:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 1"

```
If required, adapt guided_diffusion/dist_util.py to reflect your cluster layout.

## Training

To train the model, run

```
python3 scripts/inpainting_train.py --data_dir ./data/training --log_dir ./model $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS

--data_dir        : training data directory
--log_dir         : where model checkpoints saved

```

## Sampling

To generate samples using the trained model, run

```
python scripts/inpainting_sample.py  --data_dir ./data/testing --log_dir ./sampling_results --model_path ./model/model.pt --adapted_samples ./sampling_results_adapted $MODEL_FLAGS $DIFFUSION_FLAGS

--data_dir        : directory of the test data
--log_dir         : output directory
--model_path      : where requested model is saved
--adapted_samples : where post processed files are saved (post processing = prepare for evaluation by normalizing the images to the individual ranges of the input data etc.)

optional: --gt_dir ./test_data_gt 
--gt_dir          : directory of ground truth images corresponding to testing data (if eval_sam is used in file inpainting_sample.py, line 198)

```
