
# Inpainting of Healthy Brain Tissue using 2D slice-wise Diffusion Models

Baseline model, based on https://arxiv.org/pdf/2402.17307.pdf by Durrer et al.

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
python3 scripts/inpainting_train.py --data_dir ./data_3d/training --log_dir ./model $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS

--data_dir        : training data directory
--log_dir         : where model checkpoints saved

```

## Sampling

To generate samples using the trained model, run

```
python scripts/inpainting_sample.py  --data_dir ./test_data --log_dir ./sampling_results --model_path ./model/savedmodel325000.pt --adapted_samples ./sampling_results_adapted $MODEL_FLAGS $DIFFUSION_FLAGS

--data_dir        : directory of the test data
--log_dir         : output directory
--model_path      : where model is saved
--adapted_samples : where post processed files are saved (post processing as required for the evaluation)

optional: --gt_dir ./test_data_gt 
--gt_dir          : directory of ground truth images corresponding to testing data (if eval_sam is used in file inpainting_sample.py, line 201)
```
