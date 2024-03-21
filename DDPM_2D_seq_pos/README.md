
# Inpainting of Healthy Brain Tissue using slice-sequential position embedding-based Diffusion Models

Modification of the baseline model (baseline model was created by Durrer et al., https://arxiv.org/pdf/2402.17307.pdf), extended to be slice-sequential and containing positional embedding.

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

To sample, run:
```
python scripts/inpainting_sample.py --data_dir /home/user/data --log_dir ./log_folder --model_path ./model/savedmodel325000.pt --adapted_samples ./adapted_samples_325k $MODEL_FLAGS $DIFFUSION_FLAGS

--data_dir        : testing data directory
--log_dir         : where model outputs are saved
--model_dir       : where model checkpoint is saved
--adapted_samples : where post-processed model outputs are saved (post-processed as required for the evaluation)

```
