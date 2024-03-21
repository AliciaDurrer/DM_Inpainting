# Memory-efficent 3D Diffusion Model

extended and modified the work by Bieder et al., available at https://github.com/FlorentinBieder/PatchDDM-3D. for inpainting.

## Training

Adjust the "general settings", the COMMON & TRAIN flags in the run.sh file to reflect your data and setup.
To start the training, run:
```
bash run.sh
```
## Sampling

Adjust --GPU, --data_dir, --model_dir and --output_dir in the run.sh file. Model outputs will be saved in --output_dir. Final results (after postprocessing) will be saved in ./adapted_results (set as default in inpainting_sample.py).
To start the sampling, run:
```
bash run.sh
```
