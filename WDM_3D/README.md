# WDM: 3D Wavelet Diffusion Models for High-Resolution Medical Image Synthesis Adapted for Inpainting

Extended and modified the work by Friedrich et al., available at https://github.com/pfriedri/wdm-3d, for the task of inpainting.

## Training

Adjust the "general settings", the COMMON & TRAIN flags in the run.sh file to reflect your data and setup.
To start the training, run:
```
bash run.sh
```

## Sampling

Adapt run.sh:

Set the GPU to use on the second line. SAMPLE="...": change --data_dir, --model_path and --output_dir to the corresponding paths.

Then run:

```
bash run.sh
```

the final postprocessed samples will be saved in the folder "samples_postprocessed".
