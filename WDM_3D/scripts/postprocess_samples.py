import os
import sys
sys.path.append(".")
import numpy as np
import torch as th
import nibabel as nib
import argparse
from PIL import ImageFilter
from scipy.ndimage import gaussian_filter1d
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def create_argparser():
    defaults = dict(
    data_dir=None,
    data_mode=None,
    seed=None,
    image_size=None,
    use_fp16=None,
    model_path=None,
    devices=None,
    output_dir=None,
    num_samples=None,
    use_ddim=None,
    sampling_steps=None,
    clip_denoised=None,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args()

def main(input_data=args.data_dir, samples_dir=args.output_dir, adapted_samples_dir='samples_postprocessed'):

  if os.path.exists(samples_dir + '/log.txt'):
        os.remove(samples_dir + '/log.txt')

  if os.path.exists(samples_dir + '/progress.csv'):
        os.remove(samples_dir + '/progress.csv')

  for root, dirs, files in os.walk(samples_dir):

    for file_id in files:
    
      print('file', file_id)

      folder_name = file_id.split('_')[1].split('.')[0] 
      
      print('folder name', folder_name)

      nifti_file = nib.load(str(root) + '/' + str(file_id))
      
      org_nifti = nib.load(str(input_data) + '/BraTS-GLI-' + str(folder_name) + '/BraTS-GLI-' + str(folder_name) + '-t1n-voided.nii.gz')
      target_header = org_nifti.header

      np_org = np.asarray(org_nifti.dataobj)
      np_org_clipped = np.percentile(np_org, [0.5,99.5])
         
      np_redef = np.asarray(nifti_file.dataobj)  
      np_redef[np.isnan(np_redef)] = 0
  
      # normalize between ...
      
      clipped_image = np.clip(np_redef, 0, 1)
      start = np.min(np_org_clipped)
      end = np.max(np_org_clipped)
      width = end - start
      norm_img = (clipped_image - clipped_image.min())/(clipped_image.max() - clipped_image.min()) * width + start
      
      # reshape to [240, 240, 155]
      
      cropped_image = norm_img[8:-8,8:-8, 50:-51] 
      
      if not os.path.exists(adapted_samples_dir):
        os.makedirs(adapted_samples_dir)
         
      print('postprocessed and saved as: ', str(adapted_samples_dir) + '/BraTS-GLI-' + str(folder_name)+ '-t1n-inference.nii.gz')
      nib.save(nib.Nifti1Image(cropped_image, None, target_header), str(adapted_samples_dir) + '/BraTS-GLI-' + str(folder_name)+ '-t1n-inference.nii.gz')
      
    break

if __name__ == "__main__":
    main()

