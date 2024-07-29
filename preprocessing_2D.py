import os
import numpy as np
import nibabel as nib

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def crop_or_pad_image(image, target_shape):
    """
    Crop or pad the image to the target shape.
    """
    current_shape = image.shape
    cropped_padded_image = np.zeros(target_shape)

    # Calculate cropping/padding indices
    crop_start = [(current_dim - target_dim) // 2 if current_dim > target_dim else 0 for current_dim, target_dim in zip(current_shape, target_shape)]
    crop_end = [crop_start[i] + target_shape[i] for i in range(len(target_shape))]
    
    pad_start = [(target_dim - current_dim) // 2 if current_dim < target_dim else 0 for current_dim, target_dim in zip(current_shape, target_shape)]
    pad_end = [pad_start[i] + current_shape[i] for i in range(len(current_shape))]

    # Crop the image
    cropped_image = image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    # Pad the image
    cropped_padded_image[pad_start[0]:pad_end[0], pad_start[1]:pad_end[1], pad_start[2]:pad_end[2]] = cropped_image

    return cropped_padded_image

def process_images(input_folder, output_folder):
    target_shape = (224, 224, 224)

    for subdir, _, files in os.walk(input_folder):
        t1n_files = [f for f in files if f.endswith('t1n.nii.gz')] # t1 image
        healthy_files = [f for f in files if f.endswith('healthy.nii.gz')] # healthy mask

        for t1n_file in t1n_files:
            t1n_path = os.path.join(subdir, t1n_file)
            healthy_path = os.path.join(subdir, t1n_file.replace('t1n', 'mask-healthy'))

            if not os.path.exists(healthy_path):
                continue

            # Load images
            t1n_image = nib.load(t1n_path).get_fdata()
            healthy_image = nib.load(healthy_path).get_fdata()
            
            # Clipping
            t1n_image = np.clip(t1n_image, np.quantile(t1n_image, 0.001), np.quantile(t1n_image, 0.999))

            # Normalize images between 0 and 1
            t1n_image = normalize_image(t1n_image)
            healthy_image = normalize_image(healthy_image)

            # Crop or pad images to [224, 224, 224]
            t1n_image = crop_or_pad_image(t1n_image, target_shape)
            healthy_image = crop_or_pad_image(healthy_image, target_shape)

            # Iterate through each slice in the z-dimension
            for z in range(t1n_image.shape[2]):
                t1n_slice = t1n_image[:, :, z]
                healthy_slice = healthy_image[:, :, z]

                if np.any(healthy_slice != 0):  # Check if the mask slice contains non-zero values
                    # Create corresponding subdir in output folder with volume name and slice index
                    volume_name = os.path.splitext(os.path.splitext(t1n_file)[0])[0].replace('-t1n', '')  # Extract volume name without extensions and remove '-t1n'
                    output_subdir = os.path.join(output_folder, f'{volume_name}_slice_{z}')
                    os.makedirs(output_subdir, exist_ok=True)

                    # Save the processed slice
                    processed_t1n_slice_path = os.path.join(output_subdir, t1n_file)
                    processed_healthy_slice_path = os.path.join(output_subdir, t1n_file.replace('t1n.nii.gz', 'mask.nii.gz'))
                    nib.save(nib.Nifti1Image(t1n_slice, np.eye(4)), processed_t1n_slice_path)
                    nib.save(nib.Nifti1Image(healthy_slice, np.eye(4)), processed_healthy_slice_path)

                    # Mask out values in t1n_slice where healthy_slice == 1 (to create "voided" slice that needs inpainting)
                    voided_slice = t1n_slice.copy()
                    voided_slice[healthy_slice == 1] = 0

                    # Save the modified slice
                    modified_slice_path = os.path.join(output_subdir, t1n_file.replace('t1n.nii.gz', 'voided.nii.gz'))
                    nib.save(nib.Nifti1Image(voided_slice, np.eye(4)), modified_slice_path)

if __name__ == '__main__':
    input_folder = '/home/user/BraTS_data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training/'  # Replace with your actual input folder path
    output_folder = '/home/user/BraTS_data/preprocessed_training_data_2D_slices/'  # Replace with your actual output folder path
    process_images(input_folder, output_folder)
