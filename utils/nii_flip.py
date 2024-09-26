import nibabel as nib
import numpy as np

def nii_flip(nii_image, axis):
    """
    Flip a NIfTI image along the desired axis.
    """
    #Get the image data as a numpy array
    image_data = nii_image.get_fdata()
    # Flip the image along the desired axis (0: x-axis, 1: y-axis, 2: z-axis)
    # For example, flipping along the left-right (x-axis)
    flipped_data = np.flip(image_data, axis)
    # Create a new NIfTI image using the flipped data and the original header
    return nib.Nifti1Image(flipped_data, nii_image.affine, nii_image.header)