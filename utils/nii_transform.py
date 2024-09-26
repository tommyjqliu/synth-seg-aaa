import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def nii_transform(nii_image, operation):
    # Get the image data as a numpy array
    image_data = nii_image.get_fdata()
    # Flip the image along the desired axis (0: x-axis, 1: y-axis, 2: z-axis)
    # For example, flipping along the left-right (x-axis)
    processed_data = operation(image_data).copy()
    # Create a new NIfTI image using the flipped data and the original header
    return nib.Nifti1Image(processed_data, nii_image.affine, nii_image.header)


def label_transform(
    label,
    target_shape,
    offset_x=0,
    offset_y=0,
    offset_z=0,
    zoom_x=1,
    zoom_y=1,
    zoom_z=1,
    flip_x=False,
    flip_y=False,
    flip_z=False,
):
    base_image = np.zeros(target_shape)

    transform_moving = zoom(
        label,
        (
            zoom_x if zoom_x > 0 else 1,
            zoom_y if zoom_y > 0 else 1,
            zoom_z if zoom_z > 0 else 1,
        ),
    )
    if flip_x:
        transform_moving = np.flip(transform_moving, axis=0)
    if flip_y:
        transform_moving = np.flip(transform_moving, axis=1)
    if flip_z:
        transform_moving = np.flip(transform_moving, axis=2)

    base_image[
        int(offset_x) : int(offset_x) + transform_moving.shape[0],
        int(offset_y) : int(offset_y) + transform_moving.shape[1],
        int(offset_z) : int(offset_z) + transform_moving.shape[2],
    ] = transform_moving

    return base_image