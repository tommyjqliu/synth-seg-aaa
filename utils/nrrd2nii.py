import nrrd
import nibabel as nib
import numpy as np


def nrrd2nii(nrrd_file_path, nii_output_path=None):
    data, header = nrrd.read(nrrd_file_path)
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(data, affine)

    if nii_output_path is not None:
        nib.save(nii_img, nii_output_path)

    return nii_img
