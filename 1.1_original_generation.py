import os
import numpy as np
import nibabel as nib
from SynthSeg.brain_generator import BrainGenerator

input_label_path = "data/uwa_aaa_combined"
image_path = "data/Dataset001_aaa/imagesTr"
label_path = "data/Dataset001_aaa/labelsTr"
os.makedirs(image_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)
label_files = os.listdir(input_label_path)
count = 0

for label_file in label_files[:5]:
    print(label_file)
    brain_generator = BrainGenerator(os.path.join(input_label_path, label_file))
    for i in range(1):
        image, label = brain_generator.generate_brain()
        nib.save(
            nib.Nifti1Image(image, np.eye(4)),
            os.path.join(image_path, f"aaa_{count:03d}_0000.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(label, np.eye(4)),
            os.path.join(label_path, f"aaa_{count:03d}.nii.gz"),
        )
        count += 1