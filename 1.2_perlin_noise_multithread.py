import os
import numpy as np
import nibabel as nib
from SynthSegAAA.brain_generator import BrainGenerator
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
input_label_path = "data/uwa_aaa_combined"
image_path = "data/nnunet_raw/Dataset002_perlin_aaa/imagesTr"
label_path = "data/nnunet_raw/Dataset002_perlin_aaa/labelsTr"

# Create directories if they don't exist
os.makedirs(image_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)

# Get and sort label files
label_files = os.listdir(input_label_path)
label_files.sort()

def process_label_file(label_file, count_start):
    """
    Process a single label file, generating three brain images and saving them.
    Returns the updated count after processing.
    """
    brain_generator = BrainGenerator(
        os.path.join(input_label_path, label_file),
        generation_labels=np.array([0, 1, 2, 3]),
        output_labels=np.array([0, 1, 2, 0]),
    )
    count = count_start
    for i in range(3):
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
    return count

def main():
    # Limit to first 15 label files as in original script
    selected_files = label_files[:5]
    total_images = len(selected_files) * 1  # Each file generates 3 images

    # Calculate number of workers (use CPU count or a reasonable default)
    num_workers = min(multiprocessing.cpu_count(), len(selected_files))
    
    # Distribute work across threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Prepare tasks with initial count for each file
        tasks = []
        current_count = 0
        for label_file in selected_files:
            tasks.append(executor.submit(process_label_file, label_file, current_count))
            current_count += 3  # Increment count for next file (3 images per file)

        # Wait for all tasks to complete
        for future in tasks:
            future.result()  # Ensure all tasks complete and handle any exceptions

    print(f"Generated {total_images} images and labels.")

if __name__ == "__main__":
    main()