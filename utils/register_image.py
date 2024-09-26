import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatText, Checkbox, IntText, HBox, VBox, Output
import numpy as np
from scipy.ndimage import affine_transform, zoom as zoom_function

def register_image(moving, static):
    # Get dimensions of static and moving images
    static_data = static.get_fdata()
    moving_data = moving.get_fdata()

    static_shape = static_data.shape
    moving_shape = moving_data.shape
    num_slices = static_shape[-1]  # Use static shape for number of slices

    # Calculate the scaling factors to map moving image into static space
    scale_factors = np.array(static_shape) / np.array(moving_shape)

    def apply_transformations(img_data, moving_x, moving_y, moving_z, zoom_x, zoom_y, zoom_z, flip_x, flip_y, flip_z):
        # Step 1: Rescale the moving image to fit within the static space
        img_data_rescaled = zoom_function(img_data, scale_factors)

        # Step 2: Apply translation (moving) in x, y, z directions
        translation = np.eye(4)  # Identity matrix for affine transformation
        translation[:3, 3] = [moving_x, moving_y, moving_z]  # Add translation vector

        # Apply affine transformation for translation
        img_data_transformed = affine_transform(img_data_rescaled, translation)

        # Step 3: Apply zoom in x, y, z directions
        zoom_factors = [zoom_x, zoom_y, zoom_z]
        img_data_transformed = zoom_function(img_data_transformed, zoom_factors)

        # Step 4: Apply flipping along x, y, z axes
        if flip_x:
            img_data_transformed = np.flip(img_data_transformed, axis=0)
        if flip_y:
            img_data_transformed = np.flip(img_data_transformed, axis=1)
        if flip_z:
            img_data_transformed = np.flip(img_data_transformed, axis=2)

        return img_data_transformed

    def display_slice(slice_idx, moving_x, moving_y, moving_z, zoom_x, zoom_y, zoom_z, flip_x, flip_y, flip_z):
        # Apply transformations to the moving image
        transformed_moving = apply_transformations(
            moving_data, moving_x, moving_y, moving_z, zoom_x, zoom_y, zoom_z, flip_x, flip_y, flip_z
        )

        # Ensure the transformed moving image has the same size as the static image
        transformed_moving_resized = np.zeros_like(static_data)
        # Assuming the transformed image fits inside the static image, overlay it
        insert_slices = tuple(slice(0, min(ts, ss)) for ts, ss in zip(transformed_moving.shape, static_shape))
        transformed_moving_resized[insert_slices] = transformed_moving[insert_slices]

        # Plot the images side by side
        output = Output()
        with output:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            cmap_img = 'gray'

            # Static image
            axes[0].imshow(static_data[:, :, slice_idx], cmap=cmap_img)
            axes[0].set_title("Static Image")
            axes[0].axis("off")

            # Transformed and resized moving image
            axes[1].imshow(transformed_moving_resized[:, :, slice_idx], cmap=cmap_img)
            axes[1].set_title("Transformed Moving Image")
            axes[1].axis("off")

            plt.suptitle(f"Slice {slice_idx}")
            plt.show()

        return output

    # Interactive components
    slice_idx = IntText(value=0, min=0, max=num_slices - 1, description='Slice:')
    moving_x = FloatText(value=0.0, description='Move X:')
    moving_y = FloatText(value=0.0, description='Move Y:')
    moving_z = FloatText(value=0.0, description='Move Z:')
    zoom_x = FloatText(value=1.0, description='Zoom X:')
    zoom_y = FloatText(value=1.0, description='Zoom Y:')
    zoom_z = FloatText(value=1.0, description='Zoom Z:')
    flip_x = Checkbox(value=False, description='Flip X')
    flip_y = Checkbox(value=False, description='Flip Y')
    flip_z = Checkbox(value=False, description='Flip Z')

    # Organize the layout: place the input fields to the right of the plot
    input_widgets = VBox([slice_idx, moving_x, moving_y, moving_z, zoom_x, zoom_y, zoom_z, flip_x, flip_y, flip_z])

    # Create an interactive function that updates the plot
    @interact
    def update_display(slice_idx=slice_idx, moving_x=moving_x, moving_y=moving_y, moving_z=moving_z, 
                       zoom_x=zoom_x, zoom_y=zoom_y, zoom_z=zoom_z, 
                       flip_x=flip_x, flip_y=flip_y, flip_z=flip_z):
        output_plot = display_slice(slice_idx, moving_x, moving_y, moving_z, zoom_x, zoom_y, zoom_z, flip_x, flip_y, flip_z)
        display(HBox([output_plot, input_widgets]))

