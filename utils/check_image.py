from typing import Dict, List
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

try:
    from ipywidgets import interact, IntSlider
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


def check_image(options: List[Dict]):
    if IPYWIDGETS_AVAILABLE:
        """
        options: list[
            dict[
                "title":str,
                "image" : nib.Nifti1Image | np.ndarray,
                "overlay" : nib.Nifti1Image | np.ndarray,
                "is_label":bool,
                "is_overlay_label":bool,
            ]
        ]
        """
        # Assume first image defines the dimensions
        def display_slice(**kwargs):
            fig_width_per_img = 4  # Width per image
            fig_height = 4  # Fixed height
            fig, axes = plt.subplots(
                1, len(options), figsize=(fig_width_per_img * len(options), fig_height)
            )

            if len(options) == 1:
                axes = [axes]
            for i, opt in enumerate(options):
                slice_idx = kwargs[f"slice_index_{i}"]
                img_data = (
                    opt["image"].get_fdata()
                    if isinstance(opt["image"], nib.Nifti1Image)
                    else opt["image"]
                )
                cmap_img = "gray" if not opt.get("is_label", False) else "jet"

                # Display base image
                im = axes[i].imshow(img_data[:, :, slice_idx], cmap=cmap_img)
                axes[i].set_title(opt.get("title", f"Image {i}"))
                axes[i].axis("off")

                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                if opt.get("is_label", False):
                    cbar.set_label("Label Value")

                # Check if overlay exists and display it
                overlay = opt.get("overlay", None)
                if overlay is not None:
                    overlay_data = overlay.get_fdata()
                    cmap_overlay = "jet" if opt.get("is_overlay_label", False) else "hot"
                    # Overlay with transparency (alpha)
                    axes[i].imshow(
                        overlay_data[:, :, slice_idx], cmap=cmap_overlay, alpha=0.5
                    )

                    # Optionally, add a colorbar for the overlay
                    overlay_cbar = plt.colorbar(
                        axes[i].images[-1], ax=axes[i], fraction=0.046, pad=0.04
                    )
                    if opt.get("is_overlay_label", False):
                        overlay_cbar.set_label("Overlay Value")

            plt.suptitle(f"NIfTI Image Slice {slice_idx}")
            plt.show()

        interact(
            display_slice,
            **{
                f"slice_index_{i}": IntSlider(
                    min=0, max=options[i]["image"].shape[-1] - 1, step=1, value=0
                )
                for i in range(len(options))
            },
        )
    else:
        def display_slice(slice_indices, axes, fig):
            # Clear previous content
            for ax in axes:
                ax.clear()

            for i, opt in enumerate(options):
                img_data = (
                    opt["image"].get_fdata()
                    if isinstance(opt["image"], nib.Nifti1Image)
                    else opt["image"]
                )
                slice_idx = slice_indices[i]
                cmap_img = "gray" if not opt.get("is_label", False) else "jet"

                # Display base image
                im = axes[i].imshow(img_data[:, :, slice_idx], cmap=cmap_img)
                axes[i].set_title(opt.get("title", f"Image {i}"))
                axes[i].axis("off")

                # Add colorbar for base image
                cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                if opt.get("is_label", False):
                    cbar.set_label("Label Value")

                # Check and display overlay if it exists
                overlay = opt.get("overlay", None)
                if overlay is not None:
                    overlay_data = (
                        overlay.get_fdata()
                        if isinstance(overlay, nib.Nifti1Image)
                        else overlay
                    )
                    cmap_overlay = "jet" if opt.get("is_overlay_label", False) else "hot"
                    overlay_im = axes[i].imshow(
                        overlay_data[:, :, slice_idx], cmap=cmap_overlay, alpha=0.5
                    )

                    # Add colorbar for overlay
                    overlay_cbar = fig.colorbar(overlay_im, ax=axes[i], fraction=0.046, pad=0.04)
                    if opt.get("is_overlay_label", False):
                        overlay_cbar.set_label("Overlay Value")

            plt.suptitle(f"NIfTI Image Slice {slice_idx}")
            fig.canvas.draw_idle()

        # Set up figure
        fig_width_per_img = 4
        fig_height = 4
        fig, axes = plt.subplots(
            1, len(options), figsize=(fig_width_per_img * len(options), fig_height)
        )
        if len(options) == 1:
            axes = [axes]
        
        slice_indices = [
            (
                opt["image"].shape[-1]
                if isinstance(opt["image"], np.ndarray)
                else opt["image"].get_fdata().shape[-1]
            ) // 2
            for opt in options
        ]
        display_slice(slice_indices, axes, fig)
        plt.tight_layout()
        plt.show()

