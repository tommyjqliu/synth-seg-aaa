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
    """
    Display NIfTI images and their overlays, interactively with ipywidgets if available,
    otherwise show the middle slice.

    options: List[
        Dict[
            "title": str,
            "image": nib.Nifti1Image | np.ndarray,
            "overlay": nib.Nifti1Image | np.ndarray | None,
            "is_label": bool,
            "is_overlay_label": bool,
        ]
    ]
    """
    def display_slice(slice_indices, axes, fig):
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
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
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
                axes[i].imshow(
                    overlay_data[:, :, slice_idx], cmap=cmap_overlay, alpha=0.5
                )

                # Add colorbar for overlay
                overlay_cbar = plt.colorbar(
                    axes[i].images[-1], ax=axes[i], fraction=0.046, pad=0.04
                )
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

    if IPYWIDGETS_AVAILABLE:
        # Interactive display with sliders
        def interactive_display(**kwargs):
            slice_indices = [kwargs[f"slice_index_{i}"] for i in range(len(options))]
            display_slice(slice_indices, axes, fig)
            plt.show()

        interact(
            interactive_display,
            **{
                f"slice_index_{i}": IntSlider(
                    min=0, max=options[i]["image"].shape[-1] - 1, step=1, value=0
                )
                for i in range(len(options))
            },
        )
    else:
        # Static display of middle slice
        slice_indices = [
            (opt["image"].shape[-1] if isinstance(opt["image"], np.ndarray) 
             else opt["image"].get_fdata().shape[-1]) // 2 
            for opt in options
        ]
        display_slice(slice_indices, axes, fig)
        plt.tight_layout()
        plt.show()