import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider


def check_image(
    options: list[
        dict[
            "title":str,
            "image" : nib.Nifti1Image,
            "overlay" : nib.Nifti1Image,
            "is_label":bool,
            "is_overlay_label":bool,
        ]
    ]
):
    # Assume first image defines the dimensions
    def display_slice(**kwargs):
        fig_width_per_img = 6  # Width per image
        fig_height = 6  # Fixed height
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
