import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomSpatialDeformation(nn.Module):
    def __init__(
        self,
        scaling_bounds=0.15,
        rotation_bounds=10,
        shearing_bounds=0.02,
        translation_bounds=False,
        enable_90_rotations=False,
        nonlin_std=4.0,
        nonlin_scale=0.0625,
        inter_method="nearest",
        prob_deform=1,
        device="cpu",
    ):
        super(RandomSpatialDeformation, self).__init__()

        self.scaling_bounds = scaling_bounds
        self.rotation_bounds = rotation_bounds
        self.shearing_bounds = shearing_bounds
        self.translation_bounds = translation_bounds
        self.enable_90_rotations = enable_90_rotations
        self.nonlin_std = nonlin_std
        self.nonlin_scale = nonlin_scale
        self.inter_method = inter_method
        self.prob_deform = prob_deform
        self.device = device

        self.apply_affine_trans = (
            (self.scaling_bounds is not False)
            or (self.rotation_bounds is not False)
            or (self.shearing_bounds is not False)
            or (self.translation_bounds is not False)
            or self.enable_90_rotations
        )
        self.apply_elastic_trans = self.nonlin_std > 0

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        batch_size, *spatial_dims, _ = x[0].shape
        n_dims = len(spatial_dims)

        # Initialize identity transform
        identity = (
            torch.eye(n_dims + 1, dtype=x[0].dtype, device=x[0].device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Apply affine transformation
        if self.apply_affine_trans:
            affine_matrix = self.sample_affine_transform(batch_size, n_dims)
            transform = torch.matmul(identity, affine_matrix)
        else:
            transform = identity

        # Apply elastic transformation
        if self.apply_elastic_trans:
            elastic_transform = self.sample_elastic_transform(batch_size, spatial_dims)
            transform = transform + elastic_transform

        # Apply transformation with probability
        if torch.rand(1).item() < self.prob_deform:
            grid = F.affine_grid(
                transform[:, :n_dims, :], x[0].shape, align_corners=False
            )
            x = [
                F.grid_sample(tensor, grid, mode=self.inter_method, align_corners=False)
                for tensor in x
            ]

        return x[0] if len(x) == 1 else x

    def sample_affine_transform(self, batch_size, n_dims):
        affine_matrix = (
            torch.eye(n_dims + 1, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Implement scaling, rotation, shearing, and translation here
        # This is a simplified version and needs to be expanded based on the original implementation

        if self.scaling_bounds:
            scale = torch.rand(batch_size, n_dims) * (2 * self.scaling_bounds) + (
                1 - self.scaling_bounds
            )
            affine_matrix[:, :n_dims, :n_dims] *= scale.unsqueeze(2)

        if self.rotation_bounds:
            angle = (
                torch.rand(batch_size, 1) * (2 * self.rotation_bounds)
                - self.rotation_bounds
            )
            rot_mat = torch.zeros(batch_size, n_dims, n_dims, device=self.device)
            rot_mat[:, 0, 0] = rot_mat[:, 1, 1] = torch.cos(angle)
            rot_mat[:, 0, 1] = -torch.sin(angle)
            rot_mat[:, 1, 0] = torch.sin(angle)
            affine_matrix[:, :n_dims, :n_dims] = torch.matmul(
                affine_matrix[:, :n_dims, :n_dims], rot_mat
            )

        # Add shearing and translation implementations here

        return affine_matrix

    def sample_elastic_transform(self, batch_size, spatial_dims):
        small_shape = [max(int(dim * self.nonlin_scale), 3) for dim in spatial_dims]

        # Sample small displacement field
        displacement = (
            torch.randn(batch_size, len(spatial_dims), *small_shape) * self.nonlin_std
        )

        # Upsample to full size
        displacement = F.interpolate(
            displacement,
            size=spatial_dims,
            mode=self.inter_method,
        )

        # Convert displacement to transformation matrix
        identity = (
            torch.eye(
                len(spatial_dims) + 1,
                dtype=displacement.dtype,
                device=displacement.device,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        identity[:, : len(spatial_dims), -1] = (
            displacement.permute(0, 2, 3, 4, 1)
            .reshape(batch_size, -1, len(spatial_dims))
            .mean(dim=1)
        )

        return identity
