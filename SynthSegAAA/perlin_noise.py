import numpy as np
import noise
from scipy.ndimage import label
from numpy.random import randint, uniform

def generate_perline_label(
    grid_size,
    scale=5.5,
    octaves=6,
    persistence=0.5,
    lacunarity=2.0,
    threshold=0.60,
    min_cloud_size=1000,
    seed=42
):
    # print("Generating Perlin noise with parameters:")
    # print(f"Grid size: {grid_size}")
    # print(f"Scale: {scale}")
    # print(f"Octaves: {octaves}")
    # print(f"Persistence: {persistence}")
    # print(f"Lacunarity: {lacunarity}")
    # print(f"Threshold: {threshold}")
    # print(f"Minimum cloud size: {min_cloud_size}")
    # print(f"Seed: {seed}")
    
    """
    Generate a 3D binary cloud map using Perlin noise with arbitrary grid dimensions.

    Parameters:
    - grid_size (tuple or list): 3D grid dimensions (size_x, size_y, size_z).
    - scale (float): Noise scale (larger values = smoother clouds).
    - octaves (int): Number of noise layers.
    - persistence (float): Amplitude reduction per octave.
    - lacunarity (float): Frequency increase per octave.
    - threshold (float): Density threshold for cloud presence (0 to 1).
    - min_cloud_size (int): Minimum number of voxels for a cloud to be kept.
    - seed (int): Random seed for reproducibility.

    Returns:
    - binary_map (ndarray): 3D binary array (0s and 1s) of shape (size_x, size_y, size_z).
    """
    # Extract grid dimensions
    size_x, size_y, size_z = grid_size

    # Create a 3D grid of coordinates
    x = np.linspace(0, scale, size_x)
    y = np.linspace(0, scale, size_y)
    z = np.linspace(0, scale, size_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Initialize the density array
    density = np.zeros((size_x, size_y, size_z))

    # Generate Perlin noise
    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                value = noise.pnoise3(
                    X[i, j, k], Y[i, j, k], Z[i, j, k],
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=size_x,
                    repeaty=size_y,
                    repeatz=size_z,
                    base=seed
                )
                # Normalize noise from [-1, 1] to [0, 1]
                density[i, j, k] = (value + 1) / 2

    # Step 1: Apply threshold to create a binary map
    binary_density = (density > threshold).astype(np.uint8)

    # Step 2: Remove small clouds using connected component analysis
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    labeled_array, num_features = label(binary_density, structure=structure)
    # Filter out small components
    filtered_density = np.zeros_like(binary_density)
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        if component_size >= min_cloud_size:
            filtered_density[labeled_array == i] = 1

    return filtered_density

def randomize_perlin_label(label, target = 0, layer = 1):
    classes = np.unique(label)
    max_class = classes[-1]
    output = label.copy()

    for i in range(layer):
        perline_label = generate_perline_label(
            label.shape,
            scale=uniform(4, 8),
            octaves=randint(5, 7),
            persistence=uniform(0.4, 0.6),
            lacunarity=uniform(1.8, 2.2),
            threshold=uniform(0.53, 0.63),
            min_cloud_size=randint(1000, 100000),
            seed=randint(0, 300)
            )
        output[(label == target) & (perline_label == 1)] = max_class + 1 + i
    
    return output.copy()
