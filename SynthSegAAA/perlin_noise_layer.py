import tensorflow as tf
import keras.layers as KL
import numpy as np
from noise import pnoise3  # Perlin Noise implementation

class PerlinNoise(KL.Layer):
    def __init__(self, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, amplitude=0.5, **kwargs):
        """Add 3D Perlin Noise to an input tensor."""
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.amplitude = amplitude  # Controls noise strength
        super(PerlinNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_dims = len(input_shape) - 2  # Exclude batch and channel dims
        self.shape = input_shape[1:-1]  # Spatial dimensions (e.g., [H, W, D])
        super(PerlinNoise, self).build(input_shape)

    def call(self, inputs):
        # Generate Perlin Noise as a NumPy array
        def generate_noise():
            noise_array = np.zeros(self.shape, dtype=np.float32)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        noise_array[i, j, k] = pnoise3(
                            i / self.scale, j / self.scale, k / self.scale,
                            octaves=self.octaves,
                            persistence=self.persistence,
                            lacunarity=self.lacunarity,
                            repeatx=self.shape[0],
                            repeaty=self.shape[1],
                            repeatz=self.shape[2],
                            base=42  # Seed for reproducibility
                        )
            # Normalize noise to [0, 1]
            noise_array = (noise_array + 1) / 2
            return noise_array

        # Convert noise to tensor and match input shape
        noise_tensor = tf.convert_to_tensor(generate_noise(), dtype=tf.float32)
        noise_tensor = tf.expand_dims(noise_tensor, axis=0)  # Add batch dim
        noise_tensor = tf.expand_dims(noise_tensor, axis=-1)  # Add channel dim
        noise_tensor = tf.tile(noise_tensor, [tf.shape(inputs)[0], 1, 1, 1, tf.shape(inputs)[-1]])  # Match batch/channels

        # Add noise to the input image
        return inputs + self.amplitude * noise_tensor

    def compute_output_shape(self, input_shape):
        return input_shape