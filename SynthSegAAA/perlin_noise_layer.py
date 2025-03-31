import tensorflow as tf
import keras.layers as KL
import numpy as np
from noise import pnoise3

class PerlinNoise(KL.Layer):
    def __init__(self, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, amplitude=0.5, threshold = 0.55 , **kwargs):
        """Add 3D Perlin Noise to an input tensor where labels = 0."""
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.amplitude = amplitude
        self.threshold = threshold
        super(PerlinNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape[0] is the image, input_shape[1] is the labels
        self.n_dims = len(input_shape[0]) - 2  # Exclude batch and channel dims from image
        self.shape = input_shape[0][1:-1]  # Spatial dimensions (e.g., [H, W, D])
        super(PerlinNoise, self).build(input_shape)

    def call(self, inputs):
        image, labels = inputs  # Expect [image, labels] as input

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
                            base=42
                        )
            # Normalize noise to [0, 1]
            noise_array = (noise_array + 1) / 2
            noise_array = np.where(noise_array > self.threshold, noise_array, 0)
            return noise_array

        # Convert noise to tensor and match image shape
        noise_tensor = tf.convert_to_tensor(generate_noise(), dtype=tf.float32)
        noise_tensor = tf.expand_dims(noise_tensor, axis=0)  # Add batch dim
        noise_tensor = tf.expand_dims(noise_tensor, axis=-1)  # Add channel dim
        noise_tensor = tf.tile(noise_tensor, [tf.shape(image)[0], 1, 1, 1, tf.shape(image)[-1]])  # Match batch/channels

        # Create a mask where labels = 0
        mask = tf.cast(tf.equal(labels, 0), tf.float32)  # 1 where label = 0, 0 elsewhere

        # Apply noise only where mask = 1
        noise_contribution = self.amplitude * noise_tensor * mask

        # Add masked noise to the image
        return image + noise_contribution

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Return image shape