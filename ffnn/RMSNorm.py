import numpy as np

class RMSNorm:
    def __init__(self, size, epsilon=1e-8):
        """
        Initialize RMSNorm.
        Args:
            size (int): The size of the layer (number of neurons).
            epsilon (float): Small value to avoid division by zero.
        """
        self.epsilon = epsilon
        self.size = size
        self.scale = np.ones(size)  # Scaling parameter (gamma)

    def forward(self, x):
        """
        Perform RMSNorm on input x.
        Args:
            x (np.array): Input tensor.
        Returns:
            np.array: Normalized tensor.
        """
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return self.scale * (x / rms)

    def update_scale(self, new_scale):
        """
        Update the scaling parameter (gamma).
        Args:
            new_scale (np.array): New scaling values.
        """
        if new_scale.shape == self.scale.shape:
            self.scale = new_scale
        else:
            raise ValueError("New scale must match the size of the layer.")
