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
        self.normalized_x = None
        self.x = None
        self.rms = None
        self.learning_rate = 0.01 

    def forward(self, x):
        """
        Perform RMSNorm on input x.
        Args:
            x (np.array): Input tensor.
        Returns:
            np.array: Normalized tensor.
        """
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        self.rms = rms
        self.normalized_x = x / rms
        self.x = x
        return self.scale * (x / rms)
    
    def backward(self, grad_output):
        """
        Compute the gradient of the loss with respect to the input x and scale (gamma).
        Args:
            grad_output (np.array): Gradient of the loss with respect to the output.
        Returns:
            np.array: Gradient with respect to the input x.
            np.array: Gradient with respect to the scaling parameter (gamma).
        """
        # Gradient w.r.t. scale (gamma)
        grad_scale = np.sum(grad_output * self.normalized_x, axis=0)

        # Gradient w.r.t. normalized input
        grad_normalized_x = grad_output * self.scale

        # Gradient w.r.t. input (x)
        grad_rms = -np.sum(self.x * grad_normalized_x, axis=-1, keepdims=True) / (self.rms ** 2)
        grad_x = grad_normalized_x / self.rms + (2 / self.size) * self.x * grad_rms

        return grad_x, grad_scale

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
