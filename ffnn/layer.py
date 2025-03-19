import numpy as np
from .activation import Activation
from .initialization import ZeroInitializer, RandomUniformInitializer, RandomNormalInitializer

class Layer:
    """
    Neural network layer implementation
    """
    def __init__(self, input_size, output_size, activation='linear', 
                 weight_initializer='uniform', **initializer_params):
        self.input_size = input_size
        self.output_size = output_size
        
        # Set activation function
        activation_class = Activation.get_activation(activation)
        self.activation = activation_class
        
        # Initialize weights and biases
        self._initialize_weights(weight_initializer, **initializer_params)
        
        # Storage for forward pass
        self.inputs = None
        self.z = None  # pre-activation
        self.output = None  # post-activation
        
        # Storage for gradients
        self.dW = None
        self.db = None
    
    def _initialize_weights(self, initializer_name, **params):
        # Initialize weights
        if initializer_name == 'zeros':
            self.W = ZeroInitializer.initialize((self.input_size, self.output_size))
        elif initializer_name == 'uniform':
            low = params.get('low', -0.05)
            high = params.get('high', 0.05)
            seed = params.get('seed', None)
            self.W = RandomUniformInitializer.initialize(
                (self.input_size, self.output_size), low, high, seed)
        elif initializer_name == 'normal':
            mean = params.get('mean', 0.0)
            std = params.get('std', 0.05)
            seed = params.get('seed', None)
            self.W = RandomNormalInitializer.initialize(
                (self.input_size, self.output_size), mean, std, seed)
        else:
            raise ValueError(f"Initializer '{initializer_name}' not supported")
            
        # Initialize biases to zero
        self.b = np.zeros((1, self.output_size))
    
    def forward(self, inputs):
        """Forward pass through the layer"""
        self.inputs = inputs
        self.z = np.dot(inputs, self.W) + self.b
        self.output = self.activation.activate(self.z)
        return self.output
    
    def backward(self, dvalues):
        """Backward pass to compute gradients"""
        # First determine the gradient of the activation function
        if self.activation.__class__.__name__ == 'Softmax':
            # Special case for softmax
            dz = dvalues
        else:
            # Get gradient of activation function
            dz = dvalues * self.activation.derivative(self.z)
        
        # Now compute gradients on weights and biases using dz
        self.dW = np.dot(self.inputs.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
            
        # Return gradient on values entering this layer
        return np.dot(dz, self.W.T)