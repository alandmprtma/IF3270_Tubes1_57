import numpy as np
from .activation import Activation
from .initialization import ZeroInitializer, RandomUniformInitializer, RandomNormalInitializer, XavierInitializer, HeInitializer
from .RMSNorm import RMSNorm

class Layer:
    """
    Implementasi layer dalam neural network
    """
    def __init__(self, input_size, output_size, activation='linear', 
                 weight_initializer='uniform', rmsnorm=False, **initializer_params):
        self.input_size = input_size
        self.output_size = output_size
        
        # Ngeset activation function
        activation_class = Activation.get_activation(activation)
        self.activation = activation_class()
        
        # Ngeinisialisasi bobot dan bias
        self._initialize_weights(weight_initializer, **initializer_params)
        
        # Penyimpanan buat forward prog
        self.inputs = None
        self.z = None  # pre-activation
        self.output = None  # post-activation
        
        # Buat nyimpen gradient
        self.dW = None
        self.db = None

        self.rmsnorm = RMSNorm(output_size) if rmsnorm else None
    
    def _initialize_weights(self, initializer_name, **params):
        """Buat menginisialisasi bobot"""
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
        elif initializer_name == 'xavier':
            self.W = XavierInitializer.initialize((self.input_size, self.output_size))
        elif initializer_name == 'he':
            self.W = HeInitializer.initialize((self.input_size, self.output_size))
        else:
            raise ValueError(f"Initializer '{initializer_name}' not supported")
        
        # Ngeinisialisasi bias
        self.b = np.zeros((1, self.output_size))
    
    def forward(self, inputs):
        """Ngelakuin forward propagation di layer"""
        self.inputs = inputs
        self.z = np.dot(inputs, self.W) + self.b

        if self.rmsnorm:
            self.z = self.rmsnorm.forward(self.z)

        self.output = self.activation.activate(self.z)
        return self.output
    
    def backward(self, dvalues):
        """Ngelakuin backward propagation buat dapetin gradient"""
        if self.activation.__class__.__name__ == 'Softmax':
            # Kasus khusus softmax
            dz = dvalues
        else:
            dz = dvalues * self.activation.derivative(self.z)
        
        if self.rmsnorm:
            dz = self.rmsnorm.backward(dz)

        self.dW = np.dot(self.inputs.T, dz)
        self.db = np.sum(dz, axis=0, keepdims=True)
            
        return np.dot(dz, self.W.T)