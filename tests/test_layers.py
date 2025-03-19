import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffnn.layer import Layer
from ffnn.activation import Activation

class TestLayer:
    def setup_method(self):
        # Common setup for tests
        self.input_size = 3
        self.output_size = 2
        self.batch_size = 4
        self.input_data = np.random.random((self.batch_size, self.input_size))
    
    def test_layer_initialization(self):
        """Test layer initialization with different configurations"""
        # Test with default parameters
        layer = Layer(self.input_size, self.output_size)
        assert layer.W.shape == (self.input_size, self.output_size)
        assert layer.b.shape == (1, self.output_size)
        
        # Test with zeros initializer
        layer = Layer(self.input_size, self.output_size, weight_initializer='zeros')
        assert np.all(layer.W == 0)
        
        # Test with uniform initializer and custom bounds
        layer = Layer(self.input_size, self.output_size, 
                     weight_initializer='uniform', low=-0.1, high=0.1, seed=42)
        assert np.all((layer.W >= -0.1) & (layer.W <= 0.1))
        
        # Test with normal initializer
        layer = Layer(self.input_size, self.output_size,
                     weight_initializer='normal', mean=0.0, std=0.1, seed=42)
        
        # Test with custom activation
        layer = Layer(self.input_size, self.output_size, activation='sigmoid')
        assert layer.activation.__name__ == 'Sigmoid'
    
    def test_forward_pass(self):
        """Test forward propagation"""
        # Create layer with fixed weights for deterministic testing
        layer = Layer(self.input_size, self.output_size, weight_initializer='zeros')
        layer.W = np.ones((self.input_size, self.output_size))
        
        # Forward pass
        output = layer.forward(self.input_data)
        
        # Expected output with all-ones weights and zero bias: sum of inputs
        expected_z = np.sum(self.input_data, axis=1, keepdims=True) * np.ones((1, self.output_size))
        expected_output = expected_z  # Linear activation
        
        np.testing.assert_array_almost_equal(layer.z, expected_z)
        np.testing.assert_array_almost_equal(output, expected_output)
        
        # Test with ReLU activation
        layer = Layer(self.input_size, self.output_size, activation='relu', weight_initializer='zeros')
        layer.W = -np.ones((self.input_size, self.output_size))  # Negative weights
        output = layer.forward(self.input_data)
        
        # Expected output should be zeros due to ReLU activation on negative values
        expected_z = -np.sum(self.input_data, axis=1, keepdims=True) * np.ones((1, self.output_size))
        expected_output = np.zeros_like(expected_z)  # ReLU activation
        
        np.testing.assert_array_almost_equal(layer.z, expected_z)
        np.testing.assert_array_almost_equal(output, expected_output)
    
    def test_backward_pass(self):
        """Test backward propagation"""
        # Create layer with fixed weights
        layer = Layer(self.input_size, self.output_size, weight_initializer='zeros')
        layer.W = np.ones((self.input_size, self.output_size))
        
        # Forward pass to set up internal state
        layer.forward(self.input_data)
        
        # Create upstream gradient
        dvalues = np.ones((self.batch_size, self.output_size))
        
        # Backward pass
        dinputs = layer.backward(dvalues)
        
        # For linear activation and all-ones weights:
        # dW should be input_data.T @ dvalues
        # db should be sum of dvalues
        # dinputs should be dvalues @ W.T
        expected_dW = np.dot(self.input_data.T, dvalues)
        expected_db = np.sum(dvalues, axis=0, keepdims=True)
        expected_dinputs = np.dot(dvalues, layer.W.T)
        
        np.testing.assert_array_almost_equal(layer.dW, expected_dW)
        np.testing.assert_array_almost_equal(layer.db, expected_db)
        np.testing.assert_array_almost_equal(dinputs, expected_dinputs)
        
        # Test with ReLU activation
        layer = Layer(self.input_size, self.output_size, activation='relu', weight_initializer='zeros')
        layer.W = -np.ones((self.input_size, self.output_size))  # Negative weights
        layer.forward(self.input_data)
        
        # Backward pass
        dinputs = layer.backward(dvalues)
        
        # dW should be zero since ReLU derivative is zero for negative values
        expected_dW = np.zeros((self.input_size, self.output_size))
        expected_db = np.zeros((1, self.output_size))
        expected_dinputs = np.zeros((self.batch_size, self.input_size))
        
        np.testing.assert_array_almost_equal(layer.dW, expected_dW)
        np.testing.assert_array_almost_equal(layer.db, expected_db)
        np.testing.assert_array_almost_equal(dinputs, expected_dinputs)