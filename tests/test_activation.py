import sys
import os
import numpy as np
import pytest

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffnn.activation import Activation, Linear, ReLU, Sigmoid, Tanh, Softmax

# Test inputs
x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])

def test_activation_factory():
    """Test the activation factory method"""
    assert Activation.get_activation('linear') == Linear
    assert Activation.get_activation('relu') == ReLU
    assert Activation.get_activation('sigmoid') == Sigmoid
    assert Activation.get_activation('tanh') == Tanh
    assert Activation.get_activation('softmax') == Softmax
    
    # Test case-insensitivity
    assert Activation.get_activation('RELU') == ReLU
    
    # Test invalid activation name
    with pytest.raises(ValueError):
        Activation.get_activation('invalid_activation')

def test_linear():
    """Test linear activation function"""
    # Test forward pass
    output = Linear.activate(x)
    expected = x
    np.testing.assert_array_almost_equal(output, expected)
    
    # Test derivative
    derivative = Linear.derivative(x)
    expected = np.ones_like(x)
    np.testing.assert_array_almost_equal(derivative, expected)

def test_relu():
    """Test ReLU activation function"""
    # Test forward pass
    output = ReLU.activate(x)
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    np.testing.assert_array_almost_equal(output, expected)
    
    # Test derivative
    derivative = ReLU.derivative(x)
    expected = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(derivative, expected)

def test_sigmoid():
    """Test Sigmoid activation function"""
    # Test forward pass
    output = Sigmoid.activate(x)
    expected = 1 / (1 + np.exp(-x))
    np.testing.assert_array_almost_equal(output, expected)
    
    # Test derivative
    derivative = Sigmoid.derivative(x)
    s = Sigmoid.activate(x)
    expected = s * (1 - s)
    np.testing.assert_array_almost_equal(derivative, expected)

def test_tanh():
    """Test Tanh activation function"""
    # Test forward pass
    output = Tanh.activate(x)
    expected = np.tanh(x)
    np.testing.assert_array_almost_equal(output, expected)
    
    # Test derivative
    derivative = Tanh.derivative(x)
    expected = 1 - np.tanh(x)**2
    np.testing.assert_array_almost_equal(derivative, expected)

def test_softmax():
    """Test Softmax activation function"""
    # Test forward pass
    output = Softmax.activate(x)
    
    # Manually calculate expected softmax
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Check row sums equal 1
    assert np.allclose(np.sum(output, axis=1), np.ones(x.shape[0]))
    np.testing.assert_array_almost_equal(output, expected)
    
    # Basic check for derivative (simplified implementation)
    derivative = Softmax.derivative(x)
    assert derivative.shape == x.shape