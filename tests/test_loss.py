import sys
import os
import numpy as np
import pytest

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffnn.loss import Loss, MSE, BinaryCrossEntropy, CategoricalCrossEntropy

# Test data
y_true_binary = np.array([[1, 0], [0, 1]])
y_pred_binary = np.array([[0.8, 0.2], [0.3, 0.7]])

y_true_categorical = np.array([[0, 1, 0], [1, 0, 0]])
y_pred_categorical = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])

def test_loss_factory():
    """Test the loss factory method"""
    assert Loss.get_loss('mse') == MSE
    assert Loss.get_loss('binary_crossentropy') == BinaryCrossEntropy
    assert Loss.get_loss('categorical_crossentropy') == CategoricalCrossEntropy
    
    # Test case-insensitivity
    assert Loss.get_loss('MSE') == MSE
    
    # Test invalid loss name
    with pytest.raises(ValueError):
        Loss.get_loss('invalid_loss')

def test_mse():
    """Test MSE loss function"""
    # Simple test with a single value
    y_true = np.array([1.0])
    y_pred = np.array([0.8])
    
    # Test forward calculation
    loss = MSE.calculate(y_true, y_pred)
    expected_loss = np.mean(np.square(y_true - y_pred))
    assert np.isclose(loss, expected_loss)
    
    # Test derivative
    derivative = MSE.derivative(y_true, y_pred)
    expected_derivative = 2 * (y_pred - y_true) / y_true.shape[0]
    np.testing.assert_array_almost_equal(derivative, expected_derivative)

def test_binary_crossentropy():
    """Test Binary Cross Entropy loss function"""
    # Test forward calculation
    loss = BinaryCrossEntropy.calculate(y_true_binary, y_pred_binary)
    
    # Manual calculation with epsilon
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_binary, epsilon, 1 - epsilon)
    expected_loss = -np.mean(y_true_binary * np.log(y_pred_clipped) + (1 - y_true_binary) * np.log(1 - y_pred_clipped))
    
    assert np.isclose(loss, expected_loss)
    
    # Test derivative
    derivative = BinaryCrossEntropy.derivative(y_true_binary, y_pred_binary)
    assert derivative.shape == y_true_binary.shape

def test_categorical_crossentropy():
    """Test Categorical Cross Entropy loss function"""
    # Test forward calculation
    loss = CategoricalCrossEntropy.calculate(y_true_categorical, y_pred_categorical)
    
    # Manual calculation with epsilon
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_categorical, epsilon, 1.0)
    expected_loss = -np.mean(np.sum(y_true_categorical * np.log(y_pred_clipped), axis=1))
    
    assert np.isclose(loss, expected_loss)
    
    # Test derivative
    derivative = CategoricalCrossEntropy.derivative(y_true_categorical, y_pred_categorical)
    assert derivative.shape == y_true_categorical.shape
    
    # Test softmax_derivative method
    softmax_derivative = CategoricalCrossEntropy.softmax_derivative(y_true_categorical, y_pred_categorical)
    expected_derivative = (y_pred_categorical - y_true_categorical) / y_true_categorical.shape[0]
    np.testing.assert_array_almost_equal(softmax_derivative, expected_derivative)