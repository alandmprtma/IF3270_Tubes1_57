import sys
import os
import numpy as np
import pytest

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffnn.initialization import Initializer, ZeroInitializer, RandomUniformInitializer, RandomNormalInitializer

def test_initializer_factory():
    """Test the initializer factory method"""

    assert Initializer.get_initializer('zeros') == ZeroInitializer
    assert Initializer.get_initializer('uniform') == RandomUniformInitializer
    assert Initializer.get_initializer('normal') == RandomNormalInitializer
    
    # Test case-insensitivity
    assert Initializer.get_initializer('ZEROS') == ZeroInitializer
    
    # Test invalid initializer name
    with pytest.raises(ValueError):
        Initializer.get_initializer('invalid_initializer')

def test_zero_initializer():
    """Test zero initializer"""
    shape = (3, 4)
    weights = ZeroInitializer.initialize(shape)
    
    # Check shape
    assert weights.shape == shape
    
    # Check all values are zero
    assert np.all(weights == 0)

def test_uniform_initializer():
    """Test uniform initializer"""
    shape = (100, 100)
    low = -0.1
    high = 0.1
    
    # Test with default parameters
    weights = RandomUniformInitializer.initialize(shape)
    assert weights.shape == shape
    assert low <= np.min(weights) and np.max(weights) <= high
    
    # Test with custom bounds
    custom_low = -0.5
    custom_high = 0.5
    weights = RandomUniformInitializer.initialize(shape, low=custom_low, high=custom_high)
    assert custom_low <= np.min(weights) and np.max(weights) <= custom_high
    
    # Test reproducibility with seed
    weights1 = RandomUniformInitializer.initialize(shape, seed=42)
    weights2 = RandomUniformInitializer.initialize(shape, seed=42)
    np.testing.assert_array_equal(weights1, weights2)

def test_normal_initializer():
    """Test normal initializer"""
    shape = (100, 100)
    mean = 0.0
    std = 0.05
    
    # Test with default parameters
    weights = RandomNormalInitializer.initialize(shape)
    assert weights.shape == shape
    
    # Values should be roughly centered around mean with given std
    assert -0.25 <= np.mean(weights) <= 0.25  # Loose bound for mean
    assert 0.01 <= np.std(weights) <= 0.1     # Loose bound for std
    
    # Test with custom parameters
    custom_mean = 1.0
    custom_std = 0.1
    weights = RandomNormalInitializer.initialize(shape, mean=custom_mean, std=custom_std)
    assert 0.75 <= np.mean(weights) <= 1.25   # Loose bound for mean
    assert 0.05 <= np.std(weights) <= 0.15    # Loose bound for std
    
    # Test reproducibility with seed
    weights1 = RandomNormalInitializer.initialize(shape, seed=42)
    weights2 = RandomNormalInitializer.initialize(shape, seed=42)
    np.testing.assert_array_equal(weights1, weights2)