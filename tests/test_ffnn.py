# test_ffnn_model.py
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ffnn.model import FFNN

# Ensure output directory exists
os.makedirs('test_outputs', exist_ok=True)

def get_regression_data(n_features=3):
    """Generate regression data with specified feature count"""
    X, y = make_regression(n_samples=100, n_features=n_features, noise=0.1, random_state=42)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def get_classification_data():
    """Generate classification data"""
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, n_informative=3, random_state=42)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    return X_train, X_test, y_train, y_test

def test_model_creation():
    """Test model initialization and architecture"""
    print("\n=== Testing Model Creation ===")
    
    # Create a basic regression model
    model = FFNN(loss='mse')
    model.add(input_size=3, output_size=4, activation='relu', 
             weight_initializer='normal', mean=0, std=0.1)
    model.add(input_size=4, output_size=2, activation='linear',
             weight_initializer='normal', mean=0, std=0.1)
    
    # Check model structure
    assert len(model.layers) == 2
    assert model.layer_sizes == [3, 4, 2]
    assert model.layers[0].input_size == 3
    assert model.layers[0].output_size == 4
    assert model.layers[1].input_size == 4
    assert model.layers[1].output_size == 2
    
    # Create a classification model
    model2 = FFNN(loss='categorical_crossentropy')
    model2.add(input_size=4, output_size=5, activation='tanh')
    model2.add(input_size=5, output_size=3, activation='softmax')
    
    # Check model structure
    assert len(model2.layers) == 2
    assert model2.layer_sizes == [4, 5, 3]
    
    print("✅ Model creation test passed")

def test_forward_propagation():
    """Test forward pass with different activations"""
    print("\n=== Testing Forward Propagation ===")
    
    # Create a model with different activation functions
    model = FFNN(loss='mse')
    model.add(input_size=3, output_size=2, activation='relu')
    model.add(input_size=2, output_size=2, activation='sigmoid')
    model.add(input_size=2, output_size=1, activation='linear')
    
    # Create sample data
    X = np.array([
        [0.5, -0.1, 0.3],
        [-0.2, 0.7, 0.1],
        [0.1, -0.5, 0.2]
    ])
    
    # Forward pass
    output = model.forward(X)
    
    # Verify output shape and values
    assert output.shape == (3, 1)
    assert isinstance(output, np.ndarray)
    assert not np.any(np.isnan(output))
    
    # Test classification model with softmax
    model_cls = FFNN(loss='categorical_crossentropy')
    model_cls.add(input_size=3, output_size=3, activation='relu')
    model_cls.add(input_size=3, output_size=2, activation='softmax')
    
    output_cls = model_cls.forward(X)
    
    # Verify softmax properties
    assert output_cls.shape == (3, 2)
    assert np.all(output_cls >= 0) and np.all(output_cls <= 1)
    assert np.allclose(np.sum(output_cls, axis=1), 1.0)
    
    print("✅ Forward propagation test passed")

def test_backward_propagation():
    """Test backward propagation and gradient calculation"""
    print("\n=== Testing Backward Propagation ===")
    
    # Create a simple model
    model = FFNN(loss='mse')
    model.add(input_size=3, output_size=2, activation='relu')
    model.add(input_size=2, output_size=1, activation='linear')
    
    # Create sample data
    X = np.array([
        [0.5, -0.1, 0.3],
        [-0.2, 0.7, 0.1],
        [0.1, -0.5, 0.2]
    ])
    y = np.array([[0.1], [0.2], [0.3]])
    
    # Forward pass
    y_pred = model.forward(X)
    
    # Backward pass
    model.backward(y, y_pred)
    
    # Verify gradients exist and have correct shapes
    assert model.layers[0].dW is not None
    assert model.layers[0].db is not None
    assert model.layers[1].dW is not None
    assert model.layers[1].db is not None
    
    assert model.layers[0].dW.shape == (3, 2)
    assert model.layers[0].db.shape == (1, 2)
    assert model.layers[1].dW.shape == (2, 1)
    assert model.layers[1].db.shape == (1, 1)
    
    # Test weight updates
    initial_W0 = model.layers[0].W.copy()
    learning_rate = 0.1
    model.update_weights(learning_rate)
    
    # Verify