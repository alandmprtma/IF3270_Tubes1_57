import numpy as np

class Loss:
    """Base Class"""
    @staticmethod
    def calculate(y_true, y_pred):
        raise NotImplementedError
    
    @staticmethod
    def derivative(y_true, y_pred):
        raise NotImplementedError
    
    @classmethod
    def get_loss(cls, name):
        loss_functions = {
            'mse': MSE,
            'binary_crossentropy': BinaryCrossEntropy,
            'categorical_crossentropy': CategoricalCrossEntropy
        }
        
        if name.lower() in loss_functions:
            return loss_functions[name.lower()]
        else:
            raise ValueError(f"Loss function '{name}' not supported. Choose from: {list(loss_functions.keys())}")


class MSE(Loss):
    """Fungsi Mean Squeared Error"""
    @staticmethod
    def calculate(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def derivative(y_true, y_pred):
        batch_size = y_true.shape[0]
        return 2 * (y_pred - y_true) / batch_size

class BinaryCrossEntropy(Loss):
    """Fungsi Binary Cross Entropy loss """
    @staticmethod
    def calculate(y_true, y_pred):
        epsilon = 1e-15 # Untuk menghindari log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def derivative(y_true, y_pred):
        epsilon = 1e-15 # Untuk menghindari log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        batch_size = y_true.shape[0]
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / batch_size

class CategoricalCrossEntropy(Loss):
    """Categorical Cross Entropy loss function"""
    @staticmethod
    def calculate(y_true, y_pred):
        epsilon = 1e-15 # Untuk menghindari log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def derivative(y_true, y_pred):
        batch_size = y_true.shape[0]
        epsilon = 1e-8  # Nilai kecil untuk menghindari pembagian dengan nol
        return y_true / (y_pred + epsilon) / batch_size  # Removed negative sign for gradient descent
    
    @staticmethod
    def softmax_derivative(y_true, y_pred):
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size