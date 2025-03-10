import numpy as np

class Loss:
    """Base Class"""
    @staticmethod
    def calculate(y_true, y_pred):
        raise NotImplementedError
    
    @staticmethod
    def derivative(y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    """Fungsi Mean Squeared Error"""
    @staticmethod
    def calculate(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

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
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

class CategoricalCrossEntropy(Loss):
    """Categorical Cross Entropy loss function"""
    @staticmethod
    def calculate(y_true, y_pred):
        epsilon = 1e-15 # Untuk menghindari log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def softmax_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]