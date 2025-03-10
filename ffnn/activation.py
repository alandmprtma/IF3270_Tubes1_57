import numpy as np

class Activation:   
    """Base Class"""
    @staticmethod
    def activate(x):
        raise NotImplementedError
        
    @staticmethod
    def derivative(x):
        raise NotImplementedError

class Linear(Activation):
    """Fungsi Linear -> Linear(x) = x"""
    @staticmethod
    def activate(x):
        return x
    
    @staticmethod
    def derivative(x):
        return np.ones_like(x)

class ReLU(Activation):
    """Fungsi ReLU -> ReLU(x) = max(0, x)"""
    @staticmethod
    def activate(x):
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    """ Fungsi Sigmoid -> sigmoid(x) = 1 / (1 + exp(-x))"""
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def derivative(x):
        s = Sigmoid.activate(x)
        return s * (1 - s)

class Tanh(Activation):
    """Fungsi Hyperbole Tangent -> tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    @staticmethod
    def activate(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x)**2

class Softmax(Activation):
    """Fungsi Softmax -> f(x_i) = exp(x_i) / sum(exp(x_j))"""
    @staticmethod
    def activate(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def derivative(x):
        # Ini masih ragu banget, soalnya turunan Softmax kompleks, biasanya diiitung pake matriks Jacobian
        # Untuk kesederhanaan, ini pake pendekatan: s * (1 - s) untuk klasifikasi biner
        s = Softmax.activate(x)
        return s * (1 - s)