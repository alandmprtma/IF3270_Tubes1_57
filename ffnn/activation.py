import numpy as np

class Activation:
    """Basis kelas untuk fungsi aktivasi"""
    @staticmethod
    def activate(x):
        """Menghitung fungsi aktivasi"""
        raise NotImplementedError
        
    @staticmethod
    def derivative(x):
        """Menghitung turunan fungsi aktivasi"""
        raise NotImplementedError
    
    @classmethod
    def get_activation(cls, name):
        """Pabrik untuk memproduksi fungsi aktivasi berdasarkan input nama"""
        activations = {
            'linear': Linear,
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'softmax': Softmax
        }
        if name.lower() in activations:
            return activations[name.lower()]
        else:
            raise ValueError(f"Activation function '{name}' not supported. Choose from: {list(activations.keys())}")

class Linear(Activation):
    """fungsi aktivasi Linear : f(x) = x"""
    @staticmethod
    def activate(x):
        return x
    
    @staticmethod
    def derivative(x):
        return np.ones_like(x)

class ReLU(Activation):
    """fungsi aktivasi ReLU: f(x) = max(0, x)"""
    @staticmethod
    def activate(x):
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)

class Sigmoid(Activation):
    """Fungsi aktivasi sigmoid: f(x) = 1 / (1 + exp(-x))"""
    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def derivative(x):
        s = Sigmoid.activate(x)
        return s * (1 - s)

class Tanh(Activation):
    """Fungsi aktivasi hyperbolic tangent: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    @staticmethod
    def activate(x):
        return np.tanh(x)
    
    @staticmethod
    def derivative(x):
        return 1 - np.tanh(x)**2

class Softmax(Activation):
    """Fungsi aktivasi softmax: f(x_i) = exp(x_i) / sum(exp(x_j))"""
    @staticmethod
    def activate(x):
        # Membatasi nilai input untuk menghindari underflow/overflow
        x = np.clip(x, -1e10, 1e10)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def derivative(x):
        """
        Catatan : Turunan softmax adalah matriks Jacobian untuk setiap vektor input
        Implementasi ini disederhanakan untuk digunakan dengan fungsi loss categorical cross-entropy
        di mana turunan softmax yang digabungkan dengan categorical cross-entropy
        disederhanakan menjadi (y_pred - y_true)
        """
        return Softmax.activate(x)