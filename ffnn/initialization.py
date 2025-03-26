import numpy as np

class Initializer:
    """Base class untuk initializer"""
    @staticmethod
    def initialize(shape):
        raise NotImplementedError
    
    @classmethod
    def get_initializer(cls, name):
        initializers = {
            'zeros': ZeroInitializer,
            'uniform': RandomUniformInitializer,
            'normal': RandomNormalInitializer,
            'xavier': XavierInitializer,
            'he': HeInitializer
        }
        
        if name.lower() in initializers:
            return initializers[name.lower()]
        else:
            raise ValueError(f"Initializer '{name}' not supported. Choose from: {list(initializers.keys())}")

class ZeroInitializer(Initializer):
    """Inisialisasi bobot dengan nol"""
    @staticmethod
    def initialize(shape):
        return np.zeros(shape)

class RandomUniformInitializer(Initializer):
    """Inisialisasi bobot dengan nilai acak dari distribusi uniform dengan parameter low dan upper bound"""
    @staticmethod
    def initialize(shape, low=-0.05, high=0.05, seed=None):
        # Menggunakan seed untuk memastikan reproduksibilitas hasil
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, shape)

class RandomNormalInitializer(Initializer):
    """Inisialisasi bobot dengan nilai acak dari distribusi normal dengan parameter mean dan std deviasi"""
    @staticmethod
    def initialize(shape, mean=0.0, std=0.05, seed=None):
        # Menggunakan seed untuk memastikan reproduksibilitas hasil
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std, shape)

class XavierInitializer(Initializer):
    """Xavier initialization: var(W) = 1 / n_in"""
    @staticmethod
    def initialize(shape):
        fan_in, fan_out = shape
        limit = np.sqrt(1 / fan_in)
        return np.random.uniform(-limit, limit, shape)
    
class HeInitializer(Initializer):
    """He initialization: var(W) = 2 / n_in"""
    @staticmethod
    def initialize(shape):
        fan_in, _ = shape
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)