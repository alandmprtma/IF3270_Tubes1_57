import numpy as np

class Initializer:
    """Base class untuk initializer"""
    @staticmethod
    def initialize(shape):
        raise NotImplementedError

class ZeroInitializer(Initializer):
    """Inisialisasi bobot dengan nol"""
    @staticmethod
    def initialize(shape):
        return np.zeros(shape)

class RandomUniformInitializer(Initializer):
    """Inisialisasi bobot dengan nilai acak dari distribusi uniform dengan parameter low dan upper bound"""
    @staticmethod
    def initialize(shape, low=-0.05, high=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, shape)

class RandomNormalInitializer(Initializer):
    """Inisialisasi bobot dengan nilai acak dari distribusi normal dengan parameter mean dan std deviasi"""
    @staticmethod
    def initialize(shape, mean=0.0, std=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std, shape)