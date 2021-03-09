import numpy as np


class BaseKernel:
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-1, 1e3)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds


class Linear(BaseKernel):
    def __call__(self, x1, x2):
        return self.length_scale * np.dot(x1, x2.T)


class Matern1_5(BaseKernel):
    def __call__(self, x1, x2):
        return (1 + np.sqrt(3) / self.length_scale) * np.exp(-np.sqrt(3) / self.length_scale * np.linalg.norm(x1 - x2))


class Matern2_5(BaseKernel):
    def __call__(self, x1, x2):
        return (1 + np.sqrt(5) / self.length_scale * np.linalg.norm(x1 - x2) +
                5 / (3 * self.length_scale) * np.linalg.norm(x1 - x2) * np.linalg.norm(x1 - x2)) * np.exp(
            -np.sqrt(5) / self.length_scale * np.linalg.norm(x1 - x2))


class Periodic(BaseKernel):
    def __init__(self, periodicity=0.5, periodicity_bounds=(1e-1, 1e5)):
        super().__init__()
        self.periodicity = periodicity
        self.periodicity_bounds = periodicity_bounds

    def __call__(self, x1, x2):
        return np.exp(-2 * np.sin(np.pi * np.linalg.norm(x1 - x2) / self.periodicity) * np.sin(
            np.pi * np.linalg.norm(x1 - x2) / self.periodicity) / (self.length_scale ** 2))


class SquaredExponential(BaseKernel):
    def __call__(self, x1, x2):
        return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * self.length_scale ** 2))