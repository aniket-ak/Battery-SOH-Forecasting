# -*- coding: utf-8 -*-
"""
This file contains various kernels for use of GPR
"""

__author__ = "Aniket"
__copyright__ = "Aniket"
__license__ = "mit"

import numpy as np
from scipy.spatial.distance import cdist
import math


class BaseKernel:
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-1, 5e3)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds


class Linear(BaseKernel):
    def __call__(self, x1, x2):
        return self.length_scale * np.dot(x1, x2.T)


class Matern1_5(BaseKernel):
    def __call__(self, x1, x2):
        dists = cdist(x1/ self.length_scale, x2/ self.length_scale, metric='euclidean')
        K = dists * np.sqrt(3)
        K = (1. + K) * np.exp(-K)
        return K


class Matern2_5(BaseKernel):
    def __call__(self, x1, x2):
        dists = cdist(x1/ self.length_scale, x2/ self.length_scale, metric='euclidean')
        K = dists * math.sqrt(5)
        K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        return K


class Periodic(BaseKernel):
    def __init__(self, periodicity=0.5, periodicity_bounds=(1e-1, 1e5)):
        super().__init__()
        self.periodicity = periodicity
        self.periodicity_bounds = periodicity_bounds

    def __call__(self, x1, x2):
        return cdist(x1, x2,
                     lambda xi, xj: np.exp(-2 * (np.sin(np.pi * np.linalg.norm(xi - xj) / self.periodicity).T*
                                                       np.sin(np.pi * np.linalg.norm(xi - xj) / self.periodicity)) / (
                                                       self.periodicity ** 2)))

class SquaredExponential(BaseKernel):
    def __call__(self, x1, x2):
        return cdist(x1, x2, lambda xi,xj: np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * self.length_scale ** 2)))
