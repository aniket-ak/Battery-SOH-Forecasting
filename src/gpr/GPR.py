# -*- coding: utf-8 -*-
"""
This file contains the GPR implementation
"""

import argparse
import logging
import sys

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform

from gpr import __version__

__author__ = "Aniket"
__copyright__ = "Aniket"
__license__ = "mit"

_logger = logging.getLogger(__name__)


class GPR:

    def __init__(self, kernel, optimizer='L-BFGS-B', noise_var=1e-8):
        self.kernel = kernel
        self.noise_var = noise_var
        self.optimizer = optimizer

    # 'Public' methods
    def sample_prior(self, X_test, n_samples):
        pass

    def sample_posterior(self, X_test, n_samples):
        pass

    def log_marginal_likelihood(self, theta=None, eval_gradient=None):
        pass

    def optimize(self, theta, X_train, y_train):
        pass

    # 'Private' helper methods
    def _cholesky_factorise(y_cov):
        pass

    def _sample_multivariate_gaussian(y_mean, y_cov):
        pass


class Kernel:
    def __init__(self, signal_variance=1.0, signal_variance_bounds=(1e-5, 1e5)):
        self.theta = [signal_variance]
        self.bounds = [signal_variance_bounds]
    def __call__(self, x1, x2=None):
        self.k = self.theta[0] * np.dot(x1, x1.T)
        return self.k