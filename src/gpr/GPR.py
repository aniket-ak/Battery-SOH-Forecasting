# -*- coding: utf-8 -*-
"""
This file contains the GPR implementation
"""

import numpy as np
from scipy.optimize import minimize

__author__ = "Aniket"
__copyright__ = "Aniket"
__license__ = "mit"


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
        l = np.linalg.cholesky(y_cov)

    def _sample_multivariate_gaussian(y_mean, y_cov):
        pass
