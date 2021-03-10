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
    def sample_prior(self, X_test, n_samples=1):
        y_mean = np.zeros(X_test.shape[0])
        y_cov = self.kernel(X_test)
        return self._sample_multivariate_gaussian(y_mean, y_cov, n_samples)

    def sample_posterior(self, X_train, y_train, X_test, n_samples=1):

        # compute alpha
        K = self.kernel(X_train)
        K[np.diag_indices_from(K)] += self.noise_var
        L = self._cholesky_factorise(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Compute posterior mean
        K_trans = self.kernel(X_test, X_train)
        y_mean = K_trans.dot(alpha)

        # Compute posterior covariance
        v = np.linalg.solve(L, K_trans.T)  # L.T * K_inv * K_trans.T
        y_cov = self.kernel(X_test) - np.dot(v.T, v)

        return self._sample_multivariate_gaussian(y_mean, y_cov, n_samples), y_mean, y_cov

    def log_marginal_likelihood(self, X_train, y_train, theta, noise_var=None):

        if noise_var is None:
            noise_var = self.noise_var

        # Build K(X, X)
        self.kernel.theta = theta
        K = self.kernel(X_train)
        K[np.diag_indices_from(K)] += noise_var

        # Compute L and alpha for this K (theta).
        L = self._cholesky_factorise(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Compute log marginal likelihood.
        log_likelihood = -0.5 * np.dot(y_train.T, alpha)
        log_likelihood -= np.log(np.diag(L)).sum()
        log_likelihood -= K.shape[0] / 2 * np.log(2 * np.pi)

        return log_likelihood

    def optimize(self, X_train, y_train):

        def obj_func(theta, X_train, y_train):
            return -self.log_marginal_likelihood(X_train, y_train, theta)

        results = minimize(obj_func,
                           self.kernel.theta,
                           args=(X_train, y_train),
                           method=self.optimizer,
                           jac=None,
                           bounds=self.kernel.bounds)

        # Store results of optimization.
        self.max_log_marginal_likelihood_value = -results['fun']
        self.kernel.theta_MAP = results['x']

        return results['success']

    # 'Private' helper methods
    def _cholesky_factorise(self, y_cov):
        try:
            L = np.linalg.cholesky(y_cov)
        except np.linalg.LinAlgError as e:
            e.args = ("The kernel, %s, is not returning a"
                      "positive definite matrix. Try increasing"
                      " the noise variance of the GP or using"
                      " a larger value for epsilon. "
                      % self.kernel,) + e.args
            raise
        return L

    def _sample_multivariate_gaussian(self, y_mean, y_cov, n_samples=1, epsilon=1e-10):
        y_cov[np.diag_indices_from(y_cov)] += epsilon  # for numerical stability
        L = self._cholesky_factorise(y_cov)
        u = np.random.randn(y_mean.shape[0], n_samples)
        z = np.dot(L, u) + y_mean[:, np.newaxis]
        return z
