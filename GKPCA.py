import math
from typing import Callable
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from scipy import linalg


class GKPCA:
    _MAX_EPOCHS = int(1e4)  # max training epochs for each component
    _EPS = 1e-5  # threshold for convergence

    def __init__(
            self,
            n_components,
            generic_function: Callable[[float], float],
            kernel="linear",
            gamma=None,
            degree=3,
            coef0=1,
            kernel_params=None,
            n_jobs=None,
            inverse_rate=1.0
    ):
        """
        the generic_function should be already the derivative of the function we want to apply,
        since we are going to compute the gradient with it.
        Furthermore, the signature of the constructor is taken from sklearn.decomposition.kpca
        """
        if kernel == "precomputed":
            raise ValueError("Precomputed kernel is not supported")
        self.X_transformed_fit = None  # result of the transformed input
        self.dual_coef = None  # duality coefficients
        self.K = None  # kernel of the centered input
        self.alphas = None  # alphas of the kernel principal components
        self.scores = None  # scores of the kernel principal components
        self.means = None  # mean of the original fitted data
        self.kernel = kernel  # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’} or callable
        self.f = np.vectorize(generic_function)  # vectorized generic function
        self.n_components = n_components  # number of requested principal components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_jobs = n_jobs
        self.inverse_rate = inverse_rate

    def fit(self, X):
        if hasattr(X, "tocsr"):
            raise NotImplementedError(
                "Sparse matrices are not supported"
            )

        # compute mean by dimensions and normalize initial data set
        self.means = np.mean(X, axis=0)
        data = X - self.means
        # apply kernel
        self.K = self._get_kernel(data)
        # center with the Gram Equation in the kernel space
        one_normalized = np.ones(X.shape[0]) / X.shape[0]
        data = self.K - np.dot(one_normalized, self.K) - np.dot(self.K, one_normalized) + np.dot(one_normalized,
                                                                                                 np.dot(self.K,
                                                                                                        one_normalized))

        # weights mere shape initializations
        self.alphas = np.ones([self.n_components, X.shape[0]])
        self.scores = np.ones([self.n_components, X.shape[0]])

        # compute each component
        for c in range(self.n_components):
            old = np.zeros(X.shape[0])  # previous epoch current component weight
            # initialize weight (core values) as the normalized max norm sample in the kernel space
            diag = np.diag(data)
            opt_idx = np.where(diag == max(diag))
            core = data[opt_idx] / math.sqrt(data[opt_idx][0][opt_idx])
            for _ in range(GKPCA._MAX_EPOCHS):  # for at most _MAX_EPOCHS
                # if the current change was at least of _EPS entity
                if np.linalg.norm(core - old) < GKPCA._EPS:
                    break
                old = core.copy()  # keep track of previous epoch result
                # compute new core values with kernel gradient and generic function
                core = self.f(np.dot(core, data) / math.sqrt(np.dot(np.dot(core, data), core.transpose())))
            # compute alpha and score
            self.alphas[c] = core / math.sqrt(np.dot(np.dot(core, data), core.transpose()))
            self.scores[c] = np.dot(self.alphas[c].transpose(), data)

            # greedly remove previously found component from data
            data -= np.dot(np.vstack(np.dot(self.alphas[c].transpose(), data.transpose())),
                           np.vstack(np.dot(self.alphas[c].transpose(), data.transpose())).transpose())

    def transform(self, X):
        # apply kernel and center in kernel space
        data = self._get_kernel(X)
        n_samples = X.shape[0]
        data -= (
                np.tile(sum(data / n_samples, 2), [n_samples, 1]) +
                1 / n_samples * sum(self.K, 1) -
                ((1 / n_samples) ** 2) * np.concatenate(self.K).sum()
        )
        # initialize result and compute first principal component
        self.X_transformed_fit = np.zeros([X.shape[0], self.n_components])
        self.X_transformed_fit[:, 0] = np.dot(data, self.alphas[0])
        # compute progressively each next principal component
        for c in range(1, self.n_components):
            # greedly remove previous principal component
            data -= np.dot(np.vstack(self.X_transformed_fit[:, c - 1]), np.vstack(self.scores[c - 1]).transpose())
            self.X_transformed_fit[:, c] = np.dot(data, self.alphas[c])

        # compute duality coefficient for inverting the transformation (taken from sklearn.decomposition.kpca)
        k_result = self._get_kernel(self.X_transformed_fit)
        k_result.flat[:: n_samples + 1] += self.inverse_rate
        self.dual_coef = linalg.solve(k_result, X, sym_pos=True, overwrite_a=True)

        return self.X_transformed_fit

    def inverse_transform(self, X):
        return np.dot(self._get_kernel(X, self.X_transformed_fit), self.dual_coef)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _get_kernel(self, X, Y=None):
        """
        taken from sklearn.decomposition.kpca
        """
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )
