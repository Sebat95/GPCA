from typing import Callable
import numpy as np


class GPCA:
    _MAX_EPOCHS = int(1e4)  # Max training epochs for each component.
    _EPS = 1e-5  # Threshold for convergence.

    def __init__(self, n_components: int, generic_function: Callable[[float], float]):
        """
        The function should be already the derivative of the function we want to apply,
        since we are going to compute the gradient with it.
        """
        self.W = None  # Fitted weights.
        self.Wt = None  # Weights transposed.
        self.means = None  # Mean of the original fitted data.
        self.f = np.vectorize(generic_function)  # Vectorized generic function.
        self.n_components = n_components  # Number of requested principal components.

    def fit(self, X):
        # Compute mean by dimensions and normalize initial data set.
        self.means = np.mean(X, axis=0)
        data = X - self.means
        # Weights mere shape initializations.
        self.W = np.ones([self.n_components, X.shape[1]])

        # Compute each component.
        for c in range(self.n_components):
            old = np.zeros(X.shape[1])  # Previous epoch current component weight.
            # Initialize weight as the normalized max norm sample.
            vec_norms = np.linalg.norm(data, axis=1)
            self.W[c] = data[np.where(vec_norms == max(vec_norms))]
            self.W[c] /= np.linalg.norm(self.W[c])
            for _ in range(GPCA._MAX_EPOCHS):  # for at most _MAX_EPOCHS.
                # If the current change was at least of _EPS entity.
                if np.linalg.norm(self.W[c] - old) < GPCA._EPS:
                    break
                old = self.W[c].copy()  # Keep track of previous epoch result.
                # Compute gradient with the generic function.
                grad = np.sum(np.multiply(np.vstack(self.f(np.dot(data, self.W[c]))), data), 0)
                # Update weight after normalizing that.
                self.W[c] = grad / np.linalg.norm(grad)
            # Greedly remove previously found component from data.
            data = np.dot(data, np.identity(X.shape[1]) - self.W[c] * np.vstack(self.W[c]))
        self.Wt = self.W.transpose()

    def transform(self, X):
        data = X - self.means
        return np.dot(data, self.Wt)

    def inverse_transform(self, X):
        return np.dot(X, self.W) + self.means

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
