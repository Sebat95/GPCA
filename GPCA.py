from typing import Callable

import numpy as np


class GPCA:
    _MAX_EPOCHS = int(1e4)  # max training epochs for each component
    _EPS = 1e-5   # threshold for convergence

    '''
    the function should be already the derivative of the function we want to apply,
    since we are going to compute the gradient with it
    '''
    def __init__(self, n_components: int, generic_function: Callable[[float], float]):
        self.W = None  # fitted weights
        self.Wt = None  # weights transposed
        self.means = None  # mean of the original fitted data
        self.f = np.vectorize(generic_function)  # vectorized generic function
        self.n_components = n_components  # number of requested principal components

    '''
    X is the input data set and should be a matrix MxN with M number of samples and N sample dimensions
    '''
    def fit(self, X):
        # compute mean by dimensions and normalize initial data set
        self.means = np.mean(X, axis=0)
        data = X - self.means
        # weights mere shape initializations
        self.W = np.ones([self.n_components, X.shape[1]])

        # compute each component
        for c in range(self.n_components):
            old = np.zeros(X.shape[1])  # previous epoch current component weight
            # initialize weight as the normalized max norm sample
            vec_norms = np.linalg.norm(data, axis=1)
            self.W[c] = data[np.where(vec_norms == max(vec_norms))]
            self.W[c] /= np.linalg.norm(self.W[c])
            for _ in range(GPCA._MAX_EPOCHS):  # for at most _MAX_EPOCHS
                # if the current change was at least of _EPS entity
                if np.linalg.norm(self.W[c] - old) < GPCA._EPS:
                    break
                old = self.W[c].copy()  # keep track of previous epoch result
                # compute gradient with the generic function
                grad = np.sum(np.multiply(np.vstack(self.f(np.dot(data, self.W[c]))), data), 0)
                # update weight after normalizing that
                self.W[c] = grad / np.linalg.norm(grad)
            # greedly remove previously found component from data
            data = np.dot(data, np.identity(X.shape[1]) - self.W[c] * np.vstack(self.W[c]))
        self.Wt = self.W.transpose()

    '''
    X is the input data set and should be a matrix MxN with M number of samples and N sample dimensions
    '''
    def transform(self, X):
        data = X - self.means
        return np.dot(data, self.Wt)


    def inverse_transform(self, X):
        return np.dot(X, self.W) + self.means