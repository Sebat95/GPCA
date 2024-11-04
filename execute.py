import math
import numpy as np
from sklearn import datasets, decomposition
from GPCA import GPCA
from GKPCA import GKPCA
from mpmath import sech


iris = datasets.load_iris()
X = iris.data

'''
GPCA
'''
# L2: x
# L1.5: math.copysign(math.sqrt(abs(x)), x)
# L1: (0 if x == 0 else x // abs(x))
# 2tanh^2: math.copysign(2*np.tanh(abs(x))**2, x)
# 3-3sech: math.copysign((-3*(sech(abs(a))-1), x)
gpca = GPCA(3, lambda x: x)
gpca.fit(X)

pca = decomposition.PCA(n_components=3)
pca.fit(X)

X_pca = pca.transform(X)
X_gpca = gpca.transform(X)

# Reconstruction error.
print(np.mean(np.linalg.norm(X - pca.inverse_transform(X_pca), axis=1)))
print(np.mean(np.linalg.norm(X - gpca.inverse_transform(X_gpca), axis=1)))

'''
GKPCA
'''
kpca = decomposition.KernelPCA(n_components=3, fit_inverse_transform=True)
kpca.fit(X)

kgpca = GKPCA(3, lambda x: x)
kgpca.fit(X)

X_kpca = kpca.transform(X)
X_gkpca = kgpca.transform(X)

# Reconstruction error.
print(np.mean(np.linalg.norm(X - kpca.inverse_transform(X_kpca), axis=1)))
print(np.mean(np.linalg.norm(X - kgpca.inverse_transform(X_gkpca), axis=1)))
