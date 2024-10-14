import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from sklearn import datasets, decomposition
from GPCA import GPCA

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data

# L2 x
# L1.5 math.copysign(math.sqrt(abs(x)), x)
# L1 (0 if x == 0 else x // abs(x))
# tanh math.copysign(2*np.tanh(abs(x))**2, x)
gpca = GPCA(3, lambda x: math.copysign(math.sqrt(abs(x)), x))
gpca.fit(X)

pca = decomposition.PCA(n_components=3)
pca.fit(X)

X_pca = pca.transform(X)
X_gpca = gpca.transform(X)

# reconstruction error
print(np.mean(np.linalg.norm(X - pca.inverse_transform(X_pca), axis=1)))
print(np.mean(np.linalg.norm(X - gpca.inverse_transform(X_gpca), axis=1)))


'''
# plotting take from https://scikit-learn.org/dev/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()

ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])

plt.cla()

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X_pca[y == label, 0].mean(),
        X_pca[y == label, 1].mean() + 1.5,
        X_pca[y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()
'''