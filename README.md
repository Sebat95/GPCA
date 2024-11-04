# GPCA

 

This repository contains the works related to the implementation of a generalization of principal component analysis and its kernel version.
The Generalized PCA aims at maximizing the sum of an arbitrary convex function of principal components, the current solution is a gradient ascent approach.

The findings of this work have been published by [IEEE](https://ieeexplore.ieee.org/abstract/document/9054154).
The whole algorithm is explained in depth in the article, with a compelling mathematical derivation and clear results.

If you need anymore details, there are also my thesis that extend the article:
[PoliTo thesis](https://webthesis.biblio.polito.it/13126/) and [UIC thesis](https://indigo.uic.edu/articles/thesis/A_Generalization_of_Principal_Component_Analysis/13475391?file=25863297).



## Table of Contents

 

- [Introduction](#introduction)

- [Files](#files)

- [Usage](#usage)

- [License](#license)

 

## Introduction

 

The original code was written for Matlab, it had been cleaned up and generalized neatly in two python classes.

 

## Files

 

The files are as follows:

 

* README.md: this file

* LICENSE: license file

* execute.py: a toy example of the GPCA usage (taken from [this](https://scikit-learn.org/dev/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py))

* GPCA.py: generalized principal component analysis code

* GKPCA.py: generalized kernel principal component analysis code

 

## Usage

 

The classes try to follow the [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and [sklearn.decomposition.KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) respectively, so use them as such.

 

## License

 

This repository is licensed under the [MIT License](LICENSE). Feel free to use the code and materials for academic and non-commercial purposes.
