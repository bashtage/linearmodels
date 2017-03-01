=====================
Mathematical Formulas
=====================

Notation
========

Interest is in recovering the parameter vector from the model

.. math:: y_{i}=x_{i}\beta+\epsilon_{i}

The regressors :math:`x_{i}` are :math:`k` by 1 and :math:`\beta` is
:math:`k` by 1. The regressors :math:`x_{i}` can be separated in two
types of regressors, :math:`x_{1i}` which is :math:`k_{1}` by :math:`1`
and :math:`x_{2i}` which is :math:`k_{2}` by 1. :math:`x_{1i}` are
exogenous regressors in the sense that
:math:`E\left[x_{1i}\epsilon_{i}\right]=0`. :math:`x_{2i}` are
endogenous regressors. A set of :math:`p` instruments is available that
satisfy the requirements for validity where :math:`p\geq k_{2}`. The
extended model can be written as

.. math::

   \begin{aligned}
   y_{i} & =\beta_{1}^{\prime}x_{1i}+\beta_{2}^{\prime}x_{2i}+\epsilon_{i}\\
   x_{2i} & =\gamma_{1}^{\prime}x_{1i}+\gamma_{2}^{\prime}z_{i}+u_{i}\end{aligned}

:math:`z_{i}` is :math:`p` by 1. There are :math:`n` observations for
all variables. :math:`k_{c}=1` if the model contains a constant (either
explicit or implicit, i.e., including all dummy variables). The
constant, if included, is in :math:`x_{1i}`. :math:`X` is the :math:`n`
by :math:`k` matrix if regressors where row :math:`i` of :math:`X` is
:math:`x_{i}^{\prime}`. :math:`X` can be partitioned into
:math:`\left[X_{1}\;X_{2}\right]`. :math:`Z` is the :math:`n` by
:math:`p` matrix of instruments. The vector :math:`y` is :math:`n` by 1.
Projection matrices for :math:`X` is defined
:math:`P_{X}=X\left(X^{\prime}X\right)^{-1}X^{\prime}`. The projection
matrix for :math:`Z` is similarly defined only using :math:`Z`. The
annihilator matrix for :math:`X` is :math:`M_{X}=I-P_{X}`.

Parameter Estimation
====================

Two-stage Least Squares (2SLS)
------------------------------

The 2SLS estimator is

.. math:: \hat{\beta}_{2SLS}=\left(X^{\prime}P_{Z}X^{\prime}\right)\left(X^{\prime}P_{Z}y^{\prime}\right)

Limited Information Maximum Likelihood and k-class Estimators
-------------------------------------------------------------

The LIML or other k-class estimator is

.. math:: \hat{\beta}_{\kappa}=\left(X^{\prime}\left(I-\kappa M_{Z}\right)X^{\prime}\right)\left(X^{\prime}\left(I-\kappa M_{Z}\right)y^{\prime}\right)

where :math:`\kappa` is the parameter of the class. When :math:`\kappa=1` the 2SLS estimator is recovered. When :math:`\kappa=0`,
the OLS estimator is recovered. The LIML estimator is recovered for
:math:`\kappa` set to TODO

Generalized Method of Moments (GMM)
-----------------------------------

The GMM estimator is defined as

.. math:: \hat{\beta}_{GMM}=\left(X^{\prime}ZWZ^{\prime}X\right)^{-1}\left(X^{\prime}ZWZ^{\prime}y\right)

.. This is a comment

where :math:`W` is a positive definite weighting matrix.

Variance Estimation
===================

.. math:: n^{-1}s^{2}\Sigma_{xx}^{-1}

or

.. math:: \left(n-k\right)^{-1}s^{2}\Sigma_{xx}^{-1}
