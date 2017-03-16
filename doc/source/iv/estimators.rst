.. _iv-mathematical-notation:

Mathematical Formulas
=====================

Notation
--------

Interest is in recovering the parameter vector from the model

.. math::

   \begin{aligned}
   y_{i} & =\beta^{\prime}x_{i}+\epsilon_{i}\end{aligned}

where :math:`\beta` is :math:`k` by 1, :math:`x_{i}` is a :math:`k` by 1
vector of observable variables and :math:`\epsilon_{i}` is a scalar
error. :math:`x_{i}` can be separated in two types of variables. The
:math:`k_{1}` by 1 set of variables :math:`x_{1i}` are exogenous
regressors in the sense that :math:`E\left[x_{1i}\epsilon_{i}\right]=0`.
The :math:`k_{2}` by 1 variables :math:`x_{2i}` are endogenous. A set of
:math:`p` instruments is available that satisfy the requirements for
validity where :math:`p\geq k_{2}`. The extended model can be written as

.. math::

   \begin{aligned}
   y_{i} & =\underset{\textrm{exogenous}}{\underbrace{\beta_{1}^{\prime}x_{1i}}}+\underset{\textrm{endogenous}}{\underbrace{\beta_{2}^{\prime}x_{2i}}}+\epsilon_{i}\\
   x_{2i} & =\underset{\textrm{exogenous}}{\underbrace{\gamma_{1}^{\prime}z_{1i}}}+\underset{\textrm{instruments}}{\underbrace{\gamma_{2}^{\prime}z_{2i}}}+u_{i}\end{aligned}

The model can be expressed compactly

.. math::

   \begin{aligned}
   Y & =X_{1}\beta_{1}+X_{2}\beta_{1}+\epsilon=X\beta+\epsilon\\
   X_{2} & =Z_{1}\gamma_{1}+Z_{2}\gamma_{2}+u=Z\gamma+u\end{aligned}

The vector of instruments :math:`z_{i}` is :math:`p` by 1. There are
:math:`n` observations for all variables. :math:`k_{c}=1` if the model
contains a constant (either explicit or implicit, i.e., including all
categories of a dummy variable). The constant, if included, is in
:math:`x_{1i}`. :math:`X` is the :math:`n` by :math:`k` matrix if
regressors where row :math:`i` of :math:`X` is :math:`x_{i}^{\prime}`.
:math:`X` can be partitioned into :math:`\left[X_{1}\;X_{2}\right]`.
:math:`Z` is the :math:`n` by :math:`p` matrix of instruments. The
vector :math:`y` is :math:`n` by 1. Projection matrices for :math:`X` is
defined :math:`P_{X}=X\left(X^{\prime}X\right)^{-1}X^{\prime}`. The
projection matrix for :math:`Z` is similarly defined only using
:math:`Z`. The annihilator matrix for :math:`X` is
:math:`M_{X}=I-P_{X}`.

Parameter Estimation
--------------------

Two-stage Least Squares (2SLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2SLS estimator is

.. math::

   \begin{aligned}
   \hat{\beta}_{2SLS} & =\left(X^{\prime}P_{Z}X^{\prime}\right)\left(X^{\prime}P_{Z}y^{\prime}\right)\end{aligned}

Limited Information Maximum Likelihood and k-class Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LIML or other k-class estimator is

.. math::

   \begin{aligned}
   \hat{\beta}_{\kappa} & =\left(X^{\prime}\left(I-\kappa M_{Z}\right)X^{\prime}\right)\left(X^{\prime}\left(I-\kappa M_{Z}\right)y^{\prime}\right)\end{aligned}

where :math:`\kappa` is the parameter of the class. When
:math:`\kappa=1` the 2SLS estimator is recovered. When :math:`\kappa=0`,
the OLS estimator is recovered. The LIML estimator is recovered for
:math:`\kappa` set to

.. math:: \min\mathrm{eig\left[\left(W^{\prime}M_{z}W\right)^{-\frac{1}{2}}\left(W^{\prime}M_{X_{1}}W\right)\left(W^{\prime}M_{z}W\right)^{-\frac{1}{2}}\right]}

where :math:`W=\left[y\:X_{2}\right]` and :math:`\mathrm{eig}` returns
the eigenvalues.

Generalized Method of Moments (GMM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GMM estimator is defined as

.. math::

   \begin{aligned}
   \hat{\beta}_{GMM} & =\left(X^{\prime}ZWZ^{\prime}X\right)^{-1}\left(X^{\prime}ZWZ^{\prime}y\right)\end{aligned}

where :math:`W` is a positive definite weighting matrix.

Continuously Updating Generalized Method of Moments (GMM-CUE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuously updating GMM estimator solves the minimization problem

.. math:: \min_{\beta}n\bar{g}\left(\beta\right)^{\prime}W\left(\beta\right)^{-1}\bar{g}\left(\beta\right)

 where
:math:`\bar{g}\left(\beta\right)=n^{-1}\sum_{i=1}^{n}z_{i}\hat{\epsilon}_{i}`
and :math:`\hat{\epsilon}_{i}=y_{i}-x_{i}\beta`. Unlike standard GMM,
the weight matrix, :math:`W` directly depends on the model parameters
:math:`\beta` and so it is not possibly to use a closed form estimator.

Basic Statistics
----------------

Let :math:`\hat{\epsilon}=y-X\hat{\beta}`. The residual sum of squares
(RSS) is :math:`\hat{\epsilon}^{\prime}\hat{\epsilon}`, the model sum of
squares (MSS) is :math:`\hat{\beta}^{\prime}X^{\prime}X\hat{\beta}` and
the total sum of squares (TSS) is
:math:`\left(y-k_{c}\bar{y}\right)^{\prime}\left(y-k_{c}\bar{y}\right)^{\prime}`\ where
:math:`\bar{y}` is the sample average of :math:`y`. The model
:math:`R^{2}`\ is defined

.. math::

   \begin{aligned}
   R^{2} & =1-\frac{\hat{\epsilon}^{\prime}\hat{\epsilon}}{\left(y-k_{c}\bar{y}\right)^{\prime}\left(y-k_{c}\bar{y}\right)^{\prime}}=1-\frac{RSS}{TSS}\end{aligned}

and the adjusted :math:`R^{2}` is defined

.. math::

   \begin{aligned}
   \bar{R}^{2} & =1-\left(1-R^{2}\right)\frac{N-k_{c}}{N-k}.\end{aligned}

The residual variance is
:math:`s^{2}=n^{-1}\hat{\epsilon}^{\prime}\hat{\epsilon}` unless the
debiased flag is used, in which case a small sample adjusted version is
estimated
:math:`s^{2}=\left(n-k\right)^{-1}\hat{\epsilon}^{\prime}\hat{\epsilon}`.

The model F-statistic is defined

.. math::

   \begin{aligned}
   F & =\hat{\beta}_{-}^{\prime}\hat{V}_{-}^{-1}\dot{\hat{\beta}_{-}}\end{aligned}

where the notation :math:`\hat{\beta}_{-}` indicates that the constant
is excluded from :math:`\hat{\beta}` and :math:`\hat{V}_{-}` indicates
that the row and column corresponding to a constant are excluded. [1]_
The test statistic is distributed as :math:`\chi_{k-k_{c}}^{2}.` If the
debiased flag is set, then the test statistic :math:`F` is transformed
as :math:`F/\left(k-k_{c}\right)` and a :math:`F_{k-k_{c},n-k}`
distribution is used.

Parameter Covariance Estimation
-------------------------------

Two-stage LS, LIML and k-class estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four covariance estimators are available. The first is the standard
homoskedastic covariance, defined as

.. math::

   \begin{aligned}
   n^{-1}s^{2}\left(\frac{X^{\prime}\left(I-\kappa M_{z}\right)X}{n}\right)^{-1} & =n^{-1}s^{2}\hat{A}.\end{aligned}

Note that this estimator can be expressed as

.. math::

   \begin{aligned}
   n^{-1}\hat{A}^{-1}\left\{ s^{2}\hat{A}\right\} \hat{A}^{-1} & =n^{-1}\hat{A}^{-1}\hat{B}\hat{A}^{-1}.\end{aligned}

All estimators take this form and only differ in how the asymptotic
covariance of the scores, :math:`B`, is estimated. For the homoskedastic
covariance estimator, :math:`\hat{B}=s^{2}\hat{A}.` The score covariance
in the heteroskedasticity robust covariance estimator is

.. math::

   \begin{aligned}
   \hat{B} & =n^{-1}\sum_{i=1}^{n}\hat{\epsilon}_{i}^{2}\hat{x}_{i}\hat{x}_{i}^{\prime}=n^{-1}\sum_{i=1}^{n}\hat{\xi}_{i}\hat{\xi}_{i}^{\prime}.\end{aligned}

where :math:`\hat{x_{i}}` is row :math:`i` of :math:`\hat{X}=P_{Z}X` and
:math:`\hat{\xi}_{i}=\hat{\epsilon}_{i}\hat{x}_{i}`.

The kernel covariance estimator is robust to both heteroskedasticity and
autocorrelation and is defined as

.. math::

   \begin{aligned}
   \hat{B} & =\hat{\Gamma}_{0}+\sum_{i=1}^{n-1}k\left(i/h\right)\left(\hat{\Gamma}_{i}+\hat{\Gamma}_{i}^{\prime}\right)\\
   \hat{\Gamma_{j}} & =n^{-1}\sum_{i=j+1}^{n}\hat{\xi}_{i-j}\hat{\xi}_{i}^{\prime}\end{aligned}

where :math:`k\left(i/h\right)` is a kernel weighting function where
:math:`h` is the kernel bandwidth.

The one-way clustered covariance estimator is defined as

.. math::

   \begin{aligned}
   n^{-1}\sum_{j=1}^{g}\left(\sum_{i\in\mathcal{G}_{j}}\hat{\xi}_{i}\right)\left(\sum_{i\in\mathcal{G}_{j}}\hat{\xi}_{i}\right)^{\prime}\end{aligned}

where :math:`\sum_{i\in\mathcal{G}_{j}}\hat{\xi}_{i}` is the sum of the
scores for all members in group :math:`\mathcal{G}_{j}` and :math:`g` is
the number of groups.

If the debiased flag is used to perform a small-sample adjustment, all
estimators except the clustered covariance are rescaled by
:math:`\left(n-k\right)/n`. The clustered covariance is rescaled by
:math:`\left(\left(n-k\right)\left(n-1\right)/n^{2}\right)\left(\left(g-1\right)/g\right)`. [2]_

P-values
~~~~~~~~

P-values are computes using 2-sided tests,

.. math:: Pr\left(\left|z\right|>Z\right)=2-2\Phi\left(\left|z\right|\right)

If the covariance estimator was debiased, a Student’s t distribution
with :math:`n-k` degrees of freedom is used,

.. math::

   \begin{aligned}
   Pr\left(\left|z\right|>Z\right) & =2-2t_{n-k}\left(\left|z\right|\right)\end{aligned}

where :math:`t_{n-k}\left(\cdot\right)` is the CDF of a Student’s T
distribution.

Confidence Intervals 
~~~~~~~~~~~~~~~~~~~~~

Confidence intervals are constructed as

.. math:: CI_{i,1-\alpha}=\hat{\beta}_{i}\pm q_{\alpha/2}\times\hat{\sigma}_{\beta_{i}}

where :math:`q_{\alpha/2}` is the :math:`\alpha/2` quantile of a
standard Normal distribution or a Student’s t. The Student’s t is used
when a debiased covariance estimator is used.

To Do
-----

Gmm Covariance

Constant detection

k-class special cases

J-stat

C-stat

Kernel weights

Optimal BW selection

Std err, T-stats

All df’s

s2??

Linear hypothesis testing

sargan

basman

wu haussman

wooldridge score

wooldridge regression

wooldridge overid

anderson rubin

basmann f

First Stage Results -> partial r2, shea r2, f-stat

.. [1]
   If the model contains an implicit constant, e.g., all categories of a
   dummy, one of the categories is excluded when computing the test
   statistic. The choice of category to drop has no effect and is
   equivalent to reparameterizing the model with a constant and
   excluding one category of dummy.

.. [2]
   This somewhat non-obvious choice is drive by Stata compatibility.
