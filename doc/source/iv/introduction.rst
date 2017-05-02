.. _iv-introduction:

Introduction
------------

Instrumental variable models are used when regressors are endogenous
or there is measurement error on the variable.  These models make use of
instruments which are correlated with the endogenous variable but not
with the model error. All model estimated by this package can be described
as

.. math::

  y_i    & = x_{1i}\beta_1 + x_{2i}\beta_2 + \epsilon_i \\
  x_{2i} & = z_{1i}\delta + z_{2i}\gamma + \nu_i

In this expression, :math:`x_{1i}` is a set of :math:`k_1` regressors that
are exogenous while :math:`x_{2i}` is a set of :math:`k_2` regressors that are
endogenous in the sense that :math:`Cov(x_{2i},\epsilon_i)\neq 0`. In total
there are The :math:`k` regressors in the model. The :math:`p_2` element
vector :math:`z_{2i}` are instruments that explain :math:`x_{2i}` but not
:math:`y_i`.  Note that :math:`x_{1i}` and :math:`z_{1i}` are the same since
variables are also valid to use when projecting the endogenous variables.
In total there are :math:`p=p_1+p_2=k_1+p_2` variables available to use when
projecting the endogenous regressors.

There are four estimation methods available to fit models of this type.  All accept the
same four required inputs:

* ``dependent`` - The variable to be modeled, :math:`y_i` in the model
* ``exog`` - The exogenous regressors, :math:`x_{1i}` in the model. Note
  that :math:`x_{1i}` and :math:`z_{1i}` are the same since variables are
  also valid to use when projecting the endogenous variables.
* ``endog`` - The endogenous regressors, :math:`x_{2i}` in the model
* ``instruments`` - The instruments, :math:`z_{2i}` in the model


.. code-block:: python

   import pandas as pd
   import numpy as np
   import statsmodels.api as sm
   from linearmodels.iv import IV2SLS
   from linearmodels.datasets import wage
   data = wage.load()
   dependent = np.log(data.wage)
   exog = sm.add_constant(data.exper)
   endog = data.educ
   instruments = data.sibs

   mod = IV2SLS(dependent, exog, endog, instruments)
   res = mod.fit(cov_type='unadjusted')
   res

::

                              IV-2SLS Estimation Summary
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                      0.0459
    Estimator:                    IV-2SLS   Adj. R-squared:                 0.0438
    No. Observations:                 934   F-statistic:                    23.872
    Date:                Mon, Mar 13 2017   P-value (F-stat)                0.0000
    Time:                        14:52:30   Distribution:                  chi2(2)
    Cov. Estimator:            unadjusted

                                 Parameter Estimates
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          4.4912     0.4692     9.5719     0.0000      3.5716      5.4108
    exper          0.0341     0.0073     4.6649     0.0000      0.0198      0.0485
    educ           0.1405     0.0290     4.8434     0.0000      0.0837      0.1974
    ==============================================================================

    Endogenous: educ
    Instruments: sibs
    Unadjusted Covariance (Homoskedastic)
    Debiased: False

Estimators
==========
Four methods to estimate models are available.

* Two-stage least squares (2SLS) :class:`~linearmodels.iv.model.IV2SLS`
* Limited Information Maximum Likelihood (LIML) and related k-class
  estimators :class:`~linearmodels.iv.model.IVLIML`
* Generalized Method of Moments (GMM) :class:`~linearmodels.iv.model.IVGMM`
* Generalized Method of Moments using the Continuously Updating Estimator
  (CUE) :class:`~linearmodels.iv.model.IVGMMCUE`

All estimator require the same four key inputs, ``dependent``, ``exog`` ,
``endog``  and ``instruments``. In addition to these four required
parameters, optional arguments can be used to alter the default configuration.

Optional Arguments
******************

2SLS Estimation
^^^^^^^^^^^^^^^
The 2SLS estimator is the simplest and has no optional arguments. The 2SLS
estimator nests OLS and so it is possible to estimate models using OLS by
specifying both ``endog`` and ``instruments`` as ``None``.

.. code-block:: python

   mod = IV2SLS(dependent, exog, None, None)
   ols_res = mod.fit()

LIML Estimation
^^^^^^^^^^^^^^^
Two optional arguments can be used to alter the estimation method when using IVLIML

* ``fuller`` allows Fuller's :math:`\alpha` to be specified, which provides a
  finite sample correction to the usual LIML estimator.
* ``kappa`` allows a user-specified value of :math:`\kappa` to be provided in
  which case the LIML estimated value of :math:`\kappa` is ignored.

GMM and GMM-CUE Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``weight_type`` accepts a string which indicates the type of weighting
  matrix to use in the GMM estimation proceedure.  There are four classes
  if weighting matrices available:

  * 'unadjusted' - Assumes the GMM moment conditions are homoskedastic. See
    :class:`~linearmodels.iv.gmm.HomoskedasticWeightMatrix`.
  * 'robust' - Allows the GMM moment conditions to be heteroskedastic while
    assuming they are not correlated across observations. See
    :class:`~linearmodels.iv.gmm.HeteroskedasticWeightMatrix`.
  * 'kernel' - Allows for both heteroskedasticity and autocorrrelation in the
    moment conditions. See :class:`~linearmodels.iv.gmm.KernelWeightMatrix`.
  * 'cluster' - Allows for a one- and two-way cluster structure where moment
    conditions within a cluster are correlated.
    See :class:`~linearmodels.iv.gmm.ClusteredWeightMatrix`.

  Each weight type accepts a set of additional parameters which are similar to
  those for the corresponding covariance estimator.

Model Estimation and Covariance Specification
=============================================
All models are estimated using teh ``fit`` method which provides an
opportunity to customize the parameter covariance estimator used to
perform inference. Four classes of covariance estimators are available:

* 'unadjusted' - Assumes the model scores are homoskedastic. See
  :class:`~linearmodels.iv.covariance.HomoskedasticCovariance`.

* 'robust', 'heteroskedastic' - Allows the model scores to be heteroskedastic
  while assuming they are not correlated across observations. See
  :class:`~linearmodels.iv.covariance.HeteroskedasticCovariance`.

* 'kernel' - Allows for both heteroskedasticity and autocorrrelation in the
  model scores. The estimator allows the ``kernel`` to be selected from

  * 'bartlett', 'newey-west` - Triangular kernel utilized in the common
    Newey-West estimator.
  * 'parzen' - Parzen's kernel.
  * 'qs', 'quadratic-spectral' - The quadratic spectral kernel studied by
    Andrews.

  The ``bandwidth`` can also be specified.  If not provided, an estimate of
  the optimal value is used.

  See :class:`~linearmodels.iv.covariance.KernelCovariance`.

* 'clustered', 'one-way' - Allows for a one-way cluster structure where model
  scores within a cluster are correlated.
  See :class:`~linearmodels.iv.covariance.ClusteredCovariance`. Using
  clustered covariance requires passing an array containing information
  containing cluster membership information.

.. code-block:: python

   mod = IV2SLS(dependent, exog, endog, instruments)
   iq_bands = data.IQ // 20
   res = mod.fit(cov_type='clustered', clusters=iq_bands)

GMM Estimation
**************
GMM allows additional inputs that affect the method of estimation. In
particular, the default is to use two-step GMM.  One-step (inefficient)
GMM can be forced by setting ``iter_limit`` to 1.  If ``iter_limit`` is
raised above 2, then an iterative method is used where multiple steps
are used to estimate the model parameters.  If normalized model parameters
change by less than ``tol`` across successive iterations, then the estimation
is assumed to converge and the iterations are stopped.

By default, the first-step uses teh average outer-product of the instruments
as the weighting matrix.  ``initial_weight`` allows a user-specified choice of
weighting matrix to be used instead.

GMM-CUE Estimation
******************
GMM CUE uses a non-linear optimizer to optimize the GMM objective directly
where both the moment condition and the moment score estimator change with
parameter values. ``starting`` allows a user-specified set of starting values
to be used in-place of the default starting values and ``display`` controls
whether iterative output is printed during estimation.