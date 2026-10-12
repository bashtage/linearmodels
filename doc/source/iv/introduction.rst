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

There are five estimation methods available to fit models of this type.  All accept the
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
Five methods to estimate models are available.

* Two-stage least squares (2SLS) :class:`~linearmodels.iv.model.IV2SLS`
* Limited Information Maximum Likelihood (LIML) and related k-class
  estimators :class:`~linearmodels.iv.model.IVLIML`
* Generalized Method of Moments (GMM) :class:`~linearmodels.iv.model.IVGMM`
* Generalized Method of Moments using the Continuously Updating Estimator
  (CUE) :class:`~linearmodels.iv.model.IVGMMCUE`
* Jackknife IV Estimator (JIVE) :class:`~linearmodels.iv.model.IVJIVE`

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

JIVE Estimation
^^^^^^^^^^^^^^^
The JIVE estimator has no options that change how the parameters are
estimated. It uses first-stage predictions that leave out each observation as
the instruments, which reduces the bias of 2SLS when there are many
instruments. See :ref:`iv-mathematical-notation` for the definition and for the
caveats about inference when there are many instruments. The covariance
estimator is chosen using ``cov_type`` in ``fit``. The 'unadjusted', 'robust'
and 'kernel' estimators are available as for the other estimators. As for
:class:`~linearmodels.iv.model.IV2SLS` and
:class:`~linearmodels.iv.model.IVLIML`, the bandwidth of the kernel estimator
is selected using the data if ``bandwidth`` is not provided. The 'clustered'
estimator supports one-way clustering only, and a ``ValueError`` is raised if
``clusters`` has two columns.

Since JIVE is not a k-class estimator, ``kappa`` is ``None`` in its results.

GMM and GMM-CUE Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``weight_type`` accepts a string which indicates the type of weighting
  matrix to use in the GMM estimation procedure.  There are four classes
  of weighting matrices available:

  * 'unadjusted' - Assumes the GMM moment conditions are homoskedastic. See
    :class:`~linearmodels.iv.gmm.HomoskedasticWeightMatrix`.
  * 'robust' - Allows the GMM moment conditions to be heteroskedastic while
    assuming they are not correlated across observations. See
    :class:`~linearmodels.iv.gmm.HeteroskedasticWeightMatrix`.
  * 'kernel' - Allows for both heteroskedasticity and autocorrelation in the
    moment conditions. See :class:`~linearmodels.iv.gmm.KernelWeightMatrix`.
  * 'clustered', 'one-way' - Allows for a one-way cluster structure where
    moment conditions within a cluster are correlated. Two-way clustering is
    not available. See
    :class:`~linearmodels.iv.gmm.OneWayClusteredWeightMatrix`.

  Each weight type accepts a set of additional parameters which are similar to
  those for the corresponding covariance estimator.

**Kernel bandwidth.** If ``weight_type`` is 'kernel' and ``bandwidth`` is not
provided, the bandwidth of the kernel weight matrix is estimated from the data,
unless ``optimal_bw=False``, as the bandwidth of the kernel covariance of
:class:`~linearmodels.iv.model.IV2SLS` and
:class:`~linearmodels.iv.model.IVLIML` is. The same is true of the kernel
covariance that ``fit`` uses if ``cov_type`` is 'kernel'. See
:ref:`iv-mathematical-notation` for the method. The bandwidth that was used is
in ``weight_config`` and ``cov_config`` of the results.

.. versionchanged:: 7.1

   The default bandwidth of the kernel estimators of
   :class:`~linearmodels.iv.model.IVGMM` and
   :class:`~linearmodels.iv.model.IVGMMCUE` was ``nobs - 2`` in 7.0 and
   earlier. That is not a consistent estimator of the covariance of the
   moment conditions, and it gave standard errors that were too small. The
   bandwidth is now estimated from the data, and a model that uses this default
   issues a ``DefaultBandwidthWarning``. The warning says how to restore the
   previous bandwidth: pass ``optimal_bw=False``, or ``bandwidth=nobs - 2``, in
   the weight configuration for the weight matrix and to ``fit`` for the
   covariance. Passing ``bandwidth`` or ``optimal_bw`` silences the warning, and
   ``optimal_bw=True`` selects the bandwidth from the data without it.

.. code-block:: python

   from linearmodels.iv import IVGMM

   # The bandwidth of linearmodels 7.0 and earlier
   mod = IVGMM(dependent, exog, endog, instruments, weight_type="kernel",
               optimal_bw=False)
   res = mod.fit(cov_type="kernel", optimal_bw=False)

Model Estimation and Covariance Specification
=============================================
All models are estimated using the ``fit`` method which provides an
opportunity to customize the parameter covariance estimator used to
perform inference. Four classes of covariance estimators are available:

* 'unadjusted' - Assumes the model scores are homoskedastic. See
  :class:`~linearmodels.iv.covariance.HomoskedasticCovariance`.

* 'robust', 'heteroskedastic' - Allows the model scores to be heteroskedastic
  while assuming they are not correlated across observations. See
  :class:`~linearmodels.iv.covariance.HeteroskedasticCovariance`.

* 'kernel' - Allows for both heteroskedasticity and autocorrelation in the
  model scores. The estimator allows the ``kernel`` to be selected from

  * 'bartlett', 'newey-west' - Triangular kernel utilized in the common
    Newey-West estimator.
  * 'parzen' - Parzen's kernel.
  * 'qs', 'quadratic-spectral' - The quadratic spectral kernel studied by
    Andrews.

  The ``bandwidth`` can also be specified.  If not provided, an estimate of
  the optimal value is used. This is the case for all of the estimators. It
  has been the case for :class:`~linearmodels.iv.model.IVGMM` and
  :class:`~linearmodels.iv.model.IVGMMCUE` since 7.1, see the kernel
  bandwidth in `GMM and GMM-CUE Estimation`_.

  See :class:`~linearmodels.iv.covariance.KernelCovariance`.

* 'clustered', 'one-way' - Allows for a one-way cluster structure where model
  scores within a cluster are correlated.
  See :class:`~linearmodels.iv.covariance.ClusteredCovariance`. Using
  clustered covariance requires passing an array containing cluster
  membership information, ``clusters``.
  :class:`~linearmodels.iv.model.IV2SLS` and
  :class:`~linearmodels.iv.model.IVLIML` also support two-way clustering, which
  uses a ``clusters`` array with two columns. The other estimators,
  :class:`~linearmodels.iv.model.IVGMM`,
  :class:`~linearmodels.iv.model.IVGMMCUE` and
  :class:`~linearmodels.iv.model.IVJIVE`, support one-way clustering only.

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

By default, the first-step uses the average outer-product of the instruments
as the weighting matrix.  ``initial_weight`` allows a user-specified choice of
weighting matrix to be used instead.

GMM-CUE Estimation
******************
GMM CUE uses a non-linear optimizer to optimize the GMM objective directly
where both the moment condition and the moment score estimator change with
parameter values. ``starting`` allows a user-specified set of starting values
to be used in-place of the default starting values and ``display`` controls
whether iterative output is printed during estimation.

.. _iv-testing-identification:

Testing Identification
======================
The results of every IV estimator have a ``first_stage`` attribute that
contains tests of whether the instruments identify the model. The model is
identified when the matrix of coefficients on the excluded instruments in the
first-stage regressions has full column rank, so that there are at least as
many informative instruments as there are endogenous regressors.

.. code-block:: python

   res = IV2SLS(dependent, exog, endog, instruments).fit(cov_type="robust")
   first_stage = res.first_stage
   first_stage.kleibergen_paap     # rk LM test of underidentification
   first_stage.kleibergen_paap_f   # rk Wald F statistic
   first_stage.cragg_donald        # Cragg-Donald test of underidentification
   first_stage.cragg_donald_f      # Cragg-Donald Wald F statistic

The four statistics are also available as functions, which can be imported from
``linearmodels.iv``. They take 2-d arrays of the endogenous regressors, the
excluded instruments and the exogenous regressors. In a weighted model the
arrays have to be multiplied by the square roots of the weights.

.. code-block:: python

   from linearmodels.iv import (
       cragg_donald,
       cragg_donald_f,
       kleibergen_paap,
       kleibergen_paap_f,
   )

   endog_array = endog.to_frame().to_numpy()
   instruments_array = instruments.to_frame().to_numpy()
   exog_array = exog.to_numpy()
   cragg_donald(endog_array, instruments_array, exog_array)

The null hypothesis of the two tests is that the first-stage coefficient matrix
is rank deficient, so that the model is **underidentified**. The tests have a
:math:`\chi^2_{p_2 - k_2 + 1}` distribution, where :math:`p_2` is the number of
excluded instruments and :math:`k_2` is the number of endogenous regressors.
When there are no endogenous regressors, or when the statistics are not
defined, for example if the endogenous regressors are collinear, the tests are
reported as an :class:`~linearmodels.shared.hypotheses.InvalidTestStatistic`.

* :func:`~linearmodels.iv.common.cragg_donald` requires conditionally
  homoskedastic and serially uncorrelated errors. It does **not** use the
  covariance estimator that was used to fit the model.
* :func:`~linearmodels.iv.common.kleibergen_paap` uses the covariance estimator
  that was used to fit the model: unadjusted, robust, one-way clustered or
  kernel. With the unadjusted covariance it is Anderson's canonical correlation
  LM statistic. The test follows ``ranktest`` and Stata's ``ivreg2``
  "Kleibergen-Paap rk LM statistic".

The F statistics, :func:`~linearmodels.iv.common.cragg_donald_f` and
:func:`~linearmodels.iv.common.kleibergen_paap_f`, are the statistics that are
commonly compared with weak-instrument critical values. They are the
statistics that ``ivreg2`` reports as the "Cragg-Donald Wald F statistic" and
the "Kleibergen-Paap rk Wald F statistic". They do not have p-values.

.. warning::

   These statistics are easy to over-interpret.

   * The null is that the model is underidentified. Rejecting it does **not**
     show that the instruments are strong. The test rejects with probability
     near one when the instruments are weak enough for 2SLS to be badly biased
     and its tests to be badly sized. These are not tests of weak instruments.
   * Stock-Yogo critical values apply only to the Cragg-Donald F statistic
     when the errors are conditionally homoskedastic and serially
     uncorrelated. They are not valid for the Kleibergen-Paap F statistic when
     the errors are not i.i.d. and comparing them is only a heuristic. For one
     endogenous regressor, the effective F statistic of Olea and Pflueger
     (2013) is better founded. It is not implemented, and neither are
     weak-instrument robust confidence sets.
   * The Cragg-Donald statistic ignores heteroskedasticity, clustering and
     serial correlation. The Kleibergen-Paap statistic is only as robust as
     the covariance estimator used for it. If the dependence in the data is not
     modeled, both can reject far too often when the instruments and
     first-stage errors are clustered or persistent. With few clusters the
     clustered test can be conservative, and a kernel bandwidth that is too
     short does not remove the over-rejection.
   * The tests are asymptotic. The Kleibergen-Paap statistic uses a covariance
     matrix with :math:`p_2 k_2` rows, so it needs :math:`n` to be large relative to
     :math:`p_2 k_2` and many clusters when clustering. A warning is issued if
     the covariance is rank deficient, and the result should not be relied
     upon.
   * Using the outcome of a test to decide whether to proceed with IV
     estimation distorts the inference that follows.
   * The Bartlett kernel bandwidth here is one less than the ``bw()`` that
     ``ivreg2`` uses for the same weights.

.. _iv-specification-tests:

Specification Tests
===================
The results of :class:`~linearmodels.iv.model.IV2SLS` and
:class:`~linearmodels.iv.model.IVLIML` contain tests of whether the endogenous
regressors are exogenous and of the overidentifying restrictions. The results
of :class:`~linearmodels.iv.model.IVGMM` and
:class:`~linearmodels.iv.model.IVGMMCUE` contain the J-statistic and the
C-statistic. A model that uses weights is tested using its weights. The
definitions are in :ref:`iv-mathematical-notation`.

The tests differ in what they assume about the errors, and in whether they use
the covariance estimator that was used to fit the model.

Tests of whether the endogenous regressors are exogenous:

* :meth:`~linearmodels.iv.results.IVResults.durbin` and
  :meth:`~linearmodels.iv.results.IVResults.wu_hausman` assume that the errors
  are homoskedastic. They do not use ``cov_type``, so they are the same for
  every covariance estimator.
* :attr:`~linearmodels.iv.results.IVResults.wooldridge_score` is robust to
  heteroskedasticity. It also does not use ``cov_type``.
* :attr:`~linearmodels.iv.results.IVResults.wooldridge_regression` is a Wald
  test that uses the covariance estimator that was used to fit the model, and so
  it is as robust as that estimator is.
* :meth:`~linearmodels.iv.results.IVGMMResults.c_stat` is available from the GMM
  estimators. It uses the estimator of the covariance of the moment conditions
  that is selected by ``weight_type``.

Tests of the overidentifying restrictions, which require more instruments than
endogenous regressors:

* :attr:`~linearmodels.iv.results.IVResults.sargan`,
  :attr:`~linearmodels.iv.results.IVResults.basmann`,
  :attr:`~linearmodels.iv.results.IVResults.anderson_rubin` and
  :attr:`~linearmodels.iv.results.IVResults.basmann_f` assume that the errors
  are homoskedastic, and do not use ``cov_type``.
* :attr:`~linearmodels.iv.results.IVResults.wooldridge_overid` is robust to
  heteroskedasticity. It also does not use ``cov_type``.
* :attr:`~linearmodels.iv.results.IVGMMResults.j_stat` and
  :attr:`~linearmodels.iv.results.IVGMMResults.centered_j_stat` are available
  from the GMM estimators. They use the estimator of the covariance of the
  moment conditions that is selected by ``weight_type``. In iterated GMM the
  J-statistic cannot exceed the number of observations, however large the
  violation of the moment conditions is, and the centered J-statistic can.

.. code-block:: python

   res = IV2SLS(dependent, exog, endog, instruments).fit(cov_type="robust")
   res.wu_hausman()                # classical, does not use cov_type
   res.wooldridge_regression       # uses the covariance estimator of the fit
   res.wooldridge_score            # robust to heteroskedasticity
   res.sargan                      # classical, does not use cov_type
   res.wooldridge_overid           # robust to heteroskedasticity

The ``f.stat`` column of ``res.first_stage.diagnostics`` is the Wald test that
the coefficients on the excluded instruments are zero in the first-stage
regression. If ``cov_type`` is 'unadjusted' it is divided by the number of
excluded instruments, and its p-value uses an F distribution. It is the
textbook F-statistic only if ``debiased`` is ``True``, and it is :math:`n/(n-k)`
times larger if it is not, where :math:`k` is the number of exogenous regressors
and excluded instruments. For all other covariance estimators it is the Wald
statistic itself, which has a :math:`\chi^2` distribution. Software that reports
an F-statistic for a robust test, for example ``ivreg`` in R, reports the Wald
statistic divided by the number of excluded instruments.

.. _iv-absorbing-ls:

Absorbing Least Squares
=======================
:class:`~linearmodels.iv.absorbing.AbsorbingLS` estimates linear models with
high-dimensional effects, for example the dummy variables of individuals, firms
or years, without constructing the dummy variables. It absorbs any number of
categorical variables, as well as the interactions of categorical variables
with continuous variables, which are known as fixed slopes. It is modeled on
Stata's ``areg`` and ``reghdfe``. It does not have endogenous regressors. It can
be imported from ``linearmodels.iv`` and from the top-level package.

``AbsorbingLS`` is in the IV package for historical reasons. It is built on the
IV machinery, so that it shares the covariance estimators and the results
classes with the IV estimators, and it reports OLS-style statistics. The
``rsquared`` of the results includes the explanatory power of the absorbed
effects, and ``absorbed_rsquared`` does not. It is not a panel estimator, and it
does not have the statistics of
:class:`~linearmodels.panel.model.PanelOLS`. Its default for ``debiased`` is
``False``, where the panel estimators use ``True``. See
:ref:`panel-small-sample` for the small-sample adjustments of the panel
estimators.

A rule of thumb is to use :class:`~linearmodels.panel.model.PanelOLS` for up to
two effects when you want panel-style output and inference that matches Stata's
``xtreg``, and to use ``AbsorbingLS`` for more than two effects, for effects
interacted with continuous variables, and for models with very many groups.

.. _iv-many-dummies:

Models with Many Dummy Variables
================================
Do not construct the dummy variables of a high-dimensional effect and pass them
to an IV estimator as exogenous regressors. The estimators convert their inputs
to ``float64`` and keep several dense copies of the regressors and the
instruments, so that fitting a model needs many times the memory of the data.
A model with thousands of dummy variables can exhaust the memory of a computer
even if the dummy variables are stored in a smaller type.

Instead, remove the effects from the dependent variable, the exogenous and
endogenous regressors and the instruments, which is known as partialling out,
and fit the IV model to the transformed variables. By the Frisch-Waugh-Lovell
theorem the parameter estimates of :class:`~linearmodels.iv.model.IV2SLS` are
the same as those of the model that includes the dummy variables. If there is a
single set of effects, the transformation subtracts the means of the groups, for
example using ``groupby(...).transform("mean")``.
:class:`~linearmodels.iv.absorbing.AbsorbingLS` removes any number of effects, and
effects interacted with continuous variables, without constructing the dummy
variables. After ``fit``, the transformed variables are ``absorbed_dependent``
and ``absorbed_exog``. Pass all of the variables other than the dependent
variable as ``exog`` when the model is created.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from linearmodels.iv import AbsorbingLS, IV2SLS

   rng = np.random.default_rng(0)
   nobs, ngroups = 20000, 2000
   group = rng.integers(0, ngroups, nobs)
   effect = rng.standard_normal(ngroups)[group]
   z = pd.DataFrame(rng.standard_normal((nobs, 2)), columns=["z1", "z2"])
   v = rng.standard_normal(nobs)
   w = pd.Series(rng.standard_normal(nobs) + effect, name="w")
   x = pd.Series(z["z1"] + z["z2"] + effect + v, name="x")
   y = pd.Series(x + w + effect + v + rng.standard_normal(nobs), name="y")

   absorb = pd.DataFrame({"group": pd.Categorical(group)})
   variables = pd.concat([w, x, z], axis=1)
   absorbing = AbsorbingLS(y, variables, absorb=absorb)
   absorbing.fit()
   y_tilde = absorbing.absorbed_dependent
   x_tilde = absorbing.absorbed_exog
   mod = IV2SLS(y_tilde, x_tilde[["w"]], x_tilde[["x"]], x_tilde[["z1", "z2"]])
   res = mod.fit(cov_type="robust")

The unadjusted, robust and clustered covariances of the two models are the same
if ``debiased`` is ``False``. If it is ``True``, the model that includes the
dummy variables counts them in its degrees of freedom, so the standard errors of
the transformed model have to be multiplied by
:math:`\sqrt{(n-k)/(n-k-d)}`. Here :math:`n` is the number of observations,
:math:`k` is the number of regressors in the transformed model, and :math:`d` is
the number of dummy variables that were removed, not counting the redundant ones.

Aligning Inputs
===============
The rows of the inputs of the IV estimators are paired by position. An input
that carries labels, such as a pandas ``Series`` or ``DataFrame`` with an index,
must have the same labels, in the same order, as ``dependent``. See
:ref:`input-alignment` for the rule, and for the other estimators.
