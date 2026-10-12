.. _panel-introduction:

Introduction
------------

Panel data includes observations on multiple entities -- individuals, firms,
countries -- over multiple time periods.  In most classical applications of
panel data the number of entities, N, is large and the number of time periods,
T, is small (often between 2 and 5).  Most asymptotic theory for these
estimators has been developed under an assumption that N will diverge while
T is fixed.

Most panel models are designed to estimate the parameters of a model which
can be described

.. math::

  y_{it}  = x_{it}\beta + \alpha_i + \epsilon_{it}

where i indexes the entities and t indexes time.  :math:`\beta` contains the
parameters of interest.  :math:`\alpha_i` are entity-specific components that
are not usually identified in the standard setup, and so cannot be
consistently estimated and :math:`\epsilon_{it}` are idiosyncratic errors
uncorrelated with :math:`\alpha_i` and the covariates :math:`x_{it}`.

All models require two inputs

* ``dependent`` - The variable to be modeled, :math:`y_{it}` in the model
* ``exog`` - The regressors, :math:`x_{it}` in the model.

and use different techniques to address the presence of :math:`\alpha_i`.

In particular,

* :class:`~linearmodels.panel.model.PanelOLS` uses fixed effect
  (i.e., entity effects) to eliminate the entity specific components.
  This is mathematically equivalent to including a dummy variable for
  each entity, although the implementation does not do this for
  performance reasons.
* :class:`~linearmodels.panel.model.BetweenOLS` averages within an
  entity and then regresses the time-averaged values using OLS.
* :class:`~linearmodels.panel.model.FirstDifferenceOLS` takes the first
  difference to eliminate the entity specific effect.
* :class:`~linearmodels.panel.model.RandomEffects` uses a quasi-difference
  to efficiently estimate :math:`\beta` when the entity effect is
  independent from the regressors.  It is, however, not consistent when
  there is dependence between the entity effect and the regressors.
* :class:`~linearmodels.panel.model.PooledOLS` ignores the entity effect
  and is consistent but inefficient when the effect is independent of the
  regressors.

:class:`~linearmodels.panel.model.PanelOLS` is somewhat more general than the
other estimators and can be used to model 2 effects (e.g., entity and time
effects).

Model specification is similar to `statsmodels <http://www.statsmodels.org/>`_.
This example estimates a fixed effect regression on a panel of the wages of working
men modeling the log wage as a function of squared experience, a dummy if the
man is married and a dummy indicating if the man is a union member.

.. code-block:: python

   from linearmodels.panel import PanelOLS
   from linearmodels.datasets import wage_panel
   import statsmodels.api as sm
   data = wage_panel.load()
   data = data.set_index(['nr','year'])
   dependent = data.lwage
   exog = sm.add_constant(data[['expersq','married','union']])
   mod = PanelOLS(dependent, exog, entity_effects=True)
   res = mod.fit(cov_type='unadjusted')
   res


While the result contains many properties containing specific quantities of
interest (e.g., ``params`` or ``tstats``), the string representation of the
result is a summary table.

::

                              PanelOLS Estimation Summary
    ================================================================================
    Dep. Variable:                  lwage   R-squared:                        0.1365
    Estimator:                   PanelOLS   R-squared (Between):             -0.0674
    No. Observations:                4360   R-squared (Within):               0.1365
    Date:                Wed, Apr 19 2017   R-squared (Overall):              0.0270
    Time:                        17:48:58   Log-likelihood                   -1439.0
    Cov. Estimator:            Unadjusted
                                            F-statistic:                      200.87
    Entities:                         545   P-value                           0.0000
    Avg Obs:                       8.0000   Distribution:                  F(3,3812)
    Min Obs:                       8.0000
    Max Obs:                       8.0000   F-statistic (robust):             200.87
                                            P-value                           0.0000
    Time periods:                       8   Distribution:                  F(3,3812)
    Avg Obs:                       545.00
    Min Obs:                       545.00
    Max Obs:                       545.00

                                 Parameter Estimates
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          1.3953     0.0123     113.50     0.0000      1.3712      1.4194
    expersq        0.0037     0.0002     19.560     0.0000      0.0033      0.0041
    married        0.1073     0.0182     5.8992     0.0000      0.0717      0.1430
    union          0.0828     0.0198     4.1864     0.0000      0.0440      0.1215
    ==============================================================================

    F-test for Poolability: 9.3360
    P-value: 0.0000
    Distribution: F(544,3812)

    Included effects: Entity


Like statsmodels, panel models can be specified using a R-like formula. This model
is identical to the previous. Note the use of the *special* variable ``EntityEffects``
to include the fixed effects.

.. code-block:: python

    mod = PanelOLS.from_formula('lwage ~ 1 + expersq + union + married + EntityEffects',data)

.. _panel-specification-tests:

Specification Tests
===================
The results of the panel models contain two specification tests. The Hausman
test compares the random effects estimates with those of a model that is
consistent if the effects are correlated with the regressors. The Pesaran CD
test examines whether the idiosyncratic shocks are correlated across entities.

Hausman Test
************
The Hausman test [Hausman78]_ examines whether the effects are uncorrelated with
the regressors, which is the assumption that random effects needs to be
consistent. If the assumption is true, the random effects estimator is
consistent and efficient, and the fixed effects estimator is consistent but not
efficient. If it is not true, only the fixed effects estimator is consistent.
The test statistic is a Wald statistic of the difference between the two sets of
estimates,

.. math::

  H = (b_0 - b_1)^{\prime} (V_0 - V_1)^{-1} (b_0 - b_1)

where :math:`b_0` and :math:`V_0` are the coefficients and covariance of the
model that is consistent whether or not the assumption is true, and :math:`b_1`
and :math:`V_1` are those of the model that is efficient if it is true. The
statistic has a :math:`\chi^2_k` distribution, where :math:`k` is the number of
coefficients that are compared.

The test is a method of the results of a random effects model,
:meth:`~linearmodels.panel.results.RandomEffectsResults.hausman`. Its argument
is the results of the other model, which is usually a fixed effects model.

.. code-block:: python

    from linearmodels.panel import RandomEffects

    fe_res = PanelOLS(dependent, exog, entity_effects=True).fit()
    re_res = RandomEffects(dependent, exog).fit()
    re_res.hausman(fe_res)

::

    Hausman specification test
    H0: No systematic difference in coefficients between models
    Statistic: 112.1182
    P-value: 0.0000
    Distributed: chi2(3)

    Coefficients (b0: consistent under H0 and H1, b1: efficient under H0)
                   b0        b1     b0-b1  Std. Err.
    expersq  0.003699  0.003192  0.000507   0.000054
    married  0.107343  0.131555 -0.024212   0.006791
    union    0.082762  0.103451 -0.020688   0.007007

The test rejects the null in this example, so there is evidence that the effects
are correlated with the regressors and that random effects is not consistent.
The test returns a
:class:`~linearmodels.panel.results.HausmanTestResults`, which is a named tuple
with the fields ``statistic`` and ``estimates``. They can be unpacked,
``statistic, estimates = re_res.hausman(fe_res)``, or read as attributes. The
statistic is a :class:`~linearmodels.shared.hypotheses.WaldTestStatistic`, and
``estimates`` is a ``DataFrame``. Its columns ``b0`` and ``b1`` contain the
coefficients of the other model and of the random effects model, and
``Std. Err.`` is the standard error of their difference.

The test has requirements, which are checked.

* Both models must use the unadjusted covariance estimator, which is the default
  in ``fit``. The variance of the difference of the estimates is the
  difference of their variances only if the random effects estimator is
  efficient, which requires homoskedastic errors. A ``TypeError`` is raised if
  either model uses another covariance estimator.
* The models must be estimated using the same observations, the same
  dependent variable and the same weights. A ``ValueError`` is raised
  otherwise. The regressors are not compared. Only the coefficients that have
  the same name in both models are tested, and the constant is not included
  unless ``include_constant=True``.
* The covariance of the difference has to be positive definite. A warning is
  issued, and a generalized inverse is used, if it is not. If the statistic is
  negative or is not finite, an
  :class:`~linearmodels.shared.hypotheses.InvalidTestStatistic` is returned
  as the statistic. This can happen if the other model is a pooled model, whose
  unadjusted covariance ignores the effects.

By default, each model's covariance is based on its own estimate of the error
variance. As in Stata's ``hausman``, ``sigmamore=True`` bases both covariances
on the estimate of the random effects model, and ``sigmaless=True`` bases both
on the estimate of the other model. They cannot be used together.

Effects that the other model has and the random effects model does not are not
tested separately. If the other model has time effects and the random effects
model does not, the null is that the regressors are uncorrelated with the entity
effects and with the time effects, and the test can reject because of the time
effects. To test the entity effects only, include dummy variables for the time
periods as regressors in the random effects model. Their coefficients are not
part of the test, since the other model does not have them.

This is not the Wu-Hausman test of the endogeneity of regressors in IV models,
:meth:`~linearmodels.iv.results.IVResults.wu_hausman`.

Pesaran CD Test
***************
The Pesaran CD test [Pesaran21]_ examines the null that the idiosyncratic shocks
are not correlated across entities. The statistic is the scaled sum of the
correlations between the shocks of each pair of entities, which are computed
using the periods that both have. It is asymptotically standard normal under
the null. The test is the property
:attr:`~linearmodels.panel.results.PanelResults.pesaran_cd`.

.. code-block:: python

    fe_res.pesaran_cd

::

    Pesaran CD test
    H0: Idiosyncratic shocks are cross-sectionally independent
    Statistic: 7.8113
    P-value: 0.0000
    Distributed: N(0,1)

The statistic is computed from the idiosyncratic shocks of the model, which are
``idiosyncratic`` in the results. The weights of a weighted model are not applied
to them. :class:`~linearmodels.panel.model.FirstDifferenceOLS` is the exception.
It uses ``resids``, the residuals of the differenced equation, as ``plm::pcdtest``
does for a regression of the differenced data. Its ``idiosyncratic`` shocks are
in levels, and they are as persistent as the errors in levels, for example a
random walk, which is when first differences are used. The correlations are
computed over time, and so the test assumes that the shocks are not serially
correlated. It over-rejects if they are persistent.

The test is not valid for every model. For the following models
``pesaran_cd`` returns an
:class:`~linearmodels.shared.hypotheses.InapplicableTestStatistic`, and its
string says why.

* :class:`~linearmodels.panel.model.BetweenOLS`. The model is fitted to the
  entity means, so it has no residuals that vary over time.
* :class:`~linearmodels.panel.model.FamaMacBeth`. The errors of the model are the
  residuals of the cross-sectional regressions of each period. The series that
  are implied by the average of the coefficients contain a common,
  time-varying term, so they are cross-sectionally dependent by construction.
  The inference of Fama-MacBeth is already robust to cross-sectional dependence.
* Models with time effects, and models with other effects that change over time
  for some entity and that have a group of several entities missing from some
  periods, for example period or industry-by-period effects. These effects
  demean the residuals across the entities that are observed in the same
  periods, which makes the residuals negatively correlated across entities and
  biases the statistic downwards [JuodisReese]_. When the errors are
  independent and the model has time effects, the statistic is centered near
  :math:`-\sqrt{T/2}` and its standard deviation is far below 1, where :math:`T`
  is the number of periods. Effects that do not change over time for an entity,
  or that are present in every period, do not have this effect.

.. code-block:: python

    te_res = PanelOLS(dependent, exog, entity_effects=True, time_effects=True).fit()
    print(te_res.pesaran_cd)

.. code-block:: text

    Irrelevant test statistic
    The Pesaran CD test is not applicable to models with time effects. These effects demean the residuals across entities that are observed in the same periods, which makes the residuals negatively correlated across entities and biases the statistic downwards, away from its null distribution (Juodis and Reese, 2022).
    Pesaran CD test

The statistic is computed from dense :math:`N` by :math:`N` matrices of the
pairwise correlations and of the numbers of periods that the pairs have in
common, where :math:`N` is the number of entities. The memory that is needed is
proportional to :math:`N^2`. It is about 450 MB when :math:`N` is 4,000.
