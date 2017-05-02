.. _panel-introduction:

Introduction
------------

Panel data includes observations on multiple entities -- individuals, firms,
countries -- over multiple time periods.  In most classical applications of
panel data the number of entities, N, is large and the number of time periods,
T, is small (often between 2 and 5).  Most asymptoic theory for these
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
This example rrun sa fixed effect regression on a panel of the wages of working
men modeling teh log wage as a function of squared experience, a dummy if the
man is married and a dummy indicating if the man is a union member.

.. code-block:: python

   from linearmodels.panel import PanelOLS
   from linearmodels.datasets import wage_panel
   import statsmodels.api as sm
   data = wage_panel.load()
   data = data.set_index(['nr','year'])
   dependent = data.lwage
   exog = sm.add_constant(data[['expersq','married','union']])
   mod = PanelOLS(dependent, exog, entity_effect=True)
   res = mod.fit(cov_type='unadjusted')
   res


While the result contains many properties containing specific quantities of
interest (e.g., ``params`` or ``tstata``), the string representation of the
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
is identical to the previous. Note the use of the *special* variable ``EntityEffect``
to include the fixed effects.

.. code-block:: python

    mod = PanelOLS.from_formula('lwage ~ 1 + expersq + union + married + EntityEffect',data)
