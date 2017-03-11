Instrumental Variable Model Specification
-----------------------------------------

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
endogenous in the sense that :math:`Cov(x_{2i},\epsilon_i)\neq 0`. The :math:`p`
element vector :math:`z_i` are instruments that explain :math:`x_{2i}` but not
:math:`y_i`.

There are four estimation methods available to fit models of this type.  All accept the
same four required inputs:

  * ``dependent`` - The variable to be modeled, :math:`y_i` in the model
  * ``exog`` - The exogenous regressors, :math:`x_{1i}` in the model. Note
    that :math:`x_{1i}` and :math:`z_{1i}` are the same since variables are
    also valid to use when projecting the endogenous variables.
  * ``endog`` - The endogenous regressors, :math:`x_{2i}` in the model
  * ``instruments`` - The instruments, :math:`z_{2i}` in the model

..
   from urllib.request import urlopen
   import statsmodels.api as sm
   url = 'http://www.stata-press.com/data/r13/hsng.dta'
   resp = urlopen(url)
   with open('hsng.dta', 'wb') as dta:
       dta.write(resp.read())
   ivregress 2sls rent pcturban (hsngval = faminc i.region) [Equivalent]

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

                             IV-2SLS Estimation Summary
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                    0.045876
    No. Observations:                 934   Adj. R-squared:               0.043826
    Date:                Fri, Mar 10 2017   F-statistic:                    23.872
    Time:                        14:57:00   F-stat dist:                   chi2(2)
                                            F-stat p-value:                 0.0000

                                 Parameter Estimates
    ==============================================================================
               Parameters  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          4.4912    0.46921     9.5719     0.0000     [3.5716     5.4108]
    exper        0.034138   0.007318     4.6649     0.0000   [0.019795   0.048481]
    educ          0.14055   0.029018     4.8434     0.0000   [0.083672    0.19742]
    ==============================================================================

    Instrumented: educ
    Instruments: sibs
    Covariance estimator: unadjusted


.. code-block:: python

  from linearmodels.iv import IV2SLS
  mod = IV2SLS(endog, exog, instrumented, instruments)
  res = mod.fit()

The usage is similar in spirit to :class:`statsmodels.regression.linear_model.OLS`.
The estimator will reduce to OLS when ``instruments`` is equal to ``instrumented``.

.. code-block:: python

  from linearmodels.iv import IV2SLS
  ols_mod = IV2SLS(endog, exog, instrumented, instrumented)
  res = ols_mod.fit()


Two stage-least squares
=======================

Limited Information Maximum Likelihood (LIML) and k-class Estiamtion
====================================================================

Generalized Method of Moments (GMM) Estimation
==============================================

Continuously Updating GMM Estimation
====================================


