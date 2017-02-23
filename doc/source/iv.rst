Instrumental Variable Model Estimation
--------------------------------------

Instrumental variable models are used when regressors are endogenous
or there is measurement error on the variable.  These models make use of
instruments which are correlated with the endogenous variable but not
with the model error. All model estimated by this package can be described
as

.. math::

  y_i    & = x_{1i}\beta_1 + x_{2i}\beta_2 + \epsilon_i \\
  x_{2i} & = x_{1i}\delta + z_{i}\gamma + \nu_i

In this expression, :math:`x_{1i}` is a set of :math:`k_1` regressors that
are exogenous while :math:`x_{2i}` is a set of :math:`k_2` regressors that are
endogenous in the sense that :math:`Cov(x_{2i},\epsilon_i)\neq 0`. The :math:`p`
element vector :math:`z_i` are instruments that explain :math:`x_{2i}` but not
:math:`y_i`.

There are four estimation methods available to fit models of this type.  All accept the
same four required inputs:

  * ``endog`` - The variable to be modeled, :math:`y_i` in the model
  * ``exog`` - The exogenous regressors, :math:`x_{1i}` in the model
  * ``instrumented`` - The endogenous regressors, :math:`x_{2i}` in the model
  * ``instruments`` - The instruments, :math:`z_i` in the model

.. code-block:: python

  from panel.iv import IV2SLS
  mod = IV2SLS(endog, exog, instrumented, instruments)
  res = mod.fit()

The usage is similar in spirit to :class:`statsmodels.regression.linear_model.OLS`.
The estimator will reduce to OLS when ``instruments`` is equal to ``instrumented``.

.. code-block:: python

  from panel.iv import IV2SLS
  ols_mod = IV2SLS(endog, exog, instrumented, instrumented)
  res = ols_mod.fit()


Two stage-least squares
=======================

Limited Information Maximum Likelihood (LIML)
=============================================

Generalized Method of Moments (GMM) Estimation
==============================================




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   iv/models
   iv/covariance
   iv/weighting
