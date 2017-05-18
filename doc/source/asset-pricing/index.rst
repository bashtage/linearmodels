Linear Factor Models for Asset Pricing
--------------------------------------

Factor models are commonly used to test whether a set of factors, for example
the market (CAP-M) or the Fama-French 3 (Market, Size and Value) can explain
the return in a set of test portfolio.  The economic hypothesis tested is that

.. math::

    E[r_{it}] = \lambda_0 + \beta_i \lambda

where :math:`\beta_i` is a set of factor loadings and :math:`\lambda` are the
risk premia associated with the factors. :math:`\lambda_0` is the risk-free
rate.

When all of the factors are traded, a simple Seemingly Unrelated Regression
(SUR) approach can be used to test whether the factors price the test
portfolios.  This is possible since the expected average return on a factor
must equal the factor's risk premium.
:class:`~linearmodels.asset_pricing.model.TradedFactorModel` implements this
model and allows the specification to be testes with a J-statistic of the form

.. math::

    J = \hat{\alpha}^{\prime} \hat{\Sigma}_{\alpha} \hat{\alpha} \sim \chi^2_P

where there are P portfolios.

When some factors are not-traded, for example macroeconomic shocks, then a 2-step
procedure is required since the average value of the factors does not
necessarily equal the risk premium associated with the factor. The first step
estimates the factor exposures (effectively) using demeaned factors, and the second
step estimates the risk premia using the factor exposures as regressors so
that the estimated risk premia attempt to price the cross-section of assets as
well as possible.  A similar J-statistic can be used to test the model, although it
has a different distribution,

.. math::

    J = \hat{\alpha}^{\prime} \hat{\Sigma}_{\alpha} \hat{\alpha} \sim \chi^2_{P-K}

where K is the number of factors.
:class:`~linearmodels.asset_pricing.model.LinearFactorModel` implements this
two-step estimator.
:class:`~linearmodels.asset_pricing.model.LinearFactorModelGMM` contains a more
efficient estimator of the same model using over-identified GMM.

Both estimators can be used with either a heteroskedasticity-robust covariance
by setting ``cov_type='robust'`` or a HAC covariance estimator using
``cov_type='kernel'``.  The HAC estimator supports three kernel weight generators,
'bartlett` or `newey-west` which uses a triangular kernel, 'parzen' and 'qs'
(or 'quadratic-spectral').

:ref:`asset_pricing-mathematical-notation` contains a concise explanation of
the formulas used in estimating parameters, estimating covariances and
implementing hypothesis tests.


.. toctree::
   :maxdepth: 1
   :glob:

   introduction
   examples/examples.ipynb
   examples/formulas.ipynb
   reference
   mathematical-formula
