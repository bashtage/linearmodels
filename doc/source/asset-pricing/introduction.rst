Introduction
------------

There are three estimators for linear factor asset pricing models:

* :class:`~linearmodels.asset_pricing.model.TradedFactorModel` implements
  an estimator which is appropriate when all factors are traded assets. When
  this is the case, the model can be estimated using regressions as a SUR.
* :class:`~linearmodels.asset_pricing.model.LinearFactorModel` implements
  a general purpose estimator based on the 2-step strategy where the first
  step estimated the factor loadings and the second step estimates the risk
  premia using the estimates from the first step.  This model is appropriate
  for both traded and non-traded factors.
* :class:`~linearmodels.asset_pricing.model.LinearFactorModelGMM` implements
  a version of the 2-step model using GMM.  The GMM estimator is relatively
  efficient and so should be preferred. The continuously updating version can
  is available using an options when fitting the model (``use_cue=True``).

All estimators implement both standard heteroskedasticity robust inference (the
default) as well as kernel-based HAC estimators using eights based on the
Bartlett kernel (Newey-West), the Parzen kernel or the Quadratic-Spectral kernel.

The basic usage is the same for all three estimators.  Two inputs are required:

* ``portfolios`` - The test portfolios.  A T by P array of portfolio returns.
* ``factors`` - The priced factors. A T by K array of factor returns or shocks.

This example makes use of some data from
`Ken French's data library <http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>`_.
The factors are the market, the size factor and the value factor.  Sis test
portfolios are used with both small and large firm size (S1 and S5) and low
and high value (V1 and V5).  The portfolios are transformed into excess returns
prior to estimation.

.. code-block:: python

  from linearmodels.datasets import french
  data = french.load()
  factors = data[['MktRF', 'SMB', 'HML']]
  portfolios = data[['S1V1','S1V3','S1V5','S5V1','S5V3','S5V5']].copy()
  portfolios.loc[:,:] = portfolios.values - data[['RF']].values
  from linearmodels.asset_pricing import LinearFactorModel
  mod = LinearFactorModel(portfolios, factors)
  res = mod.fit(cov_type='kernel')
  print(res)

::

                          LinearFactorModel Estimation Summary
    ================================================================================
    No. Test Portfolios:                  6   R-squared:                      0.8879
    No. Factors:                          3   J-statistic:                    30.694
    No. Observations:                   819   P-value                         0.0000
    Date:                  Thu, May 18 2017   Distribution:                  chi2(3)
    Time:                          22:23:54
    Cov. Estimator:                  kernel

                                Risk Premia Estimates
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    MktRF          0.0060     0.0113     0.5305     0.5958     -0.0162      0.0282
    SMB            0.0001     0.0081     0.0178     0.9858     -0.0157      0.0160
    HML            0.0045     0.0032     1.4041     0.1603     -0.0018      0.0108
    ==============================================================================

    See full_summary for complete results