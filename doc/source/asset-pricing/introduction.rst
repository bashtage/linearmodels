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
    No. Factors:                          3   J-statistic:                    39.109
    No. Observations:                   819   P-value                         0.0000
    Date:                  Sun, May 21 2017   Distribution:                  chi2(3)
    Time:                          21:18:56
    Cov. Estimator:                  kernel

                                Risk Premia Estimates
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    MktRF          0.0060     0.0016     3.7381     0.0002      0.0029      0.0092
    SMB            0.0001     0.0011     0.1281     0.8980     -0.0021      0.0023
    HML            0.0045     0.0012     3.7904     0.0002      0.0022      0.0068
    ==============================================================================

    Covariance estimator:
    KernelCovariance, Kernel: bartlett, Bandwidth: 4
    See full_summary for complete results
