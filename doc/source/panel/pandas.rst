.. _panel-pandas-differences:

Comparison with pandas PanelOLS and FamaMacBeth
===============================================

pandas deprecated ``PanelOLS`` (``pandas.stats.plm.PanelOLS``) and
``FamaMacBeth`` (``pandas.stats.plm.FamaMacBeth``)  in 0.18 and
dropped it in 0.20.  :class:`linearmodels.panel.model.PanelOLS`
and :class:`linearmodels.panel.model.FamaMacBeth`  provide
a similar set of functionality with a few notable differences:

1. When using a MultiIndex DataFrame, this package expects the MultiIndex
   to be of the form entity-time. pandas used time-entity.  It is simple to
   transform one to the other using the one-liner.

.. code-block:: python

    data = data.reset_index().set_index(['entity','time'])


2. Effects are implemented in ``linearmodels`` using differencing
   and so even very large models (100000 entities +) can be quickly
   estimated. The version in pandas used LSDV which is not feasible in
   large models and can be slow in moderately large model.

3. Effects are not explicitly estimated nor are they reported in model summaries.
   Effects are usually not consistent (e.g., entity effects in a large-N
   panel) and so it does not usually make sense to report them with parameter
   estimates.  Effects that can be consistent estimated can be included as
   dummies (e.g. time effects in a model with fixed T but large-N).

4. R-squared definitions differ.  The default R-squared in ``linearmodels``
   reports the fit after included effects are removed.
   :class:`~linearmodels.panel.model.PanelOLS` also provides a set of R-squared measures
   that measure the fit of the model parameters using alternative models.
   These are only meaningful for models that only include entity effects.
   An R-squared measure that is defined similarly to the R-squared in pandas
   is available in the :attr:`~linearmodels.panel.results.PanelEffectsResults.rsquared_inclusive`
   property.

5. Models are first specified then explicitly fit using the :func:`~linearmodels.panel.model.PanelOLS.fit`
   method. Fit options, such as the choice of covariance estimator, are
   provided when fitting the model.

6. The intercept must be explicitly included if desired.  If a model with effects
   is estimated without an intercept, then all effects are included.  If a
   model is estimated with an intercept, the estimated model is demeaned to
   using the restriction that the sum of the effects is 0 so that the intercept
   is meaningful.

7. Other statistics, such as F-stats, differ since inconsistent effects are
   not included in the test statistic.


Here the difference is presented using the canonical Grunfeld data on
investment.


.. code-block:: python

    import numpy as np
    from statsmodels.datasets import grunfeld
    data = grunfeld.load_pandas().data
    data.year = data.year.astype(np.int64)
    from linearmodels import PanelOLS
    etdata = data.set_index(['firm','year'])
    PanelOLS(etdata.invest,etdata[['value','capital']],entity_effect=True).fit(debiased=True)


::

                              PanelOLS Estimation Summary
    ================================================================================
    Dep. Variable:                 invest   R-squared:                        0.7667
    Estimator:                   PanelOLS   R-squared (Between):              0.8223
    No. Observations:                 220   R-squared (Within):               0.7667
    Date:                Mon, Apr 17 2017   R-squared (Overall):              0.8132
    Time:                        12:21:30   Log-likelihood                   -1167.4
    Cov. Estimator:            Unadjusted
                                            F-statistic:                      340.08
    Entities:                          11   P-value                           0.0000
    Avg Obs:                       20.000   Distribution:                   F(2,207)
    Min Obs:                       20.000
    Max Obs:                       20.000   F-statistic (robust):             340.08
                                            P-value                           0.0000
    Time periods:                      20   Distribution:                   F(2,207)
    Avg Obs:                       11.000
    Min Obs:                       11.000
    Max Obs:                       11.000

                                 Parameter Estimates
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    value          0.1101     0.0113     9.7461     0.0000      0.0879      0.1324
    capital        0.3100     0.0165     18.744     0.0000      0.2774      0.3426
    ==============================================================================

    F-test for Poolability: 50.838
    P-value: 0.0000
    Distribution: F(11,207)

    Included effects: Entity
    PanelEffectsResults, id: 0x2aeec70b7f0


The call to the deprecated pandas PanelOLS is similar. Note the use of the
time-entity data format.

.. code-block:: python

    tedata = data.set_index(['year','firm'])
    from pandas.stats import plm
    plm.PanelOLS(tedata['invest'],tedata[['value','capital']],entity_effects=True)


The output format is quite different.

::

    -------------------------Summary of Regression Analysis-------------------------

    Formula: Y ~ <value> + <capital> + <FE_b'Atlantic Refining'> + <FE_b'Chrysler'>
                 + <FE_b'Diamond Match'> + <FE_b'General Electric'>
                 + <FE_b'General Motors'> + <FE_b'Goodyear'> + <FE_b'IBM'> + <FE_b'US Steel'>
                 + <FE_b'Union Oil'> + <FE_b'Westinghouse'> + <intercept>

    Number of Observations:         220
    Number of Degrees of Freedom:   13

    R-squared:         0.9461
    Adj R-squared:     0.9429

    Rmse:             50.2995

    F-stat (12, 207):   302.6388, p-value:     0.0000

    Degrees of Freedom: model 12, resid 207

    -----------------------Summary of Estimated Coefficients------------------------
          Variable       Coef    Std Err     t-stat    p-value    CI 2.5%   CI 97.5%
    --------------------------------------------------------------------------------
             value     0.1101     0.0113       9.75     0.0000     0.0880     0.1323
           capital     0.3100     0.0165      18.74     0.0000     0.2776     0.3425
    FE_b'Atlantic Refining'   -94.0243    17.1637      -5.48     0.0000  -127.6652   -60.3834
    FE_b'Chrysler'    -7.2309    17.3382      -0.42     0.6771   -41.2138    26.7520
    FE_b'Diamond Match'    14.0102    15.9436       0.88     0.3806   -17.2393    45.2596
    --------------------------------------------------------------------------------
    FE_b'General Electric'  -214.9912    25.4613      -8.44     0.0000  -264.8953  -165.0871
    FE_b'General Motors'   -49.7209    48.2801      -1.03     0.3043  -144.3498    44.9080
    FE_b'Goodyear'   -66.6363    16.3788      -4.07     0.0001   -98.7389   -34.5338
         FE_b'IBM'    -2.5820    16.3792      -0.16     0.8749   -34.6852    29.5212
    FE_b'US Steel'   122.4829    25.9595       4.72     0.0000    71.6023   173.3636
    --------------------------------------------------------------------------------
    FE_b'Union Oil'   -45.9660    16.3575      -2.81     0.0054   -78.0267   -13.9054
    FE_b'Westinghouse'   -36.9683    17.3092      -2.14     0.0339   -70.8942    -3.0424
         intercept   -20.5782    11.2978      -1.82     0.0700   -42.7219     1.5655
    ---------------------------------End of Summary---------------------------------