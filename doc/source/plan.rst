Module Plans
------------
The plan for this modules is to add some key missing linear models.  This plan is intentionally
limited since there is no intention to statsmodels.

Univariate
==========

*  OLS - :class:`statsmodels.regression.linear_model.OLS`
*  WLS - :class:`statsmodels.regression.linear_model.WLS`
*  GLS - :class:`statsmodels.regression.linear_model.GLS`
* 2SLS - :class:`linearmodels.iv.model.IV2SLS`
* LIML/k-class - :class:`linearmodels.iv.model.IVLIML`
* GMM IV Estimation - :class:`linearmodels.iv.model.IVGMM`
* GMM-CUE Estimation - :class:`linearmodels.iv.model.IVGMMCUE`

In Progress
***********
* 2-step Linear Asset Pricing Model Estimation - :class:`linearmodels.asset_pricing.model.LinearFactorModel`
* Linear Asset Pricing Model Estimation for Traded Factors - :class:`linearmodels.asset_pricing.model.TradedFactorModel`


Panel
=====

Least Squares Estimators
************************

* Fixed Effects - :class:`linearmodels.panel.model.PanelOLS`
* Random Effects - :class:`linearmodels.panel.model.RandomEffects`
* Pooled - :class:`linearmodels.panel.model.PooledOLS`
* Between - :class:`linearmodels.panel.model.BetweenOLS`
* First Difference - :class:`linearmodels.panel.model.FirstDifferenceOLS`
* Fama-MacBeth - :class:`linearmodels.panel.model.FamaMacBeth`


Instrumental Variable Estimators
********************************

Not Implemented
~~~~~~~~~~~~~~~
* 2SLS
* GMM