Linear Model Estimation
=======================

Estimation and inference in some common linear models:

**Models for panel data**

* Fixed Effects (:class:`~linearmodels.panel.model.PanelOLS`)
* Random Effects (:class:`~linearmodels.panel.model.RandomEffects`)
* First Difference (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
* Between Estimation (:class:`~linearmodels.panel.model.BetweenOLS`)
* Pooled OLS (:class:`~linearmodels.panel.model.PooledOLS`)
* Fama-MacBeth Estimation (:class:`~linearmodels.panel.model.FamaMacBeth`)

**Single equation instrumental variables (IV) models**

* Two-stage least squares (:class:`~linearmodels.iv.model.IV2SLS`)
* Limited Information ML (LIML) (:class:`~linearmodels.iv.model.IVLIML`)
* Generalized Method of Moments (GMM) (:class:`~linearmodels.iv.model.IVGMM`)
* Continuously Updating GMM (:class:`~linearmodels.iv.model.IVGMMCUE`)

**Asset Pricing Model Estimation and Testing**

* Linear Factor Model (2-step, for traded or non-traded factors)
  (:class:`~linearmodels.asset_pricing.model.LinearFactorModel`)
* Linear factor model (1-step SUR, only for traded factors)
  (:class:`~linearmodels.asset_pricing.model.TradedFactorModel`)

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :glob:

   iv/index
   panel/index
   asset_pricing/index
   utility
   plan
   contributing
   changes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
