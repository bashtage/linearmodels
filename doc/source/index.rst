Linear Model Estimation
=======================

Estimation and inference in some common linear models:

**Models for panel data**

* Fixed Effects (:class:`~linearmodels.panel.model.PanelOLS`)
* Random Effects (:class:`~linearmodels.panel.model.RandomEffects`)
* First Difference (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
* Between Estimation (:class:`~linearmodels.panel.model.BetweenOLS`)
* Pooled OLS (:class:`~linearmodels.panel.model.PooledOLS`)

**Single equation instrumental variables (IV) models**

* Two-stage least squares (:class:`~linearmodels.iv.model.IV2SLS`)
* Limited Information ML (LIML) (:class:`~linearmodels.iv.model.IVLIML`)
* Generalized Method of Moments (GMM) (:class:`~linearmodels.iv.model.IVGMM`)
* Continuously Updating GMM (:class:`~linearmodels.iv.model.IVGMMCUE`)


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
