Linear Model Estimation
=======================

Estimation and inference in some linear models:

  * Models for panel data

    * Fixed Effects (:class:`~linearmodels.panel.model.PanelOLS`)
    * Random Effects (:class:`~linearmodels.panel.model.RandomEffects`)
    * First Difference (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
    * Between Estimation (:class:`~linearmodels.panel.model.BetweenOLS`)
    * Pooled OLS (:class:`~linearmodels.panel.model.PooledOLS`)

  * Models for estimation single equation models using instrumental variables (IV)

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
   utility
   plan


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
