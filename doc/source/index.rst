Linear Model Estimation
=======================

.. note::

  `Stable documentation <https://bashtage.github.io/linearmodels/doc/>`_ for the latest release
  is located at `doc <https://bashtage.github.io/linearmodels/doc/>`_.
  Documentation for `recent developments <https://bashtage.github.io/linearmodels/devel/>`_
  is located at `devel <https://bashtage.github.io/linearmodels/devel/>`_.

Estimation and inference in some common linear models:

**Panel Data Models**

* Fixed Effects (:class:`~linearmodels.panel.model.PanelOLS`)
* Random Effects (:class:`~linearmodels.panel.model.RandomEffects`)
* First Difference (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
* Between Estimation (:class:`~linearmodels.panel.model.BetweenOLS`)
* Pooled OLS (:class:`~linearmodels.panel.model.PooledOLS`)
* Fama-MacBeth Estimation (:class:`~linearmodels.panel.model.FamaMacBeth`)


**Single equation Instrumental Variables (IV) models**

* Two-stage least squares (:class:`~linearmodels.iv.model.IV2SLS`)
* Limited Information ML (LIML) (:class:`~linearmodels.iv.model.IVLIML`)
* Generalized Method of Moments (GMM) (:class:`~linearmodels.iv.model.IVGMM`)
* Continuously Updating GMM (:class:`~linearmodels.iv.model.IVGMMCUE`)


**System Regression Estimators**

* Seemingly Unrelated Regression (:class:`~linearmodels.system.model.SUR`)


**Asset Pricing Model Estimation and Testing**

* Linear Factor Model (2-step, for traded or non-traded factors)
  (:class:`~linearmodels.asset_pricing.model.LinearFactorModel`)
* Linear Factor Model (GMM, for traded or non-traded factors)
  (:class:`~linearmodels.asset_pricing.model.LinearFactorModelGMM`)
* Linear factor model (1-step SUR, only for traded factors)
  (:class:`~linearmodels.asset_pricing.model.TradedFactorModel`)


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :glob:

   iv/index
   panel/index
   asset-pricing/index
   system/index
   utility
   plan
   contributing
   changes
   references


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
