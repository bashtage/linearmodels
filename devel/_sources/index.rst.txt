.. image:: images/logo-text.svg
   :width: 50%
   :align: left
   :alt: linearmodels logo

.. note::

  `Stable documentation <https://bashtage.github.io/linearmodels/>`_ for the latest release
  is located at `doc <https://bashtage.github.io/linearmodels/>`_.
  Documentation for `recent developments <https://bashtage.github.io/linearmodels/devel/>`_
  is located at `devel <https://bashtage.github.io/linearmodels/devel/>`_.

Estimation and inference in some common linear models that are missing
from `statsmodels <https://statsmodels.org>`_:

**Panel Data Models**

* Fixed Effects (:class:`~linearmodels.panel.model.PanelOLS`)
* Random Effects (:class:`~linearmodels.panel.model.RandomEffects`)
* First Difference (:class:`~linearmodels.panel.model.FirstDifferenceOLS`)
* Between Estimation (:class:`~linearmodels.panel.model.BetweenOLS`)
* Pooled OLS (:class:`~linearmodels.panel.model.PooledOLS`)
* Fama-MacBeth Estimation (:class:`~linearmodels.panel.model.FamaMacBeth`)


**High-dimensional Regression**

* Absorbing Least Squares (:class:`~linearmodels.iv.absorbing.AbsorbingLS`)


**Single equation Instrumental Variables (IV) models**

* Two-stage least squares (2SLS, :class:`~linearmodels.iv.model.IV2SLS`)
* Limited Information ML (LIML, :class:`~linearmodels.iv.model.IVLIML`)
* Generalized Method of Moments (GMM, :class:`~linearmodels.iv.model.IVGMM`)
* Continuously Updating GMM (CUE-GMM, :class:`~linearmodels.iv.model.IVGMMCUE`)


**System Regression Estimators**

* Seemingly Unrelated Regression (SUR, :class:`~linearmodels.system.model.SUR`)
* Three-stage Least Squares (3SLS, :class:`~linearmodels.system.model.IV3SLS`)
* Generalized Method of Moments System Estimator (GMM, :class:`~linearmodels.system.model.IVSystemGMM`)


**Asset Pricing Model Estimation and Testing**

* Linear Factor Model (2-step, for traded or non-traded factors)
  (:class:`~linearmodels.asset_pricing.model.LinearFactorModel`)
* Linear Factor Model (GMM, for traded or non-traded factors)
  (:class:`~linearmodels.asset_pricing.model.LinearFactorModelGMM`)
* Linear factor model (1-step SUR, only for traded factors)
  (:class:`~linearmodels.asset_pricing.model.TradedFactorModel`)

linearmodels
============

.. toctree::
   :maxdepth: 1
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


Indices
=======

* :ref:`genindex`
* :ref:`modindex`
